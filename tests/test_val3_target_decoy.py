"""Tests for the Val3 topology- and profile-matched Target-Decoy control.

Covers docs/val3/val3_topology_profile_matched_target_decoy_implementation_spec.md
sections 4-17 (config, candidate universe, matching / random selection, evaluability,
comparison set, triplet delta) and the regression requirements of section 20.2, plus
docs/val3/val3_decoy_selection_and_mean_paired_delta_spec.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from val3 import decoy as td
from val3 import runner, viz
from val3.config import load_val3_triplet_config
from val3.streaming import assert_distinguishable_null_types, prune_non_v23_null_types
from val3.metrics import (
    build_comparison_triplet_sets,
    compute_global_effect_size,
    compute_global_joint_triplet_comparison,
    sample_decoy_anchor_joint_baseline,
    target_decoy_summary_metadata,
    write_target_decoy_diagnostics,
)

GENE_NODE_NULL = "target_khop_restricted_paired_gene_label_permutation"
PATH_SHUFFLE_NULL = "path_count_length_matched_structural_resampling"
LEGACY_DECOY_NULL = td.DECOY_NULL_TYPE
TARGET_DECOY_NULL = td.TARGET_DECOY_NULL_TYPE
EVAL_TYPE = "path_existing_pairs"
STAT = "joint_jaccard"


# ---------------------------------------------------------------------------
# toy fixtures
# ---------------------------------------------------------------------------

def _toy_kg() -> pd.DataFrame:
    # g1 -> g2 -> g3 -> g4; g5 is an isolated sink; g6 has no outgoing edge.
    return pd.DataFrame([
        {"source": "g1", "target": "g2", "polarity": 1},
        {"source": "g2", "target": "g3", "polarity": -1},
        {"source": "g3", "target": "g4", "polarity": 1},
        {"source": "g1", "target": "g6", "polarity": 1},
        {"source": "g7", "target": "g5", "polarity": 1},
    ])


def _toy_kg_path(tmp_path: Path) -> Path:
    path = tmp_path / "kg.tsv"
    _toy_kg().to_csv(path, sep="\t", index=False)
    return path


def _toy_target_df() -> pd.DataFrame:
    rows = []
    cells = {"g1": ["A", "B", "C"], "g2": ["A"], "g3": ["A", "B"], "g4": ["A"], "g6": ["A", "B"], "g7": ["A"], "gX": ["A"]}
    for pert_id, cell_ids in cells.items():
        for cell_id in cell_ids:
            rows.append({"pert_id": pert_id, "cell_id": cell_id})
    return pd.DataFrame(rows)


def _feature_table(tmp_path: Path) -> pd.DataFrame:
    graph = td.load_directed_kg_graph(_toy_kg_path(tmp_path))
    universe = td.build_target_decoy_candidate_universe(_toy_target_df(), kg_nodes=set(graph.nodes()))
    return td.build_target_decoy_matching_features(
        universe, graph=graph, target_df=_toy_target_df(), reach_cutoff=4
    )


def _uniform_feature_table(n: int) -> pd.DataFrame:
    """All candidates share identical features, so ordering is decided by the tie-break."""
    rows = [
        {
            "target_id": f"c{i:03d}",
            "directed_out_degree": 3,
            "reachable_unique_nodes_k4": 5,
            "unique_kd_cell_lines": 2,
            "total_degree_diagnostic": 6,
        }
        for i in range(n)
    ]
    out = pd.DataFrame(rows)
    for var, rank_col in td.TARGET_DECOY_RANK_COLUMNS.items():
        out[rank_col] = td.midrank_percentile(out[var]).to_numpy(dtype=float)
    return out


def _observed_triplets(records: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"triplet_id": tid, "drug_id": drug, "target_id": target, "disease_id": disease}
            for tid, drug, target, disease in records
        ]
    )


# ---------------------------------------------------------------------------
# 20.1 (1)-(3): candidate universe and matching features
# ---------------------------------------------------------------------------

def test_candidate_universe_is_kd_targets_intersect_kg_nodes(tmp_path: Path) -> None:
    graph = td.load_directed_kg_graph(_toy_kg_path(tmp_path))
    universe = td.build_target_decoy_candidate_universe(_toy_target_df(), kg_nodes=set(graph.nodes()))
    # gX has a KD profile but is not a KG node; g5 is a KG node without a KD profile.
    assert universe == ["g1", "g2", "g3", "g4", "g6", "g7"]
    assert "gX" not in universe
    assert "g5" not in universe


def test_candidate_universe_can_be_restricted_to_labeled_targets(tmp_path: Path) -> None:
    graph = td.load_directed_kg_graph(_toy_kg_path(tmp_path))
    universe = td.build_target_decoy_candidate_universe(
        _toy_target_df(), kg_nodes=set(graph.nodes()), labeled_target_ids={"g1", "g3", "gX", "gZ"},
    )
    # gX is a labeled target but not in the KD-x-KG pool; gZ is not in the KD-x-KG pool
    # either; g2/g4/g6/g7 are in the KD-x-KG pool but never labeled, so both filters
    # must hold at once.
    assert universe == ["g1", "g3"]


def test_candidate_universe_without_labeled_target_ids_is_unrestricted(tmp_path: Path) -> None:
    graph = td.load_directed_kg_graph(_toy_kg_path(tmp_path))
    restricted = td.build_target_decoy_candidate_universe(
        _toy_target_df(), kg_nodes=set(graph.nodes()), labeled_target_ids=None,
    )
    unrestricted = td.build_target_decoy_candidate_universe(_toy_target_df(), kg_nodes=set(graph.nodes()))
    assert restricted == unrestricted == ["g1", "g2", "g3", "g4", "g6", "g7"]


def test_candidate_universe_keeps_zero_out_degree_and_zero_reach_nodes(tmp_path: Path) -> None:
    features = _feature_table(tmp_path)
    zero_out = features.loc[features["target_id"].eq("g4")].iloc[0]
    assert int(zero_out["directed_out_degree"]) == 0
    assert int(zero_out["reachable_unique_nodes_k4"]) == 0


def test_matching_features_out_degree_reach_and_kd_cells(tmp_path: Path) -> None:
    features = _feature_table(tmp_path).set_index("target_id")
    assert int(features.loc["g1", "directed_out_degree"]) == 2
    assert int(features.loc["g2", "directed_out_degree"]) == 1
    # g1 reaches g2, g6, g3, g4 within 4 directed hops (self excluded).
    assert int(features.loc["g1", "reachable_unique_nodes_k4"]) == 4
    assert int(features.loc["g2", "reachable_unique_nodes_k4"]) == 2
    assert int(features.loc["g1", "unique_kd_cell_lines"]) == 3
    assert int(features.loc["g2", "unique_kd_cell_lines"]) == 1
    # in-degree + out-degree diagnostic
    assert int(features.loc["g2", "total_degree_diagnostic"]) == 2


def test_reachability_respects_cutoff_and_direction(tmp_path: Path) -> None:
    graph = td.load_directed_kg_graph(_toy_kg_path(tmp_path))
    features = td.build_target_decoy_matching_features(
        ["g1"], graph=graph, target_df=_toy_target_df(), reach_cutoff=2
    )
    assert int(features.iloc[0]["reachable_unique_nodes_k4"]) == 3
    assert int(features.iloc[0]["reach_cutoff"]) == 2


def test_midrank_percentile_handles_ties() -> None:
    got = td.midrank_percentile(pd.Series([1, 1, 2])).tolist()
    assert got == pytest.approx([1 / 3, 1 / 3, 5 / 6])
    assert td.midrank_percentile(pd.Series([5, 1, 3])).tolist() == pytest.approx([5 / 6, 1 / 6, 0.5])


def test_matching_feature_table_records_graph_representation(tmp_path: Path) -> None:
    features = _feature_table(tmp_path)
    assert set(features["graph_type"]) == {td.TARGET_DECOY_GRAPH_TYPE}
    assert set(features["edge_count_rule"]) == {td.TARGET_DECOY_EDGE_COUNT_RULE}


# ---------------------------------------------------------------------------
# 20.1 (4): distance
# ---------------------------------------------------------------------------

def test_distance_adds_all_three_rank_terms() -> None:
    features = pd.DataFrame([
        {"target_id": "obs", "directed_out_degree": 1, "reachable_unique_nodes_k4": 1,
         "unique_kd_cell_lines": 1, "total_degree_diagnostic": 2,
         "rank_out_degree": 0.5, "rank_reachable_k4": 0.5, "rank_kd_cell_lines": 0.5},
        # Each rank differs by 0.1 -> distance = sqrt(3 * 0.01).
        {"target_id": "near", "directed_out_degree": 2, "reachable_unique_nodes_k4": 2,
         "unique_kd_cell_lines": 2, "total_degree_diagnostic": 4,
         "rank_out_degree": 0.6, "rank_reachable_k4": 0.6, "rank_kd_cell_lines": 0.6},
        # A single rank differs by 0.2 -> larger than the three-term 0.1 candidate only if
        # all three terms are summed (0.2 > sqrt(0.03) is false, so ordering proves the sum).
        {"target_id": "far", "directed_out_degree": 9, "reachable_unique_nodes_k4": 1,
         "unique_kd_cell_lines": 1, "total_degree_diagnostic": 10,
         "rank_out_degree": 0.9, "rank_reachable_k4": 0.5, "rank_kd_cell_lines": 0.5},
    ])
    assignments = td.build_target_decoy_assignments(
        _observed_triplets([("t1", "d1", "obs", "dis1")]),
        feature_table=features,
        positive_targets_by_drug_disease={},
        direction="inhibitory",
        n_decoys_per_triplet=2,
        seed=0,
    )
    by_target = dict(zip(assignments["decoy_target_id"], assignments["matching_distance"]))
    assert by_target["near"] == pytest.approx(np.sqrt(3 * 0.01))
    assert by_target["far"] == pytest.approx(0.4)
    assert list(assignments["match_rank"]) == [1, 2]
    assert list(assignments["decoy_target_id"]) == ["near", "far"]


# ---------------------------------------------------------------------------
# 20.1 (5)-(6): exclusion
# ---------------------------------------------------------------------------

def test_observed_target_and_same_drug_disease_positives_are_excluded() -> None:
    features = _uniform_feature_table(20)
    label_input = _observed_triplets([
        ("t1", "d1", "c000", "dis1"),
        ("t2", "d1", "c001", "dis1"),   # same drug-disease positive
        ("t3", "d1", "c002", "dis2"),   # same drug, other disease -> not excluded
        ("t4", "d2", "c003", "dis1"),   # other drug, same disease -> not excluded
    ])
    positives = td.build_positive_targets_by_drug_disease(label_input)
    assignments = td.build_target_decoy_assignments(
        _observed_triplets([("t1", "d1", "c000", "dis1")]),
        feature_table=features,
        positive_targets_by_drug_disease=positives,
        direction="inhibitory",
        n_decoys_per_triplet=10,
        seed=0,
    )
    selected = set(assignments["decoy_target_id"])
    assert "c000" not in selected
    assert "c001" not in selected
    assert int(assignments["n_eligible_candidates"].iloc[0]) == 18
    assert assignments["positive_exclusion_scope"].eq("same_drug_disease").all()


def test_positive_set_uses_full_label_input_not_the_primary_subset() -> None:
    features = _uniform_feature_table(20)
    full_label_input = _observed_triplets([
        ("t1", "d1", "c000", "dis1"),
        ("t9", "d1", "c005", "dis1"),   # present only in the full label input
    ])
    primary_subset = _observed_triplets([("t1", "d1", "c000", "dis1")])

    def _assign(label_df: pd.DataFrame, n_decoys: int) -> pd.DataFrame:
        # Requesting every eligible candidate makes eligibility decisive rather than
        # tie-break order.
        return td.build_target_decoy_assignments(
            primary_subset,
            feature_table=features,
            positive_targets_by_drug_disease=td.build_positive_targets_by_drug_disease(label_df),
            direction="inhibitory",
            n_decoys_per_triplet=n_decoys,
            seed=0,
        )

    from_full = _assign(full_label_input, 18)
    from_subset = _assign(primary_subset, 19)
    assert int(from_full["n_eligible_candidates"].iloc[0]) == 18
    assert int(from_subset["n_eligible_candidates"].iloc[0]) == 19
    assert "c005" not in set(from_full["decoy_target_id"])
    assert "c005" in set(from_subset["decoy_target_id"])


def test_direction_is_not_part_of_the_positive_exclusion_key() -> None:
    label_input = _observed_triplets([("t1", "d1", "c000", "dis1"), ("t2", "d1", "c001", "dis1")])
    label_input["direction"] = ["inhibitory", "activatory"]
    positives = td.build_positive_targets_by_drug_disease(label_input)
    assert positives[("d1", "dis1")] == {"c000", "c001"}


def test_insufficient_matching_support_excludes_the_triplet_instead_of_reusing_candidates() -> None:
    features = _uniform_feature_table(10)
    positives = td.build_positive_targets_by_drug_disease(
        _observed_triplets([(f"t{i}", "d1", f"c{i:03d}", "dis1") for i in range(3)])
    )
    assignments = td.build_target_decoy_assignments(
        _observed_triplets([("t0", "d1", "c000", "dis1")]),
        feature_table=features,
        positive_targets_by_drug_disease=positives,
        direction="inhibitory",
        n_decoys_per_triplet=10,
        seed=0,
    )
    assert assignments.empty


def test_insufficient_matching_support_excludes_only_that_triplet() -> None:
    features = _uniform_feature_table(12)
    positives = td.build_positive_targets_by_drug_disease(
        _observed_triplets([
            ("p0", "dB", "c002", "disB"),
            ("p1", "dB", "c003", "disB"),
            ("p2", "dB", "c004", "disB"),
        ])
    )
    assignments = td.build_target_decoy_assignments(
        _observed_triplets([
            ("tA", "dA", "c000", "disA"),
            ("tB", "dB", "c001", "disB"),
        ]),
        feature_table=features,
        positive_targets_by_drug_disease=positives,
        direction="inhibitory",
        n_decoys_per_triplet=10,
        seed=0,
    )
    assert set(assignments["original_triplet_id"].astype(str)) == {"tA"}
    assert len(assignments) == 10


def test_summarize_assignments_reports_triplets_excluded_for_insufficient_eligible() -> None:
    features = _uniform_feature_table(12)
    positives = td.build_positive_targets_by_drug_disease(
        _observed_triplets([
            ("p0", "dB", "c002", "disB"),
            ("p1", "dB", "c003", "disB"),
            ("p2", "dB", "c004", "disB"),
        ])
    )
    assignments = td.build_target_decoy_assignments(
        _observed_triplets([
            ("tA", "dA", "c000", "disA"),
            ("tB", "dB", "c001", "disB"),
        ]),
        feature_table=features,
        positive_targets_by_drug_disease=positives,
        direction="inhibitory",
        n_decoys_per_triplet=10,
        seed=0,
    )
    summary = td.summarize_target_decoy_assignments(assignments, n_input_triplets=2)
    assert summary["n_original_triplets"] == 1
    assert summary["n_triplets_excluded_insufficient_eligible"] == 1
    # Without n_input_triplets the diagnostic is simply omitted, not defaulted to 0.
    assert "n_triplets_excluded_insufficient_eligible" not in td.summarize_target_decoy_assignments(assignments)


# ---------------------------------------------------------------------------
# 20.1 (7)-(10): assignment
# ---------------------------------------------------------------------------

def test_each_triplet_gets_ten_distinct_candidates_and_reuse_across_triplets() -> None:
    features = _uniform_feature_table(30)
    observed = _observed_triplets([
        ("t1", "d1", "c000", "dis1"),
        ("t2", "d2", "c001", "dis2"),
    ])
    assignments = td.build_target_decoy_assignments(
        observed,
        feature_table=features,
        positive_targets_by_drug_disease={},
        direction="inhibitory",
        n_decoys_per_triplet=10,
        seed=0,
    )
    assert len(assignments) == 20
    for tid, sub in assignments.groupby("original_triplet_id"):
        assert len(sub) == 10
        assert sub["decoy_target_id"].nunique() == 10
        assert sorted(sub["match_rank"]) == list(range(1, 11))
        assert sub["pseudo_triplet_id"].nunique() == 10
    # Across-triplet reuse is allowed by construction.
    shared = set(assignments.loc[assignments["original_triplet_id"].eq("t1"), "decoy_target_id"]) & set(
        assignments.loc[assignments["original_triplet_id"].eq("t2"), "decoy_target_id"]
    )
    assert shared


def test_pseudo_triplet_id_is_deterministic_and_unique() -> None:
    features = _uniform_feature_table(15)
    assignments = td.build_target_decoy_assignments(
        _observed_triplets([("t1", "d1", "c000", "dis1")]),
        feature_table=features,
        positive_targets_by_drug_disease={},
        direction="inhibitory",
        n_decoys_per_triplet=10,
        seed=0,
    )
    assert assignments["pseudo_triplet_id"].is_unique
    expected = "t1__matched_decoy__" + assignments["decoy_target_id"]
    assert list(assignments["pseudo_triplet_id"]) == list(expected)


def test_stable_tie_break_is_invariant_to_input_row_order() -> None:
    features = _uniform_feature_table(25)
    observed = _observed_triplets([
        ("t1", "d1", "c000", "dis1"),
        ("t2", "d2", "c001", "dis2"),
    ])
    kwargs = dict(
        positive_targets_by_drug_disease={},
        direction="inhibitory",
        n_decoys_per_triplet=10,
        seed=7,
    )
    base = td.build_target_decoy_assignments(observed, feature_table=features, **kwargs)
    shuffled_rows = td.build_target_decoy_assignments(
        observed.iloc[::-1].reset_index(drop=True), feature_table=features, **kwargs
    )
    shuffled_features = td.build_target_decoy_assignments(
        observed, feature_table=features.sample(frac=1.0, random_state=3).reset_index(drop=True), **kwargs
    )
    key = ["original_triplet_id", "match_rank", "decoy_target_id"]
    left = base[key].sort_values(key).reset_index(drop=True)
    for other in (shuffled_rows, shuffled_features):
        right = other[key].sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)


def test_tie_break_key_is_a_platform_stable_hash() -> None:
    key = td.target_decoy_tie_break_key(0, "t1", "c001")
    assert key == td.target_decoy_tie_break_key(0, "t1", "c001")
    assert len(key) == 64
    assert key != td.target_decoy_tie_break_key(1, "t1", "c001")
    assert key != td.target_decoy_tie_break_key(0, "t2", "c001")


def test_candidate_selection_ignores_path_evaluability(tmp_path: Path) -> None:
    """Selection uses only pre-inference features.

    ``g4`` has zero out-degree, so it can never produce a path, yet it stays selectable.
    """
    features = _feature_table(tmp_path)
    assignments = td.build_target_decoy_assignments(
        _observed_triplets([("t1", "d1", "g2", "dis1")]),
        feature_table=features,
        positive_targets_by_drug_disease={},
        direction="inhibitory",
        n_decoys_per_triplet=5,
        seed=0,
    )
    assert "g4" in set(assignments["decoy_target_id"])
    assert len(assignments) == 5


# ---------------------------------------------------------------------------
# 20.1 (11): evaluability
# ---------------------------------------------------------------------------

def _assignments_with_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _uniform_feature_table(12)
    assignments = td.build_target_decoy_assignments(
        _observed_triplets([("t1", "d1", "c000", "dis1")]),
        feature_table=features,
        positive_targets_by_drug_disease={},
        direction="inhibitory",
        n_decoys_per_triplet=10,
        seed=0,
    )
    scored = assignments["pseudo_triplet_id"].tolist()[:3]
    joint = pd.DataFrame([
        {
            "triplet_id": pseudo,
            "evaluable_set_type": EVAL_TYPE,
            "n_raw_cell_pairs": 4,
            "n_path_existing_cell_pairs": 2,
            "n_drug_paths": 5,
            "n_kd_paths": 6,
            "joint_jaccard": 0.1 * (i + 1),
        }
        for i, pseudo in enumerate(scored)
    ])
    return assignments, joint


def test_non_evaluable_candidates_are_kept_without_zero_fill_or_backfill() -> None:
    assignments, joint = _assignments_with_scores()
    evaluability = td.build_target_decoy_evaluability(assignments, joint, evaluable_set_type=EVAL_TYPE)
    assert len(evaluability) == len(assignments) == 10
    assert int(evaluability["is_decoy_evaluable"].sum()) == 3
    non_evaluable = evaluability.loc[~evaluability["is_decoy_evaluable"]]
    assert non_evaluable["joint_jaccard"].isna().all()
    assert not (pd.to_numeric(non_evaluable["joint_jaccard"], errors="coerce") == 0).any()
    # No 11th candidate was added to compensate.
    assert evaluability["match_rank"].max() == 10


def test_missing_path_existing_pairs_makes_a_candidate_non_evaluable() -> None:
    assignments, joint = _assignments_with_scores()
    joint.loc[0, "n_path_existing_cell_pairs"] = 0
    evaluability = td.build_target_decoy_evaluability(assignments, joint, evaluable_set_type=EVAL_TYPE)
    assert int(evaluability["is_decoy_evaluable"].sum()) == 2


def test_evaluability_summary_reports_candidate_level_denominator() -> None:
    assignments, joint = _assignments_with_scores()
    evaluability = td.build_target_decoy_evaluability(assignments, joint, evaluable_set_type=EVAL_TYPE)
    summary = td.summarize_target_decoy_evaluability(
        evaluability,
        evaluable_set_type=EVAL_TYPE,
        observed_evaluable_triplet_ids={"t1"},
        comparison_triplet_ids={"t1"},
    )
    row = summary.iloc[0]
    assert int(row["n_selected_decoys_total"]) == 10
    assert int(row["n_evaluable_decoys_total"]) == 3
    assert float(row["candidate_level_evaluable_rate"]) == pytest.approx(0.3)
    assert row["candidate_level_evaluable_rate_denominator"] == "n_selected_decoys_total"
    assert float(row["triplet_level_coverage"]) == pytest.approx(1.0)
    assert row["n_evaluable_decoys_histogram"] == '{"3": 1}'


# ---------------------------------------------------------------------------
# 17: balance
# ---------------------------------------------------------------------------

def _manual_assignments(records: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """Minimal assignment frame: (triplet_id, observed_target_id, observed value, decoy value)."""
    rows = []
    for idx, (tid, observed_target, obs_value, decoy_value) in enumerate(records):
        row = {
            "original_triplet_id": tid,
            "pseudo_triplet_id": f"{tid}__matched_decoy__c{idx:03d}",
            "original_target_id": observed_target,
            "decoy_target_id": f"c{idx:03d}",
        }
        for var in td.TARGET_DECOY_MATCHING_VARIABLES + td.TARGET_DECOY_DIAGNOSTIC_VARIABLES:
            row[f"observed_{var}"] = obs_value
            row[f"decoy_{var}"] = decoy_value
        rows.append(row)
    return pd.DataFrame(rows)


def test_matching_balance_smd_matches_the_weighted_definition() -> None:
    assignments = _manual_assignments([
        ("t1", "obs1", 10.0, 6.0),
        ("t1", "obs1", 10.0, 8.0),
    ])
    balance = td.build_target_decoy_matching_balance(
        assignments, analysis_scope="all_profile_supported", stage="pre_inference_selected"
    )
    row = balance.loc[
        balance["weighting"].eq("triplet_weighted") & balance["variable"].eq("directed_out_degree")
    ].iloc[0]
    # observed: mean 10, population variance 0; decoy: weights 1/2 each -> mean 7, variance 1.
    assert float(row["observed_mean"]) == pytest.approx(10.0)
    assert float(row["decoy_mean"]) == pytest.approx(7.0)
    assert float(row["observed_sd"]) == pytest.approx(0.0)
    assert float(row["decoy_sd"]) == pytest.approx(1.0)
    assert float(row["smd"]) == pytest.approx(3.0 / np.sqrt(0.5))
    assert float(row["abs_smd"]) == pytest.approx(abs(float(row["smd"])))
    assert bool(row["is_matching_variable"]) is True
    assert bool(
        balance.loc[balance["variable"].eq("total_degree_diagnostic"), "is_matching_variable"].iloc[0]
    ) is False


def test_matching_balance_target_weighting_equalizes_observed_target_strata() -> None:
    # obs1 carries two triplets, obs2 one; target weighting must not let obs1 dominate.
    assignments = _manual_assignments([
        ("t1", "obs1", 4.0, 4.0),
        ("t2", "obs1", 4.0, 4.0),
        ("t3", "obs2", 10.0, 10.0),
    ])
    balance = td.build_target_decoy_matching_balance(
        assignments, analysis_scope="all_profile_supported", stage="pre_inference_selected"
    )

    def _means(weighting: str) -> tuple[float, float]:
        row = balance.loc[
            balance["weighting"].eq(weighting) & balance["variable"].eq("directed_out_degree")
        ].iloc[0]
        return float(row["observed_mean"]), float(row["decoy_mean"])

    # Triplet weighting: (4 + 4 + 10) / 3 = 6.
    assert _means("triplet_weighted")[0] == pytest.approx(6.0)
    # Target weighting: obs1 stratum weight 1, obs2 stratum weight 1 -> (4 + 10) / 2 = 7.
    assert _means("target_weighted")[0] == pytest.approx(7.0)
    assert _means("target_weighted")[1] == pytest.approx(7.0)


def test_matching_balance_triplet_weighting_gives_each_triplet_weight_one() -> None:
    features = _uniform_feature_table(20).copy()
    # Give one candidate an extreme feature value so weighting is observable.
    features.loc[features["target_id"].eq("c019"), "directed_out_degree"] = 100
    assignments = td.build_target_decoy_assignments(
        _observed_triplets([("t1", "d1", "c000", "dis1")]),
        feature_table=features,
        positive_targets_by_drug_disease={},
        direction="inhibitory",
        n_decoys_per_triplet=10,
        seed=0,
    )
    balance = td.build_target_decoy_matching_balance(
        assignments, analysis_scope="all_profile_supported", stage="pre_inference_selected"
    )
    row = balance.loc[
        balance["weighting"].eq("triplet_weighted") & balance["variable"].eq("directed_out_degree")
    ].iloc[0]
    expected = float(
        pd.to_numeric(assignments["decoy_directed_out_degree"], errors="coerce").mean()
    )
    assert float(row["decoy_mean"]) == pytest.approx(expected)
    assert int(row["n_observed_rows"]) == 1
    assert int(row["n_decoy_rows"]) == 10


def test_matching_balance_post_evaluability_uses_surviving_decoys_only() -> None:
    assignments, joint = _assignments_with_scores()
    evaluability = td.build_target_decoy_evaluability(assignments, joint, evaluable_set_type=EVAL_TYPE)
    surviving = assignments.merge(
        evaluability[["pseudo_triplet_id", "is_decoy_evaluable"]].rename(
            columns={"is_decoy_evaluable": "_is_surviving"}
        ),
        on="pseudo_triplet_id",
        how="left",
    )
    balance = td.build_target_decoy_matching_balance(
        surviving,
        analysis_scope="primary_path_evaluable",
        stage="post_evaluability_surviving",
        surviving_col="_is_surviving",
    )
    assert int(balance["n_decoy_rows"].iloc[0]) == 3


def _balance_with_scopes(smd_by_variable: dict[str, float]) -> pd.DataFrame:
    """Balance rows over every scope / stage / weighting combination the run emits."""
    rows = []
    for scope in ["all_profile_supported", "primary_path_evaluable"]:
        for stage in ["pre_inference_selected", "post_evaluability_surviving"]:
            for weighting in ["triplet_weighted", "target_weighted"]:
                for variable in list(td.TARGET_DECOY_MATCHING_VARIABLES) + list(td.TARGET_DECOY_DIAGNOSTIC_VARIABLES):
                    acceptance_row = (
                        scope == td.TARGET_DECOY_ACCEPTANCE_ANALYSIS_SCOPE
                        and stage == td.TARGET_DECOY_ACCEPTANCE_STAGE
                        and weighting == td.TARGET_DECOY_ACCEPTANCE_WEIGHTING
                        and variable in td.TARGET_DECOY_MATCHING_VARIABLES
                    )
                    # Off-acceptance rows carry a deliberately large SMD: they must not leak
                    # into the acceptance judgement.
                    smd = float(smd_by_variable.get(variable, 0.05)) if acceptance_row else 0.9
                    rows.append({
                        "analysis_scope": scope,
                        "stage": stage,
                        "weighting": weighting,
                        "variable": variable,
                        "n_observed_rows": 1570,
                        "n_decoy_rows": 15700,
                        "observed_mean": 1.0,
                        "decoy_mean": 1.0 - smd,
                        "observed_sd": 1.0,
                        "decoy_sd": 1.0,
                        "smd": smd,
                        "abs_smd": abs(smd),
                        "is_matching_variable": variable in td.TARGET_DECOY_MATCHING_VARIABLES,
                        "evaluable_set_type": EVAL_TYPE,
                    })
    return pd.DataFrame(rows)


def test_matching_acceptance_uses_only_the_primary_scope_stage_and_weighting() -> None:
    acceptance = td.build_target_decoy_matching_acceptance(_balance_with_scopes({}))
    assert set(acceptance["analysis_scope"]) == {td.TARGET_DECOY_ACCEPTANCE_ANALYSIS_SCOPE}
    assert set(acceptance["stage"]) == {td.TARGET_DECOY_ACCEPTANCE_STAGE}
    assert set(acceptance["weighting"]) == {td.TARGET_DECOY_ACCEPTANCE_WEIGHTING}
    assert set(acceptance["variable"]) == set(td.TARGET_DECOY_MATCHING_VARIABLES)
    assert "total_degree_diagnostic" not in set(acceptance["variable"])
    # The 0.9 rows of the other scopes would dominate a scope-blind maximum.
    assert float(acceptance["max_abs_smd"].max()) == pytest.approx(0.05)
    assert bool(acceptance["all_matching_variables_within_threshold"].all()) is True
    assert set(acceptance["null_type"]) == {TARGET_DECOY_NULL}


def test_matching_acceptance_flags_a_matching_variable_above_the_threshold() -> None:
    acceptance = td.build_target_decoy_matching_acceptance(
        _balance_with_scopes({"unique_kd_cell_lines": 0.14})
    )
    flags = dict(zip(acceptance["variable"].astype(str), acceptance["is_within_threshold"].astype(bool)))
    assert flags["unique_kd_cell_lines"] is False
    assert flags["directed_out_degree"] is True
    assert bool(acceptance["all_matching_variables_within_threshold"].any()) is False
    assert float(acceptance["abs_smd_threshold"].iloc[0]) == pytest.approx(
        td.TARGET_DECOY_SMD_ACCEPTANCE_THRESHOLD
    )


def test_matching_acceptance_metadata_reports_the_threshold_decision() -> None:
    passing = td.target_decoy_acceptance_metadata(
        td.build_target_decoy_matching_acceptance(_balance_with_scopes({}))
    )
    assert passing["evaluated"] is True
    assert passing["analysis_scope"] == td.TARGET_DECOY_ACCEPTANCE_ANALYSIS_SCOPE
    assert passing["abs_smd_threshold"] == pytest.approx(td.TARGET_DECOY_SMD_ACCEPTANCE_THRESHOLD)
    assert passing["all_matching_variables_within_threshold"] is True
    assert set(passing["by_evaluable_set_type"][EVAL_TYPE]["abs_smd_by_variable"]) == set(
        td.TARGET_DECOY_MATCHING_VARIABLES
    )
    failing = td.target_decoy_acceptance_metadata(
        td.build_target_decoy_matching_acceptance(_balance_with_scopes({"directed_out_degree": 0.2}))
    )
    assert failing["all_matching_variables_within_threshold"] is False
    assert td.target_decoy_acceptance_metadata(pd.DataFrame())["evaluated"] is False


def test_matching_acceptance_requires_all_three_matching_variables() -> None:
    balance = _balance_with_scopes({})
    partial = balance.loc[~balance["variable"].astype(str).eq("unique_kd_cell_lines")]
    acceptance = td.build_target_decoy_matching_acceptance(partial)
    assert int(acceptance["n_matching_variables"].iloc[0]) == 2
    assert bool(acceptance["all_matching_variables_within_threshold"].any()) is False


# ---------------------------------------------------------------------------
# 13-14: comparison set and triplet-wise delta
# ---------------------------------------------------------------------------

def _true_joint(observed_by_triplet: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "triplet_id": tid,
            "drug_id": f"drug_{tid}",
            "target_id": f"target_{tid}",
            "direction": "inhibitory",
            "evaluable_set_type": EVAL_TYPE,
            "n_cell_pairs_raw": 4,
            "n_cell_pairs_used": 2,
            STAT: value,
        }
        for tid, value in observed_by_triplet.items()
    ])


def _decoy_baseline(scores_by_triplet: dict[str, list[float]], *, null_type: str = TARGET_DECOY_NULL) -> pd.DataFrame:
    rows = []
    for tid, scores in scores_by_triplet.items():
        for idx, score in enumerate(scores):
            rows.append({
                "triplet_id": tid,
                "drug_id": f"drug_{tid}",
                "target_id": f"target_{tid}",
                "direction": "inhibitory",
                "evaluable_set_type": EVAL_TYPE,
                "null_type": null_type,
                "iteration": pd.NA,
                "decoy_anchor_id": f"decoy_{tid}_{idx}",
                "n_cell_pairs_raw": 4,
                "n_cell_pairs_used": 2,
                STAT: score,
            })
    return pd.DataFrame(rows)


def _shuffle_baseline(scores_by_triplet: dict[str, list[float]], *, null_type: str) -> pd.DataFrame:
    rows = []
    for tid, scores in scores_by_triplet.items():
        for idx, score in enumerate(scores, start=1):
            rows.append({
                "triplet_id": tid,
                "drug_id": f"drug_{tid}",
                "target_id": f"target_{tid}",
                "direction": "inhibitory",
                "evaluable_set_type": EVAL_TYPE,
                "null_type": null_type,
                "iteration": idx,
                "n_cell_pairs_raw": 4,
                "n_cell_pairs_used": 2,
                STAT: score,
            })
    return pd.DataFrame(rows)


def test_triplet_without_evaluable_decoy_is_excluded_only_from_target_decoy() -> None:
    true_joint = _true_joint({"t1": 0.5, "t2": 0.4})
    baseline = pd.concat([
        _decoy_baseline({"t1": [0.1, 0.2]}),
        _shuffle_baseline({"t1": [0.1, 0.2], "t2": [0.05, 0.15]}, null_type=GENE_NODE_NULL),
    ], ignore_index=True)
    sets = build_comparison_triplet_sets(true_joint, baseline, n_shuffle_by_null_type={GENE_NODE_NULL: 2})

    def _ids(null_type: str) -> set[str]:
        sub = sets.loc[sets["null_type"].eq(null_type) & sets["is_comparison_triplet"]]
        return set(sub["triplet_id"])

    assert _ids(TARGET_DECOY_NULL) == {"t1"}
    assert _ids(GENE_NODE_NULL) == {"t1", "t2"}


def test_triplet_delta_uses_median_of_evaluable_decoys() -> None:
    true_joint = _true_joint({"t1": 0.5})
    baseline = _decoy_baseline({"t1": [0.0, 0.0, 0.9]})
    sets = build_comparison_triplet_sets(true_joint, baseline, n_shuffle_by_null_type={})
    comparison = compute_global_joint_triplet_comparison(
        true_joint, baseline, metrics=["jaccard"], comparison_triplet_sets=sets
    )
    row = comparison.iloc[0]
    assert row["null_type"] == TARGET_DECOY_NULL
    assert row["baseline_representative"] == "median"
    assert float(row["baseline_mean"]) == pytest.approx(0.3)
    assert float(row["baseline_median"]) == pytest.approx(0.0)
    assert float(row["delta"]) == pytest.approx(0.5)
    assert float(row["ref_delta_vs_null_mean"]) == pytest.approx(0.2)
    assert int(row["n_null_values"]) == 3
    assert int(row["n_decoy_anchors"]) == 3


def test_every_null_model_uses_the_median_representative() -> None:
    true_joint = _true_joint({"t1": 0.5})
    for null_type in [LEGACY_DECOY_NULL, TARGET_DECOY_NULL]:
        baseline = _decoy_baseline({"t1": [0.0, 0.0, 0.9]}, null_type=null_type)
        sets = build_comparison_triplet_sets(true_joint, baseline, n_shuffle_by_null_type={})
        comparison = compute_global_joint_triplet_comparison(
            true_joint, baseline, metrics=["jaccard"], comparison_triplet_sets=sets
        )
        row = comparison.iloc[0]
        assert row["null_type"] == null_type
        assert row["baseline_representative"] == "median"
        assert float(row["delta"]) == pytest.approx(0.5)
        # The mean-based value is retained as a non-primary diagnostic.
        assert float(row["ref_delta_vs_null_mean"]) == pytest.approx(0.2)
    shuffles = _shuffle_baseline({"t1": [0.0, 0.0, 0.9]}, null_type=GENE_NODE_NULL)
    sets = build_comparison_triplet_sets(true_joint, shuffles, n_shuffle_by_null_type={GENE_NODE_NULL: 3})
    comparison = compute_global_joint_triplet_comparison(
        true_joint, shuffles, metrics=["jaccard"], comparison_triplet_sets=sets
    )
    row = comparison.iloc[0]
    assert row["baseline_representative"] == "median"
    assert float(row["delta"]) == pytest.approx(0.5)
    assert float(row["ref_delta_vs_null_mean"]) == pytest.approx(0.2)
    assert "p_empirical" not in comparison.columns


def test_global_effect_size_is_mean_of_observed_minus_null_median() -> None:
    """paired_delta_mean = mean over triplets of (observed - that triplet's null median)."""
    true_joint = _true_joint({"t1": 0.5, "t2": 0.4})
    baseline = _decoy_baseline({"t1": [0.0, 0.0, 0.9], "t2": [0.1, 0.5]})
    sets = build_comparison_triplet_sets(true_joint, baseline, n_shuffle_by_null_type={})
    effect = compute_global_effect_size(
        true_joint, baseline, metrics=["jaccard"], comparison_triplet_sets=sets
    ).iloc[0]

    # triplet medians: 0.0 and 0.3 -> primary deltas 0.5 and 0.1
    assert effect["baseline_representative"] == "median"
    assert effect["estimand"] == "mean_observed_minus_triplet_null_median"
    assert float(effect["null_median_mean"]) == pytest.approx(0.15)
    assert float(effect["paired_delta_mean"]) == pytest.approx((0.5 + 0.1) / 2)
    assert float(effect["paired_delta_median"]) == pytest.approx(0.3)
    # triplet means: 0.3 and 0.3 -> non-primary reference deltas 0.2 and 0.1
    assert float(effect["ref_null_mean_mean"]) == pytest.approx(0.3)
    assert float(effect["ref_paired_delta_mean"]) == pytest.approx((0.2 + 0.1) / 2)
    assert float(effect["ref_paired_delta_median"]) == pytest.approx(0.15)
    assert "baseline_mean_mean" not in effect.index


def test_jaccard_paired_wilcoxon_holm_table_uses_the_median_representative(
    tmp_path: Path,
) -> None:
    from val3 import report

    true_joint = _true_joint({"t1": 0.5, "t2": 0.4})
    baseline = _decoy_baseline({"t1": [0.0, 0.0, 0.9], "t2": [0.1, 0.5]})
    table = report._jaccard_paired_wilcoxon_holm_table(
        {"global_joint_true": true_joint, "global_joint_baseline": baseline}
    )
    row = table.iloc[0]
    assert row["baseline_representative"] == "median"
    assert float(row["null_median_mean"]) == pytest.approx(0.15)
    assert float(row["delta_mean"]) == pytest.approx(0.3)
    assert int(row["n_positive_delta"]) == 2
    assert int(row["n_negative_delta"]) == 0
    assert float(row["wilcoxon_two_sided_statistic"]) == pytest.approx(0.0)
    assert float(row["p_wilcoxon_two_sided"]) == pytest.approx(0.5)
    assert float(row["p_wilcoxon_two_sided_holm"]) == pytest.approx(0.5)
    assert "null_mean_mean" not in table.columns
    assert not any("sign" in column for column in table.columns)

    run_root = tmp_path / "run"
    (run_root / "evaluation" / "inhibitory").mkdir(parents=True)
    outputs = report.write_val3_triplet_visualization_report(
        run_root=run_root, direction="inhibitory",
    )
    written = Path(outputs["report_tables"]["jaccard_paired_wilcoxon_holm"])
    assert written.name == "jaccard_paired_wilcoxon_holm.tsv" and written.exists()


def test_delta_by_n_evaluable_decoys_diagnostic() -> None:
    true_joint = _true_joint({"t1": 0.5, "t2": 0.4})
    baseline = _decoy_baseline({"t1": [0.1, 0.3], "t2": [0.2]})
    sets = build_comparison_triplet_sets(true_joint, baseline, n_shuffle_by_null_type={})
    comparison = compute_global_joint_triplet_comparison(
        true_joint, baseline, metrics=["jaccard"], comparison_triplet_sets=sets
    )
    strata = td.summarize_target_decoy_delta_by_n_evaluable(comparison, evaluable_set_type=EVAL_TYPE)
    assert sorted(strata["n_evaluable_decoys"]) == [1, 2]
    assert set(strata["n_triplets"]) == {1}


# ---------------------------------------------------------------------------
# decoy selection: matched vs random
# ---------------------------------------------------------------------------

def _selection_kwargs(**overrides):
    kwargs = dict(
        feature_table=_uniform_feature_table(40),
        positive_targets_by_drug_disease={},
        direction="inhibitory",
        n_decoys_per_triplet=10,
        seed=42,
    )
    kwargs.update(overrides)
    return kwargs


def test_random_selection_is_reproducible_and_order_independent(monkeypatch) -> None:
    observed = _observed_triplets([("t1", "d1", "c000", "dis1"), ("t2", "d2", "c001", "dis2")])
    kwargs = _selection_kwargs()

    def _fail(*_args, **_kwargs):
        raise AssertionError("random decoy selection must not draw from np.random")

    monkeypatch.setattr(np.random, "default_rng", _fail)
    monkeypatch.setattr(np.random, "choice", _fail)
    first = td.build_target_decoy_assignments(observed, decoy_selection="random", **kwargs)
    second = td.build_target_decoy_assignments(observed, decoy_selection="random", **kwargs)
    pd.testing.assert_frame_equal(first, second)

    # Processing the triplets in the opposite order must not change either selection.
    reversed_obs = observed.iloc[::-1].reset_index(drop=True)
    reversed_out = td.build_target_decoy_assignments(reversed_obs, decoy_selection="random", **kwargs)
    for tid in ["t1", "t2"]:
        assert list(first.loc[first["original_triplet_id"].eq(tid), "decoy_target_id"]) == list(
            reversed_out.loc[reversed_out["original_triplet_id"].eq(tid), "decoy_target_id"]
        )


def test_random_selection_shares_the_universe_and_exclusions_with_matched() -> None:
    observed = _observed_triplets([("t1", "d1", "c000", "dis1")])
    kwargs = _selection_kwargs(positive_targets_by_drug_disease={("d1", "dis1"): {"c001"}})
    matched = td.build_target_decoy_assignments(observed, decoy_selection="matched", **kwargs)
    random_sel = td.build_target_decoy_assignments(observed, decoy_selection="random", **kwargs)

    assert len(random_sel) == 10
    assert random_sel["decoy_target_id"].nunique() == 10
    # Same eligible pool: the observed target and the same drug-disease positive are out.
    assert not {"c000", "c001"} & set(random_sel["decoy_target_id"])
    assert int(random_sel["n_eligible_candidates"].iloc[0]) == int(matched["n_eligible_candidates"].iloc[0])
    assert set(random_sel["selection_mode"]) == {"random"}
    assert set(matched["selection_mode"]) == {"matched"}
    assert random_sel["pseudo_triplet_id"].str.contains("__random_decoy__").all()
    assert matched["pseudo_triplet_id"].str.contains("__matched_decoy__").all()
    assert list(random_sel.columns) == td.TARGET_DECOY_ASSIGNMENT_COLUMNS
    # The rank distance is still recorded for the random draw, as a diagnostic only.
    assert random_sel["matching_distance"].notna().all()
    assert sorted(random_sel["match_rank"]) == list(range(1, 11))


def test_random_selection_ignores_the_rank_distance() -> None:
    """With graded features the matched pick is the nearest set; random is not."""
    graded = pd.DataFrame([
        {
            "target_id": f"c{i:03d}",
            "directed_out_degree": i,
            "reachable_unique_nodes_k4": i,
            "unique_kd_cell_lines": 1 + i % 3,
            "total_degree_diagnostic": 2 * i,
        }
        for i in range(40)
    ])
    for var, rank_col in td.TARGET_DECOY_RANK_COLUMNS.items():
        graded[rank_col] = td.midrank_percentile(graded[var]).to_numpy(dtype=float)
    observed = _observed_triplets([("t1", "d1", "c020", "dis1")])
    kwargs = _selection_kwargs(feature_table=graded)
    matched = td.build_target_decoy_assignments(observed, decoy_selection="matched", **kwargs)
    random_sel = td.build_target_decoy_assignments(observed, decoy_selection="random", **kwargs)

    assert set(random_sel["decoy_target_id"]) != set(matched["decoy_target_id"])
    # matched keeps the ten smallest distances; the uniform draw has no such property.
    assert float(matched["matching_distance"].max()) < float(random_sel["matching_distance"].max())
    assert matched["matching_distance"].is_monotonic_increasing


def test_random_selection_does_not_claim_a_matching_rule() -> None:
    random_meta = td.target_decoy_model_metadata(
        n_candidate_universe=40,
        reach_cutoff=4,
        n_decoys_per_triplet=10,
        tie_break_seed=0,
        is_primary_spec=True,
        decoy_selection="random",
    )
    assert random_meta["decoy_selection"] == "random"
    for key in ["matching_variables", "rank_transform", "distance"]:
        assert random_meta[key] == td.TARGET_DECOY_SELECTION_NOT_APPLICABLE
    assert random_meta["reach_cutoff"] == 4
    matched_meta = td.target_decoy_model_metadata(
        n_candidate_universe=40,
        reach_cutoff=4,
        n_decoys_per_triplet=10,
        tie_break_seed=0,
        is_primary_spec=True,
    )
    assert matched_meta["decoy_selection"] == "matched"
    assert matched_meta["matching_variables"] == list(td.TARGET_DECOY_MATCHING_VARIABLES)


def test_unknown_decoy_selection_is_rejected() -> None:
    observed = _observed_triplets([("t1", "d1", "c000", "dis1")])
    with pytest.raises(ValueError, match="decoy_selection must be one of"):
        td.build_target_decoy_assignments(observed, decoy_selection="nearest", **_selection_kwargs())


def test_assignments_are_reproducible_for_a_fixed_seed() -> None:
    features = _uniform_feature_table(30)
    observed = _observed_triplets([("t1", "d1", "c000", "dis1"), ("t2", "d2", "c001", "dis2")])
    kwargs = dict(
        feature_table=features,
        positive_targets_by_drug_disease={},
        direction="inhibitory",
        n_decoys_per_triplet=10,
        seed=42,
    )
    pd.testing.assert_frame_equal(
        td.build_target_decoy_assignments(observed, **kwargs),
        td.build_target_decoy_assignments(observed, **kwargs),
    )


# ---------------------------------------------------------------------------
# 20.2: regression - other nulls are unaffected by enabling the new null
# ---------------------------------------------------------------------------

def test_other_nulls_are_unchanged_when_target_decoy_is_enabled() -> None:
    true_joint = _true_joint({"t1": 0.5, "t2": 0.4})
    shuffles = pd.concat([
        _shuffle_baseline({"t1": [0.1, 0.2, 0.3], "t2": [0.05, 0.15, 0.25]}, null_type=GENE_NODE_NULL),
        _shuffle_baseline({"t1": [0.2, 0.3, 0.4], "t2": [0.1, 0.2, 0.3]}, null_type=PATH_SHUFFLE_NULL),
    ], ignore_index=True)
    with_decoy = pd.concat([shuffles, _decoy_baseline({"t1": [0.1, 0.2], "t2": [0.3]})], ignore_index=True)
    n_shuffle = {GENE_NODE_NULL: 3, PATH_SHUFFLE_NULL: 3}

    def _outputs(baseline: pd.DataFrame):
        sets = build_comparison_triplet_sets(true_joint, baseline, n_shuffle_by_null_type=n_shuffle)
        comparison = compute_global_joint_triplet_comparison(
            true_joint, baseline, metrics=["jaccard"], comparison_triplet_sets=sets
        )
        effect = compute_global_effect_size(
            true_joint, baseline, metrics=["jaccard"], comparison_triplet_sets=sets
        )
        keep = [GENE_NODE_NULL, PATH_SHUFFLE_NULL]
        return (
            comparison.loc[comparison["null_type"].isin(keep)].reset_index(drop=True),
            effect.loc[effect["null_type"].isin(keep)].reset_index(drop=True),
        )

    without = _outputs(shuffles)
    with_ = _outputs(with_decoy)
    for left, right in zip(without, with_):
        # Enabling the decoy control only adds decoy-specific columns (and widens the
        # n_shuffle dtype to hold pd.NA); the gene-node shuffle and path-shuffle values
        # themselves must be untouched.
        pd.testing.assert_frame_equal(left, right[left.columns], check_dtype=False)


# ---------------------------------------------------------------------------
# 3: Target-Decoy labels distinguish random selection
# ---------------------------------------------------------------------------

def test_both_decoy_controls_share_the_same_figure_label() -> None:
    assert viz.format_null_type_label(TARGET_DECOY_NULL) == "Target-Decoy"
    assert viz.format_null_type_label(LEGACY_DECOY_NULL) == "Target-Decoy"
    # Distinct sort keys keep a deterministic axis order for either control.
    assert viz.null_type_sort_key(TARGET_DECOY_NULL) != viz.null_type_sort_key(LEGACY_DECOY_NULL)


def test_random_selection_is_the_only_suffixed_target_decoy() -> None:
    assert viz.format_null_type_label(
        TARGET_DECOY_NULL, decoy_selection="random",
    ) == "Target-Decoy (random)"
    for selection in ["matched", None, "unknown"]:
        assert viz.format_null_type_label(TARGET_DECOY_NULL, decoy_selection=selection) == "Target-Decoy"
    # The legacy control never carries the suffix, whatever selection is passed.
    for selection in ["matched", "random", None]:
        assert viz.format_null_type_label(LEGACY_DECOY_NULL, decoy_selection=selection) == "Target-Decoy"
    assert viz.format_null_type_label(GENE_NODE_NULL, decoy_selection="matched") == "Gene-node shuffle"
    assert (
        viz.format_null_type_label_multiline(TARGET_DECOY_NULL, decoy_selection="random")
        == "Target-Decoy (random)"
    )
    # NULL_TYPE_LABELS itself is unchanged: the suffix is applied on top of it.
    assert viz.NULL_TYPE_LABELS[TARGET_DECOY_NULL] == "Target-Decoy"


def test_report_table_uses_the_run_specific_target_decoy_label() -> None:
    from val3 import report

    raw = pd.DataFrame({"null_type": [TARGET_DECOY_NULL], "value": [0.1]})
    matched = report._display_metric_names(raw, decoy_selection="matched")
    random = report._display_metric_names(raw, decoy_selection="random")
    assert matched.loc[0, "null_type"] == "Target-Decoy"
    assert random.loc[0, "null_type"] == "Target-Decoy (random)"


def test_report_facing_table_rejects_a_mix_of_both_decoy_controls() -> None:
    mixed = pd.DataFrame({"null_type": [TARGET_DECOY_NULL, LEGACY_DECOY_NULL], "value": [0.1, 0.2]})
    with pytest.raises(ValueError, match="share the figure label"):
        assert_distinguishable_null_types(mixed)
    with pytest.raises(ValueError, match="share the figure label"):
        prune_non_v23_null_types(mixed)


def test_legacy_alias_pair_is_not_treated_as_a_collision() -> None:
    # Both names denote the same observed-target-pool control, so they may coexist.
    aliases = pd.DataFrame({
        "null_type": [LEGACY_DECOY_NULL, "evaluable_decoy_anchor_resampling"],
        "value": [0.1, 0.2],
    })
    assert_distinguishable_null_types(aliases)
    assert len(prune_non_v23_null_types(aliases)) == 2


def test_single_decoy_control_with_other_nulls_is_accepted() -> None:
    for decoy_null in (TARGET_DECOY_NULL, LEGACY_DECOY_NULL):
        df = pd.DataFrame({"null_type": [GENE_NODE_NULL, PATH_SHUFFLE_NULL, decoy_null], "value": [0.1, 0.2, 0.3]})
        assert len(prune_non_v23_null_types(df)) == 3
        labels = [viz.format_null_type_label(n) for n in df["null_type"]]
        assert len(set(labels)) == 3


def test_candidate_diagnostics_records_the_responsible_null_type() -> None:
    features = _uniform_feature_table(15)
    assignments = td.build_target_decoy_assignments(
        _observed_triplets([("t1", "d1", "c000", "dis1")]),
        feature_table=features,
        positive_targets_by_drug_disease={},
        direction="inhibitory",
        n_decoys_per_triplet=10,
        seed=0,
    )
    diagnostics = td.summarize_decoy_candidates(assignments, null_type=TARGET_DECOY_NULL)
    assert set(diagnostics["null_type"]) == {TARGET_DECOY_NULL}
    assert int(diagnostics["n_decoy_candidates"].iloc[0]) == 10
    assert td.summarize_decoy_candidates(assignments)["null_type"].iloc[0] == LEGACY_DECOY_NULL


# ---------------------------------------------------------------------------
# 4: config
# ---------------------------------------------------------------------------

def _write_config(tmp_path: Path, **overrides) -> Path:
    null_models = {
        GENE_NODE_NULL: {"enabled": True, "min_local_bg_genes": 0},
        TARGET_DECOY_NULL: {
            "enabled": True,
            "n_decoys_per_triplet": 10,
            "exclude_known_positive_targets": True,
            "positive_exclusion_scope": "same_drug_disease",
            "is_primary_spec": True,
        },
    }
    null_models[TARGET_DECOY_NULL].update(overrides.pop("target_decoy", {}))
    for name, patch in overrides.pop("null_models", {}).items():
        null_models.setdefault(name, {}).update(patch)
    cfg = {
        "run": {"seed": 0, "date_tag": "20260729", "output_root": "./out", "output_name": "run",
                "registry_path": "./registry.yaml"},
        "data": {"kg": "signor", "triplet_source": "targetnexus_db"},
        "sections": {"prepare_inputs": True, "path_exploration": True, "evaluation": True,
                     "make_visualization_tables": False, "make_report": False},
        "analysis": {"k_star": overrides.pop("k_star", 4), "gap": 1, "directions": ["inhibitory"],
                     "cell_pair_mode": "cartesian"},
        "infer": {"path_strategy": "simple",
                  "restrict_baseline_to_observed_evaluable": overrides.pop("restrict_baseline", False)},
        # Exploratory test config: the P0-4 primary spec pins n_shuffle to 100.
        "baseline": {
            "n_shuffle": overrides.pop("n_shuffle", 10),
            "is_primary_spec": overrides.pop("baseline_is_primary_spec", False),
            "null_models": null_models,
        },
        "evaluation": dict(
            {
                "metrics": ["jaccard"],
                "evaluable_sets": {"path_existing_pairs": {"enabled": True}},
                "global_effect_size": {"enabled": True},
            },
            **overrides.pop("evaluation", {}),
        ),
        "progress": {"enabled": False, "use_tqdm": False},
    }
    path = tmp_path / "val3.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_primary_config_resolves_fixed_model_definition(tmp_path: Path) -> None:
    cfg = load_val3_triplet_config(_write_config(tmp_path))
    model = cfg.baseline["null_models"][TARGET_DECOY_NULL]
    assert model["enabled"] is True
    assert model["is_primary_spec"] is True
    assert model["n_decoys_per_triplet"] == 10
    expected_universe = td.TARGET_DECOY_CANDIDATE_POOL_TYPE_PROFILE_SUPPORTED_OBSERVED
    assert model["decoy_candidate_universe"] == expected_universe
    assert model["candidate_universe_type"] == expected_universe
    assert model["matching_variables"] == list(td.TARGET_DECOY_MATCHING_VARIABLES)
    assert model["rank_transform"] == "midrank_percentile"
    assert model["distance"] == "equal_weight_euclidean"
    assert model["reach_cutoff"] == 4
    assert model["within_triplet_replacement"] is False
    assert model["across_triplet_reuse"] is True
    assert model["path_evaluability_prefilter"] is False
    assert model["post_inference_backfill"] is False
    assert model["tie_break_seed"] == 0
    assert model["decoy_selection"] == "matched"
    assert TARGET_DECOY_NULL in cfg.baseline["enabled_null_models"]


@pytest.mark.parametrize(
    "metrics",
    [
        ["jaccard", "sensitivity"],
        ["jaccard", "precision"],
        ["sensitivity"],
    ],
)
def test_config_rejects_every_non_jaccard_metric(
    tmp_path: Path,
    metrics: list[str],
) -> None:
    with pytest.raises(ValueError, match="evaluation.metrics"):
        load_val3_triplet_config(
            _write_config(
                tmp_path,
                baseline_is_primary_spec=False,
                evaluation={"metrics": metrics},
            )
        )


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"target_decoy": {"n_decoys_per_triplet": 5}}, "n_decoys_per_triplet must be 10"),
        ({"target_decoy": {"exclude_known_positive_targets": False}}, "exclude_known_positive_targets must be"),
        ({"target_decoy": {"positive_exclusion_scope": "same_disease"}}, "positive_exclusion_scope must be"),
        ({"k_star": 3}, "analysis.k_star must be 4"),
        ({"target_decoy": {"decoy_selection": "nearest"}}, "decoy_selection must be one of"),
        ({"restrict_baseline": True}, "restrict_baseline_to_observed_evaluable must be false"),
        (
            {"target_decoy": {"decoy_candidate_universe": "l1000_kd_profile_x_signor_nodes"}},
            "decoy_candidate_universe must be",
        ),
        (
            {
                "target_decoy": {
                    "decoy_candidate_universe": "l1000_kd_profile_x_signor_nodes_x_labeled_targets"
                }
            },
            "decoy_candidate_universe must be",
        ),
        (
            {"target_decoy": {"decoy_candidate_universe": "everything"}},
            "decoy_candidate_universe must be one of",
        ),
    ],
)
def test_primary_config_rejects_off_spec_values(tmp_path: Path, overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_val3_triplet_config(_write_config(tmp_path, **overrides))


def test_exploratory_config_allows_the_plain_l1000_candidate_universe(tmp_path: Path) -> None:
    cfg = load_val3_triplet_config(
        _write_config(
            tmp_path,
            target_decoy={
                "is_primary_spec": False,
                "decoy_candidate_universe": "l1000_kd_profile_x_signor_nodes",
            },
        )
    )
    model = cfg.baseline["null_models"][TARGET_DECOY_NULL]
    assert model["is_primary_spec"] is False
    assert model["decoy_candidate_universe"] == "l1000_kd_profile_x_signor_nodes"
    assert model["candidate_universe_type"] == "l1000_kd_profile_x_signor_nodes"


def test_exploratory_config_allows_the_full_labeled_target_universe(tmp_path: Path) -> None:
    candidate_universe = td.TARGET_DECOY_CANDIDATE_POOL_TYPE_LABELED
    cfg = load_val3_triplet_config(
        _write_config(
            tmp_path,
            target_decoy={
                "is_primary_spec": False,
                "decoy_candidate_universe": candidate_universe,
            },
        )
    )
    model = cfg.baseline["null_models"][TARGET_DECOY_NULL]
    assert model["is_primary_spec"] is False
    assert model["decoy_candidate_universe"] == candidate_universe
    assert model["candidate_universe_type"] == candidate_universe


def test_exploratory_config_allows_off_spec_values_and_records_the_flag(tmp_path: Path) -> None:
    cfg = load_val3_triplet_config(
        _write_config(
            tmp_path,
            target_decoy={"n_decoys_per_triplet": 5, "is_primary_spec": False},
            k_star=3,
        )
    )
    model = cfg.baseline["null_models"][TARGET_DECOY_NULL]
    assert model["is_primary_spec"] is False
    assert model["n_decoys_per_triplet"] == 5
    metadata = td.target_decoy_model_metadata(
        n_candidate_universe=2541,
        reach_cutoff=model["reach_cutoff"],
        n_decoys_per_triplet=model["n_decoys_per_triplet"],
        tie_break_seed=model["tie_break_seed"],
        is_primary_spec=model["is_primary_spec"],
        decoy_selection=model["decoy_selection"],
    )
    assert metadata["is_primary_spec"] is False
    assert metadata["null_type"] == TARGET_DECOY_NULL


def test_enabling_the_legacy_decoy_control_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="config から有効化できません"):
        load_val3_triplet_config(
            _write_config(tmp_path, null_models={LEGACY_DECOY_NULL: {"enabled": True}})
        )


def test_legacy_decoy_control_key_is_accepted_while_disabled(tmp_path: Path) -> None:
    for patch in [{LEGACY_DECOY_NULL: {"enabled": False}}, {}]:
        cfg = load_val3_triplet_config(_write_config(tmp_path, null_models=patch))
        assert cfg.baseline["null_models"][LEGACY_DECOY_NULL]["enabled"] is False


def test_random_selection_is_allowed_as_a_primary_run(tmp_path: Path) -> None:
    cfg = load_val3_triplet_config(
        _write_config(tmp_path, target_decoy={"decoy_selection": "random"})
    )
    model = cfg.baseline["null_models"][TARGET_DECOY_NULL]
    assert model["is_primary_spec"] is True
    assert model["decoy_selection"] == "random"
    # The matching rule does not exist for a random draw and is recorded as such.
    for key in ["matching_variables", "rank_transform", "distance"]:
        assert model[key] == td.TARGET_DECOY_SELECTION_NOT_APPLICABLE
    assert model["reach_cutoff"] == 4


def test_abolished_evaluation_keys_are_ignored(tmp_path: Path) -> None:
    cfg = load_val3_triplet_config(
        _write_config(
            tmp_path,
            evaluation={
                "global_empirical": {"enabled": True},
                "triplet_paired_tests": {"enabled": True},
                "group_resampled_null": {"enabled": True, "n_resamples": 10000},
            },
        )
    )
    for key in ["global_empirical", "triplet_paired_tests", "group_resampled_null"]:
        assert key not in cfg.evaluation


def test_legacy_decoy_alias_is_not_converted_to_the_new_null_type(tmp_path: Path) -> None:
    # The pre-v2.6 alias still merges onto the observed-target-pool control, so a saved
    # config using either name reads back; enabling it is what is now rejected.
    cfg = load_val3_triplet_config(
        _write_config(
            tmp_path,
            null_models={"evaluable_decoy_anchor_resampling": {
                "enabled": False, "max_decoy_anchors_per_triplet": 7,
            }},
        )
    )
    null_models = cfg.baseline["null_models"]
    assert null_models[LEGACY_DECOY_NULL]["enabled"] is False
    assert null_models[LEGACY_DECOY_NULL]["max_decoy_anchors_per_triplet"] == 7
    assert null_models[TARGET_DECOY_NULL]["enabled"] is True
    assert td.canonical_decoy_null_type("evaluable_decoy_anchor_resampling") == LEGACY_DECOY_NULL
    assert td.canonical_decoy_null_type(TARGET_DECOY_NULL) == TARGET_DECOY_NULL
    with pytest.raises(ValueError, match="config から有効化できません"):
        load_val3_triplet_config(
            _write_config(
                tmp_path,
                null_models={"evaluable_decoy_anchor_resampling": {"enabled": True}},
            )
        )


# ---------------------------------------------------------------------------
# 16: diagnostic outputs and runner glue
# ---------------------------------------------------------------------------

def test_baseline_rows_carry_the_new_null_type() -> None:
    assignments, joint = _assignments_with_scores()
    scored = joint["triplet_id"].tolist()
    candidate_joint = pd.DataFrame([
        {
            "triplet_id": pseudo,
            "original_triplet_id": "t1",
            "original_target_id": "c000",
            "decoy_anchor_id": f"decoy_{i}",
            "disease_id": "dis1",
            "drug_id": "d1",
            "evaluable_set_type": EVAL_TYPE,
            "n_cell_pairs_raw": 4,
            "n_cell_pairs_used": 2,
            STAT: 0.1 * (i + 1),
        }
        for i, pseudo in enumerate(scored)
    ])
    baseline, diag = sample_decoy_anchor_joint_baseline(
        candidate_joint,
        _true_joint({"t1": 0.5}),
        n_shuffle=1,
        rng=np.random.default_rng(0),
        evaluable_set_types=[EVAL_TYPE],
        null_type=TARGET_DECOY_NULL,
    )
    assert set(baseline["null_type"]) == {TARGET_DECOY_NULL}
    assert set(baseline["triplet_id"]) == {"t1"}
    assert int(baseline["n_decoy_anchors_available"].iloc[0]) == 3
    assert set(diag["null_type"]) == {TARGET_DECOY_NULL}
    assert bool(diag["is_decoy_evaluable"].all())


def test_write_target_decoy_diagnostics_emits_the_required_tables(tmp_path: Path) -> None:
    assignments, joint = _assignments_with_scores()
    pair_counts = joint.rename(columns={"triplet_id": "triplet_id"})[
        ["triplet_id", "evaluable_set_type", "n_raw_cell_pairs",
         "n_path_existing_cell_pairs", "n_drug_paths", "n_kd_paths"]
    ].copy()
    candidate_joint = joint[["triplet_id", "evaluable_set_type", "joint_jaccard"]].copy()
    true_joint = _true_joint({"t1": 0.5})
    baseline = _decoy_baseline({"t1": [0.1, 0.2, 0.3]})
    sets = build_comparison_triplet_sets(true_joint, baseline, n_shuffle_by_null_type={})
    comparison = compute_global_joint_triplet_comparison(
        true_joint, baseline, metrics=["jaccard"], comparison_triplet_sets=sets
    )
    outputs = write_target_decoy_diagnostics(
        out_dir=tmp_path,
        assignments=assignments,
        candidate_pair_counts=pair_counts,
        candidate_joint_df=candidate_joint,
        global_joint_comparison=comparison,
        comparison_triplet_sets=sets,
        observed_evaluable_ids_by_set={EVAL_TYPE: {"t1"}},
        eval_set_types=[EVAL_TYPE],
    )
    assert set(outputs) == {
        "decoy_candidate_evaluability",
        "decoy_triplet_evaluability",
        "decoy_evaluability_summary",
        "decoy_matching_balance",
        "decoy_matching_acceptance",
        "decoy_delta_by_n_evaluable_decoys",
    }
    per_triplet = pd.read_csv(outputs["decoy_triplet_evaluability"], sep="\t")
    assert int(per_triplet["n_selected_decoys"].iloc[0]) == 10
    assert int(per_triplet["n_evaluable_decoys"].iloc[0]) == 3
    assert bool(per_triplet["has_evaluable_decoy"].iloc[0]) is True
    evaluability = pd.read_csv(outputs["decoy_candidate_evaluability"], sep="\t")
    assert len(evaluability) == 10
    assert list(evaluability.columns) == td.TARGET_DECOY_EVALUABILITY_COLUMNS
    assert int(evaluability["is_decoy_evaluable"].sum()) == 3

    balance = pd.read_csv(outputs["decoy_matching_balance"], sep="\t")
    assert set(balance["analysis_scope"]) == {"all_profile_supported", "primary_path_evaluable"}
    assert set(balance["stage"]) == {"pre_inference_selected", "post_evaluability_surviving"}
    assert set(balance["weighting"]) == {"triplet_weighted", "target_weighted"}
    for column in td.TARGET_DECOY_BALANCE_COLUMNS:
        assert column in balance.columns

    summary = pd.read_csv(outputs["decoy_evaluability_summary"], sep="\t")
    assert int(summary["n_comparison_triplets"].iloc[0]) == 1

    acceptance = pd.read_csv(outputs["decoy_matching_acceptance"], sep="\t")
    assert list(acceptance.columns) == td.TARGET_DECOY_ACCEPTANCE_COLUMNS
    assert set(acceptance["analysis_scope"]) == {td.TARGET_DECOY_ACCEPTANCE_ANALYSIS_SCOPE}
    assert set(acceptance["stage"]) == {td.TARGET_DECOY_ACCEPTANCE_STAGE}
    assert set(acceptance["weighting"]) == {td.TARGET_DECOY_ACCEPTANCE_WEIGHTING}
    assert set(acceptance["variable"]) == set(td.TARGET_DECOY_MATCHING_VARIABLES)


def test_random_selection_skips_the_matching_balance_tables(tmp_path: Path) -> None:
    assignments, joint = _assignments_with_scores()
    assignments = assignments.assign(selection_mode="random")
    pair_counts = joint[
        ["triplet_id", "evaluable_set_type", "n_raw_cell_pairs",
         "n_path_existing_cell_pairs", "n_drug_paths", "n_kd_paths"]
    ].copy()
    true_joint = _true_joint({"t1": 0.5})
    baseline = _decoy_baseline({"t1": [0.1, 0.2, 0.3]})
    sets = build_comparison_triplet_sets(true_joint, baseline, n_shuffle_by_null_type={})
    comparison = compute_global_joint_triplet_comparison(
        true_joint, baseline, metrics=["jaccard"], comparison_triplet_sets=sets
    )
    outputs = write_target_decoy_diagnostics(
        out_dir=tmp_path,
        assignments=assignments,
        candidate_pair_counts=pair_counts,
        candidate_joint_df=joint[["triplet_id", "evaluable_set_type", "joint_jaccard"]].copy(),
        global_joint_comparison=comparison,
        comparison_triplet_sets=sets,
        observed_evaluable_ids_by_set={EVAL_TYPE: {"t1"}},
        eval_set_types=[EVAL_TYPE],
        decoy_selection="random",
    )
    # Balance / SMD acceptance describe a matching, so a random draw emits neither.
    assert "decoy_matching_balance" not in outputs
    assert "decoy_matching_acceptance" not in outputs
    assert not (tmp_path / "decoy_matching_balance.tsv").exists()
    assert not (tmp_path / "decoy_matching_acceptance.tsv").exists()
    assert {
        "decoy_candidate_evaluability",
        "decoy_triplet_evaluability",
        "decoy_evaluability_summary",
        "decoy_delta_by_n_evaluable_decoys",
    } <= set(outputs)


def test_prepare_target_decoy_assignments_writes_universe_and_assignments(tmp_path: Path) -> None:
    kg_path = _toy_kg_path(tmp_path)
    cfg = SimpleNamespace(seed=0, analysis={"k_star": 4}, evaluation={})
    triplets = _observed_triplets([("t1", "d1", "g2", "dis1"), ("t2", "d2", "g3", "dis1")])
    prepared = {
        "target_df": _toy_target_df(),
        "all_triplets": triplets,
        "profile_supported_observed_triplets": triplets,
    }
    run_root = tmp_path / "run"
    assignments, summary = runner._prepare_target_decoy_assignments(
        cfg,
        direction="inhibitory",
        prepared=prepared,
        kg_infer_path=kg_path,
        run_root=run_root,
        null_cfg={
            "n_decoys_per_triplet": 3,
            "exclude_known_positive_targets": True,
            "is_primary_spec": True,
            "decoy_candidate_universe": "l1000_kd_profile_x_signor_nodes",
        },
    )
    assert len(assignments) == 6
    assert summary["null_type"] == TARGET_DECOY_NULL
    assert summary["n_candidate_universe"] == 6
    assert summary["n_decoys_per_triplet_min"] == 3
    # Same drug-disease positives (g2 for d1/dis1, g3 for d2/dis1) are excluded per triplet.
    for tid, observed_target in [("t1", "g2"), ("t2", "g3")]:
        selected = set(assignments.loc[assignments["original_triplet_id"].eq(tid), "decoy_target_id"])
        assert observed_target not in selected

    eval_dir = run_root / "evaluation" / "inhibitory"
    universe = pd.read_csv(eval_dir / "decoy_candidate_universe.tsv", sep="\t")
    assert len(universe) == 6
    assert {"rank_out_degree", "rank_reachable_k4", "rank_kd_cell_lines"} <= set(universe.columns)
    saved_assignments = pd.read_csv(eval_dir / "decoy_matching_assignments.tsv", sep="\t")
    assert len(saved_assignments) == 6
    assert (saved_assignments["candidate_pool_type"] == td.TARGET_DECOY_CANDIDATE_POOL_TYPE).all()
    assert (eval_dir / "decoy_matching_balance.tsv").exists()
    assert (eval_dir / "decoy_matching_summary.json").exists()


def test_prepare_target_decoy_assignments_defaults_to_profile_supported_observed_universe(
    tmp_path: Path,
) -> None:
    kg_path = _toy_kg_path(tmp_path)
    cfg = SimpleNamespace(seed=0, analysis={"k_star": 4}, evaluation={})
    supported = _observed_triplets(
        [("t1", "d1", "g2", "dis1"), ("t2", "d2", "g3", "dis2")]
    )
    # g4 is in the full label input and the KD-x-KG pool, but its triplet is not in
    # the profile-supported observed cohort. The primary universe must exclude it.
    all_triplets = pd.concat(
        [supported, _observed_triplets([("t3", "d3", "g4", "dis3")])],
        ignore_index=True,
    )
    prepared = {
        "target_df": _toy_target_df(),
        "all_triplets": all_triplets,
        "profile_supported_observed_triplets": supported,
    }
    assignments, summary = runner._prepare_target_decoy_assignments(
        cfg,
        direction="inhibitory",
        prepared=prepared,
        kg_infer_path=kg_path,
        run_root=tmp_path / "run",
        # Omitted: must default to the legacy-population-matched observed pool.
        null_cfg={"n_decoys_per_triplet": 1, "exclude_known_positive_targets": True, "is_primary_spec": True},
    )
    assert (
        summary["candidate_universe_type"]
        == td.TARGET_DECOY_CANDIDATE_POOL_TYPE_PROFILE_SUPPORTED_OBSERVED
    )
    assert summary["n_candidate_universe"] == 2
    assert summary["n_profile_supported_observed_targets"] == 2
    assert summary["candidate_universe_matches_profile_supported_observed_targets"] is True
    t1_decoys = set(assignments.loc[assignments["original_triplet_id"].eq("t1"), "decoy_target_id"])
    t2_decoys = set(assignments.loc[assignments["original_triplet_id"].eq("t2"), "decoy_target_id"])
    assert t1_decoys == {"g3"}
    assert t2_decoys == {"g2"}


def test_primary_candidate_universe_requires_exact_profile_supported_target_set(
    tmp_path: Path,
) -> None:
    triplets = _observed_triplets([("t1", "d1", "gX", "dis1")])
    with pytest.raises(ValueError, match="must exactly match"):
        runner._prepare_target_decoy_assignments(
            SimpleNamespace(seed=0, analysis={"k_star": 4}, evaluation={}),
            direction="inhibitory",
            prepared={
                "target_df": _toy_target_df(),
                "all_triplets": triplets,
                "profile_supported_observed_triplets": triplets,
            },
            kg_infer_path=_toy_kg_path(tmp_path),
            run_root=tmp_path / "run",
            null_cfg={
                "n_decoys_per_triplet": 1,
                "exclude_known_positive_targets": True,
                "is_primary_spec": True,
            },
        )


def test_null_diagnostics_summary_judges_balance_on_the_acceptance_scope(tmp_path: Path) -> None:
    from val3 import report

    balance = _balance_with_scopes({})
    balance.to_csv(tmp_path / "decoy_matching_balance.tsv", sep="\t", index=False)
    td.build_target_decoy_matching_acceptance(balance).to_csv(
        tmp_path / "decoy_matching_acceptance.tsv", sep="\t", index=False
    )
    summary = report.build_null_diagnostics_summary(tmp_path)
    row = summary.loc[summary["diagnostic"].eq("matched_decoy_balance_acceptance")].iloc[0]
    # 0.9 belongs to the target-weighted / post-evaluability / all-profile-supported rows.
    assert float(row["summary_value"]) == pytest.approx(0.05)
    assert row["summary_name"].endswith("_within_threshold")
    assert row["null_type"] == TARGET_DECOY_NULL


def test_null_diagnostics_summary_falls_back_to_the_balance_table(tmp_path: Path) -> None:
    from val3 import report

    _balance_with_scopes({"directed_out_degree": 0.2}).to_csv(
        tmp_path / "decoy_matching_balance.tsv", sep="\t", index=False
    )
    summary = report.build_null_diagnostics_summary(tmp_path)
    row = summary.loc[summary["diagnostic"].eq("matched_decoy_balance_acceptance")].iloc[0]
    assert float(row["summary_value"]) == pytest.approx(0.2)
    assert row["summary_name"].endswith("_threshold_exceeded")


def test_null_evaluable_rate_states_its_denominator() -> None:
    from val3 import posthoc

    summary, _ = posthoc.build_null_evaluability(
        true_joint=_true_joint({"t1": 0.5}),
        baseline_joint=_decoy_baseline({"t1": [0.1, 0.2, 0.3]}),
    )
    row = summary.iloc[0]
    assert row["null_type"] == TARGET_DECOY_NULL
    # Only evaluable decoys reach the comparison table, so the rate is 1.0 by construction.
    assert float(row["null_evaluable_rate"]) == pytest.approx(1.0)
    assert row["null_evaluable_rate_denominator"] == "n_null_rows_in_comparison_table"
    assert row["candidate_level_evaluable_rate_source"] == "decoy_evaluability_summary.tsv"


def test_matching_diagnostics_figure_uses_the_acceptance_scope_rows() -> None:
    balance = _balance_with_scopes({"unique_kd_cell_lines": 0.14})
    triplet_evaluability = pd.DataFrame([
        {"original_triplet_id": f"t{i}", "evaluable_set_type": EVAL_TYPE,
         "n_selected_decoys": 10, "n_evaluable_decoys": i % 11}
        for i in range(30)
    ])
    fig = viz.plot_target_decoy_matching_diagnostics(
        balance, triplet_evaluability, evaluable_set_type=EVAL_TYPE
    )
    axes = fig.get_axes()
    assert len(axes) == 2
    # One bar per variable of the acceptance scope: the 3 matching variables plus the
    # total-degree diagnostic (0.9 here), which is drawn in the lighter colour.
    bars = sorted(axes[0].patches, key=lambda p: float(p.get_width()))
    assert [round(float(p.get_width()), 3) for p in bars] == [0.05, 0.05, 0.14, 0.9]
    assert bars[-1].get_facecolor() != bars[0].get_facecolor()
    assert axes[1].get_ylabel() == "Triplets"
    assert tuple(fig.get_size_inches()) == pytest.approx((7.2, 2.4))
    empty = viz.plot_target_decoy_matching_diagnostics(pd.DataFrame(), triplet_evaluability)
    assert len(empty.get_axes()) == 1


def test_report_emits_the_target_decoy_supplementary_figure_and_section(tmp_path: Path) -> None:
    from val3 import report

    balance = _balance_with_scopes({})
    dfs = {
        "decoy_matching_balance": balance,
        "decoy_matching_acceptance": td.build_target_decoy_matching_acceptance(balance),
        "decoy_evaluability_summary": pd.DataFrame([{
            "null_type": TARGET_DECOY_NULL,
            "evaluable_set_type": EVAL_TYPE,
            "n_selected_decoys_total": 100,
            "n_evaluable_decoys_total": 40,
            "candidate_level_evaluable_rate": 0.4,
            "candidate_level_evaluable_rate_denominator": "n_selected_decoys_total",
        }]),
        "decoy_triplet_evaluability": pd.DataFrame([
            {"original_triplet_id": f"t{i}", "evaluable_set_type": EVAL_TYPE,
             "n_selected_decoys": 10, "n_evaluable_decoys": i % 11}
            for i in range(10)
        ]),
        "decoy_delta_by_n_evaluable_decoys": pd.DataFrame(),
    }
    figures = report._plot_target_decoy_matching_diagnostics(dfs, tmp_path)
    assert len(figures) == 1
    eval_type, rel_png, _rel_pdf = figures[0]
    assert eval_type == EVAL_TYPE
    assert (tmp_path / rel_png.name).exists()

    md = "".join(report._target_decoy_matching_markdown_lines(dfs, {"target_decoy_matching": figures}))
    assert "Target-Decoy matching diagnostics" in md
    assert "primary_path_evaluable" in md
    assert rel_png.as_posix() in md
    assert report._target_decoy_matching_markdown_lines({}, {}) == []


def test_evaluation_summary_carries_the_model_definition_and_acceptance(tmp_path: Path) -> None:
    model = td.target_decoy_model_metadata(
        n_candidate_universe=2541,
        reach_cutoff=4,
        n_decoys_per_triplet=10,
        tie_break_seed=0,
        is_primary_spec=True,
    )
    (tmp_path / "decoy_matching_summary.json").write_text(json.dumps(model), encoding="utf-8")
    acceptance_path = tmp_path / "decoy_matching_acceptance.tsv"
    td.build_target_decoy_matching_acceptance(_balance_with_scopes({})).to_csv(
        acceptance_path, sep="\t", index=False
    )
    summary = target_decoy_summary_metadata(
        out_dir=tmp_path, outputs={"decoy_matching_acceptance": str(acceptance_path)}
    )
    assert summary["target_decoy_model"]["n_candidate_universe"] == 2541
    assert summary["target_decoy_model"]["tie_break_seed"] == 0
    assert summary["target_decoy_model"]["decoy_selection"] == "matched"
    assert summary["target_decoy_model"]["matching_variables"] == list(td.TARGET_DECOY_MATCHING_VARIABLES)
    acceptance = summary["target_decoy_matching_acceptance"]
    assert acceptance["evaluated"] is True
    assert acceptance["all_matching_variables_within_threshold"] is True
    # A run without the diagnostics still records that acceptance was not evaluated.
    assert target_decoy_summary_metadata(out_dir=tmp_path / "missing", outputs={})[
        "target_decoy_matching_acceptance"
    ]["evaluated"] is False


def test_runner_records_the_model_metadata_in_run_metadata(tmp_path: Path) -> None:
    kg_path = _toy_kg_path(tmp_path)
    cfg = SimpleNamespace(seed=0, analysis={"k_star": 4}, evaluation={})
    triplets = _observed_triplets([("t1", "d1", "g2", "dis1")])
    _assignments, summary = runner._prepare_target_decoy_assignments(
        cfg,
        direction="inhibitory",
        prepared={
            "target_df": _toy_target_df(),
            "all_triplets": triplets,
            "profile_supported_observed_triplets": triplets,
        },
        kg_infer_path=kg_path,
        run_root=tmp_path / "run",
        null_cfg={
            "n_decoys_per_triplet": 3,
            "exclude_known_positive_targets": True,
            "is_primary_spec": True,
            "decoy_candidate_universe": "l1000_kd_profile_x_signor_nodes",
        },
    )
    manifest = td.decoy_candidate_manifest({}, null_type=TARGET_DECOY_NULL)
    manifest["model_metadata"] = dict(summary)
    context: dict = {}
    runner._record_target_decoy_model_metadata(context, manifest)
    assert context["target_decoy_model"]["null_type"] == TARGET_DECOY_NULL
    assert context["target_decoy_model"]["n_candidate_universe"] == 6
    assert context["target_decoy_model"]["tie_break_seed"] == 0
    assert context["target_decoy_model"]["matching_variables"] == list(td.TARGET_DECOY_MATCHING_VARIABLES)
    # Runs without the Target-Decoy control leave the run metadata untouched.
    other: dict = {}
    runner._record_target_decoy_model_metadata(other, {"null_type": LEGACY_DECOY_NULL})
    assert other == {}


def test_metadata_records_actual_run_values() -> None:
    metadata = td.target_decoy_model_metadata(
        n_candidate_universe=17,
        reach_cutoff=4,
        n_decoys_per_triplet=10,
        tie_break_seed=3,
        is_primary_spec=True,
    )
    assert metadata["n_candidate_universe"] == 17
    assert metadata["tie_break_seed"] == 3
    assert metadata["non_evaluable_decoy_score"] == "missing"
    assert metadata["positive_exclusion_scope"] == "same_drug_disease"
