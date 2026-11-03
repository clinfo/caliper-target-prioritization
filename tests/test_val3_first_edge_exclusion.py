"""Unit tests for the Val3 first-edge exclusion sensitivity analysis (spec §8.1)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from val3 import metrics as m  # noqa: E402
from val3.config import load_val3_triplet_config  # noqa: E402

GENE_NODE_NULL = "target_khop_restricted_paired_gene_label_permutation"
TARGET_DECOY_NULL = "topology_profile_matched_target_decoy_control"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _side_row(
    *,
    triplet_id: str,
    root: str,
    cell_id: str,
    paths: list[list[str]],
    polarity: int = -1,
) -> dict:
    """One inference row expressed as node paths from ``root``.

    Polarities alternate so that every edge is derivable, mirroring what
    ``_signed_edge_payload_from_infer_row`` reconstructs from an infer parquet.
    """
    node_paths = []
    node_polarities = []
    for path in paths:
        node_paths.append(list(path))
        pols = []
        current = polarity
        for depth, _ in enumerate(path):
            current = -current if depth % 2 == 0 else current
            pols.append(current)
        node_polarities.append(pols)
    return {
        "triplet_id": triplet_id,
        "drug_id": f"DRUG:{triplet_id}",
        "target_id": root,
        "direction": "inhibitory",
        "pert_id": root,
        "cell_id": cell_id,
        "signature_reversal_paths": node_paths,
        "signature_reversal_polarities": node_polarities,
    }


def _side_frame(rows: list[dict], *, mode: str = "knockdown") -> pd.DataFrame:
    return m.signed_edge_df_from_infer_df(
        pd.DataFrame(rows),
        mode=mode,
        include_first_edge_excluded=True,
    )


# ---------------------------------------------------------------------------
# 1-2: definition of the excluded edge set
# ---------------------------------------------------------------------------

def test_root_edges_correspond_one_to_one_with_first_hops() -> None:
    frame = _side_frame([
        _side_row(triplet_id="T1", root="R", cell_id="C1", paths=[["A", "B"], ["C"], ["A", "D", "E"]]),
    ])
    row = frame.iloc[0]
    all_edges = row["signed_edges"]
    excluded = row["signed_edges_first_edge_excluded"]
    removed = all_edges - excluded

    # Simple paths start at the root and never revisit it, so the removed edges are
    # exactly the distinct first hops.
    assert removed == {e for e in all_edges if e[0] == "R"}
    first_hops = {p[0] for p in row["signature_reversal_paths"]}
    assert {e[1] for e in removed} == first_hops
    assert all(e[0] != "R" for e in excluded)


def test_downstream_edge_signs_are_unchanged_by_exclusion() -> None:
    frame = _side_frame([
        _side_row(triplet_id="T1", root="R", cell_id="C1", paths=[["A", "B"]]),
    ])
    row = frame.iloc[0]
    downstream = {e for e in row["signed_edges"] if e[0] != "R"}
    assert downstream == row["signed_edges_first_edge_excluded"]
    assert len(downstream) == 1
    assert next(iter(downstream))[:2] == ("A", "B")


def test_length_one_paths_become_empty_after_exclusion() -> None:
    frame = _side_frame([
        _side_row(triplet_id="T1", root="R", cell_id="C1", paths=[["A"], ["B"]]),
    ])
    assert frame.iloc[0]["signed_edges_first_edge_excluded"] == set()
    assert frame.iloc[0]["n_paths"] == 2


# ---------------------------------------------------------------------------
# 3-4: degenerate pairs and invariance of the evaluation sets
# ---------------------------------------------------------------------------

def _paired_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    drug = _side_frame([
        _side_row(triplet_id="T1", root="R", cell_id="C1", paths=[["A"]]),
        _side_row(triplet_id="T1", root="R", cell_id="C2", paths=[["A", "B"]]),
        _side_row(triplet_id="T2", root="S", cell_id="C1", paths=[["X", "Y"]]),
    ])
    pert = _side_frame([
        _side_row(triplet_id="T1", root="R", cell_id="C1", paths=[["A"]]),
        _side_row(triplet_id="T2", root="S", cell_id="C1", paths=[["X", "Y"]]),
    ])
    return drug, pert


def test_both_empty_after_exclusion_yields_zero_joint_jaccard() -> None:
    drug = _side_frame([_side_row(triplet_id="T1", root="R", cell_id="C1", paths=[["A"]])])
    pert = _side_frame([_side_row(triplet_id="T1", root="R", cell_id="C1", paths=[["A"]])])
    out = m.compute_joint_triplet_metrics(
        drug, pert,
        cell_pair_mode="cartesian",
        fixed_triplet_ids={"T1"},
        metrics=["jaccard"],
        evaluable_set_type="path_existing_pairs",
        edge_column_suffix=m.FIRST_EDGE_EXCLUDED_SUFFIX,
    )
    assert float(out["joint_jaccard"].iloc[0]) == 0.0
    assert int(out["n_cell_pairs_used"].iloc[0]) == 1


def test_evaluation_sets_are_identical_across_edge_scopes() -> None:
    drug, pert = _paired_frames()
    kwargs = dict(
        cell_pair_mode="cartesian",
        fixed_triplet_ids={"T1", "T2"},
        metrics=["jaccard"],
        evaluable_set_type="path_existing_pairs",
    )
    base = m.compute_joint_triplet_metrics(drug, pert, **kwargs)
    excluded = m.compute_joint_triplet_metrics(
        drug, pert, edge_column_suffix=m.FIRST_EDGE_EXCLUDED_SUFFIX, **kwargs
    )
    shared_cols = ["triplet_id", "n_cell_pairs_raw", "n_cell_pairs_used"]
    pd.testing.assert_frame_equal(base[shared_cols], excluded[shared_cols])

    sets_all = m.build_comparison_triplet_sets(
        base,
        base.assign(null_type=GENE_NODE_NULL, iteration=1),
        n_shuffle_by_null_type={GENE_NODE_NULL: 1},
    )
    sets_excluded = m.build_comparison_triplet_sets(
        excluded,
        excluded.assign(null_type=GENE_NODE_NULL, iteration=1),
        n_shuffle_by_null_type={GENE_NODE_NULL: 1},
    )
    pd.testing.assert_frame_equal(sets_all, sets_excluded)
    assert bool(sets_all["is_comparison_triplet"].all())


# ---------------------------------------------------------------------------
# 5: decoy rows are rooted at the decoy target
# ---------------------------------------------------------------------------

def test_decoy_rows_exclude_the_decoy_root_not_the_observed_target() -> None:
    row = _side_row(triplet_id="T1__decoy", root="DECOY", cell_id="C1", paths=[["OBS", "B"]])
    row["target_id"] = "OBS"
    frame = _side_frame([row])
    all_edges = frame.iloc[0]["signed_edges"]
    excluded = frame.iloc[0]["signed_edges_first_edge_excluded"]
    assert {e for e in all_edges if e[0] == "DECOY"} == all_edges - excluded
    # The observed target only appears as an intermediate node here, so its
    # outgoing edge survives.
    assert any(e[0] == "OBS" for e in excluded)


# ---------------------------------------------------------------------------
# 6: inclusion-exclusion cardinality
# ---------------------------------------------------------------------------

def test_pair_edge_counts_match_explicit_set_algebra() -> None:
    rng = np.random.default_rng(20260729)
    universe = [(f"U{i}", f"V{j}", int(s)) for i in range(6) for j in range(6) for s in (-1, 1)]
    for _ in range(200):
        n_a = int(rng.integers(0, 12))
        n_b = int(rng.integers(0, 12))
        ea = {universe[k] for k in rng.choice(len(universe), size=n_a, replace=False)}
        eb = {universe[k] for k in rng.choice(len(universe), size=n_b, replace=False)}
        n_inter, n_union, n_drug, n_pert = m._pair_edge_counts(ea, eb)
        assert n_inter == len(ea & eb)
        assert n_union == len(ea | eb)
        assert (n_drug, n_pert) == (len(ea), len(eb))
        for metric, fn in [("jaccard", m.metric_jaccard)]:
            expected = fn(ea, eb)
            actual = m.metric_value_from_counts(
                metric, n_intersection=n_inter, n_drug_edges=n_drug, n_pert_edges=n_pert
            )
            if pd.isna(expected):
                assert pd.isna(actual)
            else:
                assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# 7: the cross-scope summary table
# ---------------------------------------------------------------------------

def _comparison_frame(scope_delta: float) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "null_type": GENE_NODE_NULL,
            "evaluable_set_type": "path_existing_pairs",
            "triplet_id": tid,
            "statistic": "joint_jaccard",
            "observed_value": 0.5 + i * 0.1,
            "baseline_representative": "median",
            "baseline_mean": 0.5 + i * 0.1 - scope_delta,
            "baseline_median": 0.5 + i * 0.1 - scope_delta,
            "delta": scope_delta,
        }
        for i, tid in enumerate(["T1", "T2"])
    ])


def test_first_edge_exclusion_summary_delta_matches_comparison_tables() -> None:
    comparison_by_scope = {
        m.EDGE_SCOPE_ALL: _comparison_frame(0.20),
        m.EDGE_SCOPE_FIRST_EDGE_EXCLUDED: _comparison_frame(0.05),
    }
    summary = m.build_first_edge_exclusion_summary(
        comparison_by_scope=comparison_by_scope,
        diagnostics=None,
    )
    row = summary.iloc[0]
    assert row["delta_mean_all_edges"] == pytest.approx(0.20)
    assert row["delta_mean_first_edge_excluded"] == pytest.approx(0.05)
    assert row["delta_mean_ratio"] == pytest.approx(0.25)
    assert int(row["n_comparison_triplets"]) == 2
    assert row["observed_mean_all_edges"] == pytest.approx(0.55)
    assert row["null_median_all_edges"] == pytest.approx(0.35)
    assert list(summary.columns) == m.FIRST_EDGE_EXCLUSION_SUMMARY_COLUMNS
    # The removed group-resampled null no longer contributes p-value columns.
    assert not [c for c in summary.columns if str(c).startswith("p_finite_null")]


def test_first_edge_exclusion_summary_rejects_changed_comparison_sets() -> None:
    changed = _comparison_frame(0.05).iloc[:1]
    with pytest.raises(AssertionError, match="comparison triplet set"):
        m.build_first_edge_exclusion_summary(
            comparison_by_scope={
                m.EDGE_SCOPE_ALL: _comparison_frame(0.20),
                m.EDGE_SCOPE_FIRST_EDGE_EXCLUDED: changed,
            },
            diagnostics=None,
        )


# ---------------------------------------------------------------------------
# 8: config
# ---------------------------------------------------------------------------

def _config_dict(**evaluation_overrides) -> dict:
    evaluation = {
        "metrics": ["jaccard"],
        "evaluable_sets": {"path_existing_pairs": {"enabled": True}},
        "global_effect_size": {"enabled": True},
    }
    evaluation.update(evaluation_overrides)
    return {
        "run": {"seed": 0, "date_tag": "20260729", "output_root": "./out", "output_name": "run",
                "registry_path": "./registry.yaml"},
        "data": {"kg": "signor", "triplet_source": "targetnexus_db"},
        "sections": {"prepare_inputs": True, "path_exploration": True, "evaluation": True,
                     "make_visualization_tables": False, "make_report": False},
        "analysis": {"k_star": 4, "gap": 1, "directions": ["inhibitory"], "cell_pair_mode": "cartesian"},
        "infer": {"path_strategy": "simple", "restrict_baseline_to_observed_evaluable": False},
        "baseline": {
            "n_shuffle": 100,
            "is_primary_spec": True,
            "null_models": {GENE_NODE_NULL: {"enabled": True, "min_local_bg_genes": 0}},
        },
        "evaluation": evaluation,
        "progress": {"enabled": False, "use_tqdm": False},
    }


def _write(tmp_path: Path, cfg: dict) -> Path:
    path = tmp_path / "val3.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_edge_scopes_and_max_workers_reach_the_resolved_config(tmp_path: Path) -> None:
    cfg = load_val3_triplet_config(_write(tmp_path, _config_dict(
        edge_scopes={"all_edges": {"enabled": True}, "first_edge_excluded": {"enabled": True}},
        max_workers=3,
    )))
    assert cfg.evaluation["edge_scopes"]["first_edge_excluded"]["enabled"] is True
    assert cfg.evaluation["max_workers"] == 3
    assert m._enabled_edge_scopes(cfg.evaluation) == [m.EDGE_SCOPE_ALL, m.EDGE_SCOPE_FIRST_EDGE_EXCLUDED]


def test_disabling_all_edges_scope_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="all_edges.enabled must be true"):
        load_val3_triplet_config(_write(tmp_path, _config_dict(
            edge_scopes={"all_edges": {"enabled": False}},
        )))


def test_config_without_edge_scopes_disables_first_edge_scope(tmp_path: Path) -> None:
    cfg = load_val3_triplet_config(_write(tmp_path, _config_dict()))
    assert cfg.evaluation["edge_scopes"]["first_edge_excluded"]["enabled"] is False
    assert cfg.evaluation["max_workers"] == 1
    assert m._enabled_edge_scopes(cfg.evaluation) == [m.EDGE_SCOPE_ALL]


def test_max_workers_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_workers must be >= 1"):
        load_val3_triplet_config(_write(tmp_path, _config_dict(max_workers=0)))


# ---------------------------------------------------------------------------
# 9: Null 2 keeps its resample stream when the first-edge scope is added
# ---------------------------------------------------------------------------

def _structural_pool() -> dict:
    return {
        "R": {
            1: [{("R", "A", 1)}, {("R", "B", -1)}],
            2: [{("R", "A", 1), ("A", "C", -1)}, {("R", "B", -1), ("B", "D", 1)}],
        }
    }


def test_structural_resampling_draws_are_unchanged_by_the_first_edge_scope() -> None:
    lengths = [1, 2, 2, 1, 2]
    pool = _structural_pool()

    baseline_rng = np.random.default_rng(5)
    sampled_all, empty = m._sample_target_edges_for_lengths(
        lengths, pool, target_id="R", side="drug", rng=baseline_rng,
    )
    assert empty == set()

    both_rng = np.random.default_rng(5)
    sampled_both, sampled_excluded = m._sample_target_edges_for_lengths(
        lengths, pool, target_id="R", side="drug", rng=both_rng, with_first_edge_excluded=True,
    )
    assert sampled_both == sampled_all
    # Same number of draws consumed, so downstream iterations see the same stream.
    assert baseline_rng.integers(0, 1 << 30, size=4).tolist() == both_rng.integers(0, 1 << 30, size=4).tolist()

    # The excluded set is derived from the very same drawn paths, not a new draw.
    assert sampled_excluded == {e for e in sampled_all if e[0] != "R"}


def test_structural_resample_index_stream_is_recorded_once_per_observed_path() -> None:
    pool = _structural_pool()
    lengths = [1, 2, 1]
    expected_rng = np.random.default_rng(17)
    expected_indices = [
        int(expected_rng.integers(0, len(pool["R"][length]))) for length in lengths
    ]
    expected_edges: set = set()
    for length, idx in zip(lengths, expected_indices):
        expected_edges |= pool["R"][length][idx]

    sampled, excluded = m._sample_target_edges_for_lengths(
        lengths, pool, target_id="R", side="pert",
        rng=np.random.default_rng(17), with_first_edge_excluded=True,
    )
    assert sampled == expected_edges
    assert excluded == {e for e in expected_edges if e[0] != "R"}


# ---------------------------------------------------------------------------
# 10: the diagnostics accumulator
# ---------------------------------------------------------------------------

def test_all_edges_path_is_untouched_without_an_accumulator() -> None:
    drug, pert = _paired_frames()
    pair = m.build_cell_pairs(drug, pert, cell_pair_mode="cartesian")
    pair["both_paths"] = True
    with_acc = m._joint_pair_values(
        pair, ["jaccard"],
        edge_column_suffix="",
        diagnostics_accumulator=m.FirstEdgeExclusionDiagnostics().scoped(
            null_type="observed", evaluable_set_type="path_existing_pairs",
        ),
    )
    without_acc = m._joint_pair_values(pair, ["jaccard"], edge_column_suffix="")
    pd.testing.assert_frame_equal(with_acc, without_acc)


def test_accumulator_matches_naive_pair_enumeration() -> None:
    drug, pert = _paired_frames()
    pair = m.build_cell_pairs(drug, pert, cell_pair_mode="cartesian")
    pair["both_paths"] = pd.to_numeric(pair["n_drug_paths"]).gt(0) & pd.to_numeric(pair["n_pert_paths"]).gt(0)

    accumulator = m.FirstEdgeExclusionDiagnostics()
    m._joint_pair_values(
        pair,
        ["jaccard"],
        edge_column_suffix=m.FIRST_EDGE_EXCLUDED_SUFFIX,
        diagnostics_accumulator=accumulator.scoped(
            null_type="observed", evaluable_set_type="path_existing_pairs",
        ),
    )
    frame = accumulator.to_frame()
    assert len(frame) == 1

    removed_drug: list[int] = []
    removed_pert: list[int] = []
    n_empty = 0
    excluded = m.FIRST_EDGE_EXCLUDED_SUFFIX
    for rec in pair.to_dict(orient="records"):
        d_excl = rec[f"signed_edges{excluded}_drug"]
        p_excl = rec[f"signed_edges{excluded}_pert"]
        removed_drug.append(len(rec["signed_edges_drug"]) - len(d_excl))
        removed_pert.append(len(rec["signed_edges_pert"]) - len(p_excl))
        if bool(rec["both_paths"]) and (not d_excl or not p_excl):
            n_empty += 1
    assert int(frame["n_pairs_empty_after_exclusion"].iloc[0]) == n_empty
    assert frame["n_edges_removed_median_drug"].iloc[0] == pytest.approx(float(np.median(removed_drug)))
    assert frame["n_edges_removed_median_pert"].iloc[0] == pytest.approx(float(np.median(removed_pert)))


def test_weighted_path_accumulator_matches_expanded_pairs() -> None:
    """Null 3 goes through the weighted route; its counts must be cell-pair weighted."""
    drug, pert = _paired_frames()
    pair = m.build_cell_pairs(drug, pert, cell_pair_mode="cartesian")
    pair["both_paths"] = pd.to_numeric(pair["n_drug_paths"]).gt(0) & pd.to_numeric(pair["n_pert_paths"]).gt(0)

    expanded = m.FirstEdgeExclusionDiagnostics()
    m._joint_pair_values(
        pair, ["jaccard"],
        edge_column_suffix=m.FIRST_EDGE_EXCLUDED_SUFFIX,
        diagnostics_accumulator=expanded.scoped(null_type="n3", evaluable_set_type="path_existing_pairs"),
    )
    weighted = m.FirstEdgeExclusionDiagnostics()
    m.compute_joint_triplet_metrics_weighted(
        drug, pert,
        cell_pair_mode="cartesian",
        fixed_triplet_ids={"T1", "T2"},
        metrics=["jaccard"],
        evaluable_set_type="path_existing_pairs",
        edge_column_suffix=m.FIRST_EDGE_EXCLUDED_SUFFIX,
        diagnostics_accumulator=weighted.scoped(null_type="n3", evaluable_set_type="path_existing_pairs"),
    )
    pd.testing.assert_frame_equal(expanded.to_frame(), weighted.to_frame())
