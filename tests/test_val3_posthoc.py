from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from val3 import posthoc
from val3.streaming import write_parquet_zstd


GENE_NODE_NULL = "target_khop_restricted_paired_gene_label_permutation"
PATH_SHUFFLE_NULL = "path_count_length_matched_structural_resampling"
DECOY_NULL = "observed_target_pool_decoy_control"
TARGET_DECOY_NULL = "topology_profile_matched_target_decoy_control"


def _comparison_df() -> pd.DataFrame:
    rows = []
    for null_type, deltas in [
        (GENE_NODE_NULL, [0.20, -0.10, 0.04]),
        (DECOY_NULL, [0.05, 0.05, -0.15]),
    ]:
        for triplet_idx, delta in enumerate(deltas, start=1):
            rows.append({
                "null_type": null_type,
                "evaluable_set_type": "path_existing_pairs",
                "triplet_id": f"trp_{triplet_idx}",
                "drug_id": f"drug_{1 if triplet_idx < 3 else 2}",
                "target_id": f"target_{triplet_idx}",
                "statistic": "joint_jaccard",
                "observed_value": 0.30,
                "baseline_median": 0.30 - delta,
                "delta": delta,
            })
    return pd.DataFrame(rows)


def _group_map() -> pd.DataFrame:
    return pd.DataFrame({
        "triplet_id": ["trp_1", "trp_2", "trp_3"],
        "drug_id": ["drug_1", "drug_1", "drug_2"],
        "target_id": ["target_1", "target_2", "target_3"],
        "disease_id": ["DOID:1", "DOID:1", "DOID:2"],
    })


def _write_cellpair_debug(
    run_root: Path,
    *,
    source: str,
    null_type: str,
    iteration: int | None,
    rows: list[dict[str, object]],
) -> None:
    iter_label = "NA" if iteration is None else str(int(iteration))
    part_dir = (
        run_root / "evaluation" / "inhibitory" / posthoc.CELLPAIR_DEBUG_DIRNAME
        / f"source={source}" / f"null_type={null_type}"
        / "evaluable_set_type=path_existing_pairs" / f"iteration={iter_label}"
    )
    part_dir.mkdir(parents=True, exist_ok=True)
    write_parquet_zstd(pd.DataFrame(rows), part_dir / "part-00000.parquet")


def test_triplet_delta_table_adds_disease_and_recomputes_missing_delta() -> None:
    comparison = _comparison_df().drop(columns=["delta"])
    out = posthoc.build_triplet_delta_table(comparison, group_map=_group_map())

    assert set(out["null_type"]) == {GENE_NODE_NULL, DECOY_NULL}
    assert out["disease_id"].tolist() == ["DOID:1", "DOID:1", "DOID:2"] * 2
    gene_node = out.loc[out["null_type"].eq(GENE_NODE_NULL)].set_index("triplet_id")
    assert gene_node.loc["trp_1", "delta"] == pytest.approx(0.20)
    assert gene_node.loc["trp_2", "delta"] == pytest.approx(-0.10)


def test_triplet_delta_table_drops_legacy_null_types_and_other_statistics() -> None:
    comparison = _comparison_df()
    comparison.loc[comparison.index[0], "null_type"] = "global_paired_gene_label_permutation"
    extra = comparison.tail(1).copy()
    extra["statistic"] = "joint_precision"
    out = posthoc.build_triplet_delta_table(
        pd.concat([comparison, extra], ignore_index=True),
        group_map=_group_map(),
        statistics=("joint_jaccard",),
    )

    assert "global_paired_gene_label_permutation" not in set(out["null_type"])
    assert set(out["statistic"]) == {"joint_jaccard"}


def test_summarize_triplet_delta_distribution_counts_signs_and_ids() -> None:
    summary = posthoc.summarize_triplet_delta_distribution(
        posthoc.build_triplet_delta_table(_comparison_df(), group_map=_group_map())
    )
    gene_node = summary.loc[summary["null_type"].eq(GENE_NODE_NULL)].iloc[0]

    assert int(gene_node["n_triplets"]) == 3
    assert int(gene_node["n_drugs"]) == 2
    assert int(gene_node["n_diseases"]) == 2
    assert int(gene_node["n_positive_delta"]) == 2
    assert int(gene_node["n_negative_delta"]) == 1
    assert float(gene_node["delta_median"]) == pytest.approx(0.04)
    assert float(gene_node["fraction_positive_delta"]) == pytest.approx(2 / 3)


def test_cluster_bootstrap_uses_requested_cluster_units() -> None:
    delta = posthoc.build_triplet_delta_table(_comparison_df(), group_map=_group_map())
    out = posthoc.build_delta_cluster_bootstrap(delta, direction="inhibitory", seed=0, n_bootstrap=200)

    gene_node = out.loc[out["null_type"].eq(GENE_NODE_NULL)].set_index("cluster_level")
    assert set(gene_node.index) == set(posthoc.CLUSTER_LEVELS)
    assert int(gene_node.loc["triplet", "n_clusters"]) == 3
    assert int(gene_node.loc["drug", "n_clusters"]) == 2
    assert int(gene_node.loc["disease", "n_clusters"]) == 2
    for level in posthoc.CLUSTER_LEVELS:
        row = gene_node.loc[level]
        assert float(row["delta_mean_hat"]) == pytest.approx(np.mean([0.20, -0.10, 0.04]))
        assert float(row["delta_mean_ci025"]) <= float(row["delta_mean_hat"]) <= float(row["delta_mean_ci975"])
    # Drug clustering pools trp_1/trp_2, so its CI cannot coincide with the triplet-level one.
    assert float(gene_node.loc["drug", "delta_mean_ci025"]) != pytest.approx(
        float(gene_node.loc["triplet", "delta_mean_ci025"])
    )


def test_cluster_bootstrap_is_deterministic_for_a_seed() -> None:
    delta = posthoc.build_triplet_delta_table(_comparison_df(), group_map=_group_map())
    first = posthoc.build_delta_cluster_bootstrap(delta, direction="inhibitory", seed=7, n_bootstrap=100)
    second = posthoc.build_delta_cluster_bootstrap(delta, direction="inhibitory", seed=7, n_bootstrap=100)
    third = posthoc.build_delta_cluster_bootstrap(delta, direction="inhibitory", seed=8, n_bootstrap=100)

    pd.testing.assert_frame_equal(first, second)
    assert not first["delta_mean_ci025"].equals(third["delta_mean_ci025"])


def test_cluster_bootstrap_rejects_unknown_level() -> None:
    delta = posthoc.build_triplet_delta_table(_comparison_df(), group_map=_group_map())
    with pytest.raises(ValueError, match="Unsupported cluster level"):
        posthoc.build_delta_cluster_bootstrap(
            delta, direction="inhibitory", seed=0, n_bootstrap=10, cluster_levels=("cell",),
        )


def _joint_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    true_joint = pd.DataFrame({
        "evaluable_set_type": ["path_existing_pairs"] * 3,
        "triplet_id": ["trp_1", "trp_2", "trp_3"],
        "drug_id": ["drug_1", "drug_1", "drug_2"],
        "target_id": ["target_1", "target_2", "target_3"],
        "n_cell_pairs_used": [10, 4, 6],
        "joint_jaccard": [0.30, 0.30, 0.30],
    })
    baseline_rows = []
    for iteration in (1, 2):
        for triplet_id, value in [("trp_1", 0.10), ("trp_2", 0.20), ("trp_3", np.nan)]:
            baseline_rows.append({
                "null_type": GENE_NODE_NULL,
                "evaluable_set_type": "path_existing_pairs",
                "iteration": iteration,
                "triplet_id": triplet_id,
                "drug_id": "drug_1" if triplet_id != "trp_3" else "drug_2",
                "target_id": triplet_id.replace("trp", "target"),
                "n_cell_pairs_used": 5,
                "joint_jaccard": value,
            })
    return true_joint, pd.DataFrame(baseline_rows)


def test_null_evaluability_reports_rate_and_dropout_triplets() -> None:
    true_joint, baseline_joint = _joint_frames()
    summary, by_triplet = posthoc.build_null_evaluability(
        true_joint=true_joint, baseline_joint=baseline_joint,
    )
    row = summary.iloc[0]

    assert int(row["n_null_rows"]) == 6
    assert int(row["n_null_rows_evaluable"]) == 4
    assert float(row["null_evaluable_rate"]) == pytest.approx(4 / 6)
    assert int(row["n_triplets_all_draws_evaluable"]) == 2
    assert int(row["n_triplets_no_null_score"]) == 1
    assert set(by_triplet.loc[~by_triplet["has_null_score"], "triplet_id"]) == {"trp_3"}


def test_null_repeat_stability_converges_to_the_full_repeat_estimate() -> None:
    true_joint, baseline_joint = _joint_frames()
    out = posthoc.build_null_repeat_stability(
        true_joint=true_joint,
        baseline_joint=baseline_joint,
        direction="inhibitory",
        checkpoints=(1, 2),
    )

    assert out["n_repeats"].tolist() == [1, 2]
    assert out["n_triplets"].tolist() == [2, 2]
    assert out["full_n_repeats"].tolist() == [2, 2]
    assert out["delta_mean"].tolist() == pytest.approx([0.15, 0.15])
    assert out["delta_mean_full"].tolist() == pytest.approx([0.15, 0.15])
    assert out["abs_delta_mean_change_from_full"].tolist() == pytest.approx([0.0, 0.0])


def test_null_evaluability_sensitivity_imputes_zero_delta() -> None:
    true_joint, baseline_joint = _joint_frames()
    _summary, by_triplet = posthoc.build_null_evaluability(
        true_joint=true_joint, baseline_joint=baseline_joint,
    )
    delta = pd.DataFrame({
        "null_type": [GENE_NODE_NULL, GENE_NODE_NULL],
        "evaluable_set_type": ["path_existing_pairs"] * 2,
        "statistic": ["joint_jaccard"] * 2,
        "triplet_id": ["trp_1", "trp_2"],
        "delta": [0.20, 0.10],
    })
    out = posthoc.build_null_evaluability_sensitivity(delta, by_triplet).iloc[0]

    assert int(out["n_triplets_primary"]) == 2
    assert float(out["delta_mean_primary"]) == pytest.approx(0.15)
    assert int(out["n_triplets_without_null_score"]) == 1
    assert int(out["n_triplets_zero_imputed"]) == 3
    assert float(out["delta_mean_zero_imputed"]) == pytest.approx(0.10)


def test_cell_scope_metrics_reproduce_triplet_mean_and_same_cell_subset(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_cellpair_debug(
        run_root,
        source="observed",
        null_type="observed",
        iteration=None,
        rows=[
            {"triplet_id": "trp_1", "cell_id_drug": "A375", "cell_id_pert": "A375", "jaccard": 0.4, "sensitivity": 0.5, "precision": 0.6},
            {"triplet_id": "trp_1", "cell_id_drug": "PC3", "cell_id_pert": "A375", "jaccard": 0.2, "sensitivity": 0.3, "precision": 0.4},
        ],
    )
    _write_cellpair_debug(
        run_root,
        source="null",
        null_type=GENE_NODE_NULL,
        iteration=1,
        rows=[
            {"triplet_id": "trp_1", "cell_id_drug": "A375", "cell_id_pert": "A375", "jaccard": 0.1, "sensitivity": 0.1, "precision": 0.1},
            {"triplet_id": "trp_1", "cell_id_drug": "PC3", "cell_id_pert": "A375", "jaccard": 0.3, "sensitivity": 0.3, "precision": 0.3},
        ],
    )

    metrics = posthoc.build_cell_scope_triplet_metrics(run_root, "inhibitory")
    jaccard = metrics.loc[metrics["statistic"].eq("joint_jaccard")].set_index(["cell_scope", "source"])

    assert float(jaccard.loc[("all_cell_pairs", "observed"), "value"]) == pytest.approx(0.3)
    assert float(jaccard.loc[("same_cell", "observed"), "value"]) == pytest.approx(0.4)
    assert int(jaccard.loc[("same_cell", "observed"), "n_cell_pairs_used"]) == 1

    by_triplet, summary = posthoc.build_cell_scope_delta(metrics)
    same_cell = summary.loc[
        summary["cell_scope"].eq("same_cell") & summary["statistic"].eq("joint_jaccard")
    ].iloc[0]
    all_pairs = summary.loc[
        summary["cell_scope"].eq("all_cell_pairs") & summary["statistic"].eq("joint_jaccard")
    ].iloc[0]

    assert float(same_cell["delta_mean"]) == pytest.approx(0.4 - 0.1)
    assert float(all_pairs["delta_mean"]) == pytest.approx(0.3 - 0.2)
    assert set(by_triplet["cell_scope"]) == {"all_cell_pairs", "same_cell"}
    assert set(summary["null_type"]) == {GENE_NODE_NULL}
    assert "baseline_median" in by_triplet.columns and "baseline_mean" not in by_triplet.columns
    assert "baseline_median_mean" in summary.columns


def test_cell_scope_delta_wilcoxon_is_computed_per_cell_scope(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    triplets = ["trp_1", "trp_2", "trp_3"]
    same_cell_observed = [0.6, 0.5, 0.7]
    cross_cell_observed = [0.2, 0.4, 0.1]
    # Same-cell deltas are 0.3 / 0.2 / 0.4 for the gene-label null, while the all-cell-pairs
    # aggregate mixes in the cross-cell pairs and gives 0.15 / 0.20 / 0.15.
    same_cell_null = {GENE_NODE_NULL: [0.3, 0.3, 0.3], DECOY_NULL: [0.59, 0.52, 0.67]}
    cross_cell_null = {GENE_NODE_NULL: [0.2, 0.2, 0.2], DECOY_NULL: [0.1, 0.1, 0.1]}

    def _rows(same: list[float], cross: list[float]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for triplet_id, s, x in zip(triplets, same, cross):
            rows.append({"triplet_id": triplet_id, "cell_id_drug": "A375", "cell_id_pert": "A375",
                         "jaccard": s, "sensitivity": s, "precision": s})
            rows.append({"triplet_id": triplet_id, "cell_id_drug": "PC3", "cell_id_pert": "A375",
                         "jaccard": x, "sensitivity": x, "precision": x})
        return rows

    _write_cellpair_debug(
        run_root, source="observed", null_type="observed", iteration=None,
        rows=_rows(same_cell_observed, cross_cell_observed),
    )
    for null_type in [GENE_NODE_NULL, DECOY_NULL]:
        _write_cellpair_debug(
            run_root, source="null", null_type=null_type, iteration=1,
            rows=_rows(same_cell_null[null_type], cross_cell_null[null_type]),
        )

    metrics = posthoc.build_cell_scope_triplet_metrics(run_root, "inhibitory")
    _by_triplet, summary = posthoc.build_cell_scope_delta(metrics)
    jaccard = summary.loc[summary["statistic"].eq("joint_jaccard")].set_index(["cell_scope", "null_type"])

    same_cell_gene = jaccard.loc[("same_cell", GENE_NODE_NULL)]
    assert int(same_cell_gene["n_triplets"]) == 3
    assert str(same_cell_gene["wilcoxon_two_sided_status"]) == "computed"
    assert int(same_cell_gene["n_positive_delta"]) == 3
    assert int(same_cell_gene["n_negative_delta"]) == 0
    assert float(same_cell_gene["wilcoxon_two_sided_statistic"]) == pytest.approx(0.0)
    assert float(same_cell_gene["p_wilcoxon_two_sided"]) == pytest.approx(0.25)

    # Holm runs over the null models of one cell scope, so the same-cell subset carries its
    # own corrected p values rather than the all-cell-pairs ones.
    p_gene = float(same_cell_gene["p_wilcoxon_two_sided"])
    p_decoy = float(jaccard.loc[("same_cell", DECOY_NULL), "p_wilcoxon_two_sided"])
    assert p_gene < p_decoy
    assert float(same_cell_gene["p_wilcoxon_two_sided_holm"]) == pytest.approx(
        min(1.0, 2 * p_gene)
    )
    assert float(
        jaccard.loc[("same_cell", DECOY_NULL), "p_wilcoxon_two_sided_holm"]
    ) == pytest.approx(
        min(1.0, max(2 * p_gene, p_decoy))
    )


def test_cell_scope_delta_wilcoxon_accepts_a_single_nonzero_triplet(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_cellpair_debug(
        run_root, source="observed", null_type="observed", iteration=None,
        rows=[{"triplet_id": "trp_1", "cell_id_drug": "A375", "cell_id_pert": "A375",
               "jaccard": 0.5, "sensitivity": 0.5, "precision": 0.5}],
    )
    _write_cellpair_debug(
        run_root, source="null", null_type=GENE_NODE_NULL, iteration=1,
        rows=[{"triplet_id": "trp_1", "cell_id_drug": "A375", "cell_id_pert": "A375",
               "jaccard": 0.2, "sensitivity": 0.2, "precision": 0.2}],
    )

    metrics = posthoc.build_cell_scope_triplet_metrics(run_root, "inhibitory")
    _by_triplet, summary = posthoc.build_cell_scope_delta(metrics)
    row = summary.loc[
        summary["cell_scope"].eq("same_cell") & summary["statistic"].eq("joint_jaccard")
    ].iloc[0]

    assert str(row["wilcoxon_two_sided_status"]) == "computed"
    assert int(row["n_nonzero_delta"]) == 1
    assert float(row["wilcoxon_two_sided_statistic"]) == pytest.approx(0.0)
    assert float(row["p_wilcoxon_two_sided"]) == pytest.approx(1.0)
    assert float(row["p_wilcoxon_two_sided_holm"]) == pytest.approx(1.0)


def test_wilcoxon_all_zero_delta_does_not_fall_back_to_another_test() -> None:
    statistic, p_value, status = posthoc.wilcoxon_signed_rank_two_sided([0.0, 0.0])
    assert np.isnan(statistic)
    assert np.isnan(p_value)
    assert status == "no_nonzero_delta"


def _decoy_map(run_root: Path, rows: list[dict[str, str]]) -> pd.DataFrame:
    evaluation = run_root / "evaluation" / "inhibitory"
    evaluation.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        evaluation / posthoc.DECOY_ASSIGNMENTS_FILENAME, sep="\t", index=False,
    )
    return posthoc.load_decoy_pseudo_triplet_map(run_root, "inhibitory")


def test_cell_scope_delta_folds_decoy_pseudo_triplets_onto_the_original(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_cellpair_debug(
        run_root, source="observed", null_type="observed", iteration=None,
        rows=[
            {"triplet_id": "trp_1", "cell_id_drug": "A375", "cell_id_pert": "A375", "jaccard": 0.6, "sensitivity": 0.6, "precision": 0.6},
            {"triplet_id": "trp_1", "cell_id_drug": "PC3", "cell_id_pert": "A375", "jaccard": 0.2, "sensitivity": 0.2, "precision": 0.2},
        ],
    )
    # Two decoys of trp_1 plus one whose assignment row is missing from the map.
    _write_cellpair_debug(
        run_root, source="null", null_type=TARGET_DECOY_NULL, iteration=None,
        rows=[
            {"triplet_id": "trp_1__matched_decoy__d1", "cell_id_drug": "A375", "cell_id_pert": "A375", "jaccard": 0.1, "sensitivity": 0.1, "precision": 0.1},
            {"triplet_id": "trp_1__matched_decoy__d2", "cell_id_drug": "A375", "cell_id_pert": "A375", "jaccard": 0.3, "sensitivity": 0.3, "precision": 0.3},
            {"triplet_id": "trp_1__matched_decoy__d9", "cell_id_drug": "A375", "cell_id_pert": "A375", "jaccard": 0.9, "sensitivity": 0.9, "precision": 0.9},
        ],
    )
    # Null 1 gets two iterations on the same triplet, aggregated by mean as well.
    for iteration, value in [(1, 0.1), (2, 0.5)]:
        _write_cellpair_debug(
            run_root, source="null", null_type=GENE_NODE_NULL, iteration=iteration,
            rows=[{"triplet_id": "trp_1", "cell_id_drug": "A375", "cell_id_pert": "A375",
                   "jaccard": value, "sensitivity": value, "precision": value}],
        )
    decoy_map = _decoy_map(run_root, [
        {"pseudo_triplet_id": "trp_1__matched_decoy__d1", "original_triplet_id": "trp_1", "decoy_target_id": "d1"},
        {"pseudo_triplet_id": "trp_1__matched_decoy__d2", "original_triplet_id": "trp_1", "decoy_target_id": "d2"},
    ])
    assert list(decoy_map.columns) == ["pseudo_triplet_id", "original_triplet_id", "decoy_target_id"]

    metrics = posthoc.build_cell_scope_triplet_metrics(run_root, "inhibitory")
    by_triplet, summary = posthoc.build_cell_scope_delta(metrics, decoy_triplet_map=decoy_map)

    decoy_row = summary.loc[
        summary["null_type"].eq(TARGET_DECOY_NULL)
        & summary["cell_scope"].eq("same_cell")
        & summary["statistic"].eq("joint_jaccard")
    ].iloc[0]
    # median(0.1, 0.3) over the two mapped decoys; the unmapped pseudo id is dropped.
    assert float(decoy_row["baseline_median_mean"]) == pytest.approx(0.2)
    assert float(decoy_row["delta_mean"]) == pytest.approx(0.6 - 0.2)
    assert int(decoy_row["n_triplets"]) == 1
    decoy_by_triplet = by_triplet.loc[
        by_triplet["null_type"].eq(TARGET_DECOY_NULL) & by_triplet["cell_scope"].eq("same_cell")
        & by_triplet["statistic"].eq("joint_jaccard")
    ].iloc[0]
    assert decoy_by_triplet["triplet_id"] == "trp_1"
    assert int(decoy_by_triplet["n_null_draws"]) == 2
    assert float(decoy_by_triplet["baseline_median"]) == pytest.approx(0.2)

    gene_row = summary.loc[
        summary["null_type"].eq(GENE_NODE_NULL)
        & summary["cell_scope"].eq("same_cell")
        & summary["statistic"].eq("joint_jaccard")
    ].iloc[0]
    # The repeat-driven null uses the same median aggregation over its iterations.
    assert float(gene_row["baseline_median_mean"]) == pytest.approx(0.3)
    assert float(gene_row["delta_mean"]) == pytest.approx(0.6 - 0.3)


def test_decoy_pseudo_triplet_map_is_empty_without_the_assignment_table(tmp_path: Path) -> None:
    out = posthoc.load_decoy_pseudo_triplet_map(tmp_path / "run", "inhibitory")
    assert out.empty
    assert list(out.columns) == ["pseudo_triplet_id", "original_triplet_id", "decoy_target_id"]


def test_cell_scope_metrics_sum_shards_before_averaging(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    part_dir = (
        run_root / "evaluation" / "inhibitory" / posthoc.CELLPAIR_DEBUG_DIRNAME
        / "source=observed" / "null_type=observed"
        / "evaluable_set_type=path_existing_pairs" / "iteration=NA"
    )
    part_dir.mkdir(parents=True)
    write_parquet_zstd(
        pd.DataFrame([
            {"triplet_id": "trp_1", "cell_id_drug": "A375", "cell_id_pert": "A375", "jaccard": 0.6, "sensitivity": 0.6, "precision": 0.6},
            {"triplet_id": "trp_1", "cell_id_drug": "PC3", "cell_id_pert": "PC3", "jaccard": 0.6, "sensitivity": 0.6, "precision": 0.6},
            {"triplet_id": "trp_1", "cell_id_drug": "HT29", "cell_id_pert": "HT29", "jaccard": 0.6, "sensitivity": 0.6, "precision": 0.6},
        ]),
        part_dir / "part-00000.parquet",
    )
    write_parquet_zstd(
        pd.DataFrame([
            {"triplet_id": "trp_1", "cell_id_drug": "MCF7", "cell_id_pert": "MCF7", "jaccard": 0.2, "sensitivity": 0.2, "precision": 0.2},
        ]),
        part_dir / "part-00001.parquet",
    )

    metrics = posthoc.build_cell_scope_triplet_metrics(run_root, "inhibitory")
    row = metrics.loc[
        metrics["statistic"].eq("joint_jaccard") & metrics["cell_scope"].eq("all_cell_pairs")
    ].iloc[0]

    # Pooled mean over 4 cell pairs, not the mean of the two shard means.
    assert float(row["value"]) == pytest.approx(0.5)
    assert int(row["n_cell_pairs_used"]) == 4


def test_evaluated_scope_summary_counts_ids_and_null_evaluability() -> None:
    true_joint, baseline_joint = _joint_frames()
    comparison_sets = pd.DataFrame({
        "null_type": [GENE_NODE_NULL] * 3,
        "evaluable_set_type": ["path_existing_pairs"] * 3,
        "triplet_id": ["trp_1", "trp_2", "trp_3"],
        "is_comparison_triplet": [True, True, False],
    })
    out = posthoc.build_evaluated_scope_summary(
        true_joint=true_joint,
        baseline_joint=baseline_joint,
        comparison_sets=comparison_sets,
        group_map=_group_map(),
    )
    observed = out.loc[out["scope"].eq("observed")].iloc[0]
    null_row = out.loc[out["scope"].eq("null")].iloc[0]

    assert int(observed["n_triplets"]) == 3
    assert int(observed["n_unique_drugs"]) == 2
    assert int(observed["n_unique_diseases"]) == 2
    assert int(observed["n_cell_pairs_used_total"]) == 20
    assert int(null_row["n_comparison_triplets"]) == 2
    assert float(null_row["null_evaluable_rate"]) == pytest.approx(4 / 6)
    assert int(null_row["n_triplets_without_null_score"]) == 1


def test_load_triplet_group_map_prefers_prepared_inputs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    prepared = run_root / "prepared_inputs" / "observed" / "inhibitory"
    prepared.mkdir(parents=True)
    pd.DataFrame({
        "triplet_id": ["trp_1"],
        "drug_id": ["drug_1"],
        "target_id": ["target_1"],
        "disease_id": ["DOID:1"],
    }).to_csv(prepared / "profile_supported_observed_triplets.tsv", sep="\t", index=False)
    evaluation = run_root / "evaluation" / "inhibitory"
    evaluation.mkdir(parents=True)
    pd.DataFrame({
        "triplet_id": ["trp_1", "trp_2"],
        "drug_id": ["other_drug", "drug_2"],
        "target_id": ["other_target", "target_2"],
        "disease_id": ["DOID:9", "DOID:2"],
    }).to_csv(evaluation / "decoy_anchor_diagnostics.tsv", sep="\t", index=False)

    out = posthoc.load_triplet_group_map(run_root, "inhibitory").set_index("triplet_id")

    assert out.loc["trp_1", "drug_id"] == "drug_1"
    assert out.loc["trp_1", "disease_id"] == "DOID:1"
    assert out.loc["trp_2", "disease_id"] == "DOID:2"
