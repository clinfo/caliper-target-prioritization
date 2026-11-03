from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from val3 import report
from val3.report import VAL3_REPORT_FILENAME, write_val3_triplet_visualization_report
from val3.runner import _output_step_exists


def test_evaluation_step_does_not_require_the_abolished_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    evaluation = run_root / "evaluation" / "inhibitory"
    evaluation.mkdir(parents=True)
    for name in [
        "summary.json",
        "triplet_metrics_true.parquet",
        "global_evaluable_triplets.tsv",
        "global_joint_triplet_metrics_true.parquet",
        "global_joint_triplet_metrics_baseline.parquet",
        "global_joint_triplet_comparison_sets.tsv",
        "global_joint_triplet_comparison.tsv",
        "global_effect_size.tsv",
    ]:
        (evaluation / name).touch()
    cfg = SimpleNamespace(evaluation={"global_effect_size": {"enabled": True}})
    assert _output_step_exists(run_root, "inhibitory", "evaluation", cfg)
    # The three abolished analyses are no longer produced and must not be required.
    for name in [
        "global_empirical_p.tsv",
        "global_triplet_paired_tests.tsv",
        "global_group_resampled_null.tsv",
        "global_group_resampled_null_qc.tsv",
    ]:
        assert not (evaluation / name).exists()


def test_val3_report_uses_val2_style_layout(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "evaluation" / "inhibitory").mkdir(parents=True)

    outputs = write_val3_triplet_visualization_report(
        run_root=run_root,
        direction="inhibitory",
    )

    report_dir = run_root / "report"
    tables_dir = report_dir / "tables"
    assert outputs["output_md"] == str(report_dir / VAL3_REPORT_FILENAME)
    assert outputs["figures_dir"] == str(report_dir / "figures")
    assert outputs["tables_dir"] == str(tables_dir)
    assert (report_dir / VAL3_REPORT_FILENAME).exists()
    assert (report_dir / "figures").is_dir()
    assert {
        "report_run_settings.tsv",
        "coverage_triplet_summary.parquet",
        "coverage_summary_by_null.tsv",
        "metric_triplet_long.parquet",
        "metric_pert_level_long.parquet",
        "null_diagnostics_summary.tsv",
        "global_joint_triplet_delta_bootstrap.tsv",
        "jaccard_paired_wilcoxon_holm.tsv",
        "jaccard_observed_vs_null_summary.tsv",
        "null_repeat_stability.tsv",
    } <= {path.name for path in tables_dir.iterdir()}
    # Every p-value source of null_effect_pvalues.tsv was removed with the three
    # abolished analyses, so the table itself is no longer written.
    assert "null_effect_pvalues.tsv" not in {path.name for path in tables_dir.iterdir()}
    report_text = (report_dir / VAL3_REPORT_FILENAME).read_text(encoding="utf-8")
    assert "## Global effect: observed − baseline" not in report_text
    assert "## Global null summaries" in report_text
    assert "Mean (Observed − null median)" in report_text
    assert _output_step_exists(run_root, "inhibitory", "report")


def test_report_cli_refreshes_run_summary_outputs_and_keeps_cleanup(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "evaluation" / "inhibitory").mkdir(parents=True)
    cleanup = [{"path": "retained", "removed": False}]
    (run_root / "run_summary.json").write_text(
        json.dumps({
            "sections": {"make_report": True},
            "report_outputs": {"cleanup_after_report": cleanup},
        }),
        encoding="utf-8",
    )

    outputs = report.main([
        "--run-root",
        str(run_root),
        "--direction",
        "inhibitory",
    ])
    updated = json.loads((run_root / "run_summary.json").read_text(encoding="utf-8"))

    assert updated["report_outputs"]["output_md"] == outputs["output_md"]
    assert (
        updated["report_outputs"]["jaccard_observed_vs_null_summary"]
        == outputs["jaccard_observed_vs_null_summary"]
    )
    assert updated["report_outputs"]["cleanup_after_report"] == cleanup


def test_observed_vs_null_summary_and_figure_use_paper_facing_labels() -> None:
    long = pd.DataFrame([
        {
            "null_type": "gene_node",
            "evaluable_set_type": "path_existing_pairs",
            "triplet_id": f"t{index}",
            "metric": "jaccard",
            "source": source,
            "value": value,
        }
        for source, values in [
            ("observed", [0.40, 0.50]),
            ("null median", [0.20, 0.30]),
        ]
        for index, value in enumerate(values, start=1)
    ])
    paired = pd.DataFrame([{
        "null_type": "gene_node",
        "evaluable_set_type": "path_existing_pairs",
        "p_wilcoxon_two_sided_holm": 0.025,
    }])

    summary = report._build_jaccard_observed_vs_null_summary(
        long,
        paired,
        seed=17,
        n_bootstrap=200,
    )
    shuffled = report._build_jaccard_observed_vs_null_summary(
        long.sample(frac=1.0, random_state=3).reset_index(drop=True),
        paired,
        seed=17,
        n_bootstrap=200,
    )
    pd.testing.assert_frame_equal(summary, shuffled)
    row = summary.iloc[0]
    assert int(row["n_triplets"]) == 2
    assert int(row["n_bootstrap"]) == 200
    assert row["resampling_unit"] == "triplet_id"
    assert row["ci_method"] == "percentile"
    assert float(row["ci_level"]) == pytest.approx(0.95)
    assert float(row["observed_mean"]) == pytest.approx(0.45)
    assert float(row["null_median_mean"]) == pytest.approx(0.25)
    assert float(row["observed_mean_ci025"]) == pytest.approx(0.40)
    assert float(row["observed_mean_ci975"]) == pytest.approx(0.50)
    assert float(row["null_median_mean_ci025"]) == pytest.approx(0.20)
    assert float(row["null_median_mean_ci975"]) == pytest.approx(0.30)
    assert (
        float(row["observed_mean_ci025"])
        - float(row["null_median_mean_ci025"])
    ) == pytest.approx(0.20)
    assert (
        float(row["observed_mean_ci975"])
        - float(row["null_median_mean_ci975"])
    ) == pytest.approx(0.20)
    assert float(row["p_wilcoxon_two_sided_holm"]) == pytest.approx(0.025)
    assert "observed_se" not in summary.columns
    assert "null_median_se" not in summary.columns

    fig = report.viz.plot_jaccard_observed_vs_null(
        summary,
        evaluable_set_type="path_existing_pairs",
    )
    try:
        assert tuple(fig.get_size_inches()) == pytest.approx((7.2, 2.4))
        assert fig.axes[0].get_ylabel() == "Mean Jaccard"
        assert fig.axes[0].get_title() == ""
        legend_labels = [
            text.get_text() for text in fig.axes[0].get_legend().get_texts()
        ]
        assert "Null" in legend_labels
        assert "Null median" not in legend_labels
        assert "95% bootstrap CI" in legend_labels
        assert "Observed vs Null" not in {
            text.get_text() for text in fig.axes[0].texts
        }
        assert fig.axes[0].patches[0].get_facecolor() == pytest.approx(
            to_rgba("#D55E00")
        )
        assert fig.axes[0].patches[1].get_facecolor() == pytest.approx(
            to_rgba("#F4C7AB")
        )
        assert [text.get_text() for text in fig.axes[0].texts] == ["P = 0.025"]
        brackets = [
            np.asarray(line.get_xydata(), dtype=float)
            for line in fig.axes[0].lines
            if np.asarray(line.get_xydata()).shape == (4, 2)
        ]
        assert len(brackets) == 1
        assert brackets[0][0, 0] == pytest.approx(-0.18)
        assert brackets[0][-1, 0] == pytest.approx(0.18)
        assert brackets[0][1, 1] == pytest.approx(brackets[0][2, 1])
        legend_bbox_axes = (
            fig.axes[0]
            .get_legend()
            .get_bbox_to_anchor()
            .transformed(fig.axes[0].transAxes.inverted())
        )
        assert legend_bbox_axes.ymax <= 1.0
    finally:
        plt.close(fig)

    compact_fig = report.viz.plot_jaccard_observed_vs_null(
        summary,
        evaluable_set_type="path_existing_pairs",
        figwidth=3.7555,
    )
    try:
        assert tuple(compact_fig.get_size_inches()) == pytest.approx((3.7555, 2.4))
        assert compact_fig.axes[0].get_legend()._ncols == 2
        assert compact_fig.axes[0].get_ylabel() == "Mean Jaccard"
        assert [text.get_text() for text in compact_fig.axes[0].texts] == [
            "P = 0.025"
        ]
    finally:
        plt.close(compact_fig)


def test_delta_sensitivity_figures_use_compact_size_and_ascii_delta_label() -> None:
    stability = pd.DataFrame([
        {
            "null_type": "gene_node",
            "evaluable_set_type": "path_existing_pairs",
            "statistic": "joint_jaccard",
            "n_repeats": n_repeats,
            "delta_mean": value,
        }
        for n_repeats, value in [(10, 0.10), (100, 0.12)]
    ])
    cluster = pd.DataFrame([
        {
            "null_type": "gene_node",
            "evaluable_set_type": "path_existing_pairs",
            "statistic": "joint_jaccard",
            "cluster_level": level,
            "delta_mean_hat": 0.10,
            "delta_mean_ci025": 0.08,
            "delta_mean_ci975": 0.12,
        }
        for level in ["triplet", "drug", "target", "disease"]
    ])

    figures = [
        report.viz.plot_null_repeat_stability(
            stability,
            evaluable_set_type="path_existing_pairs",
        ),
        report.viz.plot_delta_cluster_bootstrap(
            cluster,
            evaluable_set_type="path_existing_pairs",
        ),
    ]
    try:
        for fig in figures:
            assert tuple(fig.get_size_inches()) == pytest.approx((3.5, 2.4))
            assert fig.axes[0].get_ylabel() == "Mean delta Jaccard"
    finally:
        for fig in figures:
            plt.close(fig)


def test_fig4_sensitivity_panels_do_not_draw_sample_size_labels() -> None:
    first_edge = pd.DataFrame([{
        "null_type": "gene_node",
        "evaluable_set_type": "path_existing_pairs",
        "statistic": "joint_jaccard",
        "delta_mean_all_edges": 0.10,
        "delta_mean_ci025_all_edges": 0.08,
        "delta_mean_ci975_all_edges": 0.12,
        "delta_mean_first_edge_excluded": 0.09,
        "delta_mean_ci025_first_edge_excluded": 0.07,
        "delta_mean_ci975_first_edge_excluded": 0.11,
        "n_comparison_triplets": 1570,
    }])
    cell_scope = pd.DataFrame([
        {
            "cell_scope": scope,
            "null_type": "gene_node",
            "evaluable_set_type": "path_existing_pairs",
            "statistic": "joint_jaccard",
            "delta_mean": value,
            "delta_mean_ci025": value - 0.02,
            "delta_mean_ci975": value + 0.02,
            "n_triplets": n_triplets,
        }
        for scope, value, n_triplets in [
            ("all_cell_pairs", 0.10, 1570),
            ("same_cell", 0.08, 1062),
        ]
    ])

    figures = [
        report.viz.plot_first_edge_exclusion_delta(
            first_edge,
            evaluable_set_type="path_existing_pairs",
        ),
        report.viz.plot_cell_scope_delta(
            cell_scope,
            evaluable_set_type="path_existing_pairs",
        ),
    ]
    try:
        for fig in figures:
            assert not [
                text.get_text()
                for text in fig.axes[0].texts
                if text.get_text().startswith("n=")
            ]
    finally:
        for fig in figures:
            plt.close(fig)


def test_report_drops_the_abolished_sections_and_nonprimary_p_values(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "evaluation" / "inhibitory").mkdir(parents=True)
    write_val3_triplet_visualization_report(run_root=run_root, direction="inhibitory")
    report_text = (run_root / "report" / VAL3_REPORT_FILENAME).read_text(encoding="utf-8")

    for heading in ["### Group-level resampled null", "### Global empirical p", "### Triplet paired tests"]:
        assert heading not in report_text
    assert "## Global null summaries" in report_text
    for column in [
        "p_sign_greater", "p_t_two_sided", "p_wilcoxon_greater",
        "p_resampled", "p_empirical",
        "p_finite_null", "ref_null_median_mean", "ref_delta_mean",
    ]:
        assert column not in report_text


def test_drop_nonprimary_columns_keeps_the_primary_test_columns() -> None:
    df = pd.DataFrame([{
        "null_type": "n", "delta_mean": 0.1,
        "wilcoxon_two_sided_statistic": 3.0,
        "p_wilcoxon_two_sided": 0.01,
        "p_wilcoxon_two_sided_holm": 0.02,
        "wilcoxon_two_sided_status": "computed",
        "p_sign_greater": 0.03, "p_sign_holm": 0.04,
        "p_t_two_sided": 0.3, "p_wilcoxon_greater": 0.4, "p_empirical": 0.5,
        "ref_delta_mean": 0.6, "ref_null_median_mean": 0.7,
    }])
    out = report._drop_nonprimary_columns(df)
    assert list(out.columns) == [
        "null_type", "delta_mean", "wilcoxon_two_sided_statistic",
        "p_wilcoxon_two_sided", "p_wilcoxon_two_sided_holm",
        "wilcoxon_two_sided_status",
    ]
    # The source frame is untouched: only the report view drops columns.
    assert "p_t_two_sided" in df.columns


def test_delta_bootstrap_figures_include_only_primary_mean(monkeypatch, tmp_path: Path) -> None:
    plot_calls: list[dict[str, object]] = []

    def fake_plot(*args, **kwargs):
        plot_calls.append(kwargs)
        return object()

    def fake_save(fig, png_path: Path):
        return png_path, png_path.with_suffix(".pdf")

    monkeypatch.setattr(report.viz, "plot_global_joint_triplet_delta_bootstrap", fake_plot)
    monkeypatch.setattr(report, "_savefig_png_pdf", fake_save)

    outputs = report._plot_global_joint_delta_bootstrap(
        pd.DataFrame({"evaluable_set_type": ["path_existing_pairs"]}),
        direction="inhibitory",
        figures_dir=tmp_path,
    )

    assert plot_calls == [
        {
            "estimator": "mean",
            "evaluable_set_type": "path_existing_pairs",
            "statistics": ["joint_jaccard"],
            "decoy_selection": None,
        },
    ]
    assert outputs == [
        (
            "path_existing_pairs",
            "mean",
            Path("figures/fig_jaccard_observed_minus_null_median_mean_bootstrap_ci__path_existing_pairs.png"),
            Path("figures/fig_jaccard_observed_minus_null_median_mean_bootstrap_ci__path_existing_pairs.pdf"),
        ),
    ]


def test_primary_delta_bootstrap_recomputes_null_median_and_is_order_invariant() -> None:
    comparison = pd.DataFrame([
        {
            "direction": "inhibitory",
            "null_type": "gene_node",
            "evaluable_set_type": "path_existing_pairs",
            "statistic": "joint_jaccard",
            "triplet_id": "t1",
            "observed_value": 0.50,
            "baseline_median": 0.00,
            "delta": -9.0,
        },
        {
            "direction": "inhibitory",
            "null_type": "gene_node",
            "evaluable_set_type": "path_existing_pairs",
            "statistic": "joint_jaccard",
            "triplet_id": "t2",
            "observed_value": 0.40,
            "baseline_median": 0.30,
            "delta": -9.0,
        },
        {
            "direction": "inhibitory",
            "null_type": "gene_node",
            "evaluable_set_type": "path_existing_pairs",
            "statistic": "joint_precision",
            "triplet_id": "t1",
            "observed_value": 0.90,
            "baseline_median": 0.10,
            "delta": 0.80,
        },
    ])

    first = report.build_global_joint_triplet_delta_bootstrap(
        comparison,
        direction="inhibitory",
        seed=17,
    )
    second = report.build_global_joint_triplet_delta_bootstrap(
        comparison.sample(frac=1.0, random_state=3).reset_index(drop=True),
        direction="inhibitory",
        seed=17,
    )

    pd.testing.assert_frame_equal(first, second)
    row = first.iloc[0]
    assert row["statistic"] == "joint_jaccard"
    assert row["estimand"] == "mean_observed_minus_triplet_null_median"
    assert int(row["n_bootstrap"]) == 1000
    assert row["resampling_unit"] == "triplet_id"
    assert row["ci_method"] == "percentile"
    assert float(row["ci_level"]) == 0.95
    assert float(row["delta_mean_hat"]) == pytest.approx(0.30)
    assert float(row["delta_mean_ci025"]) <= float(row["delta_mean_ci975"])
    assert not [column for column in first.columns if "median_hat" in column]


def test_single_triplet_bootstrap_has_degenerate_ci() -> None:
    point, lower, upper, _ = report.posthoc.bootstrap_mean_ci(
        [0.25],
        seed=9,
        group_parts=("inhibitory", "null", "path_existing_pairs", "joint_jaccard"),
    )
    assert point == lower == upper == pytest.approx(0.25)


def test_primary_figure_draws_ci_endpoints_without_clipping_to_point() -> None:
    frame = pd.DataFrame([{
        "null_type": "gene_node",
        "evaluable_set_type": "path_existing_pairs",
        "statistic": "joint_jaccard",
        "delta_mean_hat": 0.00,
        # Deliberately exclude the point estimate to pin the no-clipping rule.
        "delta_mean_ci025": 0.10,
        "delta_mean_ci975": 0.20,
    }])
    fig = report.viz.plot_global_joint_triplet_delta_bootstrap(
        frame,
        estimator="mean",
        evaluable_set_type="path_existing_pairs",
        statistics=["joint_jaccard"],
    )
    try:
        assert tuple(fig.get_size_inches()) == pytest.approx((3.5, 2.4))
        segments = [
            segment
            for collection in fig.axes[0].collections
            for segment in collection.get_segments()
        ]
        assert any(
            np.allclose(segment, np.array([[0.0, 0.10], [0.0, 0.20]]))
            for segment in segments
            if np.asarray(segment).shape == (2, 2)
        )
    finally:
        plt.close(fig)


def test_null_evaluability_figure_uses_common_observed_triplet_denominator() -> None:
    summary = pd.DataFrame([
        {
            "null_type": null_type,
            "evaluable_set_type": "path_existing_pairs",
            "null_evaluable_rate": null_row_rate,
            "n_observed_evaluable_triplets": 10,
            "n_triplets_no_null_score": n_without_score,
        }
        for null_type, null_row_rate, n_without_score in [
            ("target_khop_restricted_paired_gene_label_permutation", 0.50, 0),
            ("path_count_length_matched_structural_resampling", 1.00, 0),
            ("topology_profile_matched_target_decoy_control", 1.00, 2),
        ]
    ])
    by_triplet = pd.DataFrame([
        {
            "null_type": "gene_node",
            "evaluable_set_type": "path_existing_pairs",
            "null_evaluable_fraction": 0.75,
        }
    ])

    fig = report.viz.plot_null_evaluability(
        summary,
        by_triplet,
        evaluable_set_type="path_existing_pairs",
    )
    try:
        assert tuple(fig.get_size_inches()) == pytest.approx((3.5, 2.4))
        assert len(fig.axes) == 1
        assert fig.axes[0].get_ylabel() == "Triplet coverage (%)"
        assert fig.axes[0].get_xlabel() == "Null model"
        assert [patch.get_height() for patch in fig.axes[0].patches] == pytest.approx(
            [100.0, 100.0, 80.0],
        )
        assert [text.get_text() for text in fig.axes[0].texts] == [
            "100.0%",
            "100.0%",
            "80.0%",
        ]
    finally:
        plt.close(fig)
