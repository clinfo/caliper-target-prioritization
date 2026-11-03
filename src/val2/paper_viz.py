"""Canonical paper panels derived from existing Val2 outputs.

The functions in this module never run scoring or ablation. They materialize
the approved supplementary panels and their audit tables inside the source
run's ``report`` directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .label_confidence import (
    CONFIDENCE_SCORE_GROUPS,
    assign_confidence_score_groups,
    confidence_positive_pairs,
    load_target_scores,
    load_targetnexus_confidence,
    summarize_label_confidence,
    validate_all_evidence_labels,
)
from .support_robustness import (
    EVALUABLE_CELL_SCORE_GROUPS,
    MEASURED_GENE_SCORE_GROUPS,
    compute_support_robustness,
    load_cell_scores,
    prepare_support_score_distributions,
)


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _set_style() -> None:
    plt = _load_matplotlib()
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.35,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
        }
    )


def _relative_outputs(run_root: Path, output_png: Path) -> Dict[str, str]:
    return {
        output_png.stem: output_png.relative_to(run_root).as_posix(),
        f"{output_png.stem}_pdf": output_png.with_suffix(".pdf")
        .relative_to(run_root)
        .as_posix(),
        f"{output_png.stem}_svg": output_png.with_suffix(".svg")
        .relative_to(run_root)
        .as_posix(),
    }


def _save_figure(fig, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(output_png.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    fig.savefig(output_png.with_suffix(".svg"), facecolor="white", bbox_inches="tight")


def _write_table(
    frame: pd.DataFrame,
    path: Path,
    *,
    run_root: Path,
    logical_name: str,
) -> Dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)
    return {logical_name: path.relative_to(run_root).as_posix()}


def make_label_confidence_score_distribution(
    run_root: str | Path,
    *,
    targetnexus_root: str | Path,
    k: int = 4,
) -> Dict[str, str]:
    """Generate the approved FigS2b panel and source-audit tables."""
    run_root = Path(run_root)
    targetnexus_root = Path(targetnexus_root)
    scores = load_target_scores(
        run_root / "ablation" / "val2_scores_k1-10.csv",
        k=k,
    )
    scores = scores[scores["method"].astype(str).eq("proposed")].reset_index(
        drop=True
    )
    confidence = load_targetnexus_confidence(targetnexus_root)
    audit = validate_all_evidence_labels(scores, confidence)
    all_disease_summary = summarize_label_confidence(scores, confidence)
    approved_pairs = confidence_positive_pairs(confidence, "approved")
    score_pairs = set(
        scores[["disease_id", "target_gene"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    common_diseases = sorted(
        {
            str(disease_id)
            for disease_id, _target_gene in approved_pairs & score_pairs
        }
    )
    if not common_diseases:
        raise ValueError(
            "Label-confidence figure has no diseases with an Approved positive "
            "pair in the evaluated candidate universe"
        )
    score_groups = assign_confidence_score_groups(
        scores,
        confidence,
        evaluation_diseases=common_diseases,
    )
    if score_groups.empty:
        raise ValueError("Label-confidence figure has no evidence-positive scores")
    if score_groups.duplicated(["method", "disease_id", "target_gene"]).any():
        raise ValueError("Label-confidence score groups are not mutually exclusive")

    group_summary = (
        score_groups.groupby(
            [
                "confidence_group_order",
                "confidence_group",
                "confidence_group_label",
                "method",
            ],
            as_index=False,
            sort=True,
        )
        .agg(
            n_pairs=("score", "size"),
            n_diseases=("disease_id", "nunique"),
            score_mean=("score", "mean"),
            score_median=("score", "median"),
            score_q1=("score", lambda values: values.quantile(0.25)),
            score_q3=("score", lambda values: values.quantile(0.75)),
            within_disease_percentile_median=(
                "within_disease_score_percentile",
                "median",
            ),
        )
        .sort_values("confidence_group_order", kind="mergesort")
        .reset_index(drop=True)
    )
    expected_groups = {name for name, _label in CONFIDENCE_SCORE_GROUPS}
    if set(group_summary["confidence_group"]) != expected_groups:
        raise ValueError(
            "Label-confidence figure is missing a score group: "
            f"{group_summary['confidence_group'].tolist()}"
        )

    existing = pd.read_csv(run_root / "ablation" / "val2_metrics_k1-10.csv")
    existing = (
        existing[
            existing["k"].astype(int).eq(k)
            & existing["method"].astype(str).eq("proposed")
        ]
        .groupby("method", as_index=False)
        .agg(expected_mAP=("AP", "mean"))
    )
    observed = all_disease_summary[
        all_disease_summary["stratum"].eq("all_evidence")
    ][["method", "mAP"]]
    comparison = observed.merge(existing, on="method", how="inner")
    if comparison.empty or not np.allclose(
        comparison["mAP"],
        comparison["expected_mAP"],
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError(
            "All-evidence mAP does not reproduce the saved k=4 metrics: "
            f"{comparison.to_dict(orient='records')}"
        )

    table_dir = run_root / "report" / "tables"
    stem = f"label_confidence_score_distribution_k{k}"
    summary_path = table_dir / f"{stem}_summary.tsv"
    pairs_path = table_dir / f"{stem}_pairs.tsv"
    outputs: Dict[str, str] = {}
    outputs.update(
        _write_table(
            group_summary.assign(
                label_audit_n_pairs=audit["n_pairs"],
                label_audit_n_positive_pairs=audit["n_positive_pairs"],
                label_audit_n_mismatches=audit["n_mismatches"],
                evaluation_cohort="common_approved_positive_diseases",
                n_common_diseases=len(common_diseases),
                score_definition=f"mean k={k} CALIPER score across evaluable cell lines",
                groups_are_mutually_exclusive=True,
                targetnexus_release="25.12",
                targetnexus_label_set="DrugGD-OT__ot25_12__v1",
            ),
            summary_path,
            run_root=run_root,
            logical_name=f"{stem}_summary",
        )
    )
    outputs.update(
        _write_table(
            score_groups.assign(
                evaluation_cohort="common_approved_positive_diseases",
                n_common_diseases=len(common_diseases),
                score_definition=f"mean k={k} CALIPER score across evaluable cell lines",
                groups_are_mutually_exclusive=True,
                targetnexus_release="25.12",
                targetnexus_label_set="DrugGD-OT__ot25_12__v1",
            ),
            pairs_path,
            run_root=run_root,
            logical_name=f"{stem}_pairs",
        )
    )

    _set_style()
    plt = _load_matplotlib()
    plot = score_groups[score_groups["method"].astype(str).eq("proposed")].copy()
    group_names = [name for name, _label in CONFIDENCE_SCORE_GROUPS]
    group_display = [
        "Evidence below\nPhase 2",
        "Phase 2\nonly",
        "Phase 3+\nnot approved",
        "Approved",
    ]
    values = [
        pd.to_numeric(
            plot.loc[plot["confidence_group"].eq(group_name), "score"],
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
        for group_name in group_names
    ]

    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=True)
    boxplot = ax.boxplot(
        values,
        patch_artist=True,
        showfliers=False,
        widths=0.62,
        boxprops={"edgecolor": "#333333", "linewidth": 0.7},
        medianprops={"color": "#111111", "linewidth": 1.0},
        whiskerprops={"color": "#555555", "linewidth": 0.7},
        capprops={"color": "#555555", "linewidth": 0.7},
    )
    for box in boxplot["boxes"]:
        box.set_facecolor("#0072B2")
        box.set_alpha(0.72)
    ax.set_xticks(np.arange(1, len(values) + 1, dtype=float))
    ax.set_xticklabels(
        [
            f"{label}\n(n={len(group_values):,})"
            for label, group_values in zip(group_display, values)
        ],
        fontsize=6,
    )
    ax.set_xlabel(
        "Open Targets evidence stratum\n"
        f"(common Approved-positive diseases; D={len(common_diseases)})"
    )
    ax.set_ylabel("CALIPER score")
    ax.grid(axis="y", alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", width=0.6, length=2.5, pad=2.0)
    output_png = run_root / "report" / "figures" / f"{stem}.png"
    _save_figure(fig, output_png)
    plt.close(fig)
    outputs.update(_relative_outputs(run_root, output_png))
    return outputs


def make_support_score_distribution(
    run_root: str | Path,
    *,
    k: int = 4,
) -> Dict[str, str]:
    """Generate the approved FigS2c panel and source-audit tables."""
    run_root = Path(run_root)
    cell_scores = load_cell_scores(
        run_root / "ablation" / "val2_scores_k1-10.csv",
        k=k,
        method="proposed",
    )
    score_groups = prepare_support_score_distributions(
        cell_scores,
        positive_only=True,
    )
    if score_groups.empty:
        raise ValueError("Support-score figure has no known-positive CALIPER scores")

    table_dir = run_root / "report" / "tables"
    stem = f"support_score_distribution_k{k}"
    outputs: Dict[str, str] = {}
    outputs.update(
        _write_table(
            score_groups.assign(
                score_pair_universe="known_positive_pairs",
                score_definition=f"mean k={k} CALIPER score across evaluable cell lines",
                groups_are_mutually_exclusive=True,
            ),
            table_dir / f"{stem}_pairs.tsv",
            run_root=run_root,
            logical_name=f"{stem}_pairs",
        )
    )

    summary_rows: list[dict[str, object]] = []
    group_specs = (
        (
            "measured_genes",
            "measured_genes_group",
            "measured_genes_group_label",
            "measured_genes_group_order",
            MEASURED_GENE_SCORE_GROUPS,
        ),
        (
            "evaluable_cell_lines",
            "evaluable_cells_group",
            "evaluable_cells_group_label",
            "evaluable_cells_group_order",
            EVALUABLE_CELL_SCORE_GROUPS,
        ),
    )
    for analysis, key_column, _label_column, order_column, groups in group_specs:
        for group_key, group_label in groups:
            group = score_groups[score_groups[key_column].eq(group_key)]
            values = pd.to_numeric(group["score"], errors="coerce").dropna()
            summary_rows.append(
                {
                    "analysis": analysis,
                    "group": group_key,
                    "group_label": group_label,
                    "group_order": int(
                        score_groups.loc[
                            score_groups[key_column].eq(group_key),
                            order_column,
                        ].iloc[0]
                    ),
                    "n_pairs": int(len(values)),
                    "n_diseases": int(group["disease_id"].nunique()),
                    "score_mean": float(values.mean()),
                    "score_median": float(values.median()),
                    "score_q1": float(values.quantile(0.25)),
                    "score_q3": float(values.quantile(0.75)),
                    "score_pair_universe": "known_positive_pairs",
                    "score_definition": (
                        f"mean k={k} CALIPER score across evaluable cell lines"
                    ),
                    "groups_are_mutually_exclusive": True,
                }
            )
    outputs.update(
        _write_table(
            pd.DataFrame(summary_rows),
            table_dir / f"{stem}_summary.tsv",
            run_root=run_root,
            logical_name=f"{stem}_summary",
        )
    )

    robustness = compute_support_robustness(cell_scores)
    loo = robustness[robustness["analysis"].eq("leave_one_cell_line_out")].copy()
    outputs.update(
        _write_table(
            loo,
            table_dir / f"{stem}_loo.tsv",
            run_root=run_root,
            logical_name=f"{stem}_loo",
        )
    )

    _set_style()
    plt = _load_matplotlib()
    fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_c1 = fig.add_subplot(grid[0, 0])
    ax_c2 = fig.add_subplot(grid[0, 1], sharey=ax_c1)
    ax_c3 = fig.add_subplot(grid[1, :])

    def draw_score_boxplots(
        ax,
        *,
        group_column: str,
        groups: tuple[tuple[str, str], ...],
        xlabel: str,
        panel_label: str,
    ) -> None:
        values = [
            pd.to_numeric(
                score_groups.loc[score_groups[group_column].eq(group_key), "score"],
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
            for group_key, _label in groups
        ]
        boxplot = ax.boxplot(
            values,
            patch_artist=True,
            showfliers=False,
            widths=0.62,
            boxprops={"edgecolor": "#333333", "linewidth": 0.7},
            medianprops={"color": "#111111", "linewidth": 1.0},
            whiskerprops={"color": "#555555", "linewidth": 0.7},
            capprops={"color": "#555555", "linewidth": 0.7},
        )
        for box in boxplot["boxes"]:
            box.set_facecolor("#0072B2")
            box.set_alpha(0.72)
        ax.set_xticks(np.arange(1, len(values) + 1, dtype=float))
        ax.set_xticklabels(
            [
                f"{label}\n(n={len(group_values):,})"
                for (_key, label), group_values in zip(groups, values)
            ],
            fontsize=6.5,
        )
        ax.set_xlabel(f"{xlabel}\n(known-positive pairs)")
        ax.set_ylabel("CALIPER score")
        ax.grid(axis="y", alpha=0.22, linewidth=0.35)
        ax.set_axisbelow(True)
        ax.tick_params(
            axis="both",
            which="major",
            width=0.6,
            length=2.5,
            pad=2.0,
        )
        ax.text(
            -0.13,
            1.03,
            panel_label,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
        )

    draw_score_boxplots(
        ax_c1,
        group_column="measured_genes_group",
        groups=MEASURED_GENE_SCORE_GROUPS,
        xlabel="measured genes",
        panel_label="c1",
    )
    draw_score_boxplots(
        ax_c2,
        group_column="evaluable_cells_group",
        groups=EVALUABLE_CELL_SCORE_GROUPS,
        xlabel="evaluable cell lines",
        panel_label="c2",
    )
    ax_c2.set_ylabel("")
    ax_c2.tick_params(labelleft=False)

    loo = loo.sort_values("setting", kind="mergesort")
    x = np.arange(len(loo), dtype=float)
    delta = pd.to_numeric(loo["delta_mAP"], errors="coerce").to_numpy(dtype=float)
    ax_c3.axhline(0.0, color="#666666", linestyle=":", linewidth=0.8)
    ax_c3.scatter(
        x,
        delta,
        s=16,
        color="#0072B2",
        edgecolor="black",
        linewidth=0.45,
        zorder=3,
    )
    ax_c3.set_xticks(x)
    ax_c3.set_xticklabels(
        loo["setting"].astype(str),
        rotation=40,
        ha="right",
        rotation_mode="anchor",
        fontsize=6,
    )
    ax_c3.set_xlabel("omitted cell line")
    ax_c3.set_ylabel("ΔmAP from full analysis")
    finite_delta = np.abs(delta[np.isfinite(delta)])
    if finite_delta.size:
        bound = max(float(np.max(finite_delta)) * 1.2, 1e-4)
        ax_c3.set_ylim(-bound, bound)
    ax_c3.grid(axis="y", alpha=0.22, linewidth=0.35)
    ax_c3.set_axisbelow(True)
    ax_c3.tick_params(
        axis="both",
        which="major",
        width=0.6,
        length=2.5,
        pad=2.0,
    )
    ax_c3.text(
        -0.065,
        1.03,
        "c3",
        transform=ax_c3.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )

    output_png = run_root / "report" / "figures" / f"{stem}.png"
    _save_figure(fig, output_png)
    plt.close(fig)
    outputs.update(_relative_outputs(run_root, output_png))
    return outputs
