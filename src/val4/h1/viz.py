from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from val4.paper_viz import plot_family_edge_support, plot_family_forest


MIN_EVALUABLE_ROWS = 1000

FAMILY_FIGURE_LABELS = {
    "Transcriptional_regulation": "Transcriptional regulation",
    "Post_transcriptional_control": "Post-transcriptional",
    "Translation_regulation": "Translation",
    "Phospho_state": "Phosphorylation",
    "Binding": "Binding",
    "Localization": "Localization",
    "GTPase_cycle_regulation": "GTPase cycle",
    "Ubiquitin_Ubl_state": "Ubiquitin/Ubl",
    "Other_covalent_PTM": "Acetyl/Methyl PTM",
    "Proteolytic_processing": "Proteolysis",
    "Chemical_regulation": "Chemical regulation",
    "Generic_modification": "Generic modification",
    "Stability_quantity_regulation": "Stability/quantity",
    "Catalytic_activity": "Catalytic activity",
    "Isomerization": "Isomerization",
    "Other_rare": "Other rare",
}

_REQUIRED_INPUTS = {
    "summary": "val4_h1_family_summary.csv",
    "summary_cluster": "val4_h1_family_summary_cluster.csv",
    "null": "val4_h1_family_null.csv",
    "null_cluster": "val4_h1_family_null_cluster.csv",
    "bootstrap": "val4_h1_family_bootstrap.csv",
    "bootstrap_cluster": "val4_h1_family_bootstrap_cluster.csv",
    "paired_key_deltas": "val4_h1_paired_key_deltas.csv",
    "paired_row_deltas": "val4_h1_paired_row_deltas.parquet",
}


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    return plt, mticker


def _set_plot_style() -> None:
    plt, _ = _load_matplotlib()
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 9,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.35,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.bbox": "tight",
            "svg.fonttype": "path",
        }
    )


def _family_label(value: object) -> str:
    return FAMILY_FIGURE_LABELS.get(str(value), str(value).replace("_", " "))


def _family_labels(values: pd.Series) -> list[str]:
    return [_family_label(value) for value in values]


def _relative_to_run_root(path: Path, run_root: Path) -> str:
    try:
        return path.relative_to(run_root).as_posix()
    except ValueError:
        return path.as_posix()


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except OSError:
        return pd.read_parquet(path, engine="fastparquet")


def _read_required(run_root: Path, filename: str) -> pd.DataFrame:
    path = run_root / "metrics" / filename
    if not path.exists():
        path = run_root / filename
    if not path.exists():
        raise FileNotFoundError(f"Required Val4/H1 output not found: {path}")
    if path.suffix == ".parquet":
        return _read_parquet(path)
    return pd.read_csv(path)


def _read_optional(run_root: Path, filename: str) -> pd.DataFrame | None:
    path = run_root / "metrics" / filename
    if not path.exists():
        path = run_root / filename
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        return _read_parquet(path)
    return pd.read_csv(path)


def load_paired_tables(run_root: str | Path) -> dict[str, pd.DataFrame | list[str]]:
    """Load Val4/H1 paired outputs and apply visualization filters."""
    run_root = Path(run_root)
    summary_df = _read_required(run_root, _REQUIRED_INPUTS["summary"])
    summary_cluster_df = _read_required(run_root, _REQUIRED_INPUTS["summary_cluster"])
    null_df = _read_required(run_root, _REQUIRED_INPUTS["null"])
    null_cluster_df = _read_required(run_root, _REQUIRED_INPUTS["null_cluster"])
    boot_df = _read_required(run_root, _REQUIRED_INPUTS["bootstrap"])
    boot_cluster_df = _read_required(run_root, _REQUIRED_INPUTS["bootstrap_cluster"])
    key_delta_df = _read_required(run_root, _REQUIRED_INPUTS["paired_key_deltas"])
    row_delta_df = _read_required(run_root, _REQUIRED_INPUTS["paired_row_deltas"])

    eligible_row = set(
        summary_df.loc[summary_df["n_evaluable_rows"] >= MIN_EVALUABLE_ROWS, "family"]
    )
    eligible_cluster = set(
        summary_cluster_df.loc[
            summary_cluster_df["n_evaluable_rows"] >= MIN_EVALUABLE_ROWS, "family"
        ]
    )

    summary_df_full = summary_df.copy()
    summary_cluster_df_full = summary_cluster_df.copy()

    summary_df = summary_df[summary_df["family"].isin(eligible_row)].reset_index(drop=True)
    null_df = null_df[null_df["family"].isin(eligible_row)].reset_index(drop=True)
    boot_df = boot_df[boot_df["family"].isin(eligible_row)].reset_index(drop=True)
    row_delta_df = row_delta_df[row_delta_df["family"].isin(eligible_row)].reset_index(drop=True)

    summary_cluster_df = summary_cluster_df[
        summary_cluster_df["family"].isin(eligible_cluster)
    ].reset_index(drop=True)
    null_cluster_df = null_cluster_df[
        null_cluster_df["family"].isin(eligible_cluster)
    ].reset_index(drop=True)
    boot_cluster_df = boot_cluster_df[
        boot_cluster_df["family"].isin(eligible_cluster)
    ].reset_index(drop=True)
    key_delta_df = key_delta_df[key_delta_df["family"].isin(eligible_cluster)].reset_index(drop=True)

    excluded_row = [f for f in summary_df_full["family"] if f not in eligible_row]
    excluded_cluster = [f for f in summary_cluster_df_full["family"] if f not in eligible_cluster]

    return {
        "summary_df": summary_df,
        "summary_cluster_df": summary_cluster_df,
        "null_df": null_df,
        "null_cluster_df": null_cluster_df,
        "boot_df": boot_df,
        "boot_cluster_df": boot_cluster_df,
        "key_delta_df": key_delta_df,
        "row_delta_df": row_delta_df,
        "summary_df_full": summary_df_full,
        "summary_cluster_df_full": summary_cluster_df_full,
        "excluded_row": excluded_row,
        "excluded_cluster": excluded_cluster,
    }


# Backward-compatible alias used by report.py.
def load_notebook_equivalent_tables(run_root: str | Path) -> dict[str, pd.DataFrame | list[str]]:
    return load_paired_tables(run_root)


def _save_figure(fig, figures_dir: Path, stem: str) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figures_dir / f"{stem}.pdf"
    png_path = figures_dir / f"{stem}.png"
    svg_path = figures_dir / f"{stem}.svg"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    return {"pdf": pdf_path.as_posix(), "png": png_path.as_posix(), "svg": svg_path.as_posix()}


def _plot_family_physical_edges(
    eligibility_df: pd.DataFrame | None,
    figures_dir: Path,
    *,
    threshold: int = 100,
) -> dict[str, str]:
    if eligibility_df is None or "family" not in eligibility_df.columns:
        return {}
    edge_col = "n_physical_edges" if "n_physical_edges" in eligibility_df.columns else "physical_edges"
    if edge_col not in eligibility_df.columns:
        return {}

    plot_df = eligibility_df[["family", edge_col]].copy()
    plot_df[edge_col] = pd.to_numeric(plot_df[edge_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[edge_col]).sort_values(edge_col, ascending=False)
    if plot_df.empty:
        return {}

    plt, mticker = _load_matplotlib()
    fig_height = max(4.1, 0.38 * len(plot_df) + 1.0)
    fig, ax = plt.subplots(figsize=(6.2, fig_height), constrained_layout=True)
    colors = np.where(plot_df[edge_col].to_numpy() >= threshold, "#2A6F97", "#BDBDBD")
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df[edge_col], color=colors, edgecolor="0.25", linewidth=0.4, zorder=2)
    ax.axvline(threshold, color="0.15", linestyle=":", linewidth=1.2,
               label=f"edge-support threshold = {threshold}", zorder=3)
    ax.set_yticks(y, _family_labels(plot_df["family"]))
    ax.invert_yaxis()
    ax.set_xlabel("KG-edge support")
    ax.set_ylabel("")
    ax.xaxis.label.set_size(10)
    ax.tick_params(axis="both", labelsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6, integer=True))
    ax.grid(axis="x", color="0.88", linewidth=0.6, zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    outputs = _save_figure(fig, figures_dir, "fig_val4h1_family_physical_edges")
    plt.close(fig)
    return outputs


def _plot_n_by_family(summary_df: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    plt, mticker = _load_matplotlib()
    n_df = summary_df[["family", "n_evaluable_rows", "n_perturbations"]].copy()
    n_df = n_df.sort_values("n_evaluable_rows").reset_index(drop=True)
    if n_df.empty:
        return {}
    y = np.arange(len(n_df))

    fig_h = max(3.8, 0.42 * len(n_df) + 1.25)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, fig_h), sharey=True, constrained_layout=True)
    axes[0].hlines(y, 1.0, np.maximum(n_df["n_evaluable_rows"], 1.0),
                   color="#3D5A80", linewidth=2.4, alpha=0.72, zorder=2)
    axes[0].scatter(n_df["n_evaluable_rows"], y, color="#3D5A80", edgecolor="0.12", s=32, zorder=3)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("evaluable rows (log scale)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(_family_labels(n_df["family"]))
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axes[0].xaxis.grid(True, color="0.88", linewidth=0.7)
    axes[0].set_axisbelow(True)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].hlines(y, 1.0, np.maximum(n_df["n_perturbations"], 1.0),
                   color="#98C1D9", linewidth=2.4, alpha=0.82, zorder=2)
    axes[1].scatter(n_df["n_perturbations"], y, color="#2A6F97", edgecolor="0.12", s=32, zorder=3)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("perturbation keys (log scale)")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axes[1].xaxis.grid(True, color="0.88", linewidth=0.7)
    axes[1].set_axisbelow(True)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    outputs = _save_figure(fig, figures_dir, "fig_val4h1_n_by_family")
    plt.close(fig)
    return outputs


def _plot_forest(
    summary_df: pd.DataFrame,
    *,
    delta_col: str,
    ci_low_col: str,
    ci_high_col: str,
    title: str,
    x_label: str = r"$\Delta_f$",
    font_size: float | None = None,
    title_font_size: float | None = None,
    errorbar_label: str | None = None,
    y_tick_pad: float | None = None,
    x_tick_pad: float | None = None,
    x_label_pad: float | None = None,
    scale_figure_with_font: bool = True,
    fixed_layout: bool = False,
    legend_loc: str = "lower right",
    legend_font_size: float | None = None,
    overall_scale: float = 1.0,
    figure_scale_override: float | None = None,
    figsize_override: tuple[float, float] | None = None,
    stem: str,
    figures_dir: Path,
) -> dict[str, str]:
    plt, mticker = _load_matplotlib()
    df = summary_df.sort_values(delta_col).reset_index(drop=True)
    if df.empty:
        return {}
    y = np.arange(len(df))
    values = df[delta_col].astype(float).to_numpy()
    low = df[ci_low_col].astype(float).to_numpy()
    high = df[ci_high_col].astype(float).to_numpy()
    xerr = [np.maximum(values - low, 0.0), np.maximum(high - values, 0.0)]
    colors = np.where(values >= 0, "#2A6F97", "#B95F45")

    plot_scale = max(overall_scale, 1e-6)
    plot_font_size = font_size * plot_scale if font_size is not None else None
    plot_title_font_size = title_font_size * plot_scale if title_font_size is not None else None
    plot_y_tick_pad = y_tick_pad * plot_scale if y_tick_pad is not None else None
    plot_x_tick_pad = x_tick_pad * plot_scale if x_tick_pad is not None else None
    plot_x_label_pad = x_label_pad * plot_scale if x_label_pad is not None else None
    plot_legend_font_size = (
        legend_font_size * plot_scale if legend_font_size is not None else None
    )

    text_scale = 1.0 if plot_font_size is None else max(plot_font_size / 9.0, 1.0)
    figure_scale = (
        figure_scale_override
        if figure_scale_override is not None
        else text_scale if scale_figure_with_font else 1.0
    )
    if figsize_override is None:
        fig_h = max(3.4, 0.42 * len(df) + 1.25) * figure_scale
        fig_w = 8.0 * figure_scale
    else:
        fig_w, fig_h = figsize_override
    line_scale = 1.0 if font_size is None else text_scale
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=not fixed_layout)
    if fixed_layout:
        fig.subplots_adjust(left=0.45, right=0.97, top=0.88, bottom=0.23)
    ax.barh(
        y,
        values,
        color=colors,
        edgecolor="0.18",
        linewidth=0.45 * line_scale,
        height=0.68,
    )
    ax.scatter(
        values,
        y,
        c=colors,
        edgecolor="0.12",
        linewidth=0.45 * line_scale,
        marker="D",
        s=26 * line_scale * line_scale,
        zorder=4,
    )
    ax.errorbar(values, y, xerr=xerr, fmt="none", ecolor="0.2",
                elinewidth=0.85 * line_scale, capsize=2.5 * line_scale,
                capthick=0.85 * line_scale, zorder=3,
                label=errorbar_label)
    x_stack = np.concatenate([low, high, values, np.array([0.0])])
    x_lo, x_hi = float(np.nanmin(x_stack)), float(np.nanmax(x_stack))
    span = max(x_hi - x_lo, 1e-6)
    pad = span * 0.18
    ax.set_xlim(x_lo - pad, x_hi + pad)
    ax.axvline(0.0, color="0.12", linewidth=0.8 * line_scale, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(_family_labels(df["family"]))
    ax.set_xlabel(x_label, labelpad=plot_x_label_pad)
    ax.set_ylabel("")
    title_size = plot_title_font_size if plot_title_font_size is not None else plot_font_size
    title_kwargs = {"fontsize": title_size} if title_size is not None else {}
    if plot_font_size is not None:
        ax.xaxis.label.set_size(plot_font_size)
        ax.tick_params(axis="both", labelsize=plot_font_size)
    if plot_y_tick_pad is not None:
        ax.tick_params(axis="y", pad=plot_y_tick_pad)
    if plot_x_tick_pad is not None:
        ax.tick_params(axis="x", pad=plot_x_tick_pad)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.3g}"))
    ax.xaxis.grid(True, color="0.88", linewidth=0.7 * line_scale)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if errorbar_label is not None:
        legend_size = plot_legend_font_size if plot_legend_font_size is not None else plot_font_size
        legend = ax.legend(loc=legend_loc, frameon=True, fontsize=legend_size)
        frame = legend.get_frame()
        frame.set_facecolor("0.94")
        frame.set_edgecolor("0.55")
        frame.set_linewidth(0.8 * line_scale)
        frame.set_alpha(1.0)
    outputs = _save_figure(fig, figures_dir, stem)
    plt.close(fig)
    return outputs


def _plot_delta_histograms(
    *,
    delta_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    figures_dir: Path,
    value_col: str,
    delta_col: str,
    ci_low_col: str,
    ci_high_col: str,
    title: str,
    stem: str,
    clip: tuple[float, float] = (-0.5, 0.5),
) -> dict[str, str]:
    plt, mticker = _load_matplotlib()
    families = [str(f) for f in summary_df["family"].tolist()]
    if not families or delta_df.empty or value_col not in delta_df.columns:
        return {}

    summary_by_family = summary_df.set_index("family")
    series = pd.to_numeric(delta_df[value_col], errors="coerce")
    clipped = series.clip(lower=clip[0], upper=clip[1])
    bin_edges = np.linspace(clip[0], clip[1], 41)

    n_cols = 3
    n_rows = int(np.ceil(len(families) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(10.6, max(2.3 * n_rows, 3.4)),
        sharex=True, constrained_layout=True,
    )
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, family in zip(axes_arr, families):
        mask = delta_df["family"].astype(str) == family
        vals = clipped[mask].dropna().to_numpy()
        ax.hist(vals, bins=bin_edges, color="#BDBDBD", edgecolor="white",
                linewidth=0.35, alpha=0.9)
        delta_val = float(summary_by_family.loc[family, delta_col])
        ci_l = float(summary_by_family.loc[family, ci_low_col])
        ci_h = float(summary_by_family.loc[family, ci_high_col])
        ax.axvline(0.0, color="0.55", linestyle=":", linewidth=0.9)
        ax.axvline(delta_val, color="#0072B2", linewidth=1.4, label=r"$\Delta_f$")
        ax.axvspan(ci_l, ci_h, color="#0072B2", alpha=0.18, label="bootstrap 95% CI")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4, integer=True))
        ax.xaxis.grid(True, color="0.9", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_arr[len(families):]:
        ax.set_visible(False)
    for ax in axes_arr[-n_cols:]:
        if ax.get_visible():
            ax.set_xlabel(r"$\Delta$ observability")
    for ax in axes_arr[::n_cols]:
        ax.set_ylabel("count")

    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    outputs = _save_figure(fig, figures_dir, stem)
    plt.close(fig)
    return outputs


def make_all_figures(
    *,
    run_root: str | Path,
    figures_dir: str | Path | None = None,
) -> dict[str, str]:
    run_root = Path(run_root)
    figures_path = Path(figures_dir) if figures_dir is not None else run_root / "report" / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)
    _set_plot_style()
    tables = load_paired_tables(run_root)

    output_paths: dict[str, str] = {}
    eligibility_df = _read_optional(run_root, "val4_family_eligibility.csv")
    figure_specs: list[tuple[str, dict[str, str]]] = [
        (
            "family_physical_edges",
            (
                plot_family_edge_support(
                    eligibility_df,
                    output_png=figures_path / "fig_val4h1_family_physical_edges.png",
                    family_labels=FAMILY_FIGURE_LABELS,
                    threshold=100,
                )
                if eligibility_df is not None
                else {}
            ),
        ),
        (
            "delta",
            plot_family_forest(
                tables["summary_cluster_df"],
                value_col="delta_f_cluster",
                low_col="ci_low_cluster",
                high_col="ci_high_cluster",
                xlabel="observability",
                output_png=figures_path / "fig_val4h1_delta.png",
                family_labels=FAMILY_FIGURE_LABELS,
            ),
        ),
        (
            "cluster_delta_hist",
            _plot_delta_histograms(
                delta_df=tables["key_delta_df"],
                summary_df=tables["summary_cluster_df"],
                figures_dir=figures_path,
                value_col="delta_key",
                delta_col="delta_f_cluster",
                ci_low_col="ci_low_cluster",
                ci_high_col="ci_high_cluster",
                title=r"Per-key $\Delta$ observability by family",
                stem="fig_val4h1_cluster_delta_hist",
            ),
        ),
        (
            "row_delta",
            _plot_forest(
                tables["summary_df"],
                delta_col="delta_f",
                ci_low_col="ci_low",
                ci_high_col="ci_high",
                title=r"Row-level $\Delta$ observability by family",
                stem="fig_val4h1_row_delta",
                figures_dir=figures_path,
            ),
        ),
        (
            "row_delta_hist",
            _plot_delta_histograms(
                delta_df=tables["row_delta_df"],
                summary_df=tables["summary_df"],
                figures_dir=figures_path,
                value_col="delta_row",
                delta_col="delta_f",
                ci_low_col="ci_low",
                ci_high_col="ci_high",
                title=r"Per-row $\Delta$ observability by family",
                stem="fig_val4h1_row_delta_hist",
            ),
        ),
        (
            "n_by_family",
            _plot_n_by_family(tables["summary_cluster_df"], figures_path),
        ),
    ]
    for key, paths in figure_specs:
        for ext, path in paths.items():
            output_paths[f"{key}_{ext}"] = _relative_to_run_root(Path(path), run_root)
    return output_paths


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Val4/H1 paired-delta figures.")
    parser.add_argument("--run-root", required=True, help="output/val4/h1/<run_name> directory")
    parser.add_argument("--figures-dir", default=None, help="Optional figure output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = make_all_figures(run_root=args.run_root, figures_dir=args.figures_dir)
    for key, path in outputs.items():
        print(f"{key}\t{path}")


if __name__ == "__main__":
    main()
