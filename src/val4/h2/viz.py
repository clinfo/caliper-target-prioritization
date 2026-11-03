from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from val4.paper_viz import plot_family_forest


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


def _pdf_path(path: Path) -> Path:
    return path.with_suffix(".pdf")


def _svg_path(path: Path) -> Path:
    return path.with_suffix(".svg")


def _save_figure(fig, path: Path) -> None:
    fig.savefig(path)
    fig.savefig(_pdf_path(path), format="pdf")
    fig.savefig(_svg_path(path), format="svg")


def _barh(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    path: Path,
    *,
    xlabel: str,
    ci_low_col: str | None = None,
    ci_high_col: str | None = None,
    font_size: float | None = None,
    title_font_size: float | None = None,
    errorbar_label: str | None = None,
    y_tick_pad: float | None = None,
    x_tick_pad: float | None = None,
    x_label_pad: float | None = None,
    legend_loc: str = "lower right",
    legend_font_size: float | None = None,
    overall_scale: float = 1.0,
    figure_scale_override: float | None = None,
    figsize_override: tuple[float, float] | None = None,
) -> None:
    _set_plot_style()
    plt, mticker = _load_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or len(df) == 0 or value_col not in df.columns:
        path.write_text("No data available for figure.\n", encoding="utf-8")
        return
    cols = ["family", value_col]
    has_ci = bool(ci_low_col and ci_high_col and ci_low_col in df.columns and ci_high_col in df.columns)
    if has_ci:
        cols.extend([str(ci_low_col), str(ci_high_col)])
    plot_df = df[cols].dropna(subset=[value_col]).sort_values(value_col).reset_index(drop=True)
    if len(plot_df) == 0:
        path.write_text("No data available for figure.\n", encoding="utf-8")
        return

    y = np.arange(len(plot_df))
    values = plot_df[value_col].astype(float).to_numpy()
    labels = [_family_label(family) for family in plot_df["family"]]
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
    figure_scale = figure_scale_override if figure_scale_override is not None else text_scale
    if figsize_override is None:
        fig_h = max(3.4, 0.42 * len(plot_df) + 1.25) * figure_scale
        fig_w = 8.0 * figure_scale
    else:
        fig_w, fig_h = figsize_override
    line_scale = text_scale
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
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

    if has_ci and len(plot_df):
        low = plot_df[str(ci_low_col)].astype(float).to_numpy()
        high = plot_df[str(ci_high_col)].astype(float).to_numpy()
        xerr = [np.maximum(values - low, 0.0), np.maximum(high - values, 0.0)]
        ax.errorbar(
            values,
            y,
            xerr=xerr,
            fmt="none",
            ecolor="0.2",
            elinewidth=0.85 * line_scale,
            capsize=2.5 * line_scale,
            capthick=0.85 * line_scale,
            zorder=3,
            label=errorbar_label,
        )
        x_min = np.nanmin(np.concatenate([low, values]))
        x_max = np.nanmax(np.concatenate([high, values]))
    else:
        x_min = np.nanmin(values)
        x_max = np.nanmax(values)

    span = max(x_max - x_min, 1e-6)
    pad = span * 0.18
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.axvline(0.0, color="0.12", linewidth=0.8 * line_scale)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel, labelpad=plot_x_label_pad)
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
    _save_figure(fig, path)
    plt.close(fig)


def _metric_path(run_root: Path, filename: str) -> Path:
    path = run_root / "metrics" / filename
    if path.exists():
        return path
    return run_root / filename


def make_all_figures(run_root: str | Path, figures_dir: str | Path | None = None) -> dict[str, str]:
    run_root = Path(run_root)
    figures_path = Path(figures_dir) if figures_dir is not None else run_root / "report" / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    lofo_path = _metric_path(run_root, "val4_h2_lofo_family_summary.csv")
    if lofo_path.exists():
        df = pd.read_csv(lofo_path)
        out = figures_path / "fig_val4h2_lofo_delta_ap.png"
        paper_outputs = plot_family_forest(
            df,
            value_col="delta_AP_null_adjusted_median",
            low_col="delta_AP_null_adjusted_ci_low",
            high_col="delta_AP_null_adjusted_ci_high",
            xlabel="null-adjusted ΔAP",
            output_png=out,
            family_labels=FAMILY_FIGURE_LABELS,
        )
        outputs["h2_lofo_delta_ap"] = paper_outputs.get("png", str(out))
        outputs["h2_lofo_delta_ap_pdf"] = paper_outputs.get(
            "pdf", str(_pdf_path(out))
        )
        outputs["h2_lofo_delta_ap_svg"] = paper_outputs.get(
            "svg", str(_svg_path(out))
        )
    sign_path = _metric_path(run_root, "val4_h2_sign_shuffle_family_summary.csv")
    if sign_path.exists():
        df = pd.read_csv(sign_path)
        out = figures_path / "fig_val4h2_sign_shuffle_delta_ap.png"
        _barh(
            df,
            "delta_AP_median",
            "Family sign shuffle",
            out,
            xlabel=r"$\Delta$AP",
            ci_low_col="delta_AP_ci_low",
            ci_high_col="delta_AP_ci_high",
        )
        outputs["h2_sign_shuffle_delta_ap"] = str(out)
        outputs["h2_sign_shuffle_delta_ap_pdf"] = str(_pdf_path(out))
        outputs["h2_sign_shuffle_delta_ap_svg"] = str(_svg_path(out))
    return outputs
