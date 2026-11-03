from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .summary import (
    PRIMARY_K,
    load_ablation_metrics,
    load_ablation_summary,
    resolve_k_valid,
    summarize_fixed_k_internal_ablation,
)
from .support_robustness import build_support_robustness_table
from .paper_viz import (
    make_label_confidence_score_distribution,
    make_support_score_distribution,
)


METHOD_ORDER = ["proposed", "kg_sign_noalign", "kg_nosign", "noKG_sizematch", "ncs"]
METHOD_LABEL = {
    "proposed": "CALIPER",
    "kg_sign_noalign": "w/o KD alignment",
    "kg_nosign": "w/o edge polarity",
    "noKG_sizematch": "Size-matched no-KG",
    "ncs": "L1000 NCS",
}
METHOD_COLOR = {
    "proposed": "#0072B2",
    "kg_sign_noalign": "#E69F00",
    "kg_nosign": "#009E73",
    "noKG_sizematch": "#CC79A7",
    "ncs": "#666666",
}

# `noKG` remains available in analysis metrics but is intentionally omitted
# from paper-facing Val2 outputs.
PRESENTATION_EXCLUDED_METHODS = frozenset({"noKG"})
# L1000 NCS is k-independent and is drawn as a flat, error-bar-free dotted line
# on figures with a k axis.
REFERENCE_LINE_METHODS = ("ncs",)
FALLBACK_COLORS = ["#56B4E9", "#E69F00", "#F0E442", "#999999"]
ERROR_BAR_LABEL = "Error bars: SEM"
PAPER_FIGSIZE = (7.2, 2.4)
ABLATION_FIGSIZE = (4.8, 2.4)
K_VALID_FIGSIZE = (5.0, 2.4)
FIXED_K_ABLATION_FIGSIZE = (3.5, 2.4)
MEAN_AP_SHARED_XLIM = (0.05, 0.10)
COMPARATOR_GRAY = "#8C8C8C"
DELTA_BOX_COLOR = "#56B4E9"
PAPER_TALL_FIGSIZE = (3.5, 2.8)
PAPER_AXIS_LINEWIDTH = 0.7
PAPER_ERROR_LINEWIDTH = 0.8
PAPER_ERROR_CAPSIZE = 2.5
PAPER_MARKER_EDGE_WIDTH = 0.45
EFFECT_FOREST_HORIZONTAL_FIGSIZE = (4.2, 1.9)
EFFECT_FOREST_HORIZONTAL_LABEL_SIZE = 7
EFFECT_FOREST_HORIZONTAL_TICK_SIZE = 6.5
EFFECT_FOREST_STAR_NOTE_SIZE = 5.5
SUPPORT_ROBUSTNESS_FIGSIZE = (7.2, 4.8)


def _apply_paper_rcparams() -> None:
    import matplotlib.pyplot as plt

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
            "axes.linewidth": PAPER_AXIS_LINEWIDTH,
            "grid.linewidth": 0.35,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.0,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
        }
    )


def _apply_panel_rcparams() -> None:
    """Paper rcParams for the ablation comparison panels (no top/right spines)."""
    import matplotlib.pyplot as plt

    _apply_paper_rcparams()
    plt.rcParams.update(
        {
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _draw_y_axis_break_marks(ax) -> None:
    """Draw the two wave marks that flag a truncated (non-zero) y axis."""
    wave_x = np.linspace(-0.018, 0.018, 80)
    for y_offset in (0.032, 0.054):
        wave_y = y_offset + 0.006 * np.sin(np.linspace(0, 2 * np.pi, wave_x.size))
        ax.plot(
            wave_x,
            wave_y,
            transform=ax.transAxes,
            color="black",
            linewidth=1.0,
            clip_on=False,
            solid_capstyle="round",
            zorder=5,
        )


def _style_panel_axes(ax) -> None:
    """Shared spine / tick / grid styling for the ablation comparison panels."""
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(PAPER_AXIS_LINEWIDTH)
    ax.tick_params(axis="both", which="major", width=0.6, length=2.5, pad=2.0)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.grid(axis="x", alpha=0.10, linewidth=0.8)
    ax.set_axisbelow(True)


def _save_figure(fig, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _relative_existing_outputs(run_root: Path, key: str, png_path: Path) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    for suffix_key, path in [
        (key, png_path),
        (f"{key}_pdf", png_path.with_suffix(".pdf")),
        (f"{key}_svg", png_path.with_suffix(".svg")),
    ]:
        if path.exists():
            outputs[suffix_key] = str(path.relative_to(run_root))
    return outputs


def _relative_existing_file_output(run_root: Path, key: str, path: Path) -> Dict[str, str]:
    if path.exists():
        return {key: str(path.relative_to(run_root))}
    return {}


def _method_colors(methods: list[str]) -> list[str]:
    colors: list[str] = []
    fallback_idx = 0
    for method in methods:
        if method in METHOD_COLOR:
            colors.append(METHOD_COLOR[method])
        else:
            colors.append(FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)])
            fallback_idx += 1
    return colors


def _save_lineplot(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str,
    out_path: Path,
    title: str,
    ylabel: str,
    yerr: str | None = None,
    show_errorbars: bool = True,
) -> bool:
    if df.empty or not {x, y, group}.issubset(df.columns):
        return False
    df = df[~df[group].astype(str).isin(PRESENTATION_EXCLUDED_METHODS)].copy()
    if df.empty:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_paper_rcparams()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=ABLATION_FIGSIZE, constrained_layout=True)
    no_kg_lines: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    grouped = [(str(label), grp) for label, grp in df.groupby(group)]
    colors = dict(zip([label for label, _ in grouped], _method_colors([label for label, _ in grouped])))
    for method, grp in grouped:
        grp = grp.sort_values(x)
        if method in REFERENCE_LINE_METHODS:
            no_kg_lines.append(
                (
                    pd.to_numeric(grp[x], errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(grp[y], errors="coerce").to_numpy(dtype=float),
                    METHOD_LABEL.get(method, method),
                    method,
                )
            )
        else:
            if show_errorbars and yerr and yerr in grp.columns:
                err = pd.to_numeric(grp[yerr], errors="coerce")
                ax.errorbar(
                    grp[x],
                    grp[y],
                    yerr=err,
                    marker="o",
                    markersize=3.0,
                    markeredgewidth=PAPER_MARKER_EDGE_WIDTH,
                    linewidth=1.2,
                    elinewidth=PAPER_ERROR_LINEWIDTH,
                    capsize=PAPER_ERROR_CAPSIZE,
                    capthick=PAPER_ERROR_LINEWIDTH,
                    color=colors[method],
                    ecolor=colors[method],
                    label=METHOD_LABEL.get(method, method),
                )
            else:
                ax.plot(
                    grp[x],
                    grp[y],
                    marker="o",
                    markersize=3.0,
                    markeredgewidth=PAPER_MARKER_EDGE_WIDTH,
                    linewidth=1.2,
                    color=colors[method],
                    label=METHOD_LABEL.get(method, method),
                )
    x_min, x_max = ax.get_xlim()
    for x_values, y_values, label, ref_method in no_kg_lines:
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        if not valid.any():
            continue
        x_plot = x_values[valid]
        y_plot = y_values[valid]
        order = np.argsort(x_plot)
        x_plot = x_plot[order]
        y_plot = y_plot[order]
        x_ext = np.concatenate(([x_min], x_plot, [x_max]))
        y_ext = np.concatenate(([y_plot[0]], y_plot, [y_plot[-1]]))
        ax.plot(x_ext, y_ext, linestyle=":", linewidth=1.2, color=colors.get(ref_method), label=label)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("distance k")
    ax.set_ylabel(ylabel)
    handles, labels = ax.get_legend_handles_labels()
    if show_errorbars and yerr:
        errorbar_proxy = ax.errorbar(
            [],
            [],
            yerr=[],
            fmt="none",
            ecolor="#333333",
            elinewidth=PAPER_ERROR_LINEWIDTH,
            capsize=PAPER_ERROR_CAPSIZE,
            capthick=PAPER_ERROR_LINEWIDTH,
            label=ERROR_BAR_LABEL,
        )
        handles.append(errorbar_proxy)
        labels.append(ERROR_BAR_LABEL)
    ax.legend(handles, labels, loc="best", ncol=2, frameon=True)
    _save_figure(fig, out_path)
    plt.close(fig)
    return True


def _save_barplot(df: pd.DataFrame, *, x: str, y: str, out_path: Path, title: str) -> bool:
    if df.empty or not {x, y}.issubset(df.columns):
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_paper_rcparams()

    plot_df = df[[x, y]].copy()
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[y])
    if plot_df.empty:
        return False
    plot_df = plot_df.sort_values(y)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_height = max(PAPER_TALL_FIGSIZE[1], 0.22 * len(plot_df))
    fig, ax = plt.subplots(figsize=(PAPER_TALL_FIGSIZE[0], fig_height), constrained_layout=True)
    ax.barh(plot_df[x].astype(str), plot_df[y].astype(float))
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel(y)
    ax.set_ylabel(x)
    _save_figure(fig, out_path)
    plt.close(fig)
    return True


def _save_ap_positive_gene_scatter_for_method(
    metrics: pd.DataFrame,
    *,
    out_path: Path,
    k_valid: list[int],
    method: str,
) -> bool:
    required = {"method", "k", "AP", "n_pos"}
    if metrics.empty or not required.issubset(metrics.columns):
        return False
    df = metrics[metrics["method"].astype(str) == str(method)].copy()
    df["method"] = df["method"].astype(str)
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df["AP"] = pd.to_numeric(df["AP"], errors="coerce")
    df["n_pos"] = pd.to_numeric(df["n_pos"], errors="coerce")
    k_values = [int(k) for k in k_valid] if k_valid else sorted(int(k) for k in df["k"].dropna().unique())
    df = df[df["k"].isin(k_values)].dropna(subset=["k", "AP", "n_pos"])
    if df.empty:
        return False

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_paper_rcparams()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=ABLATION_FIGSIZE, constrained_layout=True)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    marker_by_k = {k: markers[idx % len(markers)] for idx, k in enumerate(k_values)}
    colors = _method_colors([f"k={k}" for k in k_values])
    for idx, k in enumerate(k_values):
        subset = df[df["k"] == k]
        if subset.empty:
            continue
        ax.scatter(
            subset["n_pos"],
            subset["AP"],
            s=14,
            alpha=0.64,
            color=colors[idx],
            marker=marker_by_k[k],
            edgecolors="none",
            label=f"distance k={k}",
        )
    ax.set_xlabel("Positive target genes per disease")
    ax.set_ylabel("AP")
    ax.grid(axis="both", alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=True, title="Distance", borderpad=0.25, labelspacing=0.25)
    _save_figure(fig, out_path)
    plt.close(fig)
    return True


def _ordered_methods(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "method" not in df.columns:
        return df
    out = df[~df["method"].astype(str).isin(PRESENTATION_EXCLUDED_METHODS)].copy()
    if out.empty:
        return out
    order = [m for m in METHOD_ORDER if m in set(out["method"].astype(str))]
    order.extend(sorted(set(out["method"].astype(str)) - set(order)))
    out["method"] = pd.Categorical(out["method"].astype(str), categories=order, ordered=True)
    return out.sort_values("method").reset_index(drop=True)


def _caliper_vs_sizematched_nokg_standardized_delta(
    metrics: pd.DataFrame,
    *,
    metric: str,
    k: int = 4,
    n: int = 10,
    ascending: bool = False,
) -> pd.DataFrame:
    sem_col = f"{metric}_SEM"
    required = {"disease_id", "k", "method", metric, sem_col}
    if metrics.empty or not required.issubset(metrics.columns):
        return pd.DataFrame()
    df = metrics.copy()
    df["method"] = df["method"].astype(str)
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df[sem_col] = pd.to_numeric(df[sem_col], errors="coerce")
    df = df[
        (df["k"] == int(k))
        & df["method"].isin(["proposed", "noKG_sizematch"])
    ].dropna(subset=["disease_id", metric])
    if df.empty or set(df["method"]) != {"proposed", "noKG_sizematch"}:
        return pd.DataFrame()

    caliper = df[df["method"] == "proposed"].copy()
    baseline = df[df["method"] == "noKG_sizematch"].copy()
    value_cols = ["disease_id", metric]
    optional_cols = [col for col in ["chance", "n_pos", "n_neg"] if col in caliper.columns]
    merged = caliper[value_cols + optional_cols].merge(
        baseline[value_cols + [sem_col]],
        on="disease_id",
        how="inner",
        suffixes=("_caliper", "_size_matched_noKG"),
    )
    if merged.empty:
        return pd.DataFrame()
    baseline_sem_col = f"{sem_col}_size_matched_noKG" if f"{sem_col}_size_matched_noKG" in merged.columns else sem_col
    output_baseline_sem_col = f"{sem_col}_size_matched_noKG"
    if output_baseline_sem_col not in merged.columns:
        merged[output_baseline_sem_col] = merged[baseline_sem_col]
    merged[f"delta_{metric}"] = merged[f"{metric}_caliper"] - merged[f"{metric}_size_matched_noKG"]
    merged[f"standardized_delta_{metric}"] = merged[f"delta_{metric}"] / merged[baseline_sem_col].replace(0, np.nan)
    out = (
        merged.dropna(subset=[f"standardized_delta_{metric}"])
        .sort_values(f"standardized_delta_{metric}", ascending=ascending)
        .head(int(n))
        .copy()
    )
    out.insert(0, "standardized_rank", np.arange(1, len(out) + 1))
    out.insert(1, "k", int(k))
    out.insert(2, "comparison", "CALIPER - Size-matched no-KG")
    keep = [
        c
        for c in [
            "standardized_rank",
            "k",
            "comparison",
            "disease_id",
            f"{metric}_caliper",
            f"{metric}_size_matched_noKG",
            f"delta_{metric}",
            output_baseline_sem_col,
            f"standardized_delta_{metric}",
            "chance",
            "n_pos",
            "n_neg",
        ]
        if c in out.columns
    ]
    return out[keep].reset_index(drop=True)


def _save_selected_disease_score_table(table: pd.DataFrame, *, out_csv: Path) -> bool:
    if table.empty:
        return False
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    table.to_csv(out_csv.with_suffix(".tsv"), sep="\t", index=False)
    return True


def _save_k_valid_metric_figure(
    df: pd.DataFrame,
    *,
    out_png: Path,
    k_valid: list[int],
    mean_col: str,
    sem_col: str,
    ylabel: str,
    title: str,
) -> dict[str, str]:
    required = {"method", "k", mean_col, sem_col}
    if df.empty or not required.issubset(df.columns):
        return {}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = df.copy()
    plot_df["k"] = pd.to_numeric(plot_df["k"], errors="coerce")
    plot_df = plot_df[plot_df["k"].isin([int(k) for k in k_valid])].dropna(subset=["k"])
    if plot_df.empty:
        return {}

    plot_df = _ordered_methods(plot_df)
    methods = [str(m) for m in plot_df["method"].astype(str).drop_duplicates().tolist()]
    k_values = sorted(int(k) for k in plot_df["k"].dropna().unique())
    colors = _method_colors(methods)
    x = np.arange(len(k_values), dtype=float)
    bar_width = min(0.16, 0.82 / max(len(methods), 1))
    offsets = (np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2.0) * bar_width
    use_y_break = mean_col == "mAP"

    _apply_panel_rcparams()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=K_VALID_FIGSIZE, constrained_layout=True)
    no_kg_lines: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    y_lows: list[float] = []
    y_highs: list[float] = []
    for idx, method in enumerate(methods):
        method_df = plot_df[plot_df["method"].astype(str) == method].copy()
        method_df = method_df.set_index(method_df["k"].astype(int)).reindex(k_values)
        values = pd.to_numeric(method_df[mean_col], errors="coerce")
        if method in REFERENCE_LINE_METHODS:
            value_arr = values.to_numpy(dtype=float)
            finite = np.isfinite(value_arr)
            y_lows.extend(value_arr[finite].tolist())
            y_highs.extend(value_arr[finite].tolist())
            no_kg_lines.append(
                (
                    x + offsets[idx],
                    value_arr,
                    colors[idx],
                    METHOD_LABEL.get(method, method),
                )
            )
        else:
            errors = pd.to_numeric(method_df[sem_col], errors="coerce")
            value_arr = values.to_numpy(dtype=float)
            error_arr = np.nan_to_num(errors.to_numpy(dtype=float), nan=0.0)
            finite = np.isfinite(value_arr)
            y_lows.extend((value_arr[finite] - error_arr[finite]).tolist())
            y_highs.extend((value_arr[finite] + error_arr[finite]).tolist())
            ax.errorbar(
                x + offsets[idx],
                values,
                yerr=errors,
                fmt="o",
                markersize=3.0,
                markerfacecolor=colors[idx],
                markeredgecolor="black",
                markeredgewidth=PAPER_MARKER_EDGE_WIDTH,
                color=colors[idx],
                ecolor=colors[idx],
                elinewidth=PAPER_ERROR_LINEWIDTH,
                capsize=PAPER_ERROR_CAPSIZE,
                capthick=PAPER_ERROR_LINEWIDTH,
                label=METHOD_LABEL.get(method, method),
                zorder=3,
            )
    data_x_min = x[0] - 0.55
    data_x_max = x[-1] + 0.55
    ax.set_xlim(data_x_min, data_x_max)
    for x_values, y_values, color, label in no_kg_lines:
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        if not valid.any():
            continue
        x_plot = x_values[valid]
        y_plot = y_values[valid]
        order = np.argsort(x_plot)
        x_plot = x_plot[order]
        y_plot = y_plot[order]
        x_ext = np.concatenate(([data_x_min], x_plot, [data_x_max]))
        y_ext = np.concatenate(([y_plot[0]], y_plot, [y_plot[-1]]))
        ax.plot(
            x_ext,
            y_ext,
            linestyle=":",
            linewidth=1.2,
            color=color,
            label=label,
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_xlabel("candidate k")
    ax.set_ylabel(ylabel)
    if y_lows and y_highs:
        y_low = min(y_lows)
        y_high = max(y_highs)
        if use_y_break:
            y_span = max(y_high - y_low, 1e-9)
            ax.set_ylim(max(0.0, y_low - 0.12 * y_span), y_high + 0.10 * y_span)
        else:
            y_span = max(y_high, 1e-9)
            ax.set_ylim(0.0, y_high + 0.10 * y_span)
    _style_panel_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    legend_by_label = dict(zip(labels, handles))
    blank_handle = plt.Line2D([], [], linestyle="none", marker=None, alpha=0.0)
    legend_labels = [
        METHOD_LABEL.get(method, method)
        for method in methods
        if METHOD_LABEL.get(method, method) in legend_by_label
    ]
    legend_handles = [legend_by_label.get(label, blank_handle) for label in legend_labels]
    legend = ax.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#666666",
        borderaxespad=0.0,
        markerscale=0.8,
        handlelength=1.2,
        handletextpad=0.45,
        labelspacing=0.22,
        borderpad=0.25,
        ncol=1,
    )
    legend.get_frame().set_linewidth(0.5)
    if use_y_break:
        _draw_y_axis_break_marks(ax)
    if y_lows and y_highs:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer=renderer)
        axes_bbox = ax.get_window_extent(renderer=renderer)
        if legend_bbox.x0 < axes_bbox.x1:
            legend_bottom = ax.transData.inverted().transform((legend_bbox.x0, legend_bbox.y0))[1]
            y_min, y_max = ax.get_ylim()
            y_span = max(y_max - y_min, 1e-9)
            target_bottom = max(y_highs) + 0.04 * max(max(y_highs) - min(y_lows), 1e-9)
            legend_bottom_frac = (legend_bottom - y_min) / y_span
            if 0 < legend_bottom_frac < 1 and legend_bottom < target_bottom:
                ax.set_ylim(y_min, y_min + (target_bottom - y_min) / legend_bottom_frac)
    _save_figure(fig, out_png)
    out_pdf = out_png.with_suffix(".pdf")
    out_svg = out_png.with_suffix(".svg")
    plt.close(fig)
    stem = out_png.stem
    return {stem: str(out_png), f"{stem}_pdf": str(out_pdf), f"{stem}_svg": str(out_svg)}


def _save_fixed_k_internal_ablation_figure(
    df: pd.DataFrame,
    *,
    out_png: Path,
    k: int,
    mean_col: str,
    sem_col: str,
    ylabel: str,
) -> dict[str, str]:
    """Draw paper-facing CALIPER internal ablations at the fixed primary k."""
    required = {"method", "k", mean_col, sem_col}
    if df.empty or not required.issubset(df.columns):
        return {}

    plot_df = summarize_fixed_k_internal_ablation(df, int(k))
    if plot_df.empty:
        return {}
    plot_df = plot_df.dropna(subset=[mean_col])
    if plot_df.empty:
        return {}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizontal_ap = mean_col == "mAP"
    if horizontal_ap:
        plot_df = plot_df.sort_values(mean_col, ascending=False, kind="mergesort")
    methods = [str(m) for m in plot_df["method"].astype(str).tolist()]
    values = pd.to_numeric(plot_df[mean_col], errors="coerce").to_numpy(dtype=float)
    errors = np.nan_to_num(
        pd.to_numeric(plot_df[sem_col], errors="coerce").to_numpy(dtype=float), nan=0.0
    )
    colors = _method_colors(methods)
    x = np.arange(len(methods), dtype=float)
    use_y_break = horizontal_ap

    _apply_panel_rcparams()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIXED_K_ABLATION_FIGSIZE, constrained_layout=True)
    if horizontal_ap:
        y = np.arange(len(methods), dtype=float)
        for idx, method in enumerate(methods):
            is_caliper = method == "proposed"
            color = METHOD_COLOR["proposed"] if is_caliper else COMPARATOR_GRAY
            ax.errorbar(
                values[idx],
                y[idx],
                xerr=errors[idx],
                fmt="D" if is_caliper else "o",
                markersize=4.2 if is_caliper else 3.6,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=PAPER_MARKER_EDGE_WIDTH,
                color=color,
                ecolor=color,
                elinewidth=PAPER_ERROR_LINEWIDTH,
                capsize=PAPER_ERROR_CAPSIZE,
                capthick=PAPER_ERROR_LINEWIDTH,
                zorder=3,
            )
        ax.set_xlim(*MEAN_AP_SHARED_XLIM)
        ax.set_yticks(y)
        ax.set_yticklabels([METHOD_LABEL.get(method, method) for method in methods])
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        ax.grid(axis="x", alpha=0.22, linewidth=0.35)
        _style_panel_axes(ax)
        _save_figure(fig, out_png)
        out_pdf = out_png.with_suffix(".pdf")
        out_svg = out_png.with_suffix(".svg")
        plt.close(fig)
        stem = out_png.stem
        return {
            stem: str(out_png),
            f"{stem}_pdf": str(out_pdf),
            f"{stem}_svg": str(out_svg),
        }

    for idx, method in enumerate(methods):
        ax.errorbar(
            x[idx],
            values[idx],
            yerr=errors[idx],
            fmt="o",
            markersize=3.6,
            markerfacecolor=colors[idx],
            markeredgecolor="black",
            markeredgewidth=PAPER_MARKER_EDGE_WIDTH,
            color=colors[idx],
            ecolor=colors[idx],
            elinewidth=PAPER_ERROR_LINEWIDTH,
            capsize=PAPER_ERROR_CAPSIZE,
            capthick=PAPER_ERROR_LINEWIDTH,
            label=METHOD_LABEL.get(method, method),
            zorder=3,
        )
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_LABEL.get(method, method) for method in methods],
        rotation=28,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel(ylabel)
    y_low = float(np.nanmin(values - errors))
    y_high = float(np.nanmax(values + errors))
    if np.isfinite(y_low) and np.isfinite(y_high):
        if use_y_break:
            y_span = max(y_high - y_low, 1e-9)
            ax.set_ylim(max(0.0, y_low - 0.12 * y_span), y_high + 0.16 * y_span)
        else:
            ax.set_ylim(0.0, y_high + 0.10 * max(y_high, 1e-9))
    _style_panel_axes(ax)
    errorbar_proxy = ax.errorbar(
        [],
        [],
        yerr=[],
        fmt="none",
        ecolor="#333333",
        elinewidth=PAPER_ERROR_LINEWIDTH,
        capsize=PAPER_ERROR_CAPSIZE,
        capthick=PAPER_ERROR_LINEWIDTH,
        label=ERROR_BAR_LABEL,
    )
    legend = ax.legend(
        [errorbar_proxy],
        [ERROR_BAR_LABEL],
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#666666",
        borderaxespad=0.0,
        handlelength=1.2,
        handletextpad=0.45,
        borderpad=0.25,
    )
    legend.get_frame().set_linewidth(0.5)
    if use_y_break and ax.get_ylim()[0] > 0.0:
        _draw_y_axis_break_marks(ax)
    _save_figure(fig, out_png)
    out_pdf = out_png.with_suffix(".pdf")
    out_svg = out_png.with_suffix(".svg")
    plt.close(fig)
    stem = out_png.stem
    return {stem: str(out_png), f"{stem}_pdf": str(out_pdf), f"{stem}_svg": str(out_svg)}


def _save_support_robustness_figure(
    summary: pd.DataFrame,
    *,
    out_png: Path,
) -> dict[str, str]:
    """Draw the four planned k=4 evidence-support sensitivity views."""
    required = {
        "analysis",
        "setting",
        "mAP",
        "AP_SEM",
        "pair_retention",
        "delta_mAP",
    }
    if summary.empty or not required.issubset(summary.columns):
        return {}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_panel_rcparams()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=SUPPORT_ROBUSTNESS_FIGSIZE,
        constrained_layout=True,
    )
    blue = METHOD_COLOR["proposed"]

    threshold_specs = [
        (
            axes[0, 0],
            "minimum_measured_genes",
            "minimum measured genes",
            "c1",
        ),
        (
            axes[0, 1],
            "minimum_evaluable_cells",
            "minimum evaluable cell lines",
            "c2",
        ),
    ]
    for ax, analysis, xlabel, panel_label in threshold_specs:
        data = summary[summary["analysis"].eq(analysis)].copy()
        data["_setting"] = pd.to_numeric(data["setting"], errors="coerce")
        data = data.sort_values("_setting")
        x = data["_setting"].to_numpy(dtype=float)
        y = pd.to_numeric(data["mAP"], errors="coerce").to_numpy(dtype=float)
        err = pd.to_numeric(data["AP_SEM"], errors="coerce").to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=err,
            color=blue,
            marker="o",
            markerfacecolor=blue,
            markeredgecolor="black",
            markeredgewidth=PAPER_MARKER_EDGE_WIDTH,
            linewidth=1.2,
            elinewidth=PAPER_ERROR_LINEWIDTH,
            capsize=PAPER_ERROR_CAPSIZE,
            capthick=PAPER_ERROR_LINEWIDTH,
        )
        ax.set_ylabel("mAP")
        ax.set_xticks(x)
        retention = pd.to_numeric(
            data["pair_retention"], errors="coerce"
        ).to_numpy(dtype=float)
        ax.set_xticklabels(
            [
                f"{value:g}\n({fraction:.0%})"
                if np.isfinite(fraction)
                else f"{value:g}"
                for value, fraction in zip(x, retention)
            ]
        )
        ax.set_xlabel(f"{xlabel}\n(pair retention)")
        finite_high = (y + np.nan_to_num(err, nan=0.0))[np.isfinite(y)]
        if finite_high.size:
            ax.set_ylim(0.0, float(np.max(finite_high)) * 1.12)
        _style_panel_axes(ax)
        ax.text(
            -0.13,
            1.03,
            panel_label,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
        )

    aggregation = summary[summary["analysis"].eq("cell_aggregation")].copy()
    aggregation["_order"] = aggregation["setting"].map({"mean": 0, "median": 1})
    aggregation = aggregation.sort_values("_order")
    ax = axes[1, 0]
    x = np.arange(len(aggregation), dtype=float)
    y = pd.to_numeric(aggregation["mAP"], errors="coerce").to_numpy(dtype=float)
    err = pd.to_numeric(aggregation["AP_SEM"], errors="coerce").to_numpy(dtype=float)
    ax.errorbar(
        x,
        y,
        yerr=err,
        fmt="o",
        color=blue,
        markerfacecolor=blue,
        markeredgecolor="black",
        markeredgewidth=PAPER_MARKER_EDGE_WIDTH,
        elinewidth=PAPER_ERROR_LINEWIDTH,
        capsize=PAPER_ERROR_CAPSIZE,
        capthick=PAPER_ERROR_LINEWIDTH,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(aggregation["setting"].astype(str))
    ax.set_xlabel("cell-line aggregation")
    ax.set_ylabel("mAP")
    finite_high = (y + np.nan_to_num(err, nan=0.0))[np.isfinite(y)]
    if finite_high.size:
        ax.set_ylim(0.0, float(np.max(finite_high)) * 1.12)
    _style_panel_axes(ax)
    ax.text(
        -0.13,
        1.03,
        "c3",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )

    loo = summary[summary["analysis"].eq("leave_one_cell_line_out")].copy()
    loo = loo.sort_values("setting")
    ax = axes[1, 1]
    x = np.arange(len(loo), dtype=float)
    delta = pd.to_numeric(loo["delta_mAP"], errors="coerce").to_numpy(dtype=float)
    ax.axhline(0.0, color="#666666", linestyle=":", linewidth=0.8)
    ax.scatter(
        x,
        delta,
        s=16,
        color=blue,
        edgecolor="black",
        linewidth=PAPER_MARKER_EDGE_WIDTH,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        loo["setting"].astype(str),
        rotation=55,
        ha="right",
        rotation_mode="anchor",
        fontsize=6,
    )
    ax.set_xlabel("omitted cell line")
    ax.set_ylabel("ΔmAP from full analysis")
    finite_delta = np.abs(delta[np.isfinite(delta)])
    if finite_delta.size:
        bound = max(float(np.max(finite_delta)) * 1.2, 1e-4)
        ax.set_ylim(-bound, bound)
    _style_panel_axes(ax)
    ax.text(
        -0.13,
        1.03,
        "c4",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )

    _save_figure(fig, out_png)
    out_pdf = out_png.with_suffix(".pdf")
    out_svg = out_png.with_suffix(".svg")
    plt.close(fig)
    stem = out_png.stem
    return {
        stem: str(out_png),
        f"{stem}_pdf": str(out_pdf),
        f"{stem}_svg": str(out_svg),
    }


def _permutation_summary_for_plot(run_root: Path) -> pd.DataFrame:
    df = _read_csv(run_root / "permutation" / "val2_2_permutation_summary.csv")
    if df.empty:
        return df
    if {"mAP", "mAP_base"}.issubset(df.columns) and "delta_mAP" not in df.columns:
        df["delta_mAP"] = pd.to_numeric(df["mAP"], errors="coerce") - pd.to_numeric(df["mAP_base"], errors="coerce")
    if {"mean_Lift@10", "mean_Lift@10_base"}.issubset(df.columns) and "delta_mean_Lift@10" not in df.columns:
        df["delta_mean_Lift@10"] = pd.to_numeric(df["mean_Lift@10"], errors="coerce") - pd.to_numeric(df["mean_Lift@10_base"], errors="coerce")
    group_cols = [c for c in ["perm_type", "k_star"] if c in df.columns]
    value_cols = [c for c in ["delta_mAP", "delta_mean_Lift@10"] if c in df.columns]
    if not group_cols or not value_cols:
        return pd.DataFrame()
    return df.groupby(group_cols, as_index=False)[value_cols].median(numeric_only=True)


def _metric_suffix(metric: str) -> str:
    return "ap" if metric == "AP" else "lift10"


def _pairwise_forest_frame(run_root: Path, metric: str, k: int) -> pd.DataFrame:
    """Pairwise significance rows for one metric and one k (baselines only)."""
    pairwise = _read_csv(run_root / "stat_tests" / "val2_pairwise_significance.csv")
    if pairwise.empty or "metric" not in pairwise.columns or "k" not in pairwise.columns:
        return pd.DataFrame()
    ks = pd.to_numeric(pairwise["k"], errors="coerce")
    sub = pairwise[(pairwise["metric"].astype(str) == metric) & (ks == int(k))].copy()
    if "baseline" in sub.columns:
        sub = sub[
            ~sub["baseline"].astype(str).isin(PRESENTATION_EXCLUDED_METHODS)
        ].copy()
    if sub.empty:
        return pd.DataFrame()
    order = {b: i for i, b in enumerate(METHOD_ORDER)}
    sub["_ord"] = sub["baseline"].astype(str).map(lambda b: order.get(b, len(order)))
    sub = sub.sort_values(["_ord", "baseline"]).reset_index(drop=True)
    sub["label"] = sub["baseline"].astype(str).map(lambda b: f"CALIPER - {METHOD_LABEL.get(b, b)}")
    return sub


def _best_baseline_forest_frame(run_root: Path, metric: str) -> pd.DataFrame:
    """Best-baseline significance rows for one metric (k=PRIMARY_K only)."""
    best = _read_csv(run_root / "stat_tests" / "val2_best_baseline_significance.csv")
    if best.empty or "metric" not in best.columns or "k" not in best.columns:
        return pd.DataFrame()
    sub = best[best["metric"].astype(str) == metric].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["k"] = pd.to_numeric(sub["k"], errors="coerce")
    sub = sub.sort_values("k").reset_index(drop=True)
    sub["label"] = sub["k"].map(lambda k: f"k={int(k)}")
    return sub


def _forest_center_err(df: pd.DataFrame):
    center = pd.to_numeric(df["median_delta"], errors="coerce").to_numpy(dtype=float)
    los = pd.to_numeric(df.get("median_ci_low"), errors="coerce").to_numpy(dtype=float)
    his = pd.to_numeric(df.get("median_ci_high"), errors="coerce").to_numpy(dtype=float)
    err = np.clip(np.vstack([center - los, his - center]), 0.0, None)
    return center, err


def _sig_star_note(df: pd.DataFrame) -> str:
    """Star legend for the significance figures (k=PRIMARY_K paired t-test + Holm)."""
    return "* Holm p<0.05"


def _draw_effect_size_forest(ax, df: pd.DataFrame, *, metric: str, orientation: str) -> None:
    df = df.reset_index(drop=True)
    n = len(df)
    center, err = _forest_center_err(df)
    sig = df["significant"].astype(bool).to_numpy() if "significant" in df.columns else np.zeros(n, bool)
    positions = np.arange(n)
    labels = df["label"].astype(str).tolist()
    if orientation == "horizontal":
        labels = [f"{label}  *" if s else label for label, s in zip(labels, sig)]
    value_label = f"median Δ{metric} (95% CI)"
    color = METHOD_COLOR["proposed"]
    lo = center - err[0]
    hi = center + err[1]
    finite = np.concatenate([lo[np.isfinite(lo)], hi[np.isfinite(hi)], [0.0]])
    span = float(np.nanmax(finite) - np.nanmin(finite)) or 1.0
    pad = 0.04 * span

    ebar = dict(
        fmt="o", markersize=3.0, markeredgecolor="white",
        markeredgewidth=PAPER_MARKER_EDGE_WIDTH, elinewidth=PAPER_ERROR_LINEWIDTH,
        capsize=PAPER_ERROR_CAPSIZE, capthick=PAPER_ERROR_LINEWIDTH,
        color=color, ecolor=color, zorder=3,
    )
    if orientation == "vertical":  # value on y-axis, category labels on x-axis
        ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.7, zorder=1)
        ax.errorbar(positions, center, yerr=err, **ebar)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylabel(value_label)
        for x, h, s in zip(positions, hi, sig):
            if s and np.isfinite(h):
                ax.annotate("*", (x, h + pad), ha="center", va="bottom",
                            fontsize=9, color="black", annotation_clip=False)
        if sig.any():  # headroom so stars clear the top spine / note
            y0, y1 = ax.get_ylim()
            ax.set_ylim(y0, y1 + 0.10 * (y1 - y0))
    else:  # value on x-axis, category labels on y-axis
        from matplotlib.ticker import MaxNLocator

        ax.axvline(0.0, color="#888888", linestyle="--", linewidth=0.7, zorder=1)
        ax.errorbar(center, positions, xerr=err, **ebar)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_ylim(-0.5, n - 0.5)
        ax.invert_yaxis()  # first row on top (forest-plot convention)
        ax.set_xlabel(value_label)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.label.set_size(EFFECT_FOREST_HORIZONTAL_LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=EFFECT_FOREST_HORIZONTAL_TICK_SIZE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _save_effect_size_forest(
    df: pd.DataFrame,
    out_png: Path,
    *,
    metric: str,
    title: str,
    orientation: str = "horizontal",
) -> bool:
    if df.empty or "median_delta" not in df.columns:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_paper_rcparams()

    df = df.reset_index(drop=True)
    n = len(df)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if orientation == "vertical":
        figsize = (max(3.0, 0.55 * n + 1.6), 2.6)
    else:
        figsize = (
            EFFECT_FOREST_HORIZONTAL_FIGSIZE[0],
            max(EFFECT_FOREST_HORIZONTAL_FIGSIZE[1], 0.28 * n + 1.0),
        )
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _draw_effect_size_forest(ax, df, metric=metric, orientation=orientation)
    has_sig = "significant" in df.columns and bool(df["significant"].astype(bool).any())
    if has_sig:
        ax.annotate(_sig_star_note(df), xy=(0.0, 1.02), xycoords="axes fraction",
                    ha="left", va="bottom", fontsize=EFFECT_FOREST_STAR_NOTE_SIZE, color="#333333")
    _save_figure(fig, out_png)
    plt.close(fig)
    return True


def _save_best_baseline_effect_size_panel(run_root: Path, out_png: Path) -> bool:
    frames = {
        "AP": _best_baseline_forest_frame(run_root, "AP"),
        "Lift@10": _best_baseline_forest_frame(run_root, "Lift@10"),
    }
    if any(df.empty or "median_delta" not in df.columns for df in frames.values()):
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_paper_rcparams()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=PAPER_FIGSIZE, constrained_layout=True)
    any_sig = False
    for ax, metric in zip(axes, ["AP", "Lift@10"]):
        frame = frames[metric]
        _draw_effect_size_forest(ax, frame, metric=metric, orientation="vertical")
        any_sig = any_sig or ("significant" in frame.columns and bool(frame["significant"].astype(bool).any()))
    if any_sig:
        axes[0].annotate(_sig_star_note(frames["AP"]), xy=(0.0, 1.02), xycoords="axes fraction",
                         ha="left", va="bottom", fontsize=6, color="#333333")
    _save_figure(fig, out_png)
    plt.close(fig)
    return True


def _save_win_rate_bar(df: pd.DataFrame, out_png: Path, *, metric: str, title: str) -> bool:
    if df.empty or "win_rate" not in df.columns:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_paper_rcparams()

    df = df.reset_index(drop=True)
    vals = pd.to_numeric(df["win_rate"], errors="coerce").to_numpy(dtype=float)
    ys = np.arange(len(df))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig_h = max(2.2, 0.5 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(5.4, fig_h), constrained_layout=True)
    ax.barh(ys, vals, color="#0072B2")
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(df["label"].astype(str))
    ax.set_ylim(-0.5, len(df) - 0.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(f"win-rate (Δ{metric}>0)")
    _save_figure(fig, out_png)
    plt.close(fig)
    return True


def _save_delta_box_for_k(run_root: Path, out_png: Path, *, metric: str, k: int) -> bool:
    delta = _read_csv(run_root / "stat_tests" / "val2_pairwise_disease_level_delta.csv")
    delta_col = f"delta_{metric}"
    if delta.empty or delta_col not in delta.columns or "baseline" not in delta.columns or "k" not in delta.columns:
        return False
    df = delta.copy()
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df = df[df["k"] == int(k)]
    df = df[~df["baseline"].astype(str).isin(PRESENTATION_EXCLUDED_METHODS)]
    if df.empty:
        return False
    baselines = [b for b in METHOD_ORDER if b in set(df["baseline"].astype(str))]
    baselines.extend(sorted(set(df["baseline"].astype(str)) - set(baselines)))
    data = [
        pd.to_numeric(df.loc[df["baseline"].astype(str) == b, delta_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        for b in baselines
    ]
    if not any(len(d) for d in data):
        return False

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_panel_rcparams()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIXED_K_ABLATION_FIGSIZE, constrained_layout=True)
    positions = np.arange(1, len(baselines) + 1)
    bp = ax.boxplot(
        [d for d in data],
        positions=positions,
        widths=0.5,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#111111", "linewidth": 1.2},
        boxprops={"color": "#333333"},
        whiskerprops={"color": "#333333"},
        capprops={"color": "#333333"},
        zorder=3,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(DELTA_BOX_COLOR)
        patch.set_alpha(0.35)
        patch.set_edgecolor("#333333")
    for i, d in enumerate(data, start=1):
        if len(d):
            jitter = (np.random.default_rng(0).random(len(d)) - 0.5) * 0.22
            ax.scatter(
                np.full(len(d), i) + jitter,
                d,
                s=5,
                alpha=0.3,
                color="#333333",
                edgecolors="none",
                zorder=4,
            )
    ax.axhline(0.0, color="#999999", linestyle=":", linewidth=0.7, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABEL.get(b, b) for b in baselines], rotation=20, ha="right")
    ax.set_ylabel(f"Δ{metric} (CALIPER - baseline)")
    ax.grid(axis="y", alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)
    _save_figure(fig, out_png)
    plt.close(fig)
    return True


def make_all_figures(
    run_root: str | Path,
    *,
    targetnexus_root: str | Path | None = None,
) -> Dict[str, str]:
    """Create Val2 figures from existing CSV outputs only.

    This function intentionally does not call any scoring, ablation,
    or permutation routine.
    """
    run_root = Path(run_root)
    figure_dir = run_root / "report" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}
    raw_metrics = load_ablation_metrics(run_root)

    metrics = load_ablation_summary(run_root)
    out = figure_dir / "mean_ap_by_distance_k.png"
    if _save_lineplot(
        metrics,
        x="k",
        y="mAP",
        yerr="mAP_SEM",
        group="method",
        out_path=out,
        title="Mean AP by distance k",
        ylabel="Mean AP",
        show_errorbars=False,
    ):
        outputs.update(_relative_existing_outputs(run_root, "mean_ap_by_distance_k", out))

    out = figure_dir / "mean_lift10_by_distance_k.png"
    if _save_lineplot(
        metrics,
        x="k",
        y="mean_Lift@10",
        yerr="mean_Lift@10_SEM",
        group="method",
        out_path=out,
        title="Mean Lift@10 by distance k",
        ylabel="Mean Lift@10",
        show_errorbars=False,
    ):
        outputs.update(_relative_existing_outputs(run_root, "mean_lift10_by_distance_k", out))

    k_valid = resolve_k_valid(run_root)
    for args in [
        {
            "out_png": figure_dir / "candidate_k_method_comparison_ap.png",
            "mean_col": "mAP",
            "sem_col": "mAP_SEM",
            "ylabel": "Mean AP",
            "title": "Mean AP by candidate k and method",
        },
        {
            "out_png": figure_dir / "candidate_k_method_comparison_lift10.png",
            "mean_col": "mean_Lift@10",
            "sem_col": "mean_Lift@10_SEM",
            "ylabel": "Mean Lift@10",
            "title": "Mean Lift@10 by candidate k and method",
        },
    ]:
        for key, value in _save_k_valid_metric_figure(
            load_ablation_summary(run_root),
            k_valid=k_valid,
            **args,
        ).items():
            outputs[key] = str(Path(value).relative_to(run_root))

    # 主解析は k=PRIMARY_K 固定で paper-facing internal ablation を描画する。
    for args in [
        {
            "out_png": figure_dir / f"k{PRIMARY_K}_internal_ablation_ap.png",
            "mean_col": "mAP",
            "sem_col": "mAP_SEM",
            "ylabel": "Mean AP",
        },
        {
            "out_png": figure_dir / f"k{PRIMARY_K}_internal_ablation_lift10.png",
            "mean_col": "mean_Lift@10",
            "sem_col": "mean_Lift@10_SEM",
            "ylabel": "Mean Lift@10",
        },
    ]:
        for key, value in _save_fixed_k_internal_ablation_figure(
            metrics,
            k=PRIMARY_K,
            **args,
        ).items():
            outputs[key] = str(Path(value).relative_to(run_root))

    support_robustness = build_support_robustness_table(
        run_root,
        k=PRIMARY_K,
        method="proposed",
    )
    if not support_robustness.empty:
        support_table = (
            run_root
            / "report"
            / "tables"
            / f"support_robustness_k{PRIMARY_K}.tsv"
        )
        support_table.parent.mkdir(parents=True, exist_ok=True)
        support_robustness.to_csv(support_table, sep="\t", index=False)
        outputs.update(
            _relative_existing_file_output(
                run_root,
                f"support_robustness_k{PRIMARY_K}_table",
                support_table,
            )
        )
        out = figure_dir / f"support_robustness_k{PRIMARY_K}.png"
        for key, value in _save_support_robustness_figure(
            support_robustness,
            out_png=out,
        ).items():
            outputs[key] = str(Path(value).relative_to(run_root))

    outputs.update(make_support_score_distribution(run_root, k=PRIMARY_K))
    if targetnexus_root is not None:
        outputs.update(
            make_label_confidence_score_distribution(
                run_root,
                targetnexus_root=targetnexus_root,
                k=PRIMARY_K,
            )
        )

    if not raw_metrics.empty and "method" in raw_metrics.columns:
        available_methods = (
            set(raw_metrics["method"].astype(str)) - PRESENTATION_EXCLUDED_METHODS
        )
        methods = [method for method in METHOD_ORDER if method in available_methods]
        methods.extend(sorted(available_methods - set(methods)))
        for method in methods:
            out = figure_dir / f"ap_vs_positive_gene_count_{method}.png"
            if _save_ap_positive_gene_scatter_for_method(raw_metrics, out_path=out, k_valid=k_valid, method=method):
                outputs.update(_relative_existing_outputs(run_root, f"ap_vs_positive_gene_count_{method}", out))

    perm = _permutation_summary_for_plot(run_root)
    if not perm.empty and "perm_type" in perm.columns:
        label_col = "perm_type"
        if "k_star" in perm.columns:
            perm = perm.copy()
            perm["perm_type_k"] = perm["perm_type"].astype(str) + " / k*=" + perm["k_star"].astype(str)
            label_col = "perm_type_k"
        out = figure_dir / "permutation_delta_map.png"
        if _save_barplot(perm, x=label_col, y="delta_mAP", out_path=out, title="Permutation delta mAP"):
            outputs.update(_relative_existing_outputs(run_root, "permutation_delta_map", out))
        out = figure_dir / "permutation_delta_lift10.png"
        if _save_barplot(perm, x=label_col, y="delta_mean_Lift@10", out_path=out, title="Permutation delta mean Lift@10"):
            outputs.update(_relative_existing_outputs(run_root, "permutation_delta_lift10", out))

    table_dir = run_root / "report" / "tables"
    for metric, selection, ascending in [
        ("AP", "top10", False),
        ("Lift@10", "top10", False),
        ("AP", "bottom10", True),
        ("Lift@10", "bottom10", True),
    ]:
        suffix = "ap" if metric == "AP" else "lift10"
        stem = f"k4_caliper_vs_size_matched_no_kg_standardized_delta_{suffix}_{selection}"
        table = _caliper_vs_sizematched_nokg_standardized_delta(
            raw_metrics,
            metric=metric,
            k=4,
            n=10,
            ascending=ascending,
        )
        out = table_dir / f"{stem}.csv"
        if _save_selected_disease_score_table(table, out_csv=out):
            outputs.update(_relative_existing_file_output(run_root, stem, out))
            outputs.update(_relative_existing_file_output(run_root, f"{stem}_tsv", out.with_suffix(".tsv")))

    # Significance figures from stat_tests/ (disease-level paired statistics).
    # Testing is fixed to the primary distance k=PRIMARY_K, so only that k gets
    # effect-size / win-rate / Δ-distribution panels.
    for metric in ["AP", "Lift@10"]:
        suffix = _metric_suffix(metric)
        ki = PRIMARY_K
        out = figure_dir / f"val2_effect_size_forest_{suffix}_k{ki}.png"
        if _save_effect_size_forest(
            _pairwise_forest_frame(run_root, metric, ki), out,
            metric=metric, title=f"Effect size: Δ{metric} (k={ki})",
        ):
            outputs.update(_relative_existing_outputs(run_root, f"val2_effect_size_forest_{suffix}_k{ki}", out))
        out = figure_dir / f"val2_win_rate_{suffix}_k{ki}.png"
        if _save_win_rate_bar(
            _pairwise_forest_frame(run_root, metric, ki), out,
            metric=metric, title=f"Win-rate across diseases: Δ{metric} (k={ki})",
        ):
            outputs.update(_relative_existing_outputs(run_root, f"val2_win_rate_{suffix}_k{ki}", out))
        out = figure_dir / f"val2_delta_dist_{suffix}_k{ki}.png"
        if _save_delta_box_for_k(run_root, out, metric=metric, k=ki):
            outputs.update(_relative_existing_outputs(run_root, f"val2_delta_dist_{suffix}_k{ki}", out))

        # best baseline は metric ごとに 1 枚。
        out = figure_dir / f"val2_effect_size_forest_{suffix}_bestbaseline.png"
        if _save_effect_size_forest(
            _best_baseline_forest_frame(run_root, metric), out,
            metric=metric, title=f"Effect size vs best baseline: Δ{metric}", orientation="vertical",
        ):
            outputs.update(_relative_existing_outputs(run_root, f"val2_effect_size_forest_{suffix}_bestbaseline", out))
        out = figure_dir / f"val2_win_rate_{suffix}_bestbaseline.png"
        if _save_win_rate_bar(
            _best_baseline_forest_frame(run_root, metric), out,
            metric=metric, title=f"Win-rate vs best baseline: Δ{metric}",
        ):
            outputs.update(_relative_existing_outputs(run_root, f"val2_win_rate_{suffix}_bestbaseline", out))

    out = figure_dir / "val2_effect_size_forest_bestbaseline.png"
    if _save_best_baseline_effect_size_panel(run_root, out):
        outputs.update(_relative_existing_outputs(run_root, "val2_effect_size_forest_bestbaseline", out))

    return outputs
