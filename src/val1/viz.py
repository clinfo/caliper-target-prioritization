from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import math

import numpy as np
import pandas as pd


PAPER_SINGLE_FIGSIZE = (3.5, 2.4)
PAPER_MULTI_PANEL_FIGSIZE = (7.2, 2.4)
PAPER_DISTRIBUTION_FIGSIZE = (3.5, 2.8)
PAPER_RULE_AGREEMENT_FIGSIZE = (3.5, 2.6)
PAPER_LINEWIDTH = 1.2
PAPER_MARKERSIZE = 3.0
PAPER_TICK_LENGTH = 2.5
PAPER_TICK_WIDTH = 0.6
PAPER_TITLE_PAD = 4
PRIMARY_CURVE_COLOR = "#1A4E79"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.35,
            "lines.linewidth": PAPER_LINEWIDTH,
            "lines.markersize": PAPER_MARKERSIZE,
            "legend.handlelength": 1.4,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )
    return plt


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def load_val1_outputs(run_root: str | Path) -> Dict[str, Any]:
    run_root = Path(run_root)
    out: Dict[str, Any] = {"run_root": run_root}
    paths = {
        "metadata": run_root / "metadata.json",
        "k_selection": run_root / "selection" / "k_selection.json",
        "k_ranking": run_root / "selection" / "k_ranking.tsv",
        "k_metrics": run_root / "metrics" / "k_metrics.tsv",
        "coverage_summary": run_root / "metrics" / "coverage_summary.tsv",
        "ambiguity_summary": run_root / "metrics" / "ambiguity_summary.tsv",
        "target_depths": run_root / "kmax" / "target_hop_depths.tsv",
        "td_k_metrics": run_root / "metrics" / "td_k_metrics.tsv",
        "td_selection": run_root / "selection" / "td_k_selection.tsv",
        "rule_agreement": run_root / "selection_sensitivity" / "rule_agreement.tsv",
        "rule_k_metrics": run_root / "selection_sensitivity" / "rule_k_metrics.tsv",
        "resampling_frequency": run_root / "selection_sensitivity" / "resampling_frequency.tsv",
        "path_benchmark_summary": run_root / "runtime" / "per_source_summary.tsv",
        "path_benchmark_per_source": run_root / "runtime" / "per_source.tsv",
    }
    for key, path in paths.items():
        if path.exists():
            if path.suffix == ".json":
                out[key] = _read_json(path)
            else:
                out[key] = _read_tsv(path)
    return out


def _k_valid_intervals(k_valid) -> list[tuple[float, float]]:
    ks = sorted({int(k) for k in (k_valid or [])})
    if not ks:
        return []
    intervals: list[tuple[float, float]] = []
    start = prev = ks[0]
    for k in ks[1:]:
        if k == prev + 1:
            prev = k
            continue
        intervals.append((start - 0.5, prev + 0.5))
        start = prev = k
    intervals.append((start - 0.5, prev + 0.5))
    return intervals


def _finish_axes(
    ax,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    legend: bool = True,
    legend_loc: str = "best",
) -> None:
    from matplotlib.ticker import MaxNLocator
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.22)
    ax.grid(True, axis="x", alpha=0.10)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=PAPER_TICK_LENGTH, width=PAPER_TICK_WIDTH)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if legend:
        ax.legend(frameon=True, framealpha=0.92, edgecolor="#CCCCCC", loc=legend_loc)


def _save_figure(fig, output_png: str | Path) -> Path:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_png.with_suffix(".pdf"))
    fig.savefig(output_png.with_suffix(".svg"))
    return output_png


def _mark_selection(ax, selection: Dict[str, Any], *, include_labels: bool = True) -> None:
    k_star = selection.get("k_star")
    k_valid = selection.get("K_valid") or []
    for i, (left, right) in enumerate(_k_valid_intervals(k_valid)):
        ax.axvspan(
            left,
            right,
            color="#A6CEE3",
            alpha=0.26,
            linewidth=0,
            label="candidate k" if include_labels and i == 0 else None,
        )
    if k_star is not None:
        try:
            ax.axvline(
                float(k_star),
                color="#0072B2",
                linestyle="--",
                linewidth=PAPER_LINEWIDTH,
                label="selected $k^*$" if include_labels else None,
            )
        except Exception:
            pass


def _iqr_by_k(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df.empty or value_col not in df.columns or "k" not in df.columns:
        return pd.DataFrame(columns=["k", "q25", "q75"])
    work = df[["k", value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])
    if work.empty:
        return pd.DataFrame(columns=["k", "q25", "q75"])
    out = work.groupby("k", as_index=False).agg(
        q25=(value_col, lambda x: x.quantile(0.25)),
        q75=(value_col, lambda x: x.quantile(0.75)),
    )
    return out.sort_values("k").reset_index(drop=True)


def _target_ambiguity_iqr_by_k(td_k_metrics: pd.DataFrame) -> pd.DataFrame:
    if td_k_metrics.empty or not {"target_gene", "k", "B_td"} <= set(td_k_metrics.columns):
        return pd.DataFrame(columns=["k", "q25", "q75"])
    work = td_k_metrics[["target_gene", "k", "B_td"]].copy()
    work["B_td"] = pd.to_numeric(work["B_td"], errors="coerce")
    work = work.dropna(subset=["B_td"]).drop_duplicates(["target_gene", "k"])
    return _iqr_by_k(work, "B_td")


def _add_iqr_band(ax, iqr_df: pd.DataFrame, *, label: str = "IQR") -> None:
    if iqr_df.empty:
        return
    x = pd.to_numeric(iqr_df["k"], errors="coerce").to_numpy(dtype=float)
    q25 = pd.to_numeric(iqr_df["q25"], errors="coerce").to_numpy(dtype=float)
    q75 = pd.to_numeric(iqr_df["q75"], errors="coerce").to_numpy(dtype=float)
    mask = pd.notna(x) & pd.notna(q25) & pd.notna(q75)
    if not mask.any():
        return
    ax.fill_between(x[mask], q25[mask], q75[mask], color="#C6D9EC", alpha=0.55, linewidth=0, label=label)


def _add_y_axis_wave_break(ax) -> None:
    points = 80
    wave_x = [((i / (points - 1)) - 0.5) * 0.036 for i in range(points)]
    phase = [2.0 * math.pi * i / (points - 1) for i in range(points)]
    for y_offset in (0.040, 0.064):
        wave_y = [y_offset + 0.006 * math.sin(p) for p in phase]
        ax.plot(
            wave_x,
            wave_y,
            transform=ax.transAxes,
            color="black",
            linewidth=PAPER_LINEWIDTH,
            clip_on=False,
            solid_capstyle="round",
            zorder=5,
        )


def plot_coverage_vs_k(
    k_metrics: pd.DataFrame,
    k_selection: Dict[str, Any],
    output_png: str | Path,
    td_k_metrics: pd.DataFrame | None = None,
) -> Path:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    fig, ax = plt.subplots(figsize=PAPER_SINGLE_FIGSIZE, constrained_layout=True)
    if td_k_metrics is not None:
        _add_iqr_band(ax, _iqr_by_k(td_k_metrics, "C_td"), label="IQR")
    ax.plot(k_metrics["k"], k_metrics["C"], marker="o", color="#1A4E79", zorder=3, label="Coverage")
    _finish_axes(
        ax,
        title="Coverage by distance k",
        xlabel="distance k",
        ylabel="Coverage",
        legend_loc="lower right",
    )
    _save_figure(fig, output_png)
    plt.close(fig)
    return output_png


def plot_ambiguity_vs_k(
    k_metrics: pd.DataFrame,
    k_selection: Dict[str, Any],
    output_png: str | Path,
    td_k_metrics: pd.DataFrame | None = None,
) -> Path:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    fig, ax = plt.subplots(figsize=PAPER_SINGLE_FIGSIZE, constrained_layout=True)
    if td_k_metrics is not None:
        _add_iqr_band(ax, _target_ambiguity_iqr_by_k(td_k_metrics), label="IQR")
    ax.plot(k_metrics["k"], k_metrics["B"], marker="o", color="#1A4E79", zorder=3, label="Sign ambiguity")
    _finish_axes(
        ax,
        title="Sign ambiguity by distance k",
        xlabel="distance k",
        ylabel="Sign ambiguity",
    )
    _save_figure(fig, output_png)
    plt.close(fig)
    return output_png


def plot_unambiguous_fraction_vs_k(
    k_metrics: pd.DataFrame,
    k_selection: Dict[str, Any],
    output_png: str | Path,
    td_k_metrics: pd.DataFrame | None = None,
) -> Path:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    fig, ax = plt.subplots(figsize=PAPER_SINGLE_FIGSIZE, constrained_layout=True)
    if td_k_metrics is not None:
        amb_iqr = _target_ambiguity_iqr_by_k(td_k_metrics)
        if not amb_iqr.empty:
            a_iqr = amb_iqr.copy()
            a_iqr["q25"] = 1.0 - amb_iqr["q75"]
            a_iqr["q75"] = 1.0 - amb_iqr["q25"]
            _add_iqr_band(ax, a_iqr, label="IQR")
    ax.plot(k_metrics["k"], k_metrics["A"], marker="o", color="#1A4E79", zorder=3, label="Sign unambiguity")
    _finish_axes(
        ax,
        title="Sign unambiguity by distance k",
        xlabel="distance k",
        ylabel="Sign unambiguity",
    )
    _add_y_axis_wave_break(ax)
    _save_figure(fig, output_png)
    plt.close(fig)
    return output_png


def plot_harmonic_components_vs_k(k_metrics: pd.DataFrame, k_selection: Dict[str, Any], output_png: str | Path) -> Path:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    fig, ax = plt.subplots(figsize=PAPER_SINGLE_FIGSIZE, constrained_layout=True)
    _mark_selection(ax, k_selection)
    ax.plot(k_metrics["k"], k_metrics["H"], marker="o", color="#1A4E79", zorder=3, label="Harmonic score")
    _finish_axes(
        ax,
        title="Harmonic score by distance k",
        xlabel="distance k",
        ylabel="Harmonic score",
    )
    _save_figure(fig, output_png)
    plt.close(fig)
    return output_png


def _plot_metric_panel(
    ax,
    k_metrics: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    ylabel: str,
    color: str,
    k_selection: Dict[str, Any],
) -> None:
    from matplotlib.ticker import MaxNLocator

    k_star = k_selection.get("k_star")
    if k_star is not None:
        try:
            ax.axvline(float(k_star), color="#0072B2", linestyle="--", linewidth=PAPER_LINEWIDTH, zorder=2)
        except Exception:
            pass
    ax.plot(
        pd.to_numeric(k_metrics["k"], errors="coerce"),
        pd.to_numeric(k_metrics[value_col], errors="coerce"),
        marker="o",
        markersize=PAPER_MARKERSIZE,
        linewidth=PAPER_LINEWIDTH,
        color=color,
        markerfacecolor=color,
        markeredgecolor="black",
        markeredgewidth=0.35,
        zorder=3,
    )
    ax.set_xlabel("distance k")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.22)
    ax.grid(True, axis="x", alpha=0.10)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=PAPER_TICK_LENGTH, width=PAPER_TICK_WIDTH)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_distance_k_summary(
    k_metrics: pd.DataFrame,
    k_selection: Dict[str, Any],
    output_png: str | Path,
) -> Path:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=PAPER_MULTI_PANEL_FIGSIZE, constrained_layout=True, sharex=True)
    panels = [
        ("H", "Harmonic score by distance k", "Harmonic score", "#1A4E79"),
        ("C", "Coverage by distance k", "Coverage", "#2C7A4B"),
        ("A", "Sign unambiguity by distance k", "Sign unambiguity", "#9B4D3A"),
    ]
    for ax, (value_col, title, ylabel, color) in zip(axes, panels):
        _plot_metric_panel(
            ax,
            k_metrics,
            value_col=value_col,
            title=title,
            ylabel=ylabel,
            color=color,
            k_selection=k_selection,
        )
    _save_figure(fig, output_png)
    plt.close(fig)
    return output_png


def plot_target_depth_distribution(target_depths: pd.DataFrame, output_png: str | Path) -> Path:
    values = (
        pd.to_numeric(target_depths.get("D_t", pd.Series(dtype=float)), errors="coerce")
        if not target_depths.empty
        else pd.Series(dtype=float)
    )
    return _plot_discrete_distribution(
        values,
        title="Reachable Depth by Target",
        xlabel="Maximum reachable distance k",
        ylabel="Number of targets",
        output_png=output_png,
    )


def _explode_k_valid_td(td_selection: pd.DataFrame) -> pd.Series:
    if td_selection.empty or "K_valid_td" not in td_selection.columns:
        return pd.Series(dtype=float)
    values: list[int] = []
    for raw in td_selection["K_valid_td"].fillna("").astype(str):
        if raw.strip() == "":
            continue
        for item in raw.split("|"):
            item = item.strip()
            if item:
                values.append(int(float(item)))
    return pd.Series(values, dtype=float)


def _plot_discrete_distribution(
    values: pd.Series,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    output_png: str | Path,
) -> Path:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    import seaborn as sns

    fig, (ax_hist, ax_box) = plt.subplots(
        2,
        1,
        figsize=PAPER_DISTRIBUTION_FIGSIZE,
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
        constrained_layout=True,
    )
    s = pd.to_numeric(values, errors="coerce").dropna()
    if len(s):
        lo = int(s.min())
        hi = int(s.max())
        bins = [x - 0.5 for x in range(lo, hi + 2)]
        sns.histplot(
            x=s.astype(int),
            bins=bins,
            color="#1A4E79",
            edgecolor="#FFFFFF",
            linewidth=0.35,
            alpha=0.75,
            ax=ax_hist,
        )
        sns.boxplot(
            x=s.astype(float),
            ax=ax_box,
            color="#DDEAF6",
            width=0.48,
            linewidth=0.6,
            fliersize=1.5,
            boxprops={"edgecolor": "#1A4E79"},
            whiskerprops={"color": "#1A4E79", "linewidth": 0.6},
            capprops={"color": "#1A4E79", "linewidth": 0.6},
            medianprops={"color": "#C0392B", "linewidth": 1.0},
            flierprops={
                "marker": "o",
                "markerfacecolor": "#1A4E79",
                "markeredgecolor": "#1A4E79",
                "markersize": 1.5,
                "alpha": 0.40,
            },
        )
        ax_hist.set_xticks(list(range(lo, hi + 1)))
        ax_hist.text(
            0.98,
            0.95,
            f"n = {len(s):,}",
            transform=ax_hist.transAxes,
            ha="right",
            va="top",
        )
    ax_hist.set_ylabel(ylabel)
    ax_hist.grid(True, axis="y", alpha=0.22)
    ax_hist.grid(True, axis="x", alpha=0.10)
    ax_hist.set_axisbelow(True)
    ax_hist.tick_params(axis="both", length=PAPER_TICK_LENGTH, width=PAPER_TICK_WIDTH)
    ax_box.set_xlabel(xlabel)
    ax_box.set_ylabel("")
    ax_box.set_yticks([])
    ax_box.grid(True, axis="x", alpha=0.22)
    ax_box.set_axisbelow(True)
    ax_box.tick_params(axis="x", length=PAPER_TICK_LENGTH, width=PAPER_TICK_WIDTH)
    _save_figure(fig, output_png)
    plt.close(fig)
    return output_png


def plot_td_k_star_distribution(td_selection: pd.DataFrame, output_png: str | Path) -> Path:
    values = (
        pd.to_numeric(td_selection.get("k_star_td", pd.Series(dtype=float)), errors="coerce")
        if not td_selection.empty
        else pd.Series(dtype=float)
    )
    return _plot_discrete_distribution(
        values,
        title="Selected $k^*$ by Target-Disease Pair",
        xlabel="selected $k^*$",
        ylabel="Number of target-disease pairs",
        output_png=output_png,
    )


def plot_td_k_valid_distribution(td_selection: pd.DataFrame, output_png: str | Path) -> Path:
    values = _explode_k_valid_td(td_selection)
    return _plot_discrete_distribution(
        values,
        title="Candidate k by Target-Disease Pair",
        xlabel="candidate k",
        ylabel="Number of candidate-k inclusions",
        output_png=output_png,
    )


def _set_k_axis_limits(ax, k_metrics: pd.DataFrame, *, pad: float = 0.6) -> None:
    """Keep the k axis on the full candidate grid, not just the plotted values."""
    if k_metrics is None or k_metrics.empty or "k" not in k_metrics.columns:
        return
    ks = pd.to_numeric(k_metrics["k"], errors="coerce").dropna()
    if not len(ks):
        return
    ax.set_xlim(float(ks.min()) - pad, float(ks.max()) + pad)


def _rule_display_label(rule: str) -> str:
    labels = {
        "harmonic_mean_primary": "Harmonic score (primary)",
        "geometric_mean": "Geometric score",
        "pareto_knee": "Pareto knee",
        "coverage_threshold": "Coverage threshold",
        "ambiguity_ceiling": "Ambiguity ceiling",
        "leave_one_target_out_modal": "LOTO (modal)",
        "leave_one_disease_out_modal": "LODO (modal)",
        "bootstrap_targets_modal": "Bootstrap (modal)",
    }
    return labels.get(str(rule), str(rule))


def _split_k_valid(raw: Any) -> list[int]:
    if raw is None:
        return []
    text = str(raw).strip()
    if text in {"", "nan", "None"}:
        return []
    out: list[int] = []
    for item in text.split("|"):
        item = item.strip()
        if item:
            out.append(int(float(item)))
    return out


def plot_rule_agreement(
    rule_agreement: pd.DataFrame,
    k_metrics: pd.DataFrame,
    k_selection: Dict[str, Any],
    output_png: str | Path,
) -> Path:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    fig, ax = plt.subplots(figsize=PAPER_RULE_AGREEMENT_FIGSIZE, constrained_layout=True)

    rules = list(rule_agreement["rule"].astype(str))
    y_positions = list(range(len(rules) - 1, -1, -1))
    k_star_primary = k_selection.get("k_star")
    if k_star_primary is not None:
        ax.axvline(
            float(k_star_primary),
            color="#0072B2",
            linestyle="--",
            linewidth=PAPER_LINEWIDTH,
            zorder=1,
            label="primary $k^*$",
        )

    candidate_drawn = False
    selected_drawn = False
    for y, (_, row) in zip(y_positions, rule_agreement.iterrows()):
        for k in _split_k_valid(row.get("K_valid")):
            ax.plot(
                k,
                y,
                marker="o",
                markersize=PAPER_MARKERSIZE + 1.0,
                markerfacecolor="white",
                markeredgecolor="#1A4E79",
                markeredgewidth=0.7,
                linestyle="none",
                zorder=2,
                label="candidate k" if not candidate_drawn else None,
            )
            candidate_drawn = True
        k_star = pd.to_numeric(row.get("k_star"), errors="coerce")
        if pd.notna(k_star):
            ax.plot(
                float(k_star),
                y,
                marker="o",
                markersize=PAPER_MARKERSIZE + 1.0,
                markerfacecolor="#1A4E79",
                markeredgecolor="black",
                markeredgewidth=0.35,
                linestyle="none",
                zorder=3,
                label="selected $k^*$" if not selected_drawn else None,
            )
            selected_drawn = True

    ax.set_yticks(y_positions)
    ax.set_yticklabels([_rule_display_label(r) for r in rules])
    ax.set_ylim(-0.7, len(rules) - 0.3)
    _set_k_axis_limits(ax, k_metrics)
    _finish_axes(
        ax,
        title="Selection-rule agreement",
        xlabel="distance k",
        ylabel="",
        legend_loc="lower right",
    )
    _save_figure(fig, output_png)
    plt.close(fig)
    return output_png


def plot_resampling_k_star_frequency(
    resampling_frequency: pd.DataFrame,
    k_metrics: pd.DataFrame,
    k_selection: Dict[str, Any],
    output_png: str | Path,
    *,
    rule: str = "harmonic_mean",
) -> Path:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    fig, ax = plt.subplots(figsize=PAPER_SINGLE_FIGSIZE, constrained_layout=True)

    work = resampling_frequency.copy()
    if "rule" in work.columns:
        work = work[work["rule"].astype(str) == rule]
    fold_labels = {
        "leave_one_target_out": "LOTO",
        "leave_one_disease_out": "LODO",
        "bootstrap_targets": "Bootstrap",
    }
    colors = {"leave_one_target_out": "#1A4E79", "leave_one_disease_out": "#2C7A4B", "bootstrap_targets": "#9B4D3A"}
    fold_types = [f for f in ["leave_one_target_out", "leave_one_disease_out", "bootstrap_targets"] if f in set(work.get("fold_type", pd.Series(dtype=str)).astype(str))]

    k_star_primary = k_selection.get("k_star")
    if k_star_primary is not None:
        ax.axvline(float(k_star_primary), color="#0072B2", linestyle="--", linewidth=PAPER_LINEWIDTH, zorder=1, label="primary $k^*$")

    n_groups = max(len(fold_types), 1)
    width = 0.8 / n_groups
    for i, fold_type in enumerate(fold_types):
        sub = work[work["fold_type"].astype(str) == fold_type].sort_values("k")
        offset = (i - (n_groups - 1) / 2.0) * width
        ax.bar(
            pd.to_numeric(sub["k"], errors="coerce") + offset,
            pd.to_numeric(sub["frequency"], errors="coerce"),
            width=width,
            color=colors.get(fold_type, "#1A4E79"),
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
            label=fold_labels.get(fold_type, fold_type),
        )
    ax.set_ylim(0.0, 1.05)
    _set_k_axis_limits(ax, k_metrics)
    _finish_axes(
        ax,
        title="Leave-one-out selection frequency",
        xlabel="distance k",
        ylabel="Selection frequency",
    )
    _save_figure(fig, output_png)
    plt.close(fig)
    return output_png


def _mark_k_star(ax, k_selection: Dict[str, Any], *, label: str | None = "selected $k^*$") -> None:
    """Dashed selected k* only (no candidate-k band)."""
    k_star = k_selection.get("k_star")
    if k_star is None:
        return
    try:
        ax.axvline(
            float(k_star),
            color="#0072B2",
            linestyle="--",
            linewidth=PAPER_LINEWIDTH,
            zorder=2,
            label=label,
        )
    except Exception:
        pass


def _twin_legend(ax, ax_right, *, loc: str) -> None:
    handles, labels = ax.get_legend_handles_labels()
    r_handles, r_labels = ax_right.get_legend_handles_labels()
    ax.legend(
        handles + r_handles,
        labels + r_labels,
        frameon=True,
        framealpha=0.92,
        edgecolor="#CCCCCC",
        loc=loc,
    )


def _finish_twin_axes(ax, ax_right) -> None:
    from matplotlib.ticker import MaxNLocator

    ax.grid(True, axis="y", alpha=0.22)
    ax.grid(True, axis="x", alpha=0.10)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for axis in (ax, ax_right):
        axis.tick_params(axis="both", length=PAPER_TICK_LENGTH, width=PAPER_TICK_WIDTH)
        axis.spines["right"].set_visible(True)
        axis.spines["right"].set_color("#444444")
        axis.spines["right"].set_linewidth(0.7)
        axis.spines["top"].set_visible(False)


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _path_budget(path_benchmark_summary: pd.DataFrame) -> float:
    budget = _num(path_benchmark_summary, "max_paths")
    budget = budget[np.isfinite(budget)]
    return float(budget[0]) if budget.size else float("nan")


def _budget_value_label(budget: float) -> str:
    exponent = int(np.floor(np.log10(budget)))
    mantissa = budget / (10.0**exponent)
    return rf"${mantissa:.1f}\times10^{{{exponent}}}$"


def _budget_label(budget: float) -> str:
    if not np.isfinite(budget):
        return "no path budget"
    return rf"$\geq$ {_budget_value_label(budget)}"


def _exact_median_mask(path_benchmark_summary: pd.DataFrame) -> np.ndarray:
    """Distances whose median n_paths is exact (i.e. not pinned to the budget)."""
    budget = _path_budget(path_benchmark_summary)
    median = _num(path_benchmark_summary, "n_paths_median")
    if not np.isfinite(budget):
        return np.isfinite(median)
    return np.isfinite(median) & (median < budget)


def _path_growth_per_hop(path_benchmark_summary: pd.DataFrame) -> float:
    """Geometric-mean growth factor over the distances where the median is exact."""
    ratio = _num(path_benchmark_summary, "n_paths_ratio_vs_prev")
    exact = _exact_median_mask(path_benchmark_summary)
    prev_exact = np.concatenate(([False], exact[:-1])) if exact.size else exact
    usable = ratio[exact & prev_exact]
    usable = usable[np.isfinite(usable) & (usable > 0)]
    if usable.size == 0:
        return float("nan")
    return float(np.exp(np.log(usable).mean()))


def plot_path_count_vs_k(
    path_benchmark_summary: pd.DataFrame,
    k_metrics: pd.DataFrame,
    k_selection: Dict[str, Any],
    output_png: str | Path,
) -> Path:
    """Fig. 2d: candidate simple-path count by distance k (log scale).

    The figure focuses on path-count growth alone. Boxplots are avoided because
    truncated whiskers would all pin to the path-budget value.
    """
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    fig, ax = plt.subplots(figsize=PAPER_SINGLE_FIGSIZE, constrained_layout=True)

    s = path_benchmark_summary.sort_values("cutoff").reset_index(drop=True)
    x = _num(s, "cutoff")
    median = _num(s, "n_paths_median")
    q25 = _num(s, "n_paths_q25")
    q75 = _num(s, "n_paths_q75")
    budget = _path_budget(s)

    band = np.isfinite(x) & np.isfinite(q25) & np.isfinite(q75)
    if band.any():
        ax.fill_between(x[band], q25[band], q75[band], color="#C6D9EC", alpha=0.55, linewidth=0, label="IQR")
    ax.plot(x, median, marker="o", color="#1A4E79", zorder=3, label="Simple paths (median)")
    pinned = ~_exact_median_mask(s) & np.isfinite(median)
    if pinned.any():
        ax.plot(
            x[pinned],
            median[pinned],
            linestyle="none",
            marker="v",
            markersize=PAPER_MARKERSIZE + 1.5,
            markerfacecolor="white",
            markeredgecolor="#1A4E79",
            markeredgewidth=0.7,
            zorder=4,
            label=f"{_budget_label(budget)} (lower bound)",
        )
    ax.set_yscale("log")

    growth = _path_growth_per_hop(s)
    if np.isfinite(growth):
        ax.text(
            0.03,
            0.97,
            f"{growth:.1f}× paths / hop",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
        )
    _set_k_axis_limits(ax, k_metrics)
    _finish_axes(
        ax,
        title="",
        xlabel="distance k",
        ylabel="Simple paths per target",
        legend_loc="lower right",
    )
    _save_figure(fig, output_png)
    plt.close(fig)
    return output_png


def plot_path_runtime_vs_k(
    path_benchmark_summary: pd.DataFrame,
    k_metrics: pd.DataFrame,
    output_png: str | Path,
) -> Path:
    """Path-enumeration runtime by distance k (log scale)."""
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    fig, ax = plt.subplots(figsize=PAPER_SINGLE_FIGSIZE, constrained_layout=True)

    s = path_benchmark_summary.sort_values("cutoff").reset_index(drop=True)
    x = _num(s, "cutoff")

    ax.plot(
        x,
        _num(s, "runtime_sec_total"),
        marker="o",
        color=PRIMARY_CURVE_COLOR,
        zorder=3,
    )
    ax.set_yscale("log")
    _set_k_axis_limits(ax, k_metrics)
    _finish_axes(
        ax,
        title="",
        xlabel="distance k",
        ylabel="Path-enumeration runtime (s)",
        legend=False,
    )
    _save_figure(fig, output_png)
    plt.close(fig)
    return output_png


def plot_path_budget_rate_vs_k(
    path_benchmark_summary: pd.DataFrame,
    k_metrics: pd.DataFrame,
    output_png: str | Path,
) -> Path:
    """Path-budget completion rate by distance k."""
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    fig, ax = plt.subplots(figsize=PAPER_SINGLE_FIGSIZE, constrained_layout=True)

    s = path_benchmark_summary.sort_values("cutoff").reset_index(drop=True)
    x = _num(s, "cutoff")

    ax.plot(
        x,
        _num(s, "within_budget_rate"),
        marker="o",
        color=PRIMARY_CURVE_COLOR,
        zorder=3,
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    _set_k_axis_limits(ax, k_metrics)
    _finish_axes(
        ax,
        title="",
        xlabel="distance k",
        ylabel="Completion rate",
        legend=False,
    )
    _save_figure(fig, output_png)
    plt.close(fig)
    return output_png


def _with_pdf_outputs(outputs: Dict[str, Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, path in outputs.items():
        p = Path(path)
        out[key] = str(p)
        pdf = p.with_suffix(".pdf")
        if pdf.exists():
            out[f"{key}_pdf"] = str(pdf)
        svg = p.with_suffix(".svg")
        if svg.exists():
            out[f"{key}_svg"] = str(svg)
    return out


def make_all_figures(run_root: str | Path) -> Dict[str, str]:
    run_root = Path(run_root)
    data = load_val1_outputs(run_root)
    fig_dir = run_root / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    k_metrics = data["k_metrics"]
    k_selection = data.get("k_selection", {})
    target_depths = data.get("target_depths", pd.DataFrame(columns=["D_t"]))
    td_k_metrics = data.get("td_k_metrics", pd.DataFrame())
    td_selection = data.get("td_selection", pd.DataFrame())
    outputs = {
        "coverage_vs_k": plot_coverage_vs_k(
            k_metrics,
            k_selection,
            fig_dir / "coverage_vs_k.png",
            td_k_metrics=td_k_metrics,
        ),
        "ambiguity_vs_k": plot_ambiguity_vs_k(
            k_metrics,
            k_selection,
            fig_dir / "sign_ambiguity_vs_k.png",
            td_k_metrics=td_k_metrics,
        ),
        "unambiguous_fraction_vs_k": plot_unambiguous_fraction_vs_k(
            k_metrics,
            k_selection,
            fig_dir / "sign_unambiguity_vs_k.png",
            td_k_metrics=td_k_metrics,
        ),
        "harmonic_components_vs_k": plot_harmonic_components_vs_k(k_metrics, k_selection, fig_dir / "harmonic_score_vs_k.png"),
        "distance_k_summary": plot_distance_k_summary(k_metrics, k_selection, fig_dir / "distance_k_summary.png"),
        "target_depth_distribution": plot_target_depth_distribution(target_depths, fig_dir / "target_depth_distribution.png"),
    }
    if not td_selection.empty:
        outputs["td_k_star_distribution"] = plot_td_k_star_distribution(
            td_selection,
            fig_dir / "td_k_star_distribution.png",
        )
        outputs["td_k_valid_distribution"] = plot_td_k_valid_distribution(
            td_selection,
            fig_dir / "td_k_valid_distribution.png",
        )
    rule_agreement = data.get("rule_agreement", pd.DataFrame())
    if not rule_agreement.empty:
        outputs["k_selection_rule_agreement"] = plot_rule_agreement(
            rule_agreement,
            k_metrics,
            k_selection,
            fig_dir / "k_selection_rule_agreement.png",
        )
    resampling_frequency = data.get("resampling_frequency", pd.DataFrame())
    if not resampling_frequency.empty:
        outputs["loto_k_star_frequency"] = plot_resampling_k_star_frequency(
            resampling_frequency,
            k_metrics,
            k_selection,
            fig_dir / "loto_k_star_frequency.png",
        )
    path_benchmark_summary = data.get("path_benchmark_summary", pd.DataFrame())
    if not path_benchmark_summary.empty:
        outputs["path_count_vs_k"] = plot_path_count_vs_k(
            path_benchmark_summary,
            k_metrics,
            k_selection,
            fig_dir / "path_count_vs_k.png",
        )
        outputs["path_runtime_vs_k"] = plot_path_runtime_vs_k(
            path_benchmark_summary,
            k_metrics,
            fig_dir / "path_runtime_vs_k.png",
        )
        outputs["path_budget_rate_vs_k"] = plot_path_budget_rate_vs_k(
            path_benchmark_summary,
            k_metrics,
            fig_dir / "path_budget_rate_vs_k.png",
        )
    return _with_pdf_outputs(outputs)
