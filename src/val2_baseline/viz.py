"""Figures for the val2 external-baseline comparison (methodology §12)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib_venn import venn2  # noqa: E402

from val2.legacy import expected_average_precision_with_ties

from .caliper import (
    caliper_method_display,
    detect_caliper_methods,
    is_caliper_method,
    parse_caliper_method_k,
)
from . import degree_support
from .evaluate import lift_columns_in_df, lift_col

METHOD_ORDER = ["CALIPER", "NCS", "proximity", "RWR", "DADA"]
BASELINES = ["NCS", "proximity", "RWR", "DADA"]

# Display labels mirror the report body wording (L1000-NCS / Network proximity /
# Classical RWR / DADA). Internal method keys stay CALIPER/NCS/proximity/RWR/DADA.
METHOD_DISPLAY = {
    "CALIPER": "CALIPER",
    "NCS": "L1000-NCS",
    "proximity": "Network proximity",
    "RWR": "Classical RWR",
    "DADA": "DADA",
}
# DADA shares Classical RWR's hue (its raw component *is* Classical RWR) and
# separates from it by lightness (L* ~62 -> ~35), which stays distinguishable
# under every CVD type. The remaining Okabe-Ito slots are unavailable:
# #D55E00 is DEGREE_LABEL_COLORS["Positive"], #000000 is used for annotations
# and marker edges, and #56B4E9/#F0E442 are FALLBACK_COLORS (#56B4E9 is also too
# close to CALIPER's blue, and CALIPER vs DADA is a primary comparison).
METHOD_COLOR = {
    "CALIPER": "#0072B2",
    "NCS": "#E69F00",
    "proximity": "#009E73",
    "RWR": "#CC79A7",
    "DADA": "#7B3D5E",
}
FALLBACK_COLORS = ["#56B4E9", "#F0E442", "#999999", "#666666"]

# Paper-ready style constants (kept in sync with src/val2/viz.py).
K_VALID_FIGSIZE = (5.0, 2.4)
METHOD_COMPARISON_NARROW_FIGSIZE = (3.5, 2.4)
MEAN_AP_SHARED_XLIM = (0.05, 0.10)
COMPARATOR_GRAY = "#8C8C8C"
PAPER_AXIS_LINEWIDTH = 0.7
PAPER_ERROR_LINEWIDTH = 0.8
PAPER_ERROR_CAPSIZE = 2.5
PAPER_MARKER_EDGE_WIDTH = 0.45

# Max jitter points overlaid on the raw-score violins. CALIPER/RWR each have ~61k
# common pairs; the violin KDE always uses every pair, only the strip overlay is
# subsampled (deterministic) so it reads as a jitter strip rather than a solid smear.
SCORE_SCATTER_MAX = 400

# Target-degree diagnostics (CALIPER vs RWR, shared network H). "Positive" uses a
# color distinct from METHOD_COLOR (CALIPER's blue / RWR's reddish purple) so it
# doesn't collide with either when both appear in the same figure
# (degree_distribution_top10_by_method).
DEGREE_LABEL_COLORS = {"Non-positive": "#9E9E9E", "Positive": "#D55E00"}


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _apply_paper_rcparams() -> None:
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


def _save_figure(fig, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight")


def _method_display(method: str) -> str:
    if is_caliper_method(method):
        return caliper_method_display(method)
    return METHOD_DISPLAY.get(method, method)


def _method_order(present_methods: set[str], meta: dict | None = None) -> List[str]:
    caliper_methods = detect_caliper_methods(present_methods)
    if not caliper_methods and "CALIPER" in present_methods:
        caliper_methods = ["CALIPER"]
    if not caliper_methods and meta:
        caliper_methods = caliper_mod_keys_from_meta(meta)
    baselines = [m for m in BASELINES if m in present_methods]
    return caliper_methods + baselines


def caliper_mod_keys_from_meta(meta: dict) -> List[str]:
    ks = meta.get("caliper_ks") or [meta.get("k_star")]
    if not ks:
        return ["CALIPER"]
    from .caliper import caliper_method_keys

    return caliper_method_keys([int(k) for k in ks])


def _caliper_methods_for_run(run_root: Path) -> List[str]:
    meta = _load_json(run_root / "metadata.json")
    metrics = _read_tsv(run_root / "evaluation" / "metrics_by_disease_method.tsv")
    present = set(metrics["method"].astype(str)) if "method" in metrics.columns else set()
    methods = _method_order(present, meta)
    return [m for m in methods if is_caliper_method(m)]


def _k_star_caliper_method(run_root: Path, caliper_methods: List[str]) -> str | None:
    """CALIPER method key at the val1-selected ``k_star`` (None if not available).

    Used for the k_star-only variant of the method comparison figures, so a
    multi-k run also gets a figure with a single CALIPER series.
    """
    meta = _load_json(run_root / "metadata.json")
    k_star = meta.get("k_star")
    if k_star is None:
        return None
    for m in caliper_methods:
        if parse_caliper_method_k(m) == int(k_star):
            return m
    return None


def _slugify_method(method: str) -> str:
    if method == "CALIPER":
        return "caliper"
    k = method.replace("CALIPER@k", "k")
    return k.replace("@", "_").lower()


def _method_color(method: str, fallback_idx: int) -> str:
    if method == "CALIPER" or is_caliper_method(method):
        return METHOD_COLOR["CALIPER"]
    if method in METHOD_COLOR:
        return METHOD_COLOR[method]
    return FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]


def _draw_y_axis_break(ax) -> None:
    """Draw the val2-style wavy break near the y-axis origin (see src/val2/viz.py).

    Used when the y-axis does not start at 0 (origin omitted) to mark the break.
    """
    wave_x = np.linspace(-0.018, 0.018, 80)
    for y_offset in (0.032, 0.054):
        wave_y = y_offset + 0.006 * np.sin(np.linspace(0, 2 * np.pi, wave_x.size))
        ax.plot(
            wave_x,
            wave_y,
            transform=ax.transAxes,
            color="black",
            linewidth=PAPER_AXIS_LINEWIDTH,
            clip_on=False,
            solid_capstyle="round",
            zorder=5,
        )


def plot_common_universe_flow(run_root: Path, out_png: Path) -> Dict[str, str]:
    summary = _load_json(run_root / "run_summary.json")
    flow = summary.get("flow", {})
    categories = ["diseases", "pairs", "positive_pairs"]
    category_labels = ["Diseases", "Pairs", "Positive pairs"]
    original = [
        flow.get("original_n_diseases", 0),
        flow.get("original_n_pairs", 0),
        flow.get("original_n_positive_pairs", 0),
    ]
    common = [
        flow.get("common_n_diseases", 0),
        flow.get("common_n_pairs", 0),
        flow.get("common_n_positive_pairs", 0),
    ]
    x = np.arange(len(categories))
    width = 0.38
    _apply_paper_rcparams()
    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    ax.bar(x - width / 2, original, width, label="original Val2")
    ax.bar(x + width / 2, common, width, label="common universe")
    for i, (o, c) in enumerate(zip(original, common)):
        ax.text(i - width / 2, o, f"{int(o):,}", ha="center", va="bottom", fontsize=6.5)
        ax.text(i + width / 2, c, f"{int(c):,}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, rotation=15, ha="right")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)
    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png)


def _detect_k_list(run_root: Path) -> List[int]:
    meta = _load_json(run_root / "metadata.json")
    if meta.get("K_list"):
        return [int(k) for k in meta["K_list"]]
    metrics = _read_tsv(run_root / "evaluation" / "metrics_by_disease_method.tsv")
    ks = lift_columns_in_df(metrics)
    return ks or [int(meta.get("K_eff_cap", 10))]


def plot_four_method_performance(run_root: Path, out_png: Path) -> Dict[str, str]:
    metrics = _read_tsv(run_root / "evaluation" / "metrics_by_disease_method.tsv")
    k_list = _detect_k_list(run_root)
    primary_lift = lift_col(k_list[0])
    present_methods = set(metrics["method"].astype(str)) if "method" in metrics.columns else set()
    meta = _load_json(run_root / "metadata.json")
    methods = _method_order(present_methods, meta)
    _apply_paper_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
    for ax, metric in zip(axes, ["AP", primary_lift]):
        data = []
        labels = []
        for m in methods:
            vals = metrics.loc[metrics["method"] == m, metric].dropna().to_numpy() if len(metrics) else np.array([])
            data.append(vals)
            labels.append(_method_display(m))
        if any(len(d) for d in data):
            ax.boxplot(data, tick_labels=labels, showmeans=True, showfliers=False, widths=0.55)
            for i, d in enumerate(data, start=1):
                if len(d):
                    jitter = (np.random.default_rng(0).random(len(d)) - 0.5) * 0.12
                    ax.scatter(np.full(len(d), i) + jitter, d, s=7, alpha=0.36, color="#333333", edgecolors="none")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.22, linewidth=0.35)
        ax.set_axisbelow(True)
    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png)


def plot_coverage_confounding(run_root: Path, out_png: Path) -> Dict[str, str]:
    metrics = _read_tsv(run_root / "evaluation" / "metrics_by_disease_method.tsv")
    present_methods = set(metrics["method"].astype(str)) if "method" in metrics.columns else set()
    meta = _load_json(run_root / "metadata.json")
    methods = _method_order(present_methods, meta)
    x_labels = {"n_pos": "Positive targets", "n_candidate": "Candidate targets"}
    _apply_paper_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
    for ax, xcol in zip(axes, ["n_pos", "n_candidate"]):
        if len(metrics):
            for m in methods:
                sub = metrics[metrics["method"] == m]
                if len(sub):
                    ax.scatter(sub[xcol], sub["AP"], s=10, alpha=0.5, label=_method_display(m), edgecolors="none")
        ax.set_xlabel(x_labels.get(xcol, xcol))
        ax.set_ylabel("AP")
        ax.grid(alpha=0.22, linewidth=0.35)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png)


def _mark_significant_baselines(
    ax,
    run_root: Path,
    methods: list[str],
    x: np.ndarray,
    means: list[float],
    sems: list[float],
    *,
    metric: str = "AP",
    caliper_method: str | None = None,
) -> None:
    """Star baseline points whose paired t-test vs CALIPER is significant.

    Reads evaluation/ttest_fdr.tsv (compute_ttest_table): the k=4 paired t-test
    with Holm correction. Does nothing if absent or no baseline reaches
    significance for the given metric.
    """
    ttest = _read_tsv(run_root / "evaluation" / "ttest_fdr.tsv")
    if ttest.empty or "significant" not in ttest.columns:
        return
    if "metric" in ttest.columns:
        ttest = ttest[ttest["metric"].astype(str) == str(metric)]
    if caliper_method and "caliper_method" in ttest.columns:
        ttest = ttest[ttest["caliper_method"].astype(str) == str(caliper_method)]
    if ttest.empty:
        return
    label = "Holm"
    caliper_keys = [m for m in methods if is_caliper_method(m)]
    if caliper_method is not None:
        caliper_targets = [caliper_method]
    elif caliper_keys:
        caliper_targets = caliper_keys
    elif "CALIPER" in methods:
        caliper_targets = ["CALIPER"]
    else:
        return

    method_idx = {m: i for i, m in enumerate(methods)}
    y_min, y_max = ax.get_ylim()
    span = max(y_max - y_min, 1e-9)
    tops = [means[i] + sems[i] for i in range(len(methods)) if np.isfinite(means[i])]
    base_level = (max(tops) if tops else y_max) + 0.07 * span
    step = 0.11 * span
    tick = 0.02 * span
    bracket_level = 0
    any_sig = False
    for ci_method in caliper_targets:
        sub_ttest = ttest
        if "caliper_method" in ttest.columns:
            sub_ttest = ttest[ttest["caliper_method"].astype(str) == str(ci_method)]
        sig_baselines = {
            str(r["baseline"]) for r in sub_ttest.to_dict("records") if bool(r.get("significant"))
        }
        if not sig_baselines:
            continue
        ci = method_idx.get(ci_method)
        if ci is None or not np.isfinite(means[ci]):
            continue
        sig_idx = sorted(
            method_idx[b]
            for b in sig_baselines
            if b in method_idx and method_idx[b] != ci and np.isfinite(means[method_idx[b]])
        )
        for bi in sig_idx:
            any_sig = True
            y = base_level + bracket_level * step
            x0, x1 = sorted((float(x[ci]), float(x[bi])))
            ax.plot(
                [x0, x0, x1, x1],
                [y - tick, y, y, y - tick],
                color="black",
                linewidth=0.8,
                clip_on=False,
                zorder=6,
            )
            ax.annotate(
                "*",
                xy=((x0 + x1) / 2.0, y),
                ha="center",
                va="bottom",
                fontsize=11,
                color="black",
                zorder=7,
            )
            bracket_level += 1
    if not any_sig:
        return
    ax.set_ylim(y_min, max(y_max, base_level + max(0, bracket_level - 1) * step + 0.06 * span))
    ax.text(
        0.99,
        0.02,
        f"* {label}<0.05, CALIPER vs baseline (paired t-test)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color="#333333",
    )


def plot_method_comparison(
    run_root: Path,
    out_png: Path,
    *,
    metric_col: str,
    ylabel: str,
    title: str,
    mark_ttest: bool = False,
    cohort_filter: str | None = None,
    caliper_method: str | None = None,
    comparison_methods: List[str] | None = None,
    horizontal_dot_whisker: bool = False,
    xlim: tuple[float, float] | None = None,
) -> Dict[str, str]:
    """Paper-ready per-method mean +/- SEM figure.

    Mirrors the val2 ablation candidate_k_method_comparison_ap presentation but
    without the candidate-k axis (k_star is fixed in val2_baseline). Mean and SEM
    are aggregated across diseases from metrics_by_disease_method.tsv. The
    horizontal presentation sorts methods by descending mean and highlights only
    CALIPER; it is used for the final-Figure raw Mean AP comparison.
    """
    metrics = _read_tsv(run_root / "evaluation" / "metrics_by_disease_method.tsv")
    if metrics.empty or metric_col not in metrics.columns:
        return {}
    if cohort_filter and cohort_filter in metrics.columns:
        metrics = metrics[metrics[cohort_filter].astype(bool)].copy()
    elif cohort_filter == "in_lift_eval" and "in_lift_eval" in metrics.columns:
        metrics = metrics[metrics["in_lift_eval"].astype(bool)].copy()

    methods = comparison_methods or _method_order(
        set(metrics["method"].astype(str)), _load_json(run_root / "metadata.json")
    )
    methods = [m for m in methods if m in set(metrics["method"].astype(str))]
    if not methods:
        return {}
    means: list[float] = []
    sems: list[float] = []
    for m in methods:
        vals = pd.to_numeric(
            metrics.loc[metrics["method"].astype(str) == m, metric_col], errors="coerce"
        ).dropna().to_numpy()
        if len(vals):
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        else:
            means.append(np.nan)
            sems.append(0.0)
    if horizontal_dot_whisker:
        order = sorted(
            range(len(methods)),
            key=lambda i: (
                not np.isfinite(means[i]),
                -means[i] if np.isfinite(means[i]) else 0.0,
            ),
        )
        methods = [methods[i] for i in order]
        means = [means[i] for i in order]
        sems = [sems[i] for i in order]

    _apply_paper_rcparams()
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figsize = METHOD_COMPARISON_NARROW_FIGSIZE if horizontal_dot_whisker else (
        METHOD_COMPARISON_NARROW_FIGSIZE
        if len(methods) <= 2
        else (max(K_VALID_FIGSIZE[0], 0.85 * len(methods)), K_VALID_FIGSIZE[1])
    )
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    if horizontal_dot_whisker:
        y = np.arange(len(methods), dtype=float)
        for idx, method in enumerate(methods):
            is_caliper = is_caliper_method(method)
            color = METHOD_COLOR["CALIPER"] if is_caliper else COMPARATOR_GRAY
            ax.errorbar(
                means[idx],
                y[idx],
                xerr=sems[idx],
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
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_yticks(y)
        ax.set_yticklabels([_method_display(method) for method in methods])
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(PAPER_AXIS_LINEWIDTH)
        ax.grid(axis="x", alpha=0.22, linewidth=0.35)
        ax.set_axisbelow(True)
        _save_figure(fig, out_png)
        plt.close(fig)
        return _rel_outputs(run_root, out_png)

    x = np.arange(len(methods), dtype=float)
    for idx, m in enumerate(methods):
        color = _method_color(m, idx)
        ax.errorbar(
            x[idx],
            means[idx],
            yerr=sems[idx],
            fmt="o",
            markersize=3.0,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=PAPER_MARKER_EDGE_WIDTH,
            color=color,
            ecolor=color,
            elinewidth=PAPER_ERROR_LINEWIDTH,
            capsize=PAPER_ERROR_CAPSIZE,
            capthick=PAPER_ERROR_LINEWIDTH,
            zorder=3,
        )
    x_padding = 0.35 if len(methods) <= 2 else 0.55
    ax.set_xlim(x[0] - x_padding, x[-1] + x_padding)
    ax.set_xticks(x)
    ax.set_xticklabels([_method_display(m) for m in methods], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(PAPER_AXIS_LINEWIDTH)
    # Zoom to the data range (origin typically omitted). Mark the omitted origin
    # with the val2-style wavy y-axis break (see _draw_y_axis_break).
    y_lows = [m - s for m, s in zip(means, sems) if np.isfinite(m)]
    y_highs = [m + s for m, s in zip(means, sems) if np.isfinite(m)]
    if y_lows and y_highs:
        y_low, y_high = min(y_lows), max(y_highs)
        y_span = max(y_high - y_low, 1e-9)
        ax.set_ylim(max(0.0, y_low - 0.12 * y_span), y_high + 0.10 * y_span)
    ax.grid(axis="y", alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)
    if ax.get_ylim()[0] > 0:
        _draw_y_axis_break(ax)
    if mark_ttest:
        _mark_significant_baselines(
            ax, run_root, methods, x, means, sems, metric=metric_col, caliper_method=caliper_method
        )
    _save_figure(fig, out_png)
    plt.close(fig)
    stem = out_png.stem
    out = {stem: out_png, f"{stem}_pdf": out_png.with_suffix(".pdf"), f"{stem}_svg": out_png.with_suffix(".svg")}
    rel: Dict[str, str] = {}
    for k, p in out.items():
        try:
            rel[k] = str(Path(p).relative_to(run_root))
        except Exception:
            rel[k] = str(p)
    return rel


def _rel_outputs(run_root: Path, out_png: Path) -> Dict[str, str]:
    stem = out_png.stem
    out = {stem: out_png, f"{stem}_pdf": out_png.with_suffix(".pdf"), f"{stem}_svg": out_png.with_suffix(".svg")}
    rel: Dict[str, str] = {}
    for k, p in out.items():
        try:
            rel[k] = str(Path(p).relative_to(run_root))
        except Exception:
            rel[k] = str(p)
    return rel


def _rel_file_output(run_root: Path, key: str, path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        return {key: str(path.relative_to(run_root))}
    except Exception:
        return {key: str(path)}


def _median_ci_fallback(
    run_root: Path,
    metric: str,
    baseline: str,
    caliper_method: str | None,
    n_boot: int,
    seed: int,
) -> tuple[float, float] | None:
    """Median bootstrap 95% CI recomputed from paired_deltas.tsv.

    Used only when a run's bootstrap_summary.tsv predates the ``median_ci_*``
    columns. Reuses the persisted ``n_boot``/``bootstrap_seed`` so the CI matches
    what compute_bootstrap_summary would have written. Returns ``None`` when the
    delta table / column is unavailable.
    """
    deltas = _read_tsv(run_root / "evaluation" / "paired_deltas.tsv")
    if deltas.empty or "baseline" not in deltas.columns:
        return None
    from .evaluate import _filter_deltas_for_metric, bootstrap_median_ci, build_delta_col_map

    col = build_delta_col_map(_detect_k_list(run_root)).get(metric)
    if col is None or col not in deltas.columns:
        return None
    sub = deltas[deltas["baseline"].astype(str) == str(baseline)]
    if caliper_method and "caliper_method" in sub.columns:
        sub = sub[sub["caliper_method"].astype(str) == str(caliper_method)]
    if sub.empty:
        return None
    try:
        sub = _filter_deltas_for_metric(sub, metric)
    except ValueError:
        return None
    vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
    _, lo, hi = bootstrap_median_ci(vals, int(n_boot), int(seed))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return float(lo), float(hi)


def plot_effect_size_forest(
    run_root: Path,
    out_png: Path,
    *,
    metric: str,
    caliper_method: str | None = None,
) -> Dict[str, str]:
    """Forest plot of median Δ + paired bootstrap 95% CI per baseline.

    Point and CI are both based on the median (median_delta with the bootstrap CI
    of the median from bootstrap_summary.tsv), which is the robust location
    estimate for the heavy-tailed AP deltas. Overlays a `*` on baselines that
    are significant in ttest_fdr.tsv (k=4 paired t-test, Holm). For runs whose
    bootstrap_summary.tsv predates the median-CI columns, the CI is recomputed
    from paired_deltas.tsv via :func:`_median_ci_fallback`. Returns {} if the
    bootstrap summary is unavailable.
    """
    boot = _read_tsv(run_root / "evaluation" / "bootstrap_summary.tsv")
    if boot.empty or "metric" not in boot.columns:
        return {}
    sub = boot[boot["metric"].astype(str) == str(metric)]
    if caliper_method and "caliper_method" in sub.columns:
        sub = sub[sub["caliper_method"].astype(str) == str(caliper_method)]
    if sub.empty:
        return {}
    ttest = _read_tsv(run_root / "evaluation" / "ttest_fdr.tsv")
    correction_label = "Holm" if not ttest.empty else ""

    has_caliper_groups = (
        caliper_method is None
        and "caliper_method" in sub.columns
        and sub["caliper_method"].astype(str).nunique() > 1
    )
    row_specs: List[tuple[str, str, str]] = []
    if has_caliper_groups:
        caliper_methods = sorted(dict.fromkeys(sub["caliper_method"].astype(str)))
        baselines = [b for b in BASELINES if b in set(sub["baseline"].astype(str))]
        for cm in caliper_methods:
            for b in baselines:
                row_specs.append((cm, b, f"{_method_display(cm)} - {METHOD_DISPLAY.get(b, b)}"))
    else:
        baselines = [b for b in BASELINES if b in set(sub["baseline"].astype(str))]
        cm = caliper_method or (
            str(sub["caliper_method"].iloc[0]) if "caliper_method" in sub.columns else None
        )
        for b in baselines:
            label = f"{_method_display(cm) if cm else 'CALIPER'} - {METHOD_DISPLAY.get(b, b)}"
            row_specs.append((cm or "CALIPER", b, label))

    medians, los, his, ylabels = [], [], [], []
    for cm, b, label in row_specs:
        row = sub[
            (sub["baseline"].astype(str) == str(b))
            & (
                sub["caliper_method"].astype(str) == str(cm)
                if "caliper_method" in sub.columns
                else True
            )
        ]
        if row.empty:
            continue
        med = float(row["median_delta"].iloc[0])
        if "median_ci_low" in row.columns and np.isfinite(row["median_ci_low"].iloc[0]):
            lo = float(row["median_ci_low"].iloc[0])
            hi = float(row["median_ci_high"].iloc[0])
        else:
            n_boot = int(row["n_boot"].iloc[0]) if "n_boot" in row.columns else 10000
            seed = int(row["bootstrap_seed"].iloc[0]) if "bootstrap_seed" in row.columns else 42
            fb = _median_ci_fallback(run_root, metric, b, cm, n_boot, seed)
            lo, hi = fb if fb is not None else (med, med)
        medians.append(med)
        los.append(lo)
        his.append(hi)
        star = ""
        if not ttest.empty and "significant" in ttest.columns:
            trow = ttest[
                (ttest["metric"].astype(str) == str(metric))
                & (ttest["baseline"].astype(str) == str(b))
            ]
            if "caliper_method" in ttest.columns:
                trow = trow[trow["caliper_method"].astype(str) == str(cm)]
            if len(trow) and bool(trow["significant"].iloc[0]):
                star = " *"
        ylabels.append(f"{label}{star}")
    if not medians:
        return {}
    medians = np.array(medians)
    err = np.vstack([medians - np.array(los), np.array(his) - medians])
    err = np.clip(err, 0.0, None)

    fig_w = max(4.2, min(7.2, 0.75 * len(ylabels) + 3.0))
    _apply_paper_rcparams()
    fig, ax = plt.subplots(figsize=(fig_w, 2.8), constrained_layout=True)
    xs = np.arange(len(ylabels))
    ax.errorbar(
        xs,
        medians,
        yerr=err,
        fmt="o",
        markersize=3.0,
        elinewidth=PAPER_ERROR_LINEWIDTH,
        capsize=PAPER_ERROR_CAPSIZE,
        capthick=PAPER_ERROR_LINEWIDTH,
        color="#0072B2",
        ecolor="#0072B2",
    )
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=PAPER_AXIS_LINEWIDTH)
    ax.set_xticks(xs)
    ax.set_xticklabels(ylabels, rotation=20, ha="right")
    ax.set_xlim(-0.5, len(ylabels) - 0.5)
    ax.set_ylabel(f"median Δ{metric}\n(95% CI, disease bootstrap)")
    if any("*" in y for y in ylabels) and correction_label:
        ax.text(
            0.99,
            0.99,
            f"* {correction_label}<0.05 (paired t-test)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6,
            color="#333333",
        )
    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png)


def plot_win_rate(
    run_root: Path,
    out_png: Path,
    *,
    metric: str,
    caliper_method: str | None = None,
) -> Dict[str, str]:
    """Bar chart of win-rate (fraction of diseases with Δ>0) per baseline."""
    eff = _read_tsv(run_root / "evaluation" / "effect_sizes.tsv")
    if eff.empty or "win_rate" not in eff.columns or "metric" not in eff.columns:
        return {}
    sub = eff[eff["metric"].astype(str) == str(metric)]
    if caliper_method and "caliper_method" in sub.columns:
        sub = sub[sub["caliper_method"].astype(str) == str(caliper_method)]
    if sub.empty:
        return {}

    has_caliper_groups = (
        caliper_method is None
        and "caliper_method" in sub.columns
        and sub["caliper_method"].astype(str).nunique() > 1
    )
    if has_caliper_groups:
        caliper_methods = sorted(dict.fromkeys(sub["caliper_method"].astype(str)))
        baselines = [b for b in BASELINES if b in set(sub["baseline"].astype(str))]
        width = 0.8 / max(len(caliper_methods), 1)
        x = np.arange(len(baselines), dtype=float)
        _apply_paper_rcparams()
        fig, ax = plt.subplots(figsize=(max(5.0, 1.2 * len(baselines)), 2.8), constrained_layout=True)
        for j, cm in enumerate(caliper_methods):
            vals = []
            for b in baselines:
                row = sub[
                    (sub["baseline"].astype(str) == str(b))
                    & (sub["caliper_method"].astype(str) == str(cm))
                ]
                vals.append(float(row["win_rate"].iloc[0]) if len(row) else np.nan)
            offsets = x + (j - (len(caliper_methods) - 1) / 2.0) * width
            ax.bar(
                offsets,
                vals,
                width=width,
                color=_method_color(cm, j),
                label=_method_display(cm),
            )
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=PAPER_AXIS_LINEWIDTH)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_DISPLAY.get(b, b) for b in baselines], rotation=20, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(f"win-rate (Δ{metric}>0)")
        ax.legend(fontsize=6, loc="upper right")
        _save_figure(fig, out_png)
        plt.close(fig)
        return _rel_outputs(run_root, out_png)

    baselines = [b for b in BASELINES if b in set(sub["baseline"].astype(str))]
    vals = [float(sub.loc[sub["baseline"].astype(str) == b, "win_rate"].iloc[0]) for b in baselines]

    _apply_paper_rcparams()
    fig, ax = plt.subplots(figsize=(5.0, 2.6), constrained_layout=True)
    x = np.arange(len(baselines))
    ax.bar(x, vals, color=[_method_color(b, i) for i, b in enumerate(baselines)])
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=PAPER_AXIS_LINEWIDTH)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY.get(b, b) for b in baselines], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(f"win-rate (Δ{metric}>0)")
    title = f"Win-rate across diseases: Δ{metric}"
    if caliper_method:
        title = f"{title} ({_method_display(caliper_method)})"
    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png)


def plot_paired_delta_violin(run_root: Path, out_png: Path) -> Dict[str, str]:
    """Per-disease Δ distribution (violin) for AP and Lift@10 per baseline."""
    deltas = _read_tsv(run_root / "evaluation" / "paired_deltas.tsv")
    if deltas.empty or "baseline" not in deltas.columns:
        return {}
    ttest = _read_tsv(run_root / "evaluation" / "ttest_fdr.tsv")

    _apply_paper_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    col_map = {"AP": "delta_AP", "Lift@10": "delta_Lift@10"}
    for ax, metric in zip(axes, ["AP", "Lift@10"]):
        col = col_map[metric]
        if col not in deltas.columns:
            continue
        baselines = [b for b in BASELINES if b in set(deltas["baseline"].astype(str))]
        data = [
            pd.to_numeric(deltas.loc[deltas["baseline"].astype(str) == b, col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna().to_numpy()
            for b in baselines
        ]
        positions = np.arange(1, len(baselines) + 1)
        if any(len(d) for d in data):
            parts = ax.violinplot([d for d in data], positions=positions, showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_alpha(0.5)
            for i, d in enumerate(data, start=1):
                if len(d):
                    jitter = (np.random.default_rng(0).random(len(d)) - 0.5) * 0.12
                    ax.scatter(np.full(len(d), i) + jitter, d, s=6, alpha=0.36, color="#333333", edgecolors="none")
        ax.axhline(0.0, color="grey", linestyle="--", linewidth=PAPER_AXIS_LINEWIDTH)
        labels = []
        for b in baselines:
            ann = ""
            if not ttest.empty and {"metric", "baseline", "median_delta", "win_rate", "p_holm"}.issubset(ttest.columns):
                tr = ttest[(ttest["metric"].astype(str) == metric) & (ttest["baseline"].astype(str) == b)]
                if len(tr):
                    med = float(tr["median_delta"].iloc[0])
                    win = float(tr["win_rate"].iloc[0])
                    pv = float(tr["p_holm"].iloc[0]) if np.isfinite(tr["p_holm"].iloc[0]) else float("nan")
                    ann = f"\nmed={med:.3g}\nwin={win:.2f}\np={pv:.2g}"
            labels.append(f"{METHOD_DISPLAY.get(b, b)}{ann}")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=0, fontsize=6)
        ax.set_ylabel(f"Δ{metric} (CALIPER - baseline)")
        ax.grid(axis="y", alpha=0.22, linewidth=0.35)
        ax.set_axisbelow(True)
    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png)


def _draw_method_violin(
    ax, vals: np.ndarray, color: str, *, scatter: bool = False, scatter_max: int | None = None
) -> None:
    """Single-method violin at x=1 (median line, no extrema), matched to the
    paired_delta_violin styling. When ``scatter`` is set, overlays deterministic
    jittered points; ``scatter_max`` subsamples that overlay (the violin KDE still
    uses every value) so dense panels (~61k raw scores) read as a jitter strip
    rather than a solid smear. The per-disease AP panels (145 points) pass no cap.
    """
    if not len(vals):
        return
    parts = ax.violinplot([vals], positions=[1], widths=0.7, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
        pc.set_edgecolor("#333333")
        pc.set_linewidth(0.5)
    if "cmedians" in parts:
        parts["cmedians"].set_color("#111111")
        parts["cmedians"].set_linewidth(1.0)
    if scatter:
        pts = vals
        if scatter_max is not None and len(pts) > scatter_max:
            idx = np.random.default_rng(0).choice(len(pts), size=int(scatter_max), replace=False)
            pts = pts[idx]
        jitter = (np.random.default_rng(0).random(len(pts)) - 0.5) * 0.18
        ax.scatter(
            np.full(len(pts), 1.0) + jitter, pts, s=6, alpha=0.35, color="#333333", edgecolors="none", zorder=3
        )


def plot_score_violin_caliper_vs_rwr(run_root: Path, out_png: Path) -> Dict[str, str]:
    """Raw-score distribution (violin) for CALIPER and Classical RWR, drawn as
    side-by-side panels sharing one linear y-axis.

    Both panels share a linear y-axis so the two methods sit on one common raw-score
    scale. CALIPER scores span [0,1]; Classical RWR scores are all <=0.01 and so
    concentrate at the bottom of the shared axis — the shared linear axis is
    deliberately honest about the raw-score scale gap rather than normalizing it
    away. Distribution is over all common (disease, target) candidate pairs in
    scores_common_long.tsv: the violin KDE uses every pair, while the overlaid
    jitter is a deterministic subsample (SCORE_SCATTER_MAX) for readability. Returns
    {} if either method is absent from the run.
    """
    long_df = _read_tsv(run_root / "evaluation" / "scores_common_long.tsv")
    if long_df.empty or not {"method", "score"}.issubset(long_df.columns):
        return {}
    methods = ["CALIPER", "RWR"]
    if not set(methods).issubset(set(long_df["method"].astype(str))):
        return {}
    _apply_paper_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.8), sharey=True, constrained_layout=True)
    for ax, method in zip(axes, methods):
        vals = (
            pd.to_numeric(long_df.loc[long_df["method"].astype(str) == method, "score"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy()
        )
        _draw_method_violin(ax, vals, METHOD_COLOR[method], scatter=True, scatter_max=SCORE_SCATTER_MAX)
        ax.set_xticks([1])
        ax.set_xticklabels([_method_display(method)])
        ax.set_xlim(0.4, 1.6)
        ax.grid(axis="y", alpha=0.22, linewidth=0.35)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("score")
    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png)


def plot_ap_violin_caliper_vs_rwr(run_root: Path, out_png: Path) -> Dict[str, str]:
    """Per-disease AP distribution (violin) for CALIPER and Classical RWR, drawn as
    side-by-side panels sharing one linear y-axis.

    AP is bounded and comparable across methods, so both panels share the y-axis
    for direct comparison. AP is one value per disease from
    metrics_by_disease_method.tsv (val2_baseline's disease-level unit). Returns {}
    if either method is absent from the run.
    """
    metrics = _read_tsv(run_root / "evaluation" / "metrics_by_disease_method.tsv")
    if metrics.empty or "AP" not in metrics.columns or "method" not in metrics.columns:
        return {}
    methods = ["CALIPER", "RWR"]
    if not set(methods).issubset(set(metrics["method"].astype(str))):
        return {}
    _apply_paper_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.8), sharey=True, constrained_layout=True)
    for ax, method in zip(axes, methods):
        vals = (
            pd.to_numeric(metrics.loc[metrics["method"].astype(str) == method, "AP"], errors="coerce")
            .dropna()
            .to_numpy()
        )
        _draw_method_violin(ax, vals, METHOD_COLOR[method], scatter=True)
        ax.set_xticks([1])
        ax.set_xticklabels([_method_display(method)])
        ax.set_xlim(0.4, 1.6)
        ax.grid(axis="y", alpha=0.22, linewidth=0.35)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("AP (per disease)")
    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png)


def _load_target_degree(run_root: Path) -> Dict[str, int] | None:
    """Rebuild the shared network H from the run's KG and return node -> degree.

    Resolves the KG file the same way the runner does (config/resolved_config.json's
    ``registry_path`` + ``data``), so the degree values match what the run actually
    used. Returns ``None`` if the config snapshot or KG cannot be resolved.
    """
    resolved = _load_json(run_root / "config" / "resolved_config.json")
    registry_path = resolved.get("registry_path")
    data_cfg = resolved.get("data")
    if not registry_path or not data_cfg:
        return None
    try:
        from val2.io import load_signed_multidigraph_with_optional_mechanism
        from val2.registry import load_dataset_registry, resolve_dataset_spec

        from .network import build_shared_network

        registry_path = Path(registry_path)
        registry = load_dataset_registry(registry_path)
        spec = resolve_dataset_spec(data_cfg, registry, registry_path.parent)
        graph = load_signed_multidigraph_with_optional_mechanism(
            spec.ppi_path,
            src_col=spec.source_col,
            dst_col=spec.target_col,
            sign_col=spec.sign_col,
            mechanism_col=spec.mechanism_col,
            sep=spec.sep,
        )
        H, _stats = build_shared_network(graph)
    except Exception:
        return None
    return dict(H.degree())


def _build_degree_diagnostics_frame(run_root: Path) -> pd.DataFrame | None:
    """Build the paired (disease_id, target_gene) feature table used by the
    CALIPER-vs-RWR target-degree diagnostics (rank, degree_H, positive label).

    Returns ``None`` when the run lacks an RWR baseline, the long score table is
    absent, or the KG-derived degree cannot be resolved.
    """
    long_df = _read_tsv(run_root / "evaluation" / "scores_common_long.tsv")
    required_cols = {"disease_id", "target_gene", "label", "method", "score"}
    if long_df.empty or not required_cols.issubset(long_df.columns):
        return None
    present_methods = set(long_df["method"].astype(str))
    if not {"CALIPER", "RWR"}.issubset(present_methods):
        return None

    wide = long_df.pivot_table(
        index=["disease_id", "target_gene", "label"], columns="method", values="score", aggfunc="first"
    ).reset_index()
    wide.columns.name = None

    degree = _load_target_degree(run_root)
    if not degree:
        return None
    wide["degree_H"] = wide["target_gene"].map(degree).astype(float)

    # Per-disease rankings (score descending), over the identical candidate pool.
    n_by_disease = wide.groupby("disease_id")["target_gene"].transform("count")
    for method in ("CALIPER", "RWR"):
        wide[f"{method}_rank_min"] = wide.groupby("disease_id")[method].rank(method="min", ascending=False)
        wide[f"{method}_rank_pct"] = wide[f"{method}_rank_min"] / n_by_disease

    wide["positive"] = wide["label"].astype(int).eq(1)
    wide["label_group"] = np.where(wide["positive"], "Positive", "Non-positive")

    feat = wide.dropna(subset=["degree_H"]).copy()
    if feat.empty or not feat["positive"].any():
        return None

    q1, q2 = feat.loc[feat["positive"], "degree_H"].quantile([1 / 3, 2 / 3])
    bin_labels = [
        f"Low (≤{q1:.0f})",
        f"Mid ({q1:.0f}–{q2:.0f})",
        f"High (>{q2:.0f})",
    ]
    feat["degree_bin3"] = pd.cut(feat["degree_H"], bins=[0, q1, q2, np.inf], labels=bin_labels)
    feat.attrs["degree_bin_labels"] = bin_labels
    return feat


def _draw_box(ax, data: List[np.ndarray], colors: List[str], labels: List[str]) -> None:
    """Boxplot (quartiles/median) with fliers hidden (``showfliers=False``)."""
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.5, zorder=3)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor("#333333")
    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.2)


def plot_degree_distribution_by_label(feat: pd.DataFrame, run_root: Path, out_png: Path) -> Dict[str, str]:
    groups = ["Non-positive", "Positive"]
    display_groups = ["Unlabeled", "Known positive"]
    data = [feat.loc[feat["label_group"].eq(g), "degree_H"].dropna().to_numpy() for g in groups]
    if not any(len(d) for d in data):
        return {}
    _apply_paper_rcparams()
    fig, ax = plt.subplots(figsize=(3.4, 2.9), constrained_layout=True)
    _draw_box(
        ax,
        data,
        [DEGREE_LABEL_COLORS[g] for g in groups],
        display_groups,
    )
    ax.set_ylabel("target degree")
    ax.grid(axis="y", alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)
    # Preserve the approved source-run presentation. Other paper panels may
    # change the process-wide rcParams, so make this panel's full box explicit.
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png)


def plot_degree_distribution_top10(feat: pd.DataFrame, run_root: Path, out_png: Path) -> Dict[str, str]:
    top10 = {m: feat["positive"] & feat[f"{m}_rank_min"].le(10) for m in ("CALIPER", "RWR")}
    groups = ["All positives", "CALIPER top10", "RWR top10"]
    masks = {"All positives": feat["positive"], "CALIPER top10": top10["CALIPER"], "RWR top10": top10["RWR"]}
    colors = [DEGREE_LABEL_COLORS["Positive"], METHOD_COLOR["CALIPER"], METHOD_COLOR["RWR"]]
    data = [feat.loc[masks[g], "degree_H"].dropna().to_numpy() for g in groups]
    if not any(len(d) for d in data):
        return {}
    _apply_paper_rcparams()
    fig, ax = plt.subplots(figsize=(4.0, 2.9), constrained_layout=True)
    _draw_box(ax, data, colors, groups)
    ax.set_ylabel("target degree")
    ax.grid(axis="y", alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)
    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png)


def _plot_degree_vs_score_hexbin(feat: pd.DataFrame, run_root: Path, out_png: Path, mode: str) -> Dict[str, str]:
    """CALIPER-vs-RWR (2 panels) hexbin of known-positive target degree vs score.

    ``mode`` selects the y-axis rendering (the x-axis, hexbin color scale and the
    annotated Spearman ρ are identical across modes — the three figures are the
    same degree-vs-score relationship shown three ways):

    - ``"default"``: raw score; RWR on a log y-axis (spans ~4 orders of
      magnitude), CALIPER on a linear y-axis (bounded [0,1], contains exact
      zeros that a log axis cannot render).
    - ``"linear"``: raw score, both panels on a linear y-axis (RWR positives
      collapse into a thin band near zero — shows why the log axis is used).
    - ``"percentile"``: within-disease score percentile, oriented higher-is-better
      (``1 - rank_pct``, i.e. 100% = top-scoring). Unit-free and bounded [0,1],
      so the two methods are directly comparable on one shared axis; both linear,
      ticks shown as percentages, no axis inversion (up = better, like the
      raw-score panels).
    """
    positive = feat[feat["positive"]]
    if positive.empty:
        return {}
    _apply_paper_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), constrained_layout=True)
    cmaps = {"CALIPER": "Blues", "RWR": "RdPu"}
    panels = []  # (ax, hb) drawn, so the count color scale can be unified across methods
    for ax, method in zip(axes, ("CALIPER", "RWR")):
        if mode == "percentile":
            # Orient higher-is-better: 1 - rank_pct so 100% = top-scoring target.
            vals = (1.0 - positive[f"{method}_rank_pct"]).dropna()
            yscale = "linear"
        else:
            vals = positive[method].dropna()
            yscale = "log" if (mode == "default" and method == "RWR") else "linear"
        if vals.empty:
            continue
        hb = ax.hexbin(
            positive.loc[vals.index, "degree_H"],
            vals,
            gridsize=28,
            mincnt=1,
            bins="log",
            cmap=cmaps[method],
            yscale=yscale,
            linewidths=0.1,
        )
        panels.append((ax, hb))
        # Paper style: reduce the panel frame to the axis lines only (keep left/bottom,
        # drop the top/right box). Applied per-panel so the attached colorbar keeps its
        # full outline.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("target degree")
        if mode == "percentile":
            ax.set_ylabel(f"{_method_display(method)} score percentile\n(within disease)")
            # Pad limits past [0,1] so the hexagons centred at 0%/100% render fully
            # (an exact (0,1) ylim clips the top/bottom row of bins); keep clean
            # 0–100% ticks.
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        else:
            ax.set_ylabel(f"{_method_display(method)} score" + (" (log)" if yscale == "log" else ""))
        # Spearman rho (degree vs raw score, matches degree_diagnostics_spearman.tsv).
        # Computed on the raw score in every mode so all three figures report the
        # same degree-vs-score association regardless of the y-axis rendering.
        # Placed in the bottom-right corner with an opaque bbox so it stays legible and
        # clear of the hexbins (high-degree/low-score region is sparse for both methods).
        rho = positive["degree_H"].corr(positive[method], method="spearman")
        if np.isfinite(rho):
            ax.text(
                0.96,
                0.04,
                f"Spearman ρ = {rho:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999999", alpha=0.85, linewidth=0.4),
                zorder=6,
            )
    # Unify the count color scale across the two method panels: each panel keeps its
    # own colormap (CALIPER=Blues, RWR=RdPu), but bins="log" puts every hexbin's
    # get_array() on the same log10(count) scale, so applying a shared clim makes an
    # equal color mean an equal count in both panels. Set clim before the colorbars.
    arrays = [hb.get_array() for _, hb in panels if hb.get_array().size]
    if arrays:
        vmin = min(float(a.min()) for a in arrays)
        vmax = max(float(a.max()) for a in arrays)
        for _, hb in panels:
            hb.set_clim(vmin, vmax)
    for ax, hb in panels:
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("positive pairs (log)", fontsize=6)
        cbar.ax.tick_params(labelsize=6)
    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png)


def plot_positive_degree_vs_score_hexbin(feat: pd.DataFrame, run_root: Path, out_png: Path) -> Dict[str, str]:
    return _plot_degree_vs_score_hexbin(feat, run_root, out_png, mode="default")


def plot_positive_degree_vs_score_hexbin_linear(feat: pd.DataFrame, run_root: Path, out_png: Path) -> Dict[str, str]:
    return _plot_degree_vs_score_hexbin(feat, run_root, out_png, mode="linear")


def plot_positive_degree_vs_score_percentile(feat: pd.DataFrame, run_root: Path, out_png: Path) -> Dict[str, str]:
    return _plot_degree_vs_score_hexbin(feat, run_root, out_png, mode="percentile")


def plot_positive_degree_histogram(feat: pd.DataFrame, run_root: Path, out_png: Path) -> Dict[str, str]:
    """Simple histogram of known-positive target degree (linear x-axis).

    Shows the shape of the positive degree distribution (right-skewed toward low
    degree), context for the degree-stratified diagnostics that follow. Dashed
    lines mark the tertile cuts (q1, q2) of this same distribution, i.e. the
    Low/Mid/High degree_bin3 edges used elsewhere (_build_degree_diagnostics_frame).
    """
    positive = feat[feat["positive"]]
    deg = positive["degree_H"].dropna()
    if deg.empty:
        return {}
    q1, q2 = deg.quantile([1 / 3, 2 / 3])
    _apply_paper_rcparams()
    fig, ax = plt.subplots(figsize=(4.6, 2.9), constrained_layout=True)
    ax.hist(
        deg,
        bins=40,
        color=DEGREE_LABEL_COLORS["Positive"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.3,
    )
    for q in (q1, q2):
        ax.axvline(q, color="#333333", linestyle="--", linewidth=0.8, zorder=2)
    ax.set_xlabel("target degree")
    ax.set_ylabel("known positives (pairs)")
    ax.grid(axis="y", alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)
    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png)


def plot_hit_rate_by_degree_bin(
    feat: pd.DataFrame, run_root: Path, out_png: Path
) -> tuple[Dict[str, str], pd.DataFrame]:
    """% of known positives recovered in each method's top-10, by Low/Mid/High
    target-degree stratum (the degree_bin3 tertiles of the positive degree
    distribution). Coarse, count-based counterpart of the degree-decile rank
    percentile figure: it reads directly as recovery rate rather than rank."""
    bin_labels = list(feat.attrs.get("degree_bin_labels") or feat["degree_bin3"].cat.categories)
    positive = feat[feat["positive"]]
    rows = []
    for method in ("CALIPER", "RWR"):
        hit = positive[f"{method}_rank_min"].le(10)
        for degree_bin in bin_labels:
            mask = positive["degree_bin3"].eq(degree_bin)
            n = int(mask.sum())
            n_hit = int((mask & hit).sum())
            rows.append(
                {
                    "method": method,
                    "degree_bin": degree_bin,
                    "n_positive": n,
                    "n_hit": n_hit,
                    "hit_rate_pct": 100 * n_hit / n if n else float("nan"),
                }
            )
    hit_rate_df = pd.DataFrame(rows)
    if hit_rate_df.empty or not np.isfinite(hit_rate_df["hit_rate_pct"]).any():
        return {}, hit_rate_df

    _apply_paper_rcparams()
    fig, ax = plt.subplots(figsize=METHOD_COMPARISON_NARROW_FIGSIZE, constrained_layout=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    x = np.arange(len(bin_labels), dtype=float)
    width = 0.35
    # Headroom for the value labels above the bars: the tallest bar plus its label
    # has to stay inside the axes frame, so set the limit before annotating and
    # scale the label offset to the axis range (not a fixed number of percent).
    y_max = float(np.nanmax(hit_rate_df["hit_rate_pct"].to_numpy(dtype=float)))
    y_top = y_max * 1.12 if y_max > 0 else 1.0
    ax.set_ylim(0, y_top)
    label_offset = 0.02 * y_top
    for i, method in enumerate(("CALIPER", "RWR")):
        sub = hit_rate_df[hit_rate_df["method"] == method].set_index("degree_bin").loc[bin_labels]
        bars = ax.bar(
            x + (i - 0.5) * width,
            sub["hit_rate_pct"],
            width,
            label=_method_display(method),
            color=METHOD_COLOR[method],
        )
        for bar, value in zip(bars, sub["hit_rate_pct"]):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + label_offset,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )
    ax.set_xticks(x)
    # n_positive is shared across methods within a bin (same positive set); annotate
    # it under each bin label so the hit-rate % reads against its denominator.
    n_by_bin = hit_rate_df.drop_duplicates("degree_bin").set_index("degree_bin")["n_positive"]
    ax.set_xticklabels(
        [f"{lab}\nn={int(n_by_bin.get(lab, 0))}" for lab in bin_labels], rotation=0, fontsize=6.5
    )
    ax.set_ylabel("% of known positives in top-10")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)
    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png), hit_rate_df


def _expected_ap(y: np.ndarray, s: np.ndarray) -> float:
    """Tie-aware expected average precision (val2's default tie policy) so this
    figure's AP matches the AP the main val2_baseline pipeline reports. Tied score
    blocks are treated as uniformly randomized internally rather than broken by an
    arbitrary sort order."""
    return float(expected_average_precision_with_ties(y, s)[0])


def _median_ci95(vals: np.ndarray) -> tuple[float, float]:
    """Distribution-free 95% CI for the median via the normal approximation to
    the binomial order statistics (deterministic; no resampling)."""
    v = np.sort(np.asarray(vals, dtype=float))
    n = v.size
    if n < 8:
        return (float("nan"), float("nan"))
    z = 1.959963985
    j = max(int(np.floor(n / 2.0 - z * np.sqrt(n) / 2.0)), 1)
    k = min(int(np.ceil(1.0 + n / 2.0 + z * np.sqrt(n) / 2.0)), n)
    return (float(v[j - 1]), float(v[k - 1]))


def plot_recovery_percentile_by_degree(
    feat: pd.DataFrame, run_root: Path, out_png: Path, n_bins: int = 10, xscale: str = "log"
) -> tuple[Dict[str, str], pd.DataFrame]:
    """Median within-disease score percentile of known positives vs target degree
    (equal-count degree deciles). Direct evidence that CALIPER's recovery is
    degree-robust (roughly flat across degree) while RWR's is strongly
    degree-dependent (rises steeply with degree). The Spearman rho of (degree,
    score) per method (reported in degree_diagnostics_spearman.tsv) quantifies
    this: rho~0 means degree-independent (even) recovery, high rho means
    degree-biased. Bands are distribution-free 95% CIs of the median.
    """
    positive = feat[feat["positive"]].copy()
    if positive.empty:
        return {}, pd.DataFrame()
    try:
        positive["ddec"] = pd.qcut(positive["degree_H"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        return {}, pd.DataFrame()

    rows = []
    for dec, grp in positive.groupby("ddec"):
        deg_med = float(grp["degree_H"].median())
        for method in ("CALIPER", "RWR"):
            pct = (1.0 - grp[f"{method}_rank_pct"]).to_numpy() * 100.0
            lo, hi = _median_ci95(pct)
            rows.append(
                {
                    "degree_decile": int(dec),
                    "degree_median": deg_med,
                    "method": method,
                    "median_percentile": float(np.median(pct)),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_positive": int(len(grp)),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return {}, df

    _apply_paper_rcparams()
    fig, ax = plt.subplots(figsize=METHOD_COMPARISON_NARROW_FIGSIZE, constrained_layout=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(50, color="#999999", linestyle=":", linewidth=0.7, zorder=1)
    for method in ("CALIPER", "RWR"):
        sub = df[df["method"] == method].sort_values("degree_median")
        ax.fill_between(sub["degree_median"], sub["ci_lo"], sub["ci_hi"], color=METHOD_COLOR[method], alpha=0.15, linewidth=0, zorder=2)
        ax.plot(sub["degree_median"], sub["median_percentile"], marker="o", markersize=3, color=METHOD_COLOR[method], label=_method_display(method), zorder=3)
    if xscale == "log":
        ax.set_xscale("log")
    ax.set_xlabel("target degree" + (" (log scale)" if xscale == "log" else ""))
    ax.set_ylabel("median score percentile\nof known positives")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.grid(axis="y", alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)
    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png), df


def plot_degree_matched_ap_macro(
    feat: pd.DataFrame, run_root: Path, out_png: Path, xscale: str = "linear", n_bins: int = 10
) -> tuple[Dict[str, str], pd.DataFrame]:
    """Per-disease AP averaged across diseases (macro) within each equal-count
    degree decile — the disease-level aggregation the main val2_baseline pipeline
    uses. For each decile the tie-aware expected AP is computed per disease (over
    that disease's candidates in the decile, diseases with >=1 positive) then meaned
    across diseases; the random line is the mean per-disease prevalence; bands are
    mean +/- 1.96*SEM.
    """
    df_all = feat.copy()
    try:
        df_all["ddec"] = pd.qcut(df_all["degree_H"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        return {}, pd.DataFrame()

    rows = []
    for dec, grp in df_all.groupby("ddec"):
        deg_med = float(grp["degree_H"].median())
        per_disease = {"CALIPER": [], "RWR": []}
        prevs = []
        for _, gd in grp.groupby("disease_id"):
            y = gd["positive"].to_numpy()
            if y.sum() == 0:
                continue
            prevs.append(float(y.mean()))
            for name in ("CALIPER", "RWR"):
                per_disease[name].append(_expected_ap(y, gd[name].to_numpy()))
        n_dz = len(prevs)
        if n_dz == 0:
            continue
        for name in ("CALIPER", "RWR"):
            vals = np.asarray(per_disease[name], dtype=float)
            mean = float(vals.mean())
            sem = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
            half = 1.959963985 * sem
            rows.append(
                {
                    "degree_decile": int(dec),
                    "degree_median": deg_med,
                    "method": name,
                    "ap_macro": mean,
                    "ci_lo": mean - half,
                    "ci_hi": mean + half,
                    "random_prevalence": float(np.mean(prevs)),
                    "n_diseases": int(n_dz),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty or not np.isfinite(df["ap_macro"]).any():
        return {}, df

    _apply_paper_rcparams()
    fig, ax = plt.subplots(figsize=(4.6, 2.9), constrained_layout=True)
    prev = df.drop_duplicates("degree_decile").sort_values("degree_median")
    ax.plot(prev["degree_median"], prev["random_prevalence"], linestyle=":", color="#999999", linewidth=0.9, label="random", zorder=1)
    for name in ("CALIPER", "RWR"):
        sub = df[df["method"] == name].sort_values("degree_median")
        ax.fill_between(sub["degree_median"], sub["ci_lo"], sub["ci_hi"], color=METHOD_COLOR[name], alpha=0.15, linewidth=0, zorder=2)
        ax.plot(sub["degree_median"], sub["ap_macro"], marker="o", markersize=3, color=METHOD_COLOR[name], label=_method_display(name), zorder=3)
    if xscale == "log":
        ax.set_xscale("log")
    ax.set_xlabel("target degree" + (" (log scale)" if xscale == "log" else ""))
    ax.set_ylabel("mean per-disease AP")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.grid(axis="y", alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)
    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png), df


def _compute_degree_corrected_ap(
    feat: pd.DataFrame, methods: List[str], n_bins: int = 10, n_boot: int = 1000, seed: int = 20260711
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-disease raw vs degree-standardized AP per method (one value per disease),
    plus paired ΔAP (CALIPER - baseline).

    Each disease gets a degree-standardized AP = mean (equal weight) of its expected
    AP within each global degree-decile it has positives in — removing the degree
    confound while keeping the disease as the unit (parallel to the raw per-disease
    mAP used by method_comparison). ``per_method_df`` carries mean and SEM across
    diseases (so error bars match the SEM used by plot_method_comparison);
    ``delta_df`` carries the paired mean Δ with disease-bootstrap 95% CIs.
    """
    df = feat.copy()
    try:
        df["ddec"] = pd.qcut(df["degree_H"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame(), pd.DataFrame()

    raw: Dict[str, Dict[str, float]] = {m: {} for m in methods}
    std: Dict[str, Dict[str, float]] = {m: {} for m in methods}
    for dz, g in df.groupby(df["disease_id"].astype(str)):
        if g["positive"].sum() == 0:
            continue
        y = g["positive"].to_numpy()
        for m in methods:
            raw[m][dz] = _expected_ap(y, g[m].to_numpy())
        dec_ap = {m: [] for m in methods}
        for _, gd in g.groupby("ddec"):
            if gd["positive"].sum() > 0:
                yy = gd["positive"].to_numpy()
                for m in methods:
                    dec_ap[m].append(_expected_ap(yy, gd[m].to_numpy()))
        for m in methods:
            std[m][dz] = float(np.mean(dec_ap[m])) if dec_ap[m] else float("nan")

    dz_list = [dz for dz in raw[methods[0]]]
    if not dz_list:
        return pd.DataFrame(), pd.DataFrame()
    arr = lambda d: np.array([d.get(dz, np.nan) for dz in dz_list], dtype=float)

    def mean_sem(a):
        a = a[np.isfinite(a)]
        n = a.size
        return (float(a.mean()), float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)

    per_method_rows = []
    for m in methods:
        rmean, rsem = mean_sem(arr(raw[m]))
        smean, ssem = mean_sem(arr(std[m]))
        per_method_rows.append({"method": m, "raw_ap": rmean, "raw_sem": rsem, "std_ap": smean, "std_sem": ssem})
    per_method_df = pd.DataFrame(per_method_rows)

    # Delta uses the median + bootstrap-median CI + Wilcoxon signed-rank: this is
    # the degree-diagnostics estimator shared with the per-disease degree tables
    # (degree_support), not the k=4 paired t-test that decides significance in
    # the method-comparison figures.
    rng = np.random.default_rng(seed)
    raw_cal, std_cal = arr(raw["CALIPER"]), arr(std["CALIPER"])

    def med_ci_p(dd):
        return degree_support.paired_median_ci_p(dd, rng, n_boot)

    delta_rows = []
    for b in [m for m in methods if m != "CALIPER"]:
        rmed, rlo, rhi, rp = med_ci_p(raw_cal - arr(raw[b]))
        smed, slo, shi, sp = med_ci_p(std_cal - arr(std[b]))
        delta_rows.append(
            {
                "baseline": b,
                "raw_delta": rmed, "raw_lo": rlo, "raw_hi": rhi, "raw_wilcoxon_p": rp,
                "std_delta": smed, "std_lo": slo, "std_hi": shi, "std_wilcoxon_p": sp,
            }
        )
    return per_method_df, pd.DataFrame(delta_rows)


def plot_method_comparison_ap_degree_corrected(
    per_method_df: pd.DataFrame, run_root: Path, out_png: Path, methods: tuple = ("CALIPER", "RWR")
) -> Dict[str, str]:
    """Per-method degree-standardized AP (mean +/- SEM across diseases) — the
    degree-corrected analogue of method_comparison_ap_caliper_vs_classical_rwr,
    using the same mean+/-SEM error bars as plot_method_comparison."""
    present = [m for m in methods if m in set(per_method_df["method"])]
    sub = per_method_df.set_index("method").reindex(present).reset_index()
    if sub.empty or not np.isfinite(sub["std_ap"]).any():
        return {}
    means = sub["std_ap"].astype(float).to_numpy()
    sems = sub["std_sem"].astype(float).to_numpy()
    _apply_paper_rcparams()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=METHOD_COMPARISON_NARROW_FIGSIZE, constrained_layout=True)
    x = np.arange(len(sub), dtype=float)
    for i, row in sub.iterrows():
        color = _method_color(row["method"], i)
        ax.errorbar(
            x[i], row["std_ap"], yerr=row["std_sem"], fmt="o", markersize=3.0, markerfacecolor=color, markeredgecolor="black",
            markeredgewidth=PAPER_MARKER_EDGE_WIDTH, color=color, ecolor=color, elinewidth=PAPER_ERROR_LINEWIDTH,
            capsize=PAPER_ERROR_CAPSIZE, capthick=PAPER_ERROR_LINEWIDTH, zorder=3,
        )
    ax.set_xlim(-0.35, len(sub) - 1 + 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([_method_display(m) for m in sub["method"]], rotation=20, ha="right")
    ax.set_ylabel("Mean degree-standardized AP")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(PAPER_AXIS_LINEWIDTH)
    y_lows = [m - s for m, s in zip(means, sems) if np.isfinite(m)]
    y_highs = [m + s for m, s in zip(means, sems) if np.isfinite(m)]
    if y_lows and y_highs:
        y_low, y_high = min(y_lows), max(y_highs)
        y_span = max(y_high - y_low, 1e-9)
        ax.set_ylim(max(0.0, y_low - 0.12 * y_span), y_high + 0.10 * y_span)
    ax.grid(axis="y", alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)
    if ax.get_ylim()[0] > 0:
        _draw_y_axis_break(ax)
    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png)


def plot_effect_size_forest_ap_degree_corrected(delta_df: pd.DataFrame, run_root: Path, out_png: Path) -> Dict[str, str]:
    """ΔAP (CALIPER - baseline) raw vs degree-standardized, with disease-bootstrap
    95% CI — the degree-corrected analogue of effect_size_forest_ap. Shows that the
    degree correction shifts (and for RWR flips) the effect: raw ~0/negative,
    degree-corrected positive. `*` = degree-corrected 95% CI excludes 0."""
    if delta_df.empty:
        return {}
    order = [b for b in ("RWR",) if b in set(delta_df["baseline"])]
    sub = delta_df.set_index("baseline").reindex(order).reset_index()
    if sub.empty or not np.isfinite(sub["std_delta"]).any():
        return {}
    _apply_paper_rcparams()
    fig, ax = plt.subplots(figsize=(max(4.2, 1.2 * len(sub) + 2.4), 2.8), constrained_layout=True)
    x = np.arange(len(sub), dtype=float)
    off = 0.16
    raw_err = np.clip(np.vstack([sub["raw_delta"] - sub["raw_lo"], sub["raw_hi"] - sub["raw_delta"]]), 0.0, None)
    std_err = np.clip(np.vstack([sub["std_delta"] - sub["std_lo"], sub["std_hi"] - sub["std_delta"]]), 0.0, None)
    ax.errorbar(
        x - off, sub["raw_delta"], yerr=raw_err, fmt="s", markersize=3.0, color="#999999", ecolor="#999999",
        elinewidth=PAPER_ERROR_LINEWIDTH, capsize=PAPER_ERROR_CAPSIZE, capthick=PAPER_ERROR_LINEWIDTH, label="raw", zorder=3,
    )
    ax.errorbar(
        x + off, sub["std_delta"], yerr=std_err, fmt="o", markersize=3.0, color="#0072B2", ecolor="#0072B2",
        elinewidth=PAPER_ERROR_LINEWIDTH, capsize=PAPER_ERROR_CAPSIZE, capthick=PAPER_ERROR_LINEWIDTH,
        label="degree-corrected", zorder=3,
    )
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=PAPER_AXIS_LINEWIDTH)
    for i, row in sub.iterrows():
        if np.isfinite(row["std_lo"]) and (row["std_lo"] > 0 or row["std_hi"] < 0):
            ax.annotate("*", (x[i] + off, row["std_hi"]), ha="center", va="bottom", fontsize=9, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"CALIPER − {_method_display(b)}" for b in sub["baseline"]], rotation=20, ha="right")
    ax.set_xlim(-0.5, len(sub) - 0.5)
    ax.set_ylabel("median ΔAP\n(95% CI, disease bootstrap)")
    ax.legend(frameon=False, fontsize=6.5, loc="best")
    ax.grid(axis="y", alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)
    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png)


def _draw_jitter_v(ax, data: List[np.ndarray], seed: int = 0, scatter_max: int | None = 400) -> None:
    """Deterministic jitter strip alongside a vertical boxplot at x = 1..n."""
    for i, vals in enumerate(data, start=1):
        pts = np.asarray(vals, dtype=float)
        pts = pts[np.isfinite(pts)]
        if not pts.size:
            continue
        if scatter_max is not None and pts.size > scatter_max:
            pts = pts[np.random.default_rng(seed).choice(pts.size, size=int(scatter_max), replace=False)]
        jitter = (np.random.default_rng(seed).random(pts.size) - 0.5) * 0.22
        ax.scatter(np.full(pts.size, float(i)) + jitter, pts, s=5, alpha=0.3, color="#333333", edgecolors="none", zorder=4)


def plot_degree_spearman_by_disease(
    by_disease: pd.DataFrame,
    run_root: Path,
    out_png: Path,
    *,
    candidate_set: str = "all",
    methods: tuple = ("CALIPER", "RWR"),
) -> Dict[str, str]:
    """Within-disease Spearman ρ(degree, score) per method, with the disease as the unit.

    Single paper panel: one box per method over the ρ computed *inside* each
    disease (box = quartiles, strip = one point per disease), ρ on the y axis and
    the method on the x axis. The pooled ρ (one correlation over all pairs of all
    diseases) is left to the report table, since between-disease differences in
    degree and prevalence can inflate or deflate it. ``candidate_set='all'``
    correlates over every common candidate of the disease; ``'positive'``
    restricts to known positives (pooled-table definition).
    """
    if by_disease.empty:
        return {}
    sel = by_disease[(by_disease["candidate_set"] == candidate_set) & by_disease["included"].astype(bool)]
    if sel.empty:
        return {}
    present = set(sel["method"].astype(str))
    methods = [m for m in methods if m in present]
    if not methods:
        return {}
    data = [sel.loc[sel["method"].eq(m), "spearman_rho"].dropna().to_numpy(dtype=float) for m in methods]
    if not any(len(d) for d in data):
        return {}

    _apply_paper_rcparams()
    fig, ax = plt.subplots(figsize=METHOD_COMPARISON_NARROW_FIGSIZE, constrained_layout=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    colors = [_method_color(m, i) for i, m in enumerate(methods)]
    _draw_box(ax, data, colors, [_method_display(m) for m in methods])
    _draw_jitter_v(ax, data)
    ax.axhline(0.0, color="#999999", linestyle=":", linewidth=0.7, zorder=1)
    # Keep ρ = 0 (the no-degree-dependence reference) in view without spending
    # half the panel on the unused part of the [-1, 1] range.
    vals = np.concatenate([d for d in data if len(d)])
    pad = max(0.05, 0.08 * float(np.ptp(vals)))
    ax.set_ylim(min(0.0, float(vals.min())) - pad, max(0.0, float(vals.max())) + pad)
    ax.set_ylabel("within-disease ρ (degree, score)")
    ax.grid(axis="y", alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)

    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png)


def plot_topk_overlap(
    overlap_by_disease: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    run_root: Path,
    out_png: Path,
    *,
    method_a: str = "CALIPER",
    method_b: str = "RWR",
) -> Dict[str, str]:
    """How little the two methods' top-K candidate sets overlap (Venn per K).

    One area-weighted Venn per K over the pooled top-K (disease, target) pairs
    (label-free set definition: ``rank_min <= K``): method-exclusive lobes and
    the shared intersection. A near-empty intersection means the two rankings
    surface largely different candidates, which is what makes the support
    profile of the exclusive candidates worth tabulating. The per-disease
    Jaccard and the group profiles stay in the report tables.
    """
    if overlap_by_disease.empty or overlap_summary.empty:
        return {}
    summary = overlap_summary.sort_values("K")
    k_values = [int(k) for k in summary["K"]]

    _apply_paper_rcparams()
    fig, axes = plt.subplots(
        1, len(k_values), figsize=(2.4 * len(k_values), 2.4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, k_values):
        row = summary[summary["K"].eq(k)].iloc[0]
        subsets = (
            int(row["pooled_n_caliper_only"]),
            int(row["pooled_n_rwr_only"]),
            int(row["pooled_n_both"]),
        )
        v = venn2(
            subsets=subsets,
            # Set labels name the two top-K sets; the exclusive counts sit inside
            # the non-overlapping lobes and the shared count in the intersection.
            set_labels=(_method_display(method_a), _method_display(method_b)),
            set_colors=(METHOD_COLOR[method_a], METHOD_COLOR[method_b]),
            alpha=0.55,
            ax=ax,
        )
        for label in (v.set_labels or []):
            if label is not None:
                label.set_fontsize(7)
        for sid in ("10", "01", "11"):
            label = v.get_label_by_id(sid)
            if label is not None:
                label.set_fontsize(6.5)
        ax.set_title(f"K = {k}", fontsize=8)

    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png)


def plot_topk_overlap_degree_support(
    overlap_summary: pd.DataFrame,
    group_pairs: pd.DataFrame,
    run_root: Path,
    out_png: Path,
    *,
    k: int = 10,
) -> Dict[str, str]:
    """Approved FigS3b view: pooled overlap plus exclusive-pair profiles."""
    if overlap_summary.empty or group_pairs.empty:
        return {}
    overlap_at_k = overlap_summary[
        pd.to_numeric(overlap_summary["K"], errors="coerce").eq(k)
    ]
    pairs_at_k = group_pairs[
        pd.to_numeric(group_pairs["K"], errors="coerce").eq(k)
    ].copy()
    if overlap_at_k.empty or pairs_at_k.empty:
        return {}
    overlap = overlap_at_k.iloc[0]

    _apply_paper_rcparams()
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = ["#0072B2", "#CC79A7"]
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.2, 2.4),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0, 1.0]},
        constrained_layout=True,
    )
    venn = venn2(
        subsets=(
            int(overlap["pooled_n_caliper_only"]),
            int(overlap["pooled_n_rwr_only"]),
            int(overlap["pooled_n_both"]),
        ),
        set_labels=(f"CALIPER\ntop-{k}", f"Classical RWR\ntop-{k}"),
        set_colors=tuple(colors),
        alpha=0.35,
        ax=axes[0],
    )
    for patch in venn.patches:
        if patch is not None:
            patch.set_edgecolor("#333333")
            patch.set_linewidth(0.7)
    for label in venn.subset_labels:
        if label is not None:
            label.set_fontsize(7)
    for label in venn.set_labels:
        if label is not None:
            label.set_fontsize(7)
    axes[0].set_axis_off()

    metrics = [
        ("degree_H", "target degree"),
        ("mean_support", "evaluable genes"),
        ("n_common_cells", "evaluable cell lines"),
    ]
    groups = ["caliper_only", "rwr_only"]
    group_labels = ["CALIPER-\nonly", "Classical\nRWR-only"]
    for ax, (metric, ylabel) in zip(axes[1:], metrics):
        data = [
            pd.to_numeric(
                pairs_at_k.loc[
                    pairs_at_k["group"].astype(str).eq(group),
                    metric,
                ],
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
            for group in groups
        ]
        boxplot = ax.boxplot(
            data,
            tick_labels=group_labels,
            patch_artist=True,
            showfliers=False,
            widths=0.5,
            zorder=3,
        )
        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor("#333333")
        for median in boxplot["medians"]:
            median.set_color("#111111")
            median.set_linewidth(1.2)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22, linewidth=0.35)
        ax.set_axisbelow(True)

    _save_figure(fig, out_png)
    plt.close(fig)
    return _rel_outputs(run_root, out_png)


def write_pooled_ap_table(run_root: Path) -> Dict[str, str]:
    """Pooled (micro) expected AP per method over ALL common (disease, target)
    pairs — scores ranked across diseases together, no per-disease aggregation.
    Complements the per-disease-mean (macro) AP shown in method_comparison_ap.
    """
    long_df = _read_tsv(run_root / "evaluation" / "scores_common_long.tsv")
    required = {"disease_id", "target_gene", "method", "score", "label"}
    if long_df.empty or not required.issubset(long_df.columns):
        return {}
    present = set(long_df["method"].astype(str))
    ordered = [m for m in METHOD_ORDER if m in present] + sorted(present - set(METHOD_ORDER))
    rows = []
    for m in ordered:
        sub = long_df[long_df["method"].astype(str) == m]
        y = sub["label"].astype(int).to_numpy()
        s = sub["score"].astype(float).to_numpy()
        if y.sum() == 0:
            continue
        rows.append(
            {
                "method": _method_display(m),
                "pooled_AP_micro": _expected_ap(y, s),
                "random_prevalence": float(y.mean()),
                "n_pos": int(y.sum()),
                "n_total": int(y.size),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    eval_dir = run_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out = eval_dir / "pooled_ap_micro.tsv"
    df.to_csv(out, sep="\t", index=False)
    return _rel_file_output(run_root, "pooled_ap_micro", out)


def plot_degree_diagnostics(run_root: Path, fig_dir: Path) -> Dict[str, str]:
    """CALIPER-vs-RWR target-degree diagnostics (shared network H).

    Checks whether CALIPER recovers known positives at low target degree in H
    that a degree/hub-biased method (RWR) tends to miss. Skips silently
    (returns {}) if RWR is absent from this run or the KG-derived degree
    cannot be resolved.
    """
    run_root = Path(run_root)
    feat = _build_degree_diagnostics_frame(run_root)
    if feat is None:
        return {}

    rel: Dict[str, str] = {}
    rel.update(plot_degree_distribution_by_label(feat, run_root, fig_dir / "degree_distribution_negative_vs_positive.png"))
    rel.update(plot_degree_distribution_top10(feat, run_root, fig_dir / "degree_distribution_top10_by_method.png"))
    rel.update(plot_positive_degree_vs_score_hexbin(feat, run_root, fig_dir / "positive_degree_vs_score_hexbin.png"))
    rel.update(plot_positive_degree_vs_score_hexbin_linear(feat, run_root, fig_dir / "positive_degree_vs_score_hexbin_linear.png"))
    rel.update(plot_positive_degree_vs_score_percentile(feat, run_root, fig_dir / "positive_degree_vs_score_percentile.png"))
    rel.update(plot_positive_degree_histogram(feat, run_root, fig_dir / "positive_degree_histogram.png"))
    hit_rel, hit_rate_df = plot_hit_rate_by_degree_bin(feat, run_root, fig_dir / "hit_rate_by_degree_bin.png")
    rel.update(hit_rel)
    recovery_rel, recovery_df = plot_recovery_percentile_by_degree(feat, run_root, fig_dir / "recovery_percentile_by_degree.png", xscale="log")
    rel.update(recovery_rel)
    ap_macro_rel, ap_macro_df = plot_degree_matched_ap_macro(feat, run_root, fig_dir / "degree_matched_ap_macro_by_degree_linear.png", xscale="log")
    rel.update(ap_macro_rel)

    eval_dir = run_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    if not hit_rate_df.empty:
        hit_rate_path = eval_dir / "degree_diagnostics_hit_rate_by_bin.tsv"
        hit_rate_df.to_csv(hit_rate_path, sep="\t", index=False)
        rel.update(_rel_file_output(run_root, "degree_diagnostics_hit_rate_by_bin", hit_rate_path))
    if not recovery_df.empty:
        recovery_path = eval_dir / "degree_diagnostics_recovery_by_degree.tsv"
        recovery_df.to_csv(recovery_path, sep="\t", index=False)
        rel.update(_rel_file_output(run_root, "degree_diagnostics_recovery_by_degree", recovery_path))
    if not ap_macro_df.empty:
        ap_macro_path = eval_dir / "degree_diagnostics_degree_matched_ap_macro.tsv"
        ap_macro_df.to_csv(ap_macro_path, sep="\t", index=False)
        rel.update(_rel_file_output(run_root, "degree_diagnostics_degree_matched_ap_macro", ap_macro_path))

    # Degree-corrected (direct-standardized, per-disease/macro) AP: separate figures
    # analogous to method_comparison_ap_caliper_vs_classical_rwr and effect_size_forest_ap.
    dc_methods = [m for m in ("CALIPER", "RWR", "NCS", "proximity") if m in feat.columns]
    per_method_df, delta_df = _compute_degree_corrected_ap(feat, dc_methods)
    if not per_method_df.empty:
        rel.update(
            plot_method_comparison_ap_degree_corrected(
                per_method_df, run_root, fig_dir / "method_comparison_ap_degree_corrected_caliper_vs_classical_rwr.png"
            )
        )
        rel.update(
            plot_effect_size_forest_ap_degree_corrected(
                delta_df, run_root, fig_dir / "effect_size_forest_ap_degree_corrected.png"
            )
        )
        dc_method_path = eval_dir / "degree_corrected_ap_by_method.tsv"
        per_method_df.to_csv(dc_method_path, sep="\t", index=False)
        rel.update(_rel_file_output(run_root, "degree_corrected_ap_by_method", dc_method_path))
        dc_delta_path = eval_dir / "degree_corrected_ap_delta.tsv"
        delta_df.to_csv(dc_delta_path, sep="\t", index=False)
        rel.update(_rel_file_output(run_root, "degree_corrected_ap_delta", dc_delta_path))

    from scipy.stats import mannwhitneyu  # type: ignore

    top10 = {m: feat["positive"] & feat[f"{m}_rank_min"].le(10) for m in ("CALIPER", "RWR")}
    groups = ["All positives", "CALIPER top10", "RWR top10"]
    masks = {"All positives": feat["positive"], "CALIPER top10": top10["CALIPER"], "RWR top10": top10["RWR"]}

    summary_rows = []
    for g in groups:
        vals = feat.loc[masks[g], "degree_H"].dropna()
        if vals.empty:
            continue
        desc = vals.describe(percentiles=[0.25, 0.5, 0.75])
        summary_rows.append({"group": g, **{k: float(v) for k, v in desc.items()}})
    degree_summary = pd.DataFrame(summary_rows)
    if not degree_summary.empty:
        degree_summary_path = eval_dir / "degree_diagnostics_summary.tsv"
        degree_summary.to_csv(degree_summary_path, sep="\t", index=False)
        rel.update(_rel_file_output(run_root, "degree_diagnostics_summary", degree_summary_path))

    neg = feat.loc[~feat["positive"], "degree_H"].dropna()
    pos = feat.loc[feat["positive"], "degree_H"].dropna()
    if len(neg) and len(pos):
        _, p = mannwhitneyu(neg, pos, alternative="two-sided")
        neg_vs_pos_path = eval_dir / "degree_diagnostics_neg_vs_pos_test.tsv"
        pd.DataFrame(
            [{"group_a": "Non-positive", "group_b": "Positive", "n_a": len(neg), "n_b": len(pos), "mannwhitney_p": p}]
        ).to_csv(neg_vs_pos_path, sep="\t", index=False)
        rel.update(_rel_file_output(run_root, "degree_diagnostics_neg_vs_pos_test", neg_vs_pos_path))

    pair_tests = [("All positives", "CALIPER top10"), ("All positives", "RWR top10"), ("CALIPER top10", "RWR top10")]
    test_rows = []
    for g1, g2 in pair_tests:
        x1 = feat.loc[masks[g1], "degree_H"].dropna()
        x2 = feat.loc[masks[g2], "degree_H"].dropna()
        if len(x1) and len(x2):
            _, p = mannwhitneyu(x1, x2, alternative="two-sided")
            test_rows.append({"group_a": g1, "group_b": g2, "n_a": len(x1), "n_b": len(x2), "mannwhitney_p": p})
    test_df = pd.DataFrame(test_rows)
    if not test_df.empty:
        test_df["mannwhitney_p_bonferroni"] = (test_df["mannwhitney_p"] * len(test_df)).clip(upper=1.0)
        group_tests_path = eval_dir / "degree_diagnostics_group_tests.tsv"
        test_df.to_csv(group_tests_path, sep="\t", index=False)
        rel.update(_rel_file_output(run_root, "degree_diagnostics_group_tests", group_tests_path))

    positive = feat[feat["positive"]]
    spearman_rows = [
        {"method": method, "spearman_rho_degree_vs_score": positive["degree_H"].corr(positive[method], method="spearman")}
        for method in ("CALIPER", "RWR")
    ]
    spearman_path = eval_dir / "degree_diagnostics_spearman.tsv"
    pd.DataFrame(spearman_rows).to_csv(spearman_path, sep="\t", index=False)
    rel.update(_rel_file_output(run_root, "degree_diagnostics_spearman", spearman_path))

    rel.update(_degree_support_diagnostics(feat, run_root, fig_dir, eval_dir))
    return rel


def _degree_support_diagnostics(
    feat: pd.DataFrame, run_root: Path, fig_dir: Path, eval_dir: Path
) -> Dict[str, str]:
    """Per-disease degree dependence and method-exclusive top-K support profiles.

    Two additions to the pooled CALIPER-vs-RWR degree diagnostics above: the
    degree-score correlation recomputed with the disease as the unit of analysis,
    and a description of how the candidates each method uniquely puts in its
    top-K differ in degree, coverage and cell-line support. Computation lives in
    ``degree_support``; this only writes the tables and figures. The group
    profiles are written as tables only (the report tabulates them); the single
    figure of that part is the top-K overlap Venn.
    """
    rel: Dict[str, str] = {}

    by_disease_rho, rho_summary = degree_support.compute_per_disease_degree_spearman(
        feat, methods=("CALIPER", "RWR")
    )
    if not by_disease_rho.empty:
        rho_path = eval_dir / "degree_diagnostics_spearman_by_disease.tsv"
        by_disease_rho.to_csv(rho_path, sep="\t", index=False)
        rel.update(_rel_file_output(run_root, "degree_diagnostics_spearman_by_disease", rho_path))
        rho_summary_path = eval_dir / "degree_diagnostics_spearman_summary.tsv"
        rho_summary.to_csv(rho_summary_path, sep="\t", index=False)
        rel.update(_rel_file_output(run_root, "degree_diagnostics_spearman_summary", rho_summary_path))
        # Only the primary candidate_set = all view is drawn; the positives-only
        # numbers stay in the report table (degree_diagnostics_spearman_summary).
        rel.update(
            plot_degree_spearman_by_disease(
                by_disease_rho, run_root, fig_dir / "degree_spearman_by_disease.png", candidate_set="all"
            )
        )

    k_list = _detect_k_list(run_root)
    overlap_by_disease, overlap_summary = degree_support.compute_topk_overlap(feat, k_list=k_list)
    if not overlap_by_disease.empty:
        overlap_path = eval_dir / "topk_overlap_by_disease.tsv"
        overlap_by_disease.to_csv(overlap_path, sep="\t", index=False)
        rel.update(_rel_file_output(run_root, "topk_overlap_by_disease", overlap_path))
        overlap_summary_path = eval_dir / "topk_overlap_summary.tsv"
        overlap_summary.to_csv(overlap_summary_path, sep="\t", index=False)
        rel.update(_rel_file_output(run_root, "topk_overlap_summary", overlap_summary_path))
        rel.update(
            plot_topk_overlap(
                overlap_by_disease, overlap_summary, run_root, fig_dir / "topk_overlap_caliper_vs_rwr.png"
            )
        )

    support = degree_support.load_support_features(run_root)
    pair_df, profile_df, tests_df = degree_support.compute_topk_group_profile(
        feat, support, k_list=k_list
    )
    if not profile_df.empty:
        profile_path = eval_dir / "topk_group_profile.tsv"
        profile_df.to_csv(profile_path, sep="\t", index=False)
        rel.update(_rel_file_output(run_root, "topk_group_profile", profile_path))
        pairs_path = eval_dir / "topk_group_pairs.tsv"
        pair_df.to_csv(pairs_path, sep="\t", index=False)
        rel.update(_rel_file_output(run_root, "topk_group_pairs", pairs_path))
        if not tests_df.empty:
            tests_path = eval_dir / "topk_group_tests.tsv"
            tests_df.to_csv(tests_path, sep="\t", index=False)
            rel.update(_rel_file_output(run_root, "topk_group_tests", tests_path))
        rel.update(
            plot_topk_overlap_degree_support(
                overlap_summary,
                pair_df,
                run_root,
                fig_dir / "topk_overlap_degree_support_k10.png",
                k=10,
            )
        )
    return rel


def make_all_figures(run_root: str | Path) -> Dict[str, str]:
    run_root = Path(run_root)
    fig_dir = run_root / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    caliper_methods = _caliper_methods_for_run(run_root)
    multi_caliper = len(caliper_methods) > 1
    rel: Dict[str, str] = {}
    rel.update(plot_common_universe_flow(run_root, fig_dir / "common_universe_flow.png"))
    rel.update(plot_four_method_performance(run_root, fig_dir / "four_method_performance.png"))
    rel.update(plot_coverage_confounding(run_root, fig_dir / "coverage_confounding.png"))

    rel.update(
        plot_method_comparison(
            run_root,
            fig_dir / "method_comparison_ap.png",
            metric_col="AP",
            ylabel="Mean AP",
            title="Mean AP by method",
            mark_ttest=multi_caliper,
            horizontal_dot_whisker=True,
            xlim=MEAN_AP_SHARED_XLIM,
        )
    )
    metrics = _read_tsv(run_root / "evaluation" / "metrics_by_disease_method.tsv")
    present_methods = set(metrics["method"].astype(str)) if "method" in metrics.columns else set()
    # k_star-only companion of method_comparison_ap: a multi-k run keeps every
    # CALIPER@k series in the figure above, so also emit the single-k (k_star) view.
    k_star_method = _k_star_caliper_method(run_root, caliper_methods) if multi_caliper else None
    if k_star_method:
        baselines = [m for m in _method_order(present_methods) if not is_caliper_method(m)]
        rel.update(
            plot_method_comparison(
                run_root,
                fig_dir / f"method_comparison_ap_{_slugify_method(k_star_method)}.png",
                metric_col="AP",
                ylabel="Mean AP",
                title=f"Mean AP by method ({_method_display(k_star_method)})",
                mark_ttest=True,
                caliper_method=k_star_method,
                comparison_methods=[k_star_method] + baselines,
            )
        )
    if {"CALIPER", "RWR"}.issubset(present_methods):
        rel.update(
            plot_method_comparison(
                run_root,
                fig_dir / "method_comparison_ap_caliper_vs_classical_rwr.png",
                metric_col="AP",
                ylabel="Mean AP",
                title="Mean AP: CALIPER vs Classical RWR",
                comparison_methods=["CALIPER", "RWR"],
            )
        )
        rel.update(
            plot_score_violin_caliper_vs_rwr(
                run_root, fig_dir / "score_violin_caliper_vs_classical_rwr.png"
            )
        )
        rel.update(
            plot_ap_violin_caliper_vs_rwr(
                run_root, fig_dir / "ap_violin_caliper_vs_classical_rwr.png"
            )
        )
    # Lift@K (method comparison / effect size forest / win-rate) と AP win-rate の
    # figure は生成しない。数値は evaluation/*.tsv と report の表に残る
    # (plot_win_rate / plot_method_comparison 自体は残してあるので再開は容易)。
    rel.update(plot_effect_size_forest(run_root, fig_dir / "effect_size_forest_ap.png", metric="AP"))
    rel.update(plot_paired_delta_violin(run_root, fig_dir / "paired_delta_violin.png"))
    rel.update(plot_degree_diagnostics(run_root, fig_dir))
    rel.update(write_pooled_ap_table(run_root))
    return rel
