from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

DEFAULT_METRICS: tuple[str, ...] = ("jaccard",)

NULL_TYPE_LABELS: dict[str, str] = {
    "target_khop_restricted_paired_gene_label_permutation": "Gene-node shuffle",
    "global_paired_gene_label_permutation": "Global gene-label shuffle",
    "path_count_length_matched_structural_resampling": "Path-shuffle",
    # Both decoy controls carry the same figure label; a config error prevents them
    # from ever appearing in the same report.
    "topology_profile_matched_target_decoy_control": "Target-Decoy",
    "observed_target_pool_decoy_control": "Target-Decoy",
    "observed": "Observed",
}

NULL_TYPE_ORDER: tuple[str, ...] = (
    "target_khop_restricted_paired_gene_label_permutation",
    "path_count_length_matched_structural_resampling",
    "topology_profile_matched_target_decoy_control",
    "observed_target_pool_decoy_control",
    "global_paired_gene_label_permutation",
)

TARGET_DECOY_NULL_TYPE = "topology_profile_matched_target_decoy_control"
# The random suffix is a figure-only qualifier: the null_type string is the same for
# both selection modes, so the mode has to be passed in explicitly by the caller.
TARGET_DECOY_RANDOM_LABEL_SUFFIX = " (random)"


def format_null_type_label(null_type: Any, *, decoy_selection: Any = None) -> str:
    """Return a compact, publication-friendly label for a Val3 null model.

    The matched topology/profile control is the unqualified ``Target-Decoy``.
    ``decoy_selection="random"`` gets the ``(random)`` suffix; an unknown or missing
    selection in an older run stays unqualified rather than being guessed.
    """
    raw = str(null_type)
    label = NULL_TYPE_LABELS.get(raw, raw.replace("_", " ").strip().title())
    if raw == TARGET_DECOY_NULL_TYPE and str(decoy_selection) == "random":
        label += TARGET_DECOY_RANDOM_LABEL_SUFFIX
    return label


def format_metric_label(metric: Any) -> str:
    raw = str(metric)
    if raw.startswith("joint_"):
        raw = raw.removeprefix("joint_")
    labels = {
        "jaccard": "Jaccard",
    }
    return labels.get(raw, raw.replace("_", " ").strip().title())


def format_null_type_label_multiline(null_type: Any, *, decoy_selection: Any = None) -> str:
    """Return the publication label without line breaks for axis ticks."""
    return format_null_type_label(null_type, decoy_selection=decoy_selection)


def null_type_sort_key(null_type: Any) -> tuple[int, str]:
    raw = str(null_type)
    try:
        return (NULL_TYPE_ORDER.index(raw), raw)
    except ValueError:
        return (len(NULL_TYPE_ORDER), raw)

# Shared publication-ready defaults.
_VAL3_RCPARAMS: dict[str, Any] = {
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
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "path",
}


def apply_val3_paper_style() -> None:
    """Apply the shared paper-ready Val3 Matplotlib style."""
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="paper")
    except Exception:
        pass
    plt.rcParams.update(_VAL3_RCPARAMS)


def _draw_vertical_ci(
    ax: Any,
    xs: Sequence[float],
    lows: Sequence[float],
    highs: Sequence[float],
    *,
    cap_half_width: float,
    color: str = "#1f2933",
) -> None:
    """Draw exact vertical CI endpoints without forcing the interval through the point."""
    x = np.asarray(xs, dtype=float)
    lo = np.asarray(lows, dtype=float)
    hi = np.asarray(highs, dtype=float)
    ax.vlines(x, lo, hi, color=color, linewidth=0.8, zorder=4)
    ax.hlines(lo, x - cap_half_width, x + cap_half_width, color=color, linewidth=0.8, zorder=4)
    ax.hlines(hi, x - cap_half_width, x + cap_half_width, color=color, linewidth=0.8, zorder=4)


def _draw_horizontal_ci(
    ax: Any,
    ys: Sequence[float],
    lows: Sequence[float],
    highs: Sequence[float],
    *,
    cap_half_height: float = 0.08,
    color: str = "C0",
    label: str | None = None,
) -> None:
    """Draw exact horizontal CI endpoints without clipping them to the point estimate."""
    y = np.asarray(ys, dtype=float)
    lo = np.asarray(lows, dtype=float)
    hi = np.asarray(highs, dtype=float)
    ax.hlines(y, lo, hi, color=color, linewidth=1.0, zorder=3, label=label)
    ax.vlines(lo, y - cap_half_height, y + cap_half_height, color=color, linewidth=1.0, zorder=3)
    ax.vlines(hi, y - cap_half_height, y + cap_half_height, color=color, linewidth=1.0, zorder=3)


def _hist_axis_val3_observed_style(
    ax: Any,
    arr: np.ndarray,
    *,
    ylabel: str,
    hist_title: str = "Histogram (per-triplet values)",
) -> None:
    """ヒストグラム部分のみ（旧ノート ``plot_true_only_boxplot`` の右パネルと同じスタイル）。"""
    ax.hist(arr, bins=30, alpha=0.8, color="C0")
    ax.set_xlabel(ylabel)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)


def _draw_box_strip_val3_style(ax_box: Any, arr: np.ndarray, *, ylabel: str, title: str) -> None:
    """旧ノート ``plot_true_only_boxplot`` の左パネル（箱＋strip）。"""
    try:
        import seaborn as sns
    except Exception as sns_import_error:  # pragma: no cover
        raise RuntimeError("observed box/strip plots require seaborn; pip install seaborn") from sns_import_error

    n = int(len(arr))
    plot_df = pd.DataFrame({"value": arr, "group": "Observed"})
    sns.boxplot(
        data=plot_df,
        x="group",
        y="value",
        order=["Observed"],
        ax=ax_box,
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "x",
            "markeredgecolor": "black",
            "markerfacecolor": "black",
            "markersize": 3,
        },
    )
    sns.stripplot(
        data=plot_df,
        x="group",
        y="value",
        order=["Observed"],
        ax=ax_box,
        size=2,
        alpha=0.6,
        jitter=0.25,
        color="C0",
    )
    ax_box.set_xlabel("")
    ax_box.set_xticks([0])
    ax_box.set_xticklabels([f"Observed\n(n={n})"])
    ax_box.set_ylabel(ylabel)
    ax_box.grid(axis="y", alpha=0.25)


def _draw_true_only_box_and_hist_axes(
    ax_box: Any,
    ax_hist: Any,
    values: np.ndarray,
    *,
    ylabel: str,
    title: str,
    hist_title: str = "Histogram (per-triplet values)",
) -> None:
    """旧ノート ``plot_true_only_boxplot`` と同じ箱ひげ＋strip＋ヒスト（1 指標分）。"""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        for ax in (ax_box, ax_hist):
            ax.text(0.5, 0.5, "no valid values", ha="center", va="center", transform=ax.transAxes)
        return

    _draw_box_strip_val3_style(ax_box, arr, ylabel=ylabel, title=title)
    _hist_axis_val3_observed_style(ax_hist, arr, ylabel=ylabel, hist_title=hist_title)


def _plot_observed_metrics_box_hist_faceted_rows(
    metric_df: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    figsize_row: float = 2.4,
    figwidth: float = 7.2,
    hist_title: str = "Histogram (per-triplet values)",
) -> Any:
    """metric_df の各行を 1 観測として箱＋ヒストを縦に並べる（triplet 行／pert 集約行の両方に使用）。"""
    import matplotlib.pyplot as plt

    apply_val3_paper_style()
    present = [m for m in metrics if m in metric_df.columns]
    n = len(present)
    if n == 0:
        fig, ax = plt.subplots(figsize=(figwidth, figsize_row))
        ax.text(0.5, 0.5, "no metric columns", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    if n == 1:
        metric = present[0]
        fig, (ax_box, ax_hist) = plt.subplots(
            1,
            2,
            figsize=(figwidth, figsize_row),
            gridspec_kw={"width_ratios": [1.05, 1.0]},
        )
        s = pd.to_numeric(metric_df[metric], errors="coerce").dropna().to_numpy()
        metric_label = format_metric_label(metric)
        _draw_true_only_box_and_hist_axes(
            ax_box, ax_hist, s, ylabel=metric_label, title=metric_label, hist_title=hist_title
        )
    else:
        fig, axes = plt.subplots(
            n,
            2,
            figsize=(figwidth, figsize_row * n),
            gridspec_kw={"width_ratios": [1.05, 1.0]},
            squeeze=False,
        )
        for i, metric in enumerate(present):
            ax_box, ax_hist = axes[i, 0], axes[i, 1]
            s = pd.to_numeric(metric_df[metric], errors="coerce").dropna().to_numpy()
            metric_label = format_metric_label(metric)
            _draw_true_only_box_and_hist_axes(
                ax_box, ax_hist, s, ylabel=metric_label, title=metric_label, hist_title=hist_title
            )
    fig.tight_layout()
    return fig


def plot_observed_triplet_metric_box_hist_pair(
    triplet_true_df: pd.DataFrame,
    metric: str,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (7.2, 2.4),
) -> Any:
    """1 指標について箱＋ヒストの 1 行 2 列（旧 ``plot_true_only_boxplot`` と同レイアウト）。"""
    import matplotlib.pyplot as plt

    apply_val3_paper_style()
    fig, (ax_box, ax_hist) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.05, 1.0]}
    )
    if metric not in triplet_true_df.columns:
        for ax in (ax_box, ax_hist):
            ax.text(0.5, 0.5, f"column {metric!r} missing", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    s = pd.to_numeric(triplet_true_df[metric], errors="coerce").dropna().to_numpy()
    metric_label = format_metric_label(metric)
    ttl = title or metric_label
    _draw_true_only_box_and_hist_axes(ax_box, ax_hist, s, ylabel=metric_label, title=ttl)
    fig.tight_layout()
    return fig


def plot_observed_triplet_metrics_box_hist_faceted(
    triplet_true_df: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    figsize_row: float = 2.4,
    figwidth: float = 7.2,
) -> Any:
    """各指標ごとに [箱+strip | ヒスト] の行を縦に並べる（旧ノートと同フォーマット）。"""
    return _plot_observed_metrics_box_hist_faceted_rows(
        triplet_true_df,
        metrics=metrics,
        figsize_row=figsize_row,
        figwidth=figwidth,
        hist_title="Histogram (per-triplet values)",
    )


def aggregate_observed_metrics_by_pert(
    triplet_true_df: pd.DataFrame,
    *,
    pert_col: str = "target_id",
) -> pd.DataFrame:
    """triplet 行を pert 単位に平均（既定は ``target_id`` = ノート準備段の pert_id に相当）。

    出力は ``pert_id``（元の ``pert_col`` の値）、``n_triplets``、各指標の平均。
    """
    if triplet_true_df.empty:
        return pd.DataFrame()
    if pert_col not in triplet_true_df.columns:
        raise ValueError(f"triplet_true_df missing column {pert_col!r}")
    df = triplet_true_df.copy()
    metrics = [m for m in DEFAULT_METRICS if m in df.columns]
    if not metrics:
        return pd.DataFrame()
    g = df.groupby(pert_col, dropna=False)
    agg_spec: dict[str, Any] = {m: "mean" for m in metrics}
    if "triplet_id" in df.columns:
        agg_spec["triplet_id"] = "count"
    out = g.agg(agg_spec).reset_index()
    if "triplet_id" in out.columns:
        out = out.rename(columns={"triplet_id": "n_triplets"})
    else:
        out["n_triplets"] = g.size().values
    out = out.rename(columns={pert_col: "pert_id"})
    return out


def plot_observed_pert_metrics_box_hist_faceted(
    triplet_true_df: pd.DataFrame,
    *,
    pert_col: str = "target_id",
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    figsize_row: float = 2.4,
    figwidth: float = 7.2,
) -> Any:
    """triplet を pert_id（既定: ``target_id``）単位で平均したうえで、triplet 版と同レイアウトで可視化。"""
    pert_df = aggregate_observed_metrics_by_pert(triplet_true_df, pert_col=pert_col)
    return _plot_observed_metrics_box_hist_faceted_rows(
        pert_df,
        metrics=metrics,
        figsize_row=figsize_row,
        figwidth=figwidth,
        hist_title="Distribution across targets",
    )


def load_val3_triplet_outputs(run_root: str | Path, direction: str) -> dict[str, pd.DataFrame]:
    root = Path(run_root) / "evaluation" / direction
    files = {
        "triplet_true": root / "triplet_metrics_true.parquet",
        "diag_true": root / "cellpair_filter_diagnostics_true.parquet",
        "global_joint_true": root / "global_joint_triplet_metrics_true.parquet",
        "global_joint_baseline": root / "global_joint_triplet_metrics_baseline.parquet",
    }
    out = {k: pd.read_parquet(v) if v.exists() else pd.DataFrame() for k, v in files.items()}
    for key in ["cellpair_true", "cellpair_baseline", "triplet_baseline", "diag_baseline", "decoy_candidate_cellpair", "decoy_candidate_triplet", "decoy_candidate_diag"]:
        out[key] = pd.DataFrame()
    tsv_files = {
        "global_evaluable_triplets": root / "global_evaluable_triplets.tsv",
        "global_joint_triplet_comparison_sets": root / "global_joint_triplet_comparison_sets.tsv",
        "global_joint_triplet_comparison": root / "global_joint_triplet_comparison.tsv",
        "global_effect_size": root / "global_effect_size.tsv",
        "null1_local_shuffle_diagnostics": root / "null1_local_shuffle_diagnostics.tsv",
        "decoy_candidate_diagnostics": root / "decoy_candidate_diagnostics.tsv",
        "decoy_anchor_diagnostics": root / "decoy_anchor_diagnostics.tsv",
        "decoy_candidate_joint_metrics": root / "decoy_candidate_joint_metrics.tsv",
    }
    out.update({k: pd.read_csv(v, sep="\t") if v.exists() else pd.DataFrame() for k, v in tsv_files.items()})
    return out


def summarize_metric_distributions(comparison_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if comparison_df.empty:
        return pd.DataFrame()
    group_cols = ["null_type"] if "null_type" in comparison_df.columns else []
    grouped = comparison_df.groupby(group_cols, dropna=False) if group_cols else [("", comparison_df)]
    for key, sub in grouped:
        null_type = key if isinstance(key, str) else (key[0] if isinstance(key, tuple) else str(key))
        for metric in ["jaccard"]:
            if metric not in sub.columns:
                continue
            s = pd.to_numeric(sub[metric], errors="coerce")
            delta = pd.to_numeric(sub.get(f"delta_{metric}"), errors="coerce")
            p = pd.to_numeric(sub.get(f"p_{metric}"), errors="coerce")
            row = {
                "metric": metric,
                "n": int(s.notna().sum()),
                "observed_mean": float(s.mean()) if s.notna().any() else float("nan"),
                "observed_median": float(s.median()) if s.notna().any() else float("nan"),
                "delta_mean": float(delta.mean()) if delta.notna().any() else float("nan"),
                "delta_median": float(delta.median()) if delta.notna().any() else float("nan"),
                "p_le_0_05": int((p <= 0.05).sum()) if p.notna().any() else 0,
                "fraction_p_le_0_05": float((p <= 0.05).mean()) if p.notna().any() else float("nan"),
            }
            if group_cols:
                row["null_type"] = str(null_type)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_edge_count_diagnostics(triplet_true_df: pd.DataFrame, baseline_triplet_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, df in [("observed", triplet_true_df), ("baseline", baseline_triplet_df)]:
        for col in ["mean_n_drug_paths", "mean_n_pert_paths", "mean_n_drug_edges", "mean_n_pert_edges"]:
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            rows.append({"source": source, "quantity": col, "n": int(s.notna().sum()), "mean": float(s.mean()) if s.notna().any() else float("nan"), "median": float(s.median()) if s.notna().any() else float("nan")})
    return pd.DataFrame(rows)


def summarize_evaluable_pair_diagnostics(cellpair_diag_true: pd.DataFrame, cellpair_diag_baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, df in [("observed", cellpair_diag_true), ("baseline", cellpair_diag_baseline)]:
        if df.empty:
            continue
        raw = pd.to_numeric(df.get("n_cell_pairs_raw"), errors="coerce").sum()
        evaluable = pd.to_numeric(df.get("n_cell_pairs_evaluable"), errors="coerce").sum()
        rows.append({
            "source": source,
            "n_cell_pairs_raw": int(raw),
            "n_cell_pairs_evaluable": int(evaluable),
            "fraction_evaluable": float(evaluable / raw) if raw else float("nan"),
            "n_excluded_no_drug_path": int(pd.to_numeric(df.get("n_cell_pairs_excluded_no_drug_path"), errors="coerce").sum()),
            "n_excluded_no_pert_path": int(pd.to_numeric(df.get("n_cell_pairs_excluded_no_pert_path"), errors="coerce").sum()),
            "n_excluded_no_both_path": int(pd.to_numeric(df.get("n_cell_pairs_excluded_no_both_path"), errors="coerce").sum()),
        })
    return pd.DataFrame(rows)


def summarize_observed_triplet_metrics(
    triplet_true_df: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """triplet 集約（観測のみ）の各指標について基本統計量を返す。"""
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        if metric not in triplet_true_df.columns:
            continue
        s = pd.to_numeric(triplet_true_df[metric], errors="coerce")
        rows.append(
            {
                "metric": metric,
                "n": int(s.notna().sum()),
                "mean": float(s.mean()) if s.notna().any() else float("nan"),
                "median": float(s.median()) if s.notna().any() else float("nan"),
                "std": float(s.std()) if s.notna().sum() > 1 else float("nan"),
                "q25": float(s.quantile(0.25)) if s.notna().any() else float("nan"),
                "q75": float(s.quantile(0.75)) if s.notna().any() else float("nan"),
                "min": float(s.min()) if s.notna().any() else float("nan"),
                "max": float(s.max()) if s.notna().any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def plot_observed_triplet_metric_histogram(
    triplet_true_df: pd.DataFrame,
    metric: str,
    *,
    ax: Any = None,
    bins: int = 30,
    title: str | None = None,
) -> Any:
    """観測 triplet 集約の 1 指標のヒストグラムのみ（旧ノート右パネルと同スタイル）。"""
    import matplotlib.pyplot as plt

    apply_val3_paper_style()
    ax = ax or plt.gca()
    if metric not in triplet_true_df.columns:
        ax.text(0.5, 0.5, f"column {metric!r} missing", ha="center", va="center", transform=ax.transAxes)
        return ax
    s = pd.to_numeric(triplet_true_df[metric], errors="coerce").dropna()
    if s.empty:
        ax.text(0.5, 0.5, "no valid values", ha="center", va="center", transform=ax.transAxes)
        return ax
    arr = s.to_numpy()
    ax.clear()
    ax.hist(arr, bins=bins, alpha=0.8, color="C0")
    ax.set_xlabel(format_metric_label(metric))
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    return ax


def plot_observed_triplet_metrics_grid(
    triplet_true_df: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    bins: int = 30,
    figsize: tuple[float, float] = (12.0, 3.5),
) -> tuple[Any, Any]:
    """観測のみ: ヒストのみを横並び（スタイルは旧ノートのヒストパネルに準拠）。"""
    import matplotlib.pyplot as plt

    apply_val3_paper_style()
    present = [m for m in metrics if m in triplet_true_df.columns]
    n = max(len(present), 1)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0], figsize[1]), squeeze=False)
    axr = axes[0]
    for i, metric in enumerate(present if present else [metrics[0]]):
        plot_observed_triplet_metric_histogram(triplet_true_df, metric, ax=axr[i], bins=bins)
    if not present:
        axr[0].text(0.5, 0.5, "no metric columns", ha="center", va="center", transform=axr[0].transAxes)
    fig.tight_layout()
    return fig, axr


def plot_observed_triplet_metric_boxplot(
    triplet_true_df: pd.DataFrame,
    metric: str,
    *,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """観測 triplet 集約の箱＋strip のみ 1 軸（旧ノート左パネルと同スタイル）。"""
    import matplotlib.pyplot as plt

    apply_val3_paper_style()
    ax = ax or plt.gca()
    if metric not in triplet_true_df.columns:
        ax.text(0.5, 0.5, f"column {metric!r} missing", ha="center", va="center", transform=ax.transAxes)
        return ax
    s = pd.to_numeric(triplet_true_df[metric], errors="coerce").dropna()
    if s.empty:
        ax.text(0.5, 0.5, "no valid values", ha="center", va="center", transform=ax.transAxes)
        return ax
    ax.clear()
    arr = s.to_numpy()
    metric_label = format_metric_label(metric)
    ttl = title if title is not None else metric_label
    _draw_box_strip_val3_style(ax, arr, ylabel=metric_label, title=ttl)
    return ax


def plot_observed_triplet_metrics_boxplot_grid(
    triplet_true_df: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    figsize: tuple[float, float] = (12.0, 3.5),
) -> tuple[Any, Any]:
    """観測のみ: 箱＋strip のみを横並び。"""
    import matplotlib.pyplot as plt

    present = [m for m in metrics if m in triplet_true_df.columns]
    n = max(len(present), 1)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0], figsize[1]), squeeze=False)
    axr = axes[0]
    apply_val3_paper_style()
    for i, metric in enumerate(present if present else [metrics[0]]):
        plot_observed_triplet_metric_boxplot(triplet_true_df, metric, ax=axr[i])
    if not present:
        axr[0].text(0.5, 0.5, "no metric columns", ha="center", va="center", transform=axr[0].transAxes)
    fig.tight_layout()
    return fig, axr


def plot_metric_observed_vs_baseline(comparison_df: pd.DataFrame, metric: str = "jaccard", ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    x = comparison_df.get(f"baseline_median_{metric}")
    y = comparison_df.get(metric)
    ax.scatter(x, y, s=16)
    ax.set_xlabel(f"baseline_median_{metric}")
    ax.set_ylabel(metric)
    return ax


def plot_delta_metric_distribution(comparison_df: pd.DataFrame, metric: str = "jaccard", ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    s = pd.to_numeric(comparison_df.get(f"delta_{metric}"), errors="coerce").dropna()
    ax.hist(s, bins=30)
    ax.set_xlabel(f"delta_{metric}")
    ax.set_ylabel("count")
    return ax


def plot_pvalue_distribution(comparison_df: pd.DataFrame, metric: str = "jaccard", ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    s = pd.to_numeric(comparison_df.get(f"p_{metric}"), errors="coerce").dropna()
    ax.hist(s, bins=20, range=(0, 1))
    ax.set_xlabel(f"p_{metric}")
    ax.set_ylabel("count")
    return ax


def plot_edge_count_diagnostics(edge_diag_df: pd.DataFrame, ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    pivot = edge_diag_df.pivot_table(index="quantity", columns="source", values="median", aggfunc="first")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("median")
    return ax


def plot_evaluable_pair_diagnostics(pair_diag_df: pd.DataFrame, ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    plot_df = pair_diag_df.set_index("source")[["fraction_evaluable"]]
    plot_df.plot(kind="bar", ax=ax, legend=False)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction_evaluable")
    return ax


def plot_jaccard_observed_vs_null(
    summary_df: pd.DataFrame,
    *,
    evaluable_set_type: str | None = None,
    decoy_selection: str | None = None,
    figwidth: float = 7.2,
) -> Any:
    """Mean observed Jaccard versus the triplet-specific null median for three nulls."""
    import matplotlib.pyplot as plt

    apply_val3_paper_style()
    required = {
        "null_type",
        "evaluable_set_type",
        "observed_mean",
        "observed_mean_ci025",
        "observed_mean_ci975",
        "null_median_mean",
        "null_median_mean_ci025",
        "null_median_mean_ci975",
        "p_wilcoxon_two_sided_holm",
    }
    if summary_df.empty or not required <= set(summary_df.columns):
        return _empty_figure("no observed-versus-null summary rows", figheight=2.8)
    df = _filter_by_columns(
        summary_df,
        {"evaluable_set_type": evaluable_set_type},
    )
    numeric_columns = [
        "observed_mean",
        "observed_mean_ci025",
        "observed_mean_ci975",
        "null_median_mean",
        "null_median_mean_ci025",
        "null_median_mean_ci975",
        "p_wilcoxon_two_sided_holm",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["observed_mean", "null_median_mean"])
    if df.empty:
        return _empty_figure("no finite observed-versus-null estimates", figheight=2.8)

    df["_null_order"] = df["null_type"].map(null_type_sort_key)
    df = df.sort_values("_null_order", kind="stable").reset_index(drop=True)
    nulls = df["null_type"].astype(str).tolist()
    centers = np.arange(len(df), dtype=float)
    bar_width = 0.36
    observed_x = centers - bar_width / 2.0
    null_x = centers + bar_width / 2.0

    compact = figwidth <= 3.8
    fig, ax = plt.subplots(figsize=(figwidth, 2.4), constrained_layout=False)
    ax.bar(
        observed_x,
        df["observed_mean"].to_numpy(dtype=float),
        width=bar_width,
        color="#D55E00",
        edgecolor="#1f2933",
        linewidth=0.7,
        label="Observed",
        zorder=3,
    )
    ax.bar(
        null_x,
        df["null_median_mean"].to_numpy(dtype=float),
        width=bar_width,
        color="#F4C7AB",
        edgecolor="#1f2933",
        linewidth=0.7,
        label="Null",
        zorder=3,
    )
    _draw_vertical_ci(
        ax,
        observed_x,
        df["observed_mean_ci025"].to_numpy(dtype=float),
        df["observed_mean_ci975"].to_numpy(dtype=float),
        cap_half_width=bar_width * 0.10,
    )
    _draw_vertical_ci(
        ax,
        null_x,
        df["null_median_mean_ci025"].to_numpy(dtype=float),
        df["null_median_mean_ci975"].to_numpy(dtype=float),
        cap_half_width=bar_width * 0.10,
    )

    finite_ci_upper = np.concatenate([
        df["observed_mean_ci975"].to_numpy(dtype=float),
        df["null_median_mean_ci975"].to_numpy(dtype=float),
    ])
    finite_ci_upper = finite_ci_upper[np.isfinite(finite_ci_upper)]
    comparison_scale = max(float(np.max(finite_ci_upper)), 1.0e-6)
    bracket_gap = comparison_scale * 0.035
    bracket_height = comparison_scale * 0.025
    annotation_tops: list[float] = []
    for index, row in df.iterrows():
        upper = max(
            float(row["observed_mean_ci975"]),
            float(row["null_median_mean_ci975"]),
        )
        p_value = float(row["p_wilcoxon_two_sided_holm"])
        if np.isfinite(p_value):
            p_label = "P < 0.001" if p_value < 0.001 else f"P = {p_value:.3g}"
            bracket_y = upper + bracket_gap
            bracket_top = bracket_y + bracket_height
            ax.plot(
                [
                    float(observed_x[index]),
                    float(observed_x[index]),
                    float(null_x[index]),
                    float(null_x[index]),
                ],
                [bracket_y, bracket_top, bracket_top, bracket_y],
                color="#1f2933",
                linewidth=0.7,
                clip_on=False,
                zorder=4,
            )
            ax.text(
                float(centers[index]),
                bracket_top + bracket_gap * 0.15,
                p_label,
                ha="center",
                va="bottom",
                fontsize=7,
            )
            annotation_tops.append(bracket_top + bracket_gap)
        else:
            annotation_tops.append(upper)

    ax.set_xticks(centers)
    ax.set_xticklabels([
        format_null_type_label_multiline(
            null_type,
            decoy_selection=decoy_selection,
        )
        for null_type in nulls
    ])
    ax.set_xlabel("Null model")
    ax.set_ylabel("Mean Jaccard")
    ax.grid(axis="y", alpha=0.22)
    ax.grid(axis="x", alpha=0.08)
    ax.set_axisbelow(True)
    ax.set_xlim(centers[0] - 0.58, centers[-1] + 0.58)
    y_high = max(annotation_tops) if annotation_tops else 1.0
    ax.set_ylim(0.0, y_high * (1.58 if compact else 1.34))
    ax.errorbar(
        [np.nan],
        [np.nan],
        yerr=[[0.1], [0.1]],
        fmt="none",
        ecolor="#1f2933",
        elinewidth=0.8,
        capsize=2.5,
        capthick=0.8,
        label="95% bootstrap CI",
    )
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2 if compact else 3,
        frameon=True,
        framealpha=0.94,
        edgecolor="#555555",
        borderaxespad=0.1,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    legend.get_frame().set_linewidth(0.7)
    fig.subplots_adjust(
        left=0.19 if compact else 0.095,
        right=0.98 if compact else 0.99,
        bottom=0.24,
        top=0.82,
    )
    return fig


def plot_global_joint_triplet_delta_bootstrap(
    delta_bootstrap_df: pd.DataFrame,
    *,
    estimator: str = "mean",
    evaluable_set_type: str | None = None,
    statistics: list[str] | None = None,
    decoy_selection: str | None = None,
    figsize_per_row: float = 0.45,
    min_figheight: float = 2.4,
    figwidth: float = 7.2,
) -> Any:
    """Plot global observed-minus-null paired delta bootstrap CIs.

    Parameters
    ----------
    delta_bootstrap_df:
        Output table ``global_joint_triplet_delta_bootstrap.tsv``.
    estimator:
        Must be ``"mean"``.  The plotted point is ``delta_mean_hat`` and the
        error bar is its 95% triplet-bootstrap CI.
    """
    import matplotlib.pyplot as plt

    if estimator != "mean":
        raise ValueError("estimator must be 'mean'.")

    apply_val3_paper_style()
    value_col = f"delta_{estimator}_hat"
    lo_col = f"delta_{estimator}_ci025"
    hi_col = f"delta_{estimator}_ci975"
    required = {"null_type", "evaluable_set_type", "statistic", value_col, lo_col, hi_col}
    missing = required - set(delta_bootstrap_df.columns)
    if delta_bootstrap_df.empty or missing:
        fig, ax = plt.subplots(figsize=(figwidth, min_figheight))
        msg = "no bootstrap rows" if delta_bootstrap_df.empty else f"missing columns: {sorted(missing)}"
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout()
        return fig

    df = delta_bootstrap_df.copy()
    if evaluable_set_type is not None:
        df = df.loc[df["evaluable_set_type"].astype(str).eq(str(evaluable_set_type))].copy()
    for col in [value_col, lo_col, hi_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[value_col, lo_col, hi_col])
    if df.empty:
        fig, ax = plt.subplots(figsize=(figwidth, min_figheight))
        ax.text(0.5, 0.5, "no finite bootstrap estimates", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout()
        return fig
    if statistics is not None:
        wanted = {str(s) for s in statistics}
        df = df.loc[df["statistic"].astype(str).isin(wanted)].copy()
        if df.empty:
            fig, ax = plt.subplots(figsize=(figwidth, min_figheight))
            ax.text(0.5, 0.5, "no matching bootstrap statistics", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            fig.tight_layout()
            return fig

    if str(evaluable_set_type) == "path_existing_pairs":
        return _plot_path_existing_pairs_delta_bootstrap_grouped(
            df,
            value_col=value_col,
            lo_col=lo_col,
            hi_col=hi_col,
            estimator=estimator,
            statistics=statistics,
            decoy_selection=decoy_selection,
        )

    preferred_stats = ["joint_jaccard", "jaccard"]
    present_stats = list(df["statistic"].astype(str).drop_duplicates())
    statistics = [s for s in preferred_stats if s in set(present_stats)]
    statistics.extend([s for s in present_stats if s not in set(statistics)])
    n_panels = max(len(statistics), 1)
    max_nulls = max(int(df.groupby("statistic", dropna=False)["null_type"].nunique().max()), 1)
    figheight = max(min_figheight, 1.15 + max_nulls * float(figsize_per_row))
    fig, axes = plt.subplots(1, n_panels, figsize=(figwidth, figheight), squeeze=False, sharex=True, sharey=True)

    finite_bounds = pd.to_numeric(df[lo_col], errors="coerce").tolist() + pd.to_numeric(df[hi_col], errors="coerce").tolist()
    finite_bounds = [float(x) for x in finite_bounds if np.isfinite(float(x))]
    if finite_bounds:
        lo, hi = min(finite_bounds), max(finite_bounds)
        span = max(float(hi - lo), 1.0e-6)
        xlim = (min(0.0, lo) - span * 0.08, max(0.0, hi) + span * 0.08)
    else:
        xlim = None

    for i, stat in enumerate(statistics):
        ax = axes[0, i]
        sub = df.loc[df["statistic"].astype(str).eq(str(stat))].copy()
        sub["label"] = sub["null_type"].map(
            lambda n: format_null_type_label(n, decoy_selection=decoy_selection)
        )
        sub["_null_order"] = sub["null_type"].map(null_type_sort_key)
        sub = sub.sort_values("_null_order", ascending=True).reset_index(drop=True)
        y_pos = np.arange(len(sub))[::-1]
        x = sub[value_col].to_numpy(dtype=float)
        lo = sub[lo_col].to_numpy(dtype=float)
        hi = sub[hi_col].to_numpy(dtype=float)
        ax.plot(x, y_pos, "o", color="C0", zorder=4)
        _draw_horizontal_ci(
            ax,
            y_pos,
            lo,
            hi,
            color="C0",
            label="95% bootstrap CI" if i == 0 else None,
        )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_yticks(y_pos)
        if i == 0:
            ax.set_yticklabels(sub["label"].tolist())
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_ylim(-0.65, len(sub) - 0.35)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", alpha=0.08)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, bbox_to_anchor=(0.985, 0.995))
    fig.supxlabel(f"Observed − Null ({estimator})", y=0.04)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.18, top=0.84, wspace=0.22)
    return fig


def _plot_path_existing_pairs_delta_bootstrap_grouped(
    df: pd.DataFrame,
    *,
    value_col: str,
    lo_col: str,
    hi_col: str,
    estimator: str,
    statistics: list[str] | None = None,
    decoy_selection: str | None = None,
) -> Any:
    """Paper layout: null models as x groups with the Jaccard primary effect."""
    import matplotlib.pyplot as plt

    apply_val3_paper_style()

    preferred_stats = statistics or ["joint_jaccard"]
    present_stats = list(df["statistic"].astype(str).drop_duplicates())
    statistics = [s for s in preferred_stats if s in set(present_stats)]
    statistics.extend([s for s in present_stats if s not in set(statistics)])
    nulls = sorted(df["null_type"].astype(str).drop_duplicates(), key=null_type_sort_key)
    single_metric = len(statistics) == 1

    x_tick_labels = [
        format_null_type_label_multiline(n, decoy_selection=decoy_selection) for n in nulls
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=False)
    centers = np.arange(len(nulls), dtype=float)
    bar_width = 0.42 if single_metric else min(0.28, 0.84 / max(len(statistics), 1))
    offsets = np.array([0.0]) if single_metric else (np.arange(len(statistics)) - (len(statistics) - 1) / 2.0) * bar_width
    colors = ["#deebf7", "#9ecae1", "#3182bd", "#08519c", "#6baed6"]

    y_lows: list[float] = []
    y_highs: list[float] = []
    for stat_idx, stat in enumerate(statistics):
        xs: list[float] = []
        ys: list[float] = []
        ci_lows: list[float] = []
        ci_highs: list[float] = []
        for null_idx, null_type in enumerate(nulls):
            row = df.loc[
                df["statistic"].astype(str).eq(str(stat))
                & df["null_type"].astype(str).eq(str(null_type))
            ]
            if row.empty:
                continue
            rec = row.iloc[0]
            y = float(rec[value_col])
            lo = float(rec[lo_col])
            hi = float(rec[hi_col])
            if not (np.isfinite(y) and np.isfinite(lo) and np.isfinite(hi)):
                continue
            xs.append(float(centers[null_idx] + offsets[stat_idx]))
            ys.append(y)
            ci_lows.append(lo)
            ci_highs.append(hi)
            y_lows.append(lo)
            y_highs.append(hi)
        if not xs:
            continue
        ax.bar(
            xs,
            ys,
            width=bar_width * 0.92,
            color=colors[stat_idx % len(colors)],
            edgecolor="#1f2933",
            linewidth=0.7,
            label=format_metric_label(stat),
            zorder=3,
        )
        _draw_vertical_ci(
            ax,
            xs,
            ci_lows,
            ci_highs,
            cap_half_width=bar_width * 0.12,
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(centers)
    ax.set_xticklabels(x_tick_labels, rotation=0, ha="center")
    ax.set_xlabel("Null model")
    ax.set_ylabel("Mean delta Jaccard")
    ax.grid(axis="y", alpha=0.22)
    ax.grid(axis="x", alpha=0.08)
    ax.set_axisbelow(True)
    ax.set_xlim(centers[0] - 0.58, centers[-1] + 0.58)
    if y_lows and y_highs:
        y_low = min(min(y_lows), 0.0)
        y_high = max(max(y_highs), 0.0)
        span = max(y_high - y_low, 1.0e-6)
        top_pad = 0.18 if single_metric else 0.50
        ax.set_ylim(y_low - 0.12 * span, y_high + top_pad * span)
    ax.errorbar(
        [np.nan],
        [np.nan],
        yerr=[[0.1], [0.1]],
        fmt="none",
        ecolor="#1f2933",
        elinewidth=0.8,
        capsize=2.5,
        capthick=0.8,
        label="95% bootstrap CI",
    )
    handles, labels = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    ordered_labels = [*[format_metric_label(stat) for stat in statistics], "95% bootstrap CI"]
    ordered_handles = [handle_by_label[label] for label in ordered_labels if label in handle_by_label]
    ordered_labels = [label for label in ordered_labels if label in handle_by_label]
    legend = ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(ordered_labels) if ordered_labels else 1,
        frameon=True,
        framealpha=0.94,
        edgecolor="#555555",
        borderaxespad=0.1,
        columnspacing=0.7,
        handletextpad=0.4,
    )
    legend.get_frame().set_linewidth(0.7)
    if single_metric:
        fig.subplots_adjust(left=0.19, right=0.99, bottom=0.24, top=0.88)
    else:
        fig.subplots_adjust(left=0.19, right=0.99, bottom=0.24, top=0.80)
    return fig


def _empty_figure(message: str, *, figwidth: float = 7.2, figheight: float = 2.4) -> Any:
    import matplotlib.pyplot as plt

    apply_val3_paper_style()
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _filter_by_columns(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for column, value in filters.items():
        if value is None or column not in out.columns:
            continue
        out = out.loc[out[column].astype(str).eq(str(value))]
    return out


def plot_triplet_delta_distribution(
    triplet_delta_df: pd.DataFrame,
    *,
    statistic: str = "joint_jaccard",
    evaluable_set_type: str | None = None,
    decoy_selection: str | None = None,
    bins: int = 40,
) -> Any:
    """Triplet-wise ``Observed − null median`` distribution, one panel per null model."""
    import matplotlib.pyplot as plt

    apply_val3_paper_style()
    required = {"null_type", "statistic", "delta"}
    if triplet_delta_df.empty or not required <= set(triplet_delta_df.columns):
        return _empty_figure("no triplet delta rows")
    df = _filter_by_columns(
        triplet_delta_df,
        {"statistic": statistic, "evaluable_set_type": evaluable_set_type},
    )
    df = df.assign(delta=pd.to_numeric(df["delta"], errors="coerce")).dropna(subset=["delta"])
    if df.empty:
        return _empty_figure("no finite triplet delta values")

    nulls = sorted(df["null_type"].astype(str).drop_duplicates(), key=null_type_sort_key)
    span = float(np.nanmax(np.abs(df["delta"].to_numpy(dtype=float))))
    span = span if np.isfinite(span) and span > 0 else 1.0e-3
    fig, axes = plt.subplots(1, len(nulls), figsize=(7.2, 2.4), sharex=True, sharey=True, squeeze=False)
    for ax, null_type in zip(axes[0], nulls):
        values = pd.to_numeric(df.loc[df["null_type"].astype(str).eq(null_type), "delta"], errors="coerce").dropna()
        ax.hist(
            values,
            bins=bins,
            range=(-span, span),
            color="#5B8DEF",
            alpha=0.82,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, zorder=4)
        median = float(values.median()) if not values.empty else float("nan")
        if np.isfinite(median):
            ax.axvline(median, color="#c0392b", linestyle="-", linewidth=1.0, zorder=5)
        ax.text(
            0.5,
            0.97,
            format_null_type_label(null_type, decoy_selection=decoy_selection),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontweight="semibold",
        )
        ax.set_xlabel("ΔJaccard")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.text(
            0.02,
            0.80,
            f"n={int(len(values))}\nmedian={median:.3f}\n>0: {float((values > 0).mean()):.0%}",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
    axes[0, 0].set_ylabel("Triplets")
    fig.tight_layout()
    return fig


def plot_null_repeat_stability(
    stability_df: pd.DataFrame,
    *,
    statistic: str = "joint_jaccard",
    evaluable_set_type: str | None = None,
    decoy_selection: str | None = None,
) -> Any:
    """Cumulative stability of mean paired ΔJaccard across null-repeat counts."""
    import matplotlib.pyplot as plt

    apply_val3_paper_style()
    required = {
        "null_type",
        "evaluable_set_type",
        "statistic",
        "n_repeats",
        "delta_mean",
    }
    if stability_df.empty or not required <= set(stability_df.columns):
        return _empty_figure(
            "no null-repeat stability rows",
            figwidth=3.5,
            figheight=2.4,
        )
    df = _filter_by_columns(
        stability_df,
        {"statistic": statistic, "evaluable_set_type": evaluable_set_type},
    )
    df["n_repeats"] = pd.to_numeric(df["n_repeats"], errors="coerce")
    df["delta_mean"] = pd.to_numeric(df["delta_mean"], errors="coerce")
    df = df.dropna(subset=["n_repeats", "delta_mean"])
    if df.empty:
        return _empty_figure(
            "no finite null-repeat stability estimates",
            figwidth=3.5,
            figheight=2.4,
        )

    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=False)
    nulls = sorted(
        df["null_type"].astype(str).drop_duplicates(),
        key=null_type_sort_key,
    )
    colors = ["#3182bd", "#7a5195", "#ef5675"]
    for index, null_type in enumerate(nulls):
        sub = df.loc[
            df["null_type"].astype(str).eq(null_type)
        ].sort_values("n_repeats")
        ax.plot(
            sub["n_repeats"].to_numpy(dtype=float),
            sub["delta_mean"].to_numpy(dtype=float),
            marker="o",
            color=colors[index % len(colors)],
            label=format_null_type_label(
                null_type,
                decoy_selection=decoy_selection,
            ),
            zorder=3,
        )
    checkpoints = sorted(df["n_repeats"].astype(int).unique())
    ax.set_xticks(checkpoints)
    ax.set_xlabel("Null repeats")
    ax.set_ylabel("Mean delta Jaccard")
    ax.grid(axis="both", alpha=0.22)
    ax.set_axisbelow(True)
    legend = ax.legend(
        loc="center",
        bbox_to_anchor=(0.64, 0.50),
        frameon=True,
        framealpha=0.94,
        edgecolor="#555555",
        borderaxespad=0.1,
    )
    legend.get_frame().set_linewidth(0.7)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.22, top=0.80)
    return fig


def plot_delta_cluster_bootstrap(
    cluster_bootstrap_df: pd.DataFrame,
    *,
    statistic: str = "joint_jaccard",
    evaluable_set_type: str | None = None,
    decoy_selection: str | None = None,
    estimator: str = "mean",
) -> Any:
    """Cluster bootstrap CIs of the paired delta, grouped by null model and cluster level."""
    import matplotlib.pyplot as plt

    apply_val3_paper_style()
    if estimator != "mean":
        raise ValueError("estimator must be 'mean'.")
    value_col = f"delta_{estimator}_hat"
    lo_col = f"delta_{estimator}_ci025"
    hi_col = f"delta_{estimator}_ci975"
    required = {"null_type", "cluster_level", "statistic", value_col, lo_col, hi_col}
    if cluster_bootstrap_df.empty or not required <= set(cluster_bootstrap_df.columns):
        return _empty_figure("no cluster bootstrap rows")
    df = _filter_by_columns(
        cluster_bootstrap_df,
        {"statistic": statistic, "evaluable_set_type": evaluable_set_type},
    )
    for col in [value_col, lo_col, hi_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[value_col, lo_col, hi_col])
    if df.empty:
        return _empty_figure("no finite cluster bootstrap estimates")

    nulls = sorted(df["null_type"].astype(str).drop_duplicates(), key=null_type_sort_key)
    level_order = ["triplet", "drug", "target", "disease"]
    present = list(df["cluster_level"].astype(str).drop_duplicates())
    levels = [lv for lv in level_order if lv in set(present)]
    levels.extend([lv for lv in present if lv not in set(levels)])
    colors = ["#deebf7", "#9ecae1", "#3182bd", "#08519c"]

    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=False)
    centers = np.arange(len(nulls), dtype=float)
    bar_width = min(0.22, 0.86 / max(len(levels), 1))
    offsets = (np.arange(len(levels)) - (len(levels) - 1) / 2.0) * bar_width
    y_lows: list[float] = []
    y_highs: list[float] = []
    for level_idx, level in enumerate(levels):
        xs: list[float] = []
        ys: list[float] = []
        ci_lows: list[float] = []
        ci_highs: list[float] = []
        for null_idx, null_type in enumerate(nulls):
            row = df.loc[
                df["null_type"].astype(str).eq(null_type)
                & df["cluster_level"].astype(str).eq(str(level))
            ]
            if row.empty:
                continue
            rec = row.iloc[0]
            y, lo, hi = float(rec[value_col]), float(rec[lo_col]), float(rec[hi_col])
            xs.append(float(centers[null_idx] + offsets[level_idx]))
            ys.append(y)
            ci_lows.append(lo)
            ci_highs.append(hi)
            y_lows.append(lo)
            y_highs.append(hi)
        if not xs:
            continue
        # Cluster counts differ per null model, so they stay in the table rather than the legend.
        label = f"{str(level).capitalize()} cluster"
        ax.bar(
            xs,
            ys,
            width=bar_width * 0.92,
            color=colors[level_idx % len(colors)],
            edgecolor="#1f2933",
            linewidth=0.7,
            label=label,
            zorder=3,
        )
        _draw_vertical_ci(
            ax,
            xs,
            ci_lows,
            ci_highs,
            cap_half_width=bar_width * 0.12,
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(centers)
    ax.set_xticklabels(
        [format_null_type_label_multiline(n, decoy_selection=decoy_selection) for n in nulls],
        rotation=0,
        ha="center",
    )
    ax.set_xlabel("Null model")
    ax.set_ylabel("Mean delta Jaccard")
    ax.grid(axis="y", alpha=0.22)
    ax.grid(axis="x", alpha=0.08)
    ax.set_axisbelow(True)
    ax.set_xlim(centers[0] - 0.58, centers[-1] + 0.58)
    if y_lows and y_highs:
        y_low = min(min(y_lows), 0.0)
        y_high = max(max(y_highs), 0.0)
        span = max(y_high - y_low, 1.0e-6)
        ax.set_ylim(y_low - 0.12 * span, y_high + 0.42 * span)
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2 if len(levels) > 1 else 1,
        frameon=True,
        framealpha=0.94,
        edgecolor="#555555",
        borderaxespad=0.1,
        columnspacing=0.7,
        handletextpad=0.4,
    )
    legend.get_frame().set_linewidth(0.7)
    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.24, top=0.78)
    return fig


def plot_null_evaluability(
    summary_df: pd.DataFrame,
    by_triplet_df: pd.DataFrame,
    *,
    evaluable_set_type: str | None = None,
    decoy_selection: str | None = None,
) -> Any:
    """Share of the common observed-evaluable triplets with at least one null score."""
    import matplotlib.pyplot as plt

    apply_val3_paper_style()
    required = {
        "null_type",
        "n_observed_evaluable_triplets",
        "n_triplets_no_null_score",
    }
    if summary_df.empty or not required <= set(summary_df.columns):
        return _empty_figure("no null evaluability rows", figwidth=3.5)
    summary = _filter_by_columns(summary_df, {"evaluable_set_type": evaluable_set_type})
    for column in ["n_observed_evaluable_triplets", "n_triplets_no_null_score"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
    summary = summary.dropna(
        subset=["n_observed_evaluable_triplets", "n_triplets_no_null_score"],
    )
    if summary.empty:
        return _empty_figure("no finite triplet coverage values", figwidth=3.5)
    denominators = summary["n_observed_evaluable_triplets"]
    if (
        denominators.le(0).any()
        or denominators.nunique(dropna=True) != 1
        or summary["n_triplets_no_null_score"].lt(0).any()
        or summary["n_triplets_no_null_score"].gt(denominators).any()
    ):
        return _empty_figure("inconsistent observed-triplet denominators", figwidth=3.5)
    summary["triplet_coverage_percent"] = (
        100.0
        * (
            summary["n_observed_evaluable_triplets"]
            - summary["n_triplets_no_null_score"]
        )
        / summary["n_observed_evaluable_triplets"]
    )
    nulls = sorted(summary["null_type"].astype(str).drop_duplicates(), key=null_type_sort_key)

    # Keep by_triplet_df in the public API because it remains a report audit table,
    # but omit its distribution subplot from the paper-facing figure.
    _ = by_triplet_df
    fig, ax_rate = plt.subplots(figsize=(3.5, 2.4), constrained_layout=False)
    centers = np.arange(len(nulls), dtype=float)
    rates = [
        float(
            summary.loc[
                summary["null_type"].astype(str).eq(n),
                "triplet_coverage_percent",
            ].iloc[0]
        )
        for n in nulls
    ]
    ax_rate.bar(centers, rates, width=0.6, color="#3182bd", edgecolor="#1f2933", linewidth=0.7, zorder=3)
    for x, rate in zip(centers, rates):
        ax_rate.text(x, rate + 1.2, f"{rate:.1f}%", ha="center", va="bottom")
    ax_rate.set_xticks(centers)
    ax_rate.set_xticklabels(
        [format_null_type_label_multiline(n, decoy_selection=decoy_selection) for n in nulls],
        rotation=0,
        ha="center",
    )
    ax_rate.set_ylim(0.0, 110.0)
    ax_rate.set_ylabel("Triplet coverage (%)")
    ax_rate.set_xlabel("Null model")
    ax_rate.grid(axis="y", alpha=0.22)
    ax_rate.set_axisbelow(True)
    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.24, top=0.90)
    return fig


# Figure labels follow the `figure` column of docs/caliper_naming_crosswalk_draft.md.
TARGET_DECOY_MATCHING_VARIABLE_LABELS: dict[str, str] = {
    "directed_out_degree": "Out-degree",
    "reachable_unique_nodes_k4": "Reachable nodes",
    "unique_kd_cell_lines": "KD cell lines",
    "total_degree_diagnostic": "Total degree (diagnostic)",
}


def plot_target_decoy_matching_diagnostics(
    balance_df: pd.DataFrame,
    triplet_evaluability_df: pd.DataFrame,
    *,
    evaluable_set_type: str | None = None,
    decoy_selection: str | None = None,
    smd_threshold: float = 0.1,
    analysis_scope: str = "primary_path_evaluable",
    stage: str = "pre_inference_selected",
    weighting: str = "triplet_weighted",
) -> Any:
    """Supplementary Target-Decoy diagnostics: matching balance and evaluable-decoy counts.

    Left: standardized mean differences between observed and matched decoy targets on the
    acceptance scope only, with the |SMD| threshold drawn.  Right: how many of the ten
    selected decoys per triplet turned out evaluable.  These are matching diagnostics and
    stay out of the main Target-Decoy result figure.
    """
    import matplotlib.pyplot as plt

    apply_val3_paper_style()
    required = {"analysis_scope", "stage", "weighting", "variable", "smd"}
    if balance_df.empty or not required <= set(balance_df.columns):
        return _empty_figure("no Target-Decoy matching balance rows")
    balance = balance_df.loc[
        balance_df["analysis_scope"].astype(str).eq(str(analysis_scope))
        & balance_df["stage"].astype(str).eq(str(stage))
        & balance_df["weighting"].astype(str).eq(str(weighting))
    ].copy()
    balance = _filter_by_columns(balance, {"evaluable_set_type": evaluable_set_type})
    balance["smd"] = pd.to_numeric(balance["smd"], errors="coerce")
    balance = balance.dropna(subset=["smd"]).drop_duplicates("variable")
    if balance.empty:
        return _empty_figure(f"no {analysis_scope} / {stage} / {weighting} balance rows")

    order = [v for v in TARGET_DECOY_MATCHING_VARIABLE_LABELS if v in set(balance["variable"].astype(str))]
    order += [v for v in balance["variable"].astype(str) if v not in order]
    smd_by_var = dict(zip(balance["variable"].astype(str), balance["smd"]))
    is_matching = {
        str(v): bool(str(f).strip().lower() in {"true", "1", "1.0", "yes"})
        for v, f in zip(balance["variable"].astype(str), balance.get("is_matching_variable", pd.Series(dtype=object)))
    } if "is_matching_variable" in balance.columns else {}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), squeeze=False)
    ax_smd = axes[0, 0]
    # Bars run downward in the fixed variable order, so the matching variables that gate
    # acceptance stay at the top regardless of which of them happens to be largest.
    positions = np.arange(len(order), dtype=float)[::-1]
    values = [float(smd_by_var[v]) for v in order]
    colors = ["#3182bd" if is_matching.get(v, True) else "#9ecae1" for v in order]
    ax_smd.barh(positions, values, height=0.6, color=colors, edgecolor="#1f2933", linewidth=0.7, zorder=3)
    limit = max(float(smd_threshold) * 1.6, float(np.max(np.abs(values))) * 1.45) if values else float(smd_threshold)
    for x, value in zip(positions, values):
        offset = limit * 0.03
        ax_smd.text(
            value + (offset if value >= 0 else -offset),
            x,
            f"{value:+.3f}",
            ha="left" if value >= 0 else "right",
            va="center",
        )
    ax_smd.axvline(0.0, color="#555555", linewidth=0.9, zorder=2)
    for sign in (-1.0, 1.0):
        ax_smd.axvline(sign * float(smd_threshold), color="#c0392b", linestyle="--", linewidth=1.0, zorder=2)
    # The threshold is labelled directly above the bars: a legend box would sit on top of
    # the value label of the bottom bar, and the variable names already carry identity.
    top = float(positions.max()) if len(positions) else 0.0
    ax_smd.text(
        float(smd_threshold) - limit * 0.03,
        top + 0.62,
        f"|SMD| = {float(smd_threshold):g}",
        color="#c0392b",
        ha="right",
        va="bottom",
    )
    ax_smd.set_ylim(-0.7, top + 0.95)
    ax_smd.set_yticks(positions)
    ax_smd.set_yticklabels([TARGET_DECOY_MATCHING_VARIABLE_LABELS.get(v, v.replace("_", " ")) for v in order])
    ax_smd.set_xlim(-limit, limit)
    ax_smd.set_xlabel("Observed − decoy SMD")
    ax_smd.grid(axis="x", alpha=0.22)
    ax_smd.set_axisbelow(True)

    ax_hist = axes[0, 1]
    counts = pd.Series(dtype=float)
    if not triplet_evaluability_df.empty and "n_evaluable_decoys" in triplet_evaluability_df.columns:
        sub = _filter_by_columns(triplet_evaluability_df, {"evaluable_set_type": evaluable_set_type})
        n_evaluable = pd.to_numeric(sub["n_evaluable_decoys"], errors="coerce").dropna().astype(int)
        if not n_evaluable.empty:
            counts = n_evaluable.value_counts().sort_index()
    if counts.empty:
        ax_hist.text(0.5, 0.5, "no evaluable-decoy counts", ha="center", va="center", transform=ax_hist.transAxes)
    else:
        centers = counts.index.to_numpy(dtype=float)
        ax_hist.bar(centers, counts.to_numpy(dtype=float), width=0.7, color="#3182bd", edgecolor="#1f2933", linewidth=0.7, zorder=3)
        ax_hist.set_xticks(centers)
        n_triplets = int(counts.sum())
        n_without = int(counts.get(0, 0))
        ax_hist.text(
            0.98,
            0.94,
            f"n triplets = {n_triplets}\n$|E_i|$ = 0: {n_without}",
            ha="right",
            va="top",
            transform=ax_hist.transAxes,
        )
    ax_hist.set_xlabel("Evaluable decoys per triplet ($|E_i|$)")
    ax_hist.set_ylabel("Triplets")
    ax_hist.grid(axis="y", alpha=0.25)
    ax_hist.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_cell_scope_delta(
    cell_scope_summary_df: pd.DataFrame,
    *,
    statistic: str = "joint_jaccard",
    evaluable_set_type: str | None = None,
    decoy_selection: str | None = None,
) -> Any:
    """Paired delta on all cell pairs vs the same-cell-line subset, per null model."""
    import matplotlib.pyplot as plt

    apply_val3_paper_style()
    required = {
        "cell_scope", "null_type", "statistic",
        "delta_mean", "delta_mean_ci025", "delta_mean_ci975",
    }
    if cell_scope_summary_df.empty or not required <= set(cell_scope_summary_df.columns):
        return _empty_figure("no same-cell subset rows")
    df = _filter_by_columns(
        cell_scope_summary_df,
        {"statistic": statistic, "evaluable_set_type": evaluable_set_type},
    )
    for column in ["delta_mean", "delta_mean_ci025", "delta_mean_ci975"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["delta_mean", "delta_mean_ci025", "delta_mean_ci975"])
    if df.empty:
        return _empty_figure("no finite same-cell subset estimates")

    nulls = sorted(df["null_type"].astype(str).drop_duplicates(), key=null_type_sort_key)
    scope_order = [s for s in ["all_cell_pairs", "same_cell"] if s in set(df["cell_scope"].astype(str))]
    scope_labels = {"all_cell_pairs": "All cell pairs", "same_cell": "Same cell line"}
    colors = {"all_cell_pairs": "#deebf7", "same_cell": "#3182bd"}

    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=False)
    centers = np.arange(len(nulls), dtype=float)
    bar_width = 0.38
    offsets = (np.arange(len(scope_order)) - (len(scope_order) - 1) / 2.0) * bar_width
    y_lows: list[float] = []
    y_highs: list[float] = []
    for scope_idx, scope in enumerate(scope_order):
        xs: list[float] = []
        ys: list[float] = []
        ci_lows: list[float] = []
        ci_highs: list[float] = []
        for null_idx, null_type in enumerate(nulls):
            row = df.loc[
                df["null_type"].astype(str).eq(null_type)
                & df["cell_scope"].astype(str).eq(str(scope))
            ]
            if row.empty:
                continue
            rec = row.iloc[0]
            y = float(rec["delta_mean"])
            lo = float(rec["delta_mean_ci025"])
            hi = float(rec["delta_mean_ci975"])
            xs.append(float(centers[null_idx] + offsets[scope_idx]))
            ys.append(y)
            ci_lows.append(lo)
            ci_highs.append(hi)
            y_lows.append(lo)
            y_highs.append(hi)
        if not xs:
            continue
        ax.bar(
            xs,
            ys,
            width=bar_width * 0.92,
            color=colors.get(scope, "#9ecae1"),
            edgecolor="#1f2933",
            linewidth=0.7,
            label=f"{scope_labels.get(scope, scope)}",
            zorder=3,
        )
        _draw_vertical_ci(
            ax,
            xs,
            ci_lows,
            ci_highs,
            cap_half_width=bar_width * 0.12,
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(centers)
    ax.set_xticklabels(
        [format_null_type_label_multiline(n, decoy_selection=decoy_selection) for n in nulls],
        rotation=0,
        ha="center",
    )
    ax.set_xlabel("Null model")
    ax.set_ylabel("Mean delta Jaccard")
    ax.grid(axis="y", alpha=0.22)
    ax.grid(axis="x", alpha=0.08)
    ax.set_axisbelow(True)
    ax.set_xlim(centers[0] - 0.58, centers[-1] + 0.58)
    if y_lows and y_highs:
        y_low = min(min(y_lows), 0.0)
        y_high = max(max(y_highs), 0.0)
        span = max(y_high - y_low, 1.0e-6)
        ax.set_ylim(y_low - 0.12 * span, y_high + 0.45 * span)
    ax.errorbar(
        [np.nan],
        [np.nan],
        yerr=[[0.1], [0.1]],
        fmt="none",
        ecolor="#1f2933",
        elinewidth=0.8,
        capsize=2.5,
        capthick=0.8,
        label="95% bootstrap CI",
    )
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=max(len(scope_order), 1),
        frameon=True,
        framealpha=0.94,
        edgecolor="#555555",
        borderaxespad=0.2,
        columnspacing=0.8,
        handlelength=1.5,
    )
    legend.get_frame().set_linewidth(0.7)
    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.24, top=0.82)
    return fig


EDGE_SCOPE_LABELS: dict[str, str] = {
    "all_edges": "All edges",
    "first_edge_excluded": "First-edge excluded",
}


def plot_first_edge_exclusion_delta(
    summary_df: pd.DataFrame,
    *,
    statistic: str = "joint_jaccard",
    evaluable_set_type: str | None = None,
    decoy_selection: str | None = None,
) -> Any:
    """Observed − null delta with and without the edges leaving the shared target.

    The evaluable triplet set and the contributing cell pairs are identical between
    the two scopes, so a shrinking but positive bar means the overlap signal is not
    carried by the root-adjacent edges alone.
    """
    import matplotlib.pyplot as plt

    apply_val3_paper_style()
    scope_columns = {
        "all_edges": "delta_mean_all_edges",
        "first_edge_excluded": "delta_mean_first_edge_excluded",
    }
    scope_ci_columns = {
        scope: (
            f"delta_mean_ci025_{scope}",
            f"delta_mean_ci975_{scope}",
        )
        for scope in scope_columns
    }
    required = (
        {"null_type", "statistic"}
        | set(scope_columns.values())
        | {column for pair in scope_ci_columns.values() for column in pair}
    )
    if summary_df is None or summary_df.empty or not required <= set(summary_df.columns):
        return _empty_figure("no first-edge exclusion rows")
    df = _filter_by_columns(
        summary_df,
        {"statistic": statistic, "evaluable_set_type": evaluable_set_type},
    )
    for column in [*scope_columns.values(), *[c for pair in scope_ci_columns.values() for c in pair]]:
        df = df.assign(**{column: pd.to_numeric(df[column], errors="coerce")})
    df = df.dropna(subset=[scope_columns["all_edges"]])
    if df.empty:
        return _empty_figure("no finite first-edge exclusion estimates")

    nulls = sorted(df["null_type"].astype(str).drop_duplicates(), key=null_type_sort_key)
    scope_order = list(scope_columns)
    colors = {"all_edges": "#deebf7", "first_edge_excluded": "#3182bd"}

    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=False)
    centers = np.arange(len(nulls), dtype=float)
    bar_width = 0.38
    offsets = (np.arange(len(scope_order)) - (len(scope_order) - 1) / 2.0) * bar_width
    y_lows: list[float] = []
    y_highs: list[float] = []
    for scope_idx, scope in enumerate(scope_order):
        xs: list[float] = []
        ys: list[float] = []
        ci_lows: list[float] = []
        ci_highs: list[float] = []
        for null_idx, null_type in enumerate(nulls):
            row = df.loc[df["null_type"].astype(str).eq(null_type)]
            if row.empty:
                continue
            rec = row.iloc[0]
            value = rec[scope_columns[scope]]
            if pd.isna(value):
                continue
            lo = float(rec[scope_ci_columns[scope][0]])
            hi = float(rec[scope_ci_columns[scope][1]])
            xs.append(float(centers[null_idx] + offsets[scope_idx]))
            ys.append(float(value))
            ci_lows.append(lo)
            ci_highs.append(hi)
            y_lows.append(lo)
            y_highs.append(hi)
        if not xs:
            continue
        ax.bar(
            xs,
            ys,
            width=bar_width * 0.92,
            color=colors.get(scope, "#9ecae1"),
            edgecolor="#1f2933",
            linewidth=0.7,
            label=EDGE_SCOPE_LABELS.get(scope, scope),
            zorder=3,
        )
        _draw_vertical_ci(
            ax,
            xs,
            ci_lows,
            ci_highs,
            cap_half_width=bar_width * 0.12,
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(centers)
    ax.set_xticklabels(
        [format_null_type_label_multiline(n, decoy_selection=decoy_selection) for n in nulls],
        rotation=0,
        ha="center",
    )
    ax.set_xlabel("Null model")
    ax.set_ylabel("Mean delta Jaccard")
    ax.grid(axis="y", alpha=0.22)
    ax.grid(axis="x", alpha=0.08)
    ax.set_axisbelow(True)
    ax.set_xlim(centers[0] - 0.58, centers[-1] + 0.58)
    if y_lows and y_highs:
        y_low = min(min(y_lows), 0.0)
        y_high = max(max(y_highs), 0.0)
        span = max(y_high - y_low, 1.0e-6)
        ax.set_ylim(y_low - 0.12 * span, y_high + 0.45 * span)
    ax.errorbar(
        [np.nan],
        [np.nan],
        yerr=[[0.1], [0.1]],
        fmt="none",
        ecolor="#1f2933",
        elinewidth=0.8,
        capsize=2.5,
        capthick=0.8,
        label="95% bootstrap CI",
    )
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=max(len(scope_order), 1),
        frameon=True,
        framealpha=0.94,
        edgecolor="#555555",
        borderaxespad=0.2,
        columnspacing=0.8,
        handlelength=1.5,
    )
    legend.get_frame().set_linewidth(0.7)
    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.24, top=0.82)
    return fig
