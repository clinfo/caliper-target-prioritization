"""Paper-scale mechanism-family figures shared by Val4/H1 and Val4/H2.

These helpers render only from already-computed metric tables. They live under
``src/val4`` so final-Figure scripts only compose canonical run outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


PAPER_FAMILY_FOREST_FIGSIZE = (7.0, 3.0)
PAPER_EDGE_SUPPORT_FIGSIZE = (7.0, 5.2)


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    return plt, mticker


def _set_style() -> None:
    plt, _ = _load_matplotlib()
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
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "savefig.bbox": None,
        }
    )


def _save_panel(fig, output_png: Path) -> dict[str, str]:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = output_png.with_suffix(".pdf")
    output_svg = output_png.with_suffix(".svg")
    fig.savefig(output_png, format="png", dpi=300, bbox_inches=None, pad_inches=0)
    fig.savefig(output_pdf, format="pdf", bbox_inches=None, pad_inches=0)
    fig.savefig(output_svg, format="svg", bbox_inches=None, pad_inches=0)
    return {
        "png": output_png.as_posix(),
        "pdf": output_pdf.as_posix(),
        "svg": output_svg.as_posix(),
    }


def plot_family_forest(
    summary: pd.DataFrame,
    *,
    value_col: str,
    low_col: str,
    high_col: str,
    xlabel: str,
    output_png: Path,
    family_labels: Mapping[str, str],
) -> dict[str, str]:
    """Render the approved full-width forest-panel presentation."""
    required = {"family", value_col, low_col, high_col}
    if summary.empty or not required.issubset(summary.columns):
        return {}

    values = summary[["family", value_col, low_col, high_col]].copy()
    for column in (value_col, low_col, high_col):
        values[column] = pd.to_numeric(values[column], errors="coerce")
    values = (
        values.dropna(subset=[value_col, low_col, high_col])
        .sort_values([value_col, "family"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if values.empty:
        return {}
    values["family_label"] = values["family"].astype(str).map(family_labels)
    if values["family_label"].isna().any():
        missing = values.loc[values["family_label"].isna(), "family"].tolist()
        raise ValueError(f"Missing mechanism-family labels: {missing}")

    _set_style()
    plt, mticker = _load_matplotlib()
    y = np.arange(len(values))
    estimate = values[value_col].to_numpy(dtype=float)
    low = values[low_col].to_numpy(dtype=float)
    high = values[high_col].to_numpy(dtype=float)
    colors = np.where(estimate >= 0, "#2A6F97", "#B95F45")

    fig, ax = plt.subplots(figsize=PAPER_FAMILY_FOREST_FIGSIZE)
    fig.subplots_adjust(left=0.30, right=0.985, top=0.965, bottom=0.19)
    ax.barh(
        y,
        estimate,
        color=colors,
        edgecolor="0.18",
        linewidth=0.45,
        height=0.64,
        zorder=2,
    )
    ax.errorbar(
        estimate,
        y,
        xerr=[
            np.maximum(estimate - low, 0.0),
            np.maximum(high - estimate, 0.0),
        ],
        fmt="none",
        ecolor="0.2",
        elinewidth=0.85,
        capsize=2.5,
        capthick=0.85,
        label="95% bootstrap CI",
        zorder=3,
    )
    ax.scatter(
        estimate,
        y,
        c=colors,
        edgecolor="0.12",
        linewidth=0.45,
        marker="D",
        s=18,
        zorder=4,
    )
    extent = np.concatenate([low, high, estimate, np.array([0.0])])
    x_min = float(np.nanmin(extent))
    x_max = float(np.nanmax(extent))
    span = max(x_max - x_min, 1e-9)
    ax.set_xlim(x_min - 0.12 * span, x_max + 0.12 * span)
    ax.axvline(0.0, color="0.10", linestyle="-", linewidth=0.75, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(values["family_label"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.3g}"))
    ax.grid(axis="x", color="0.88", linewidth=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.legend(loc="lower right", frameon=True)
    legend.get_frame().set_facecolor("0.94")
    legend.get_frame().set_edgecolor("0.55")
    legend.get_frame().set_linewidth(0.7)
    legend.get_frame().set_alpha(1.0)
    outputs = _save_panel(fig, output_png)
    plt.close(fig)
    return outputs


def plot_family_edge_support(
    eligibility: pd.DataFrame,
    *,
    output_png: Path,
    family_labels: Mapping[str, str],
    threshold: int,
) -> dict[str, str]:
    """Render the approved full-width mechanism-family edge-support panel."""
    edge_col = (
        "n_physical_edges"
        if "n_physical_edges" in eligibility.columns
        else "physical_edges"
    )
    if eligibility.empty or not {"family", edge_col}.issubset(eligibility.columns):
        return {}

    values = eligibility[["family", edge_col]].copy()
    values[edge_col] = pd.to_numeric(values[edge_col], errors="coerce")
    values = (
        values.dropna(subset=[edge_col])
        .sort_values([edge_col, "family"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if values.empty:
        return {}
    values["family_label"] = values["family"].astype(str).map(family_labels)
    if values["family_label"].isna().any():
        missing = values.loc[values["family_label"].isna(), "family"].tolist()
        raise ValueError(f"Missing mechanism-family labels: {missing}")

    _set_style()
    plt, mticker = _load_matplotlib()
    y = np.arange(len(values))
    edge_support = values[edge_col].to_numpy(dtype=float)
    colors = np.where(edge_support >= threshold, "#2A6F97", "#BDBDBD")

    fig, ax = plt.subplots(figsize=PAPER_EDGE_SUPPORT_FIGSIZE)
    fig.subplots_adjust(left=0.29, right=0.985, top=0.975, bottom=0.13)
    ax.barh(
        y,
        edge_support,
        color=colors,
        edgecolor="0.25",
        linewidth=0.45,
        height=0.68,
        zorder=2,
    )
    ax.axvline(
        threshold,
        color="0.10",
        linestyle=":",
        linewidth=0.9,
        label=f"edge-support threshold = {threshold}",
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(values["family_label"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("KG-edge support")
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6, integer=True))
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="x", color="0.88", linewidth=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    outputs = _save_panel(fig, output_png)
    plt.close(fig)
    return outputs
