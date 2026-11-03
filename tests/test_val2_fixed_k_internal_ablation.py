"""Tests for the paper-facing fixed k=4 internal ablation panel."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from val2 import viz
from val2.summary import PRIMARY_K, summarize_fixed_k_internal_ablation


def _ablation_summary() -> pd.DataFrame:
    """method × k summary rows in the shape produced by load_ablation_summary."""
    rows = []
    values = {
        "proposed": 0.082,
        "kg_sign_noalign": 0.071,
        "kg_nosign": 0.066,
        "noKG_sizematch": 0.068,
        "noKG": 0.060,
        "ncs": 0.074,
    }
    for k in (3, 4, 5):
        for method, mAP in values.items():
            rows.append(
                {
                    "method": method,
                    "k": k,
                    "n_diseases": 145,
                    "mAP": mAP + 0.001 * (k - 4),
                    "mAP_SEM": 0.004,
                    "mean_Lift@10": 10.0 * mAP,
                    "mean_Lift@10_SEM": 0.05,
                }
            )
    return pd.DataFrame(rows)


def test_summarize_fixed_k_omits_global_nokg_and_ncs():
    out = summarize_fixed_k_internal_ablation(_ablation_summary(), PRIMARY_K)

    assert out["method"].tolist() == [
        "proposed",
        "kg_sign_noalign",
        "kg_nosign",
        "noKG_sizematch",
    ]
    assert set(out["k"]) == {PRIMARY_K}


def test_summarize_fixed_k_returns_empty_when_k_absent():
    summary = _ablation_summary()
    summary = summary[summary["k"] != PRIMARY_K]

    assert summarize_fixed_k_internal_ablation(summary, PRIMARY_K).empty


def test_summarize_fixed_k_is_unchanged_when_global_nokg_is_missing():
    summary = _ablation_summary()
    summary = summary[summary["method"] != "noKG"]

    out = summarize_fixed_k_internal_ablation(summary, PRIMARY_K)

    assert out["method"].tolist() == [
        "proposed",
        "kg_sign_noalign",
        "kg_nosign",
        "noKG_sizematch",
    ]


@pytest.fixture()
def captured_figure(monkeypatch):
    """Keep the figure alive past ``plt.close`` so its artists can be inspected."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured: list = []
    real_close = plt.close

    def _capture(fig=None):
        captured.append(fig)

    monkeypatch.setattr(plt, "close", _capture)
    yield captured
    for fig in captured:
        real_close(fig)


def test_fixed_k_figure_omits_global_nokg(tmp_path, captured_figure):
    out_png = tmp_path / f"k{PRIMARY_K}_internal_ablation_ap.png"

    outputs = viz._save_fixed_k_internal_ablation_figure(
        _ablation_summary(),
        out_png=out_png,
        k=PRIMARY_K,
        mean_col="mAP",
        sem_col="mAP_SEM",
        ylabel="Mean AP",
    )

    stem = out_png.stem
    assert set(outputs) == {stem, f"{stem}_pdf", f"{stem}_svg"}
    for suffix in (".png", ".pdf", ".svg"):
        assert out_png.with_suffix(suffix).stat().st_size > 0

    ax = captured_figure[0].axes[0]
    assert [label.get_text() for label in ax.get_yticklabels()] == [
        "CALIPER",
        "w/o KD alignment",
        "Size-matched no-KG",
        "w/o edge polarity",
    ]
    # One horizontal error bar per displayed method, no legend proxy, and the
    # shared final-Figure Mean AP axis range.
    assert len(ax.containers) == 4
    assert ax.get_legend() is None
    assert ax.get_xlim() == pytest.approx(viz.MEAN_AP_SHARED_XLIM)
    assert ax.get_xlabel() == "Mean AP"
    markers = [
        (line.get_marker(), line.get_color())
        for line in ax.get_lines()
        if line.get_marker() in {"D", "o"}
    ]
    assert markers == [
        ("D", viz.METHOD_COLOR["proposed"]),
        ("o", viz.COMPARATOR_GRAY),
        ("o", viz.COMPARATOR_GRAY),
        ("o", viz.COMPARATOR_GRAY),
    ]
    assert not [line for line in ax.get_lines() if line.get_linestyle() == ":"]


def test_fixed_k_figure_skips_when_metric_column_missing(tmp_path):
    summary = _ablation_summary().drop(columns=["mean_Lift@10", "mean_Lift@10_SEM"])

    outputs = viz._save_fixed_k_internal_ablation_figure(
        summary,
        out_png=tmp_path / "lift10.png",
        k=PRIMARY_K,
        mean_col="mean_Lift@10",
        sem_col="mean_Lift@10_SEM",
        ylabel="Mean Lift@10",
    )

    assert outputs == {}
    assert not (tmp_path / "lift10.png").exists()
