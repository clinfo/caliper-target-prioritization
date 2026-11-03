from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from val1 import viz


@pytest.mark.parametrize(
    ("plotter", "filename"),
    [
        (viz.plot_path_runtime_vs_k, "path_runtime_vs_k.png"),
        (viz.plot_path_budget_rate_vs_k, "path_budget_rate_vs_k.png"),
    ],
)
def test_path_diagnostic_curves_use_primary_val1_color(
    tmp_path: Path,
    plotter,
    filename: str,
) -> None:
    summary = pd.DataFrame(
        {
            "cutoff": [1, 2],
            "runtime_sec_total": [0.1, 1.0],
            "within_budget_rate": [1.0, 0.5],
        }
    )
    k_metrics = pd.DataFrame({"k": [1, 2]})

    plotter(summary, k_metrics, tmp_path / filename)

    svg = (tmp_path / filename).with_suffix(".svg").read_text(encoding="utf-8")
    assert "stroke: #1a4e79" in svg.lower()
