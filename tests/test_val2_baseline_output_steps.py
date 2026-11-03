from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from val2_baseline import report, runner, viz  # noqa: E402


def test_figure_report_only_preserves_analysis_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    metric_path = tmp_path / "evaluation" / "metrics_by_disease_method.tsv"
    metric_path.parent.mkdir(parents=True)
    metric_path.write_bytes(b"disease_id\tmethod\tAP\nD1\tCALIPER\t0.5\n")
    before = hashlib.sha256(metric_path.read_bytes()).hexdigest()
    (tmp_path / "output_manifest.json").write_text(
        json.dumps({"sections": {}, "all_outputs": []})
    )
    figure_path = tmp_path / "report" / "figures" / "figure.svg"
    report_path = tmp_path / "report" / "val2_baseline_report.md"

    def fake_figures(_run_root):
        figure_path.parent.mkdir(parents=True)
        figure_path.write_text("<svg/>")
        return {"figure_svg": "report/figures/figure.svg"}

    def fake_report(_run_root):
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("report\n")
        return report_path

    monkeypatch.setattr(viz, "make_all_figures", fake_figures)
    monkeypatch.setattr(report, "write_report", fake_report)

    result = runner._run_output_steps(tmp_path, {"figures", "report"})

    after = hashlib.sha256(metric_path.read_bytes()).hexdigest()
    assert result["steps"] == ["figures", "report"]
    assert after == before
    manifest = json.loads((tmp_path / "output_manifest.json").read_text())
    assert "figures" in manifest["sections"]
    assert "report" in manifest["sections"]
