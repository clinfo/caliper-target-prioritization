from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from val2 import runner


def test_report_only_refreshes_manifest_after_final_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_path = tmp_path / "report" / "val2_report.md"
    report_path.parent.mkdir()
    report_path.write_text("initial\n")
    manifest = {
        "sections": {
            "report": [
                {
                    "section": "report",
                    "logical_name": "val2_report",
                    "relative_path": "report/val2_report.md",
                    "exists": True,
                    "size_bytes": report_path.stat().st_size,
                }
            ]
        },
        "all_outputs": [
            {
                "section": "report",
                "logical_name": "val2_report",
                "relative_path": "report/val2_report.md",
                "exists": True,
                "size_bytes": report_path.stat().st_size,
            }
        ],
    }
    (tmp_path / "output_manifest.json").write_text(json.dumps(manifest))

    writes = iter(("first report\n", "final report with manifest table\n"))

    def fake_write_report(*, run_root: Path) -> dict[str, str]:
        report_path.write_text(next(writes))
        return {"report_md": str(report_path)}

    monkeypatch.setattr(runner, "make_all_stats", lambda run_root: {})
    monkeypatch.setattr(runner, "write_val2_report", fake_write_report)

    runner._run_output_steps(tmp_path, {"report"})

    refreshed = json.loads((tmp_path / "output_manifest.json").read_text())
    report_entry = refreshed["sections"]["report"][0]
    assert report_entry["size_bytes"] == report_path.stat().st_size


def test_figure_report_only_threads_targetnexus_and_preserves_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    metric_path = tmp_path / "ablation" / "val2_metrics_k1-10.csv"
    metric_path.parent.mkdir()
    metric_path.write_bytes(b"k,method,AP\n4,proposed,0.25\n")
    before = hashlib.sha256(metric_path.read_bytes()).hexdigest()
    (tmp_path / "output_manifest.json").write_text(
        json.dumps({"sections": {}, "all_outputs": []})
    )
    targetnexus_root = tmp_path / "targetnexus"
    seen: dict[str, Path] = {}

    monkeypatch.setattr(runner, "make_all_stats", lambda _run_root: {})

    def fake_figures(run_root, *, targetnexus_root=None):
        seen["targetnexus_root"] = Path(targetnexus_root)
        figure = Path(run_root) / "report" / "figures" / "figure.svg"
        figure.parent.mkdir(parents=True)
        figure.write_text("<svg/>")
        return {"figure_svg": "report/figures/figure.svg"}

    report_path = tmp_path / "report" / "val2_report.md"

    def fake_report(*, run_root):
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("report\n")
        return {"report_md": str(report_path)}

    monkeypatch.setattr(runner, "make_all_figures", fake_figures)
    monkeypatch.setattr(runner, "write_val2_report", fake_report)

    runner._run_output_steps(
        tmp_path,
        {"figures", "report"},
        targetnexus_root=targetnexus_root,
    )

    after = hashlib.sha256(metric_path.read_bytes()).hexdigest()
    assert seen["targetnexus_root"] == targetnexus_root
    assert after == before
