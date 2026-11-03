from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from val1.metadata import refresh_posthoc_run_metadata


def test_refresh_posthoc_metadata_preserves_run_lifecycle_and_stabilizes_manifest(
    tmp_path: Path,
) -> None:
    lifecycle = {
        "run_id": "val1-existing",
        "started_at": "2026-06-29T00:00:00+00:00",
        "finished_at": "2026-06-29T00:10:00+00:00",
        "duration_sec": 600.0,
        "section_outputs": {"metrics": {"k_metrics": "metrics/k_metrics.tsv"}},
    }
    (tmp_path / "metadata.json").write_text(
        json.dumps(lifecycle),
        encoding="utf-8",
    )
    figures_dir = tmp_path / "report" / "figures"
    figures_dir.mkdir(parents=True)
    (figures_dir / "coverage_vs_k.png").write_bytes(b"figure")

    refresh_posthoc_run_metadata(
        tmp_path,
        section_outputs={
            "figures": {"coverage_vs_k": "report/figures/coverage_vs_k.png"}
        },
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == lifecycle["run_id"]
    assert metadata["started_at"] == lifecycle["started_at"]
    assert metadata["finished_at"] == lifecycle["finished_at"]
    assert metadata["duration_sec"] == lifecycle["duration_sec"]
    assert metadata["section_outputs"]["metrics"] == lifecycle["section_outputs"]["metrics"]
    assert metadata["section_outputs"]["figures"]["coverage_vs_k"].endswith(
        "coverage_vs_k.png"
    )

    manifest = json.loads(
        (tmp_path / "output_manifest.json").read_text(encoding="utf-8")
    )
    by_path = {row["path"]: row["size_bytes"] for row in manifest["files"]}
    assert manifest["n_files"] == len(
        [path for path in tmp_path.rglob("*") if path.is_file()]
    )
    assert by_path["metadata.json"] == (tmp_path / "metadata.json").stat().st_size
    assert by_path["output_manifest.json"] == (
        tmp_path / "output_manifest.json"
    ).stat().st_size
    assert by_path["report/figures/coverage_vs_k.png"] == len(b"figure")
