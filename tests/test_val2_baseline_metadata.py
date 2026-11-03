from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from val2_baseline import metadata as metadata_mod


def test_build_output_manifest_shape(tmp_path: Path) -> None:
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "method_summary.tsv").write_text("a\tb\n1\t2\n")

    section_outputs = {
        "evaluation": {
            "method_summary": "evaluation/method_summary.tsv",
            "missing_file": "evaluation/does_not_exist.tsv",
        }
    }

    manifest = metadata_mod.build_output_manifest(section_outputs, run_root=tmp_path)

    assert manifest["sections"] == section_outputs
    by_logical = {entry["logical_name"]: entry for entry in manifest["all_outputs"]}
    assert set(by_logical) == {"method_summary", "missing_file"}

    present = by_logical["method_summary"]
    assert present["section"] == "evaluation"
    assert present["relative_path"] == "evaluation/method_summary.tsv"
    assert present["exists"] is True
    assert present["size_bytes"] == (tmp_path / "evaluation" / "method_summary.tsv").stat().st_size

    missing = by_logical["missing_file"]
    assert missing["exists"] is False
    assert missing["size_bytes"] is None


def test_write_run_metadata_writes_three_files(tmp_path: Path) -> None:
    metadata = {"run_id": "r1", "primary_analysis": "native_shared_network"}
    output_manifest = metadata_mod.build_output_manifest({}, run_root=tmp_path)
    run_summary = {"run_root": str(tmp_path), "sections": {"scores": True}}

    metadata_mod.write_run_metadata(metadata, output_manifest, run_summary, run_root=tmp_path)

    written_metadata = json.loads((tmp_path / "metadata.json").read_text())
    written_manifest = json.loads((tmp_path / "output_manifest.json").read_text())
    written_summary = json.loads((tmp_path / "run_summary.json").read_text())

    assert written_metadata == metadata
    assert written_manifest == output_manifest
    assert written_summary == run_summary


def test_refresh_output_manifest_updates_file_size(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    result = evaluation / "result.tsv"
    result.write_text("old\n")
    manifest = metadata_mod.build_output_manifest(
        {"evaluation": {"result": "evaluation/result.tsv"}},
        run_root=tmp_path,
    )
    metadata_mod.write_json(manifest, tmp_path / "output_manifest.json")

    result.write_text("new and longer\n")
    metadata_mod.refresh_output_manifest(tmp_path)

    refreshed = json.loads((tmp_path / "output_manifest.json").read_text())
    entry = refreshed["all_outputs"][0]
    assert entry["relative_path"] == "evaluation/result.tsv"
    assert entry["exists"] is True
    assert entry["size_bytes"] == result.stat().st_size
