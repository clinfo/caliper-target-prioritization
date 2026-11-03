"""metadata.json / output_manifest.json / run_summary.json writers.

Centralizes run-output manifest writing, mirroring the pattern used by
``val1.metadata`` and ``val2.metadata``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from common.metadata import write_json_safe as write_json


def build_output_manifest(section_outputs: Dict[str, Dict[str, str]], *, run_root: Path) -> Dict[str, Any]:
    manifest_sections = {name: outs for name, outs in section_outputs.items()}
    all_outputs = []
    for name, outs in manifest_sections.items():
        for logical, rel in outs.items():
            p = run_root / rel
            all_outputs.append(
                {
                    "section": name,
                    "logical_name": logical,
                    "relative_path": rel,
                    "exists": p.exists(),
                    "size_bytes": int(p.stat().st_size) if p.exists() and p.is_file() else None,
                }
            )
    return {"sections": manifest_sections, "all_outputs": all_outputs}


def write_run_metadata(
    metadata: Dict[str, Any],
    output_manifest: Dict[str, Any],
    run_summary: Dict[str, Any],
    *,
    run_root: Path,
) -> None:
    write_json(metadata, run_root / "metadata.json")
    write_json(output_manifest, run_root / "output_manifest.json")
    write_json(run_summary, run_root / "run_summary.json")


def refresh_output_manifest(run_root: Path) -> Path:
    """Refresh existence and size fields without changing recorded outputs."""
    manifest_path = run_root / "output_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        existing = json.load(handle)
    sections = existing.get("sections", {}) if isinstance(existing, dict) else {}
    if not isinstance(sections, dict):
        raise ValueError(f"Invalid manifest sections in {manifest_path}")
    refreshed = build_output_manifest(sections, run_root=run_root)
    write_json(refreshed, manifest_path)
    return manifest_path
