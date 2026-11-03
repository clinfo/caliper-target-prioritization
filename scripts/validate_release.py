#!/usr/bin/env python3
"""Validate public CALIPER metadata, templates, imports, and synthetic fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


EXPECTED_DOCS = {
    "docs/reproducibility.md",
    "docs/method_overview.md",
    "docs/software_versions.tsv",
    "docs/analysis_to_script_map.tsv",
    "docs/data_provenance.md",
    "docs/data_licenses.md",
    "docs/source_data_manifest.tsv",
}
EXPECTED_CONFIGS = {
    "config/dataset_registry.example.yaml",
    "config/paper/val1.yaml",
    "config/paper/val2.yaml",
    "config/paper/val2_baseline.yaml",
    "config/paper/val3.yaml",
    "config/paper/val4/common.yaml",
    "config/paper/val4/h1.yaml",
    "config/paper/val4/h2.yaml",
}
MANIFEST_COLUMNS = [
    "asset_path", "paper_component", "panel_or_table", "canonical_run_alias",
    "source_run_root", "canonical_source_file", "extraction_method", "rows",
    "columns", "format", "size_bytes", "sha256", "upstream_resources",
    "contains_third_party_rows", "redistribution_status", "release_license", "notes",
]


def validate_manifest(path: Path, root: Path | None) -> None:
    header = path.read_text(encoding="utf-8").splitlines()[0].split("\t")
    if header != MANIFEST_COLUMNS:
        raise ValueError(f"Unexpected manifest header: {path}")
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != len(MANIFEST_COLUMNS):
            raise ValueError(f"Incorrect field count: {path}")
        row = dict(zip(MANIFEST_COLUMNS, fields, strict=True))
        if len(row["sha256"]) != 64 or any(char not in "0123456789abcdef" for char in row["sha256"]):
            raise ValueError(f"Invalid SHA-256: {path}::{row['asset_path']}")
        if row["redistribution_status"] not in {"PUBLIC_OK", "EXCLUDE", "BLOCKED"}:
            raise ValueError(f"Invalid redistribution status: {path}::{row['asset_path']}")
        if root is not None:
            asset = root / row["asset_path"]
            if not asset.is_file():
                raise FileNotFoundError(asset)
            digest = hashlib.sha256(asset.read_bytes()).hexdigest()
            if row["sha256"] != digest:
                raise ValueError(f"SHA-256 mismatch: {asset}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-data", type=Path)
    parser.add_argument("--run-fixture", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root / "src"))

    missing = sorted(path for path in EXPECTED_DOCS | EXPECTED_CONFIGS if not (root / path).is_file())
    if missing:
        raise FileNotFoundError("Missing release files: " + ", ".join(missing))
    for path in sorted(EXPECTED_CONFIGS):
        with (root / path).open(encoding="utf-8") as handle:
            if not isinstance(yaml.safe_load(handle), dict):
                raise ValueError(f"YAML top level is not a mapping: {path}")
    json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
    with (root / "CITATION.cff").open(encoding="utf-8") as handle:
        citation = yaml.safe_load(handle)
    if citation.get("title") != "CALIPER" or str(citation.get("version")) != "1.0.0":
        raise ValueError("CITATION.cff title/version mismatch")
    # This is a metadata-only index of the separate source-data archive.
    validate_manifest(root / "docs/source_data_manifest.tsv", None)

    for module in ("common", "inference", "val1", "val2", "val2_baseline", "val3", "val4"):
        __import__(module)
    if args.source_data:
        validate_manifest(args.source_data / "MANIFEST.tsv", args.source_data)
    if args.run_fixture:
        subprocess.run([sys.executable, "scripts/val3_equivalence_fixture.py", "check"], cwd=root, check=True)
    print("release validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
