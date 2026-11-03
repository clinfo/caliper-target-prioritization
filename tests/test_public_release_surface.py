from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(relative: str) -> dict:
    with (REPO_ROOT / relative).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_paper_templates_record_reported_settings() -> None:
    val1 = load_yaml("config/paper/val1.yaml")
    val2 = load_yaml("config/paper/val2.yaml")
    val2_baseline = load_yaml("config/paper/val2_baseline.yaml")
    val3 = load_yaml("config/paper/val3.yaml")
    h1 = load_yaml("config/paper/val4/h1.yaml")
    h2 = load_yaml("config/paper/val4/h2.yaml")
    common = load_yaml("config/paper/val4/common.yaml")

    assert val1["run"]["seed"] == 0
    assert val1["analysis"]["path_benchmark"]["max_paths"] == 100000
    assert val1["analysis"]["path_benchmark"]["timeout_sec"] == 600
    assert val2["ablation"]["sizematch_reps"] == 100
    assert val2_baseline["external_baseline"]["proximity"]["n_random"] == 1000
    assert val2_baseline["external_baseline"]["rwr"]["restart_probability"] == 0.5
    assert val3["analysis"]["k_star"] == 4
    assert val3["analysis"]["gap"] == 1
    assert val3["baseline"]["n_shuffle"] == 100
    assert val3["baseline"]["null_models"]["topology_profile_matched_target_decoy_control"]["n_decoys_per_triplet"] == 10
    assert h1["analysis"] == {"n_null": 1000, "bootstrap_reps": 1000, "save_observation_rows": True}
    assert h2["analysis"]["n_null"] == 100
    assert h2["analysis"]["bootstrap_reps"] == 1000
    assert common["analysis"]["k_star"] == 4
    assert common["analysis"]["gap"] == 1


def test_release_metadata_is_complete_and_has_no_url_or_doi_claim() -> None:
    metadata = json.loads((REPO_ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    assert metadata["version"] == "1.0.0"
    assert "doi" not in metadata
    assert "url" not in metadata
    assert len(metadata["creators"]) == 4


def test_release_scan_has_no_findings() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/scan_release.py", "--repo-root", "."],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["finding_count"] == 0
