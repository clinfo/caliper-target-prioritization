from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import copy
import json
import os
import platform
import shutil
import sys
import time
import uuid

import networkx as nx
import numpy as np
import pandas as pd

from .config import Val1Config, resolved_config_dict
from .io import ensure_dir, write_json


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment_block() -> Dict[str, Any]:
    """Hardware / threading / software versions (P0-8 item 5).

    Must be recorded before the run starts; it cannot be reconstructed later.
    """
    return {
        "package": "caliper-td",
        "component": "val1",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "threads": {k: os.environ.get(k) for k in THREAD_ENV_VARS},
        "execution": "serial",
        "python": sys.version.split()[0],
        "packages": {
            "networkx": nx.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return obj


def initialize_run_metadata(cfg: Val1Config, run_root: str | Path, *, ks: list[int] | None = None) -> Dict[str, Any]:
    run_root = ensure_dir(run_root)
    ensure_dir(run_root / "config")
    try:
        shutil.copy2(cfg.config_path, run_root / "config" / cfg.config_path.name)
    except Exception:
        pass

    resolved = resolved_config_dict(cfg)
    write_json(resolved, run_root / "config" / "resolved_config.json")

    run_id = f"val1-{uuid.uuid4().hex[:12]}"
    started_at = iso_now()
    canonical_definition = bool(cfg.analysis.get("ks_policy") == "kmax")
    meta: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": None,
        "duration_sec": None,
        "duration_min": None,
        "run_root": str(run_root),
        "config_path": str(cfg.config_path),
        "registry_path": str(cfg.registry_path),
        "data": dict(cfg.data),
        "params": {
            "ks": list(ks or []),
            "coverage": dict(cfg.metrics["coverage"]),
            "ambiguity": dict(cfg.metrics["ambiguity"]),
            "integration": dict(cfg.metrics["integration"]),
            "selection": dict(cfg.metrics["selection"]),
            "selection_sensitivity": copy.deepcopy(cfg.metrics["selection_sensitivity"]),
            "path_benchmark": copy.deepcopy(cfg.analysis.get("path_benchmark", {})),
        },
        "canonical_definition": canonical_definition,
        "cov_ratio_empty_base_policy": "nan",
        "cov_ratio_primary_aggregation": "median",
        "environment": environment_block(),
        "section_outputs": {},
    }
    write_json(meta, run_root / "metadata.json")
    meta["_perf_start"] = time.perf_counter()
    return meta


def record_path_benchmark_metadata(
    run_root: str | Path,
    *,
    path_benchmark: Dict[str, Any],
    ks: list[int],
    section_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    """Record the path-benchmark parameters and execution environment in place.

    Used by ``--steps pathbench`` runs, which do not re-initialize the run
    metadata: an existing ``metadata.json`` (and the ``run_summary.json`` of the
    run it belongs to) must survive untouched apart from these keys.
    """
    run_root = ensure_dir(run_root)
    metadata_path = run_root / "metadata.json"
    meta: Dict[str, Any] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            meta = loaded
    meta.setdefault("run_root", str(run_root))
    params = meta.get("params")
    if not isinstance(params, dict):
        params = {}
    params["path_benchmark"] = copy.deepcopy(path_benchmark)
    params["path_benchmark_ks"] = [int(k) for k in ks]
    meta["params"] = params
    # The environment must describe the machine that ran the benchmark.
    meta["environment"] = environment_block()
    existing_sections = meta.get("section_outputs")
    sections = dict(existing_sections) if isinstance(existing_sections, dict) else {}
    sections["path_benchmark"] = _jsonable(section_outputs)
    meta["section_outputs"] = sections
    write_json(meta, metadata_path)
    return meta


def build_output_manifest(run_root: str | Path) -> Dict[str, Any]:
    run_root = Path(run_root)
    files = []
    for p in sorted(run_root.rglob("*")):
        if p.is_file():
            try:
                size = p.stat().st_size
            except OSError:
                size = None
            files.append({"path": str(p.relative_to(run_root)), "size_bytes": size})
    return {"run_root": str(run_root), "n_files": len(files), "files": files}


def refresh_posthoc_run_metadata(
    run_root: str | Path,
    *,
    section_outputs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Refresh metadata/manifest after rules, figures, or report-only steps.

    Post-hoc output generation must preserve the original run lifecycle fields.
    The manifest is written to a size fixed point because it includes itself.
    """
    run_root = Path(run_root)
    metadata_path = run_root / "metadata.json"
    meta: Dict[str, Any] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            meta = loaded

    existing_sections = meta.get("section_outputs")
    sections = dict(existing_sections) if isinstance(existing_sections, dict) else {}
    if section_outputs is not None:
        sections.update(_jsonable(section_outputs))
    meta["section_outputs"] = sections
    write_json(meta, metadata_path)

    manifest_path = run_root / "output_manifest.json"
    manifest: Dict[str, Any] = {}
    for _ in range(8):
        manifest = build_output_manifest(run_root)
        write_json(manifest, manifest_path)
        if build_output_manifest(run_root) == manifest:
            break
    else:
        raise RuntimeError(f"Output manifest did not reach a stable size: {manifest_path}")
    return {"metadata": meta, "output_manifest": manifest}


def finalize_run_metadata(
    meta: Dict[str, Any],
    run_root: str | Path,
    *,
    section_outputs: Dict[str, Any] | None = None,
    summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    run_root = Path(run_root)
    finished_at = iso_now()
    started_perf = meta.get("_perf_start")
    if started_perf is not None:
        duration_sec = float(time.perf_counter() - float(started_perf))
    else:
        duration_sec = None

    clean_meta = {k: v for k, v in meta.items() if not k.startswith("_")}
    clean_meta["finished_at"] = finished_at
    clean_meta["duration_sec"] = duration_sec
    clean_meta["duration_min"] = None if duration_sec is None else float(duration_sec / 60.0)
    if section_outputs is not None:
        clean_meta["section_outputs"] = _jsonable(section_outputs)

    manifest = build_output_manifest(run_root)
    write_json(manifest, run_root / "output_manifest.json")
    clean_meta["output_manifest"] = "output_manifest.json"
    write_json(clean_meta, run_root / "metadata.json")

    run_summary = dict(summary or {})
    run_summary.update(
        {
            "run_id": clean_meta.get("run_id"),
            "run_root": str(run_root),
            "started_at": clean_meta.get("started_at"),
            "finished_at": finished_at,
            "duration_sec": duration_sec,
            "duration_min": clean_meta.get("duration_min"),
            "metadata": "metadata.json",
            "output_manifest": "output_manifest.json",
        }
    )
    write_json(run_summary, run_root / "run_summary.json")
    return clean_meta


# Backward-compatible names used by the previous runner.
init_run_metadata = initialize_run_metadata
finalize_metadata = finalize_run_metadata
