from __future__ import annotations

import shutil
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Val3TripletConfig
from .io import ensure_dir, write_json
from .registry import Val3RegistrySpec


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _public_context(context: dict[str, Any]) -> dict[str, Any]:
    return {str(k): v for k, v in context.items() if not str(k).startswith("_")}


def _write_run_metadata(context: dict[str, Any], run_root: str | Path) -> None:
    write_json(_public_context(context), Path(run_root) / "metadata.json")


def _git_commit(repo_root: str | Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return None


def initialize_run_context(*, cfg: Val3TripletConfig, spec: Val3RegistrySpec, run_root: str | Path, repo_root: str | Path) -> dict[str, Any]:
    run_root = ensure_dir(run_root)
    cfg_dir = ensure_dir(Path(run_root) / "config")
    if cfg.config_path.exists():
        shutil.copy2(cfg.config_path, cfg_dir / cfg.config_path.name)
    started_at = _iso_now()
    context = {
        "started_at": started_at,
        "finished_at": None,
        "duration_sec": None,
        "duration_min": None,
        "section_timings": [],
        "section_runtime_sec": {},
        "config_path": str(cfg.config_path),
        "registry_path": str(cfg.registry_path),
        "run_root": str(run_root),
        "seed": int(cfg.seed),
        "data": cfg.data,
        "resolved_data": {
            **cfg.data,
            "target_profile": spec.target_profile_key,
        },
        "sections": cfg.sections,
        "analysis": cfg.analysis,
        "baseline": cfg.baseline,
        "evaluation": cfg.evaluation,
        "progress_enabled": bool(cfg.progress.get("enabled", True)),
        "storage": cfg.storage,
        "kg_path": str(spec.kg_path),
        "triplet_tsv": str(spec.triplet_tsv),
        "git_commit": _git_commit(repo_root),
    }
    write_json(_public_context(context), cfg_dir / "resolved_config.json")
    context["_perf_start"] = time.perf_counter()
    _write_run_metadata(context, run_root)
    return context


@contextmanager
def timed_run_section(context: dict[str, Any], *, run_root: str | Path, section: str):
    started_at = _iso_now()
    started_perf = time.perf_counter()
    status = "completed"
    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        finished_at = _iso_now()
        duration_sec = float(time.perf_counter() - started_perf)
        record = {
            "section": str(section),
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_sec": duration_sec,
            "duration_min": float(duration_sec / 60.0),
            "status": status,
        }
        context.setdefault("section_timings", []).append(record)
        runtime = context.setdefault("section_runtime_sec", {})
        runtime[str(section)] = float(runtime.get(str(section), 0.0) + duration_sec)
        context["last_updated_at"] = finished_at
        _write_run_metadata(context, run_root)


def finalize_run_context(context: dict[str, Any], *, run_root: str | Path) -> dict[str, Any]:
    finished_at = _iso_now()
    started_perf = context.get("_perf_start")
    duration_sec = None
    if started_perf is not None:
        duration_sec = float(time.perf_counter() - float(started_perf))
    context["finished_at"] = finished_at
    context["duration_sec"] = duration_sec
    context["duration_min"] = None if duration_sec is None else float(duration_sec / 60.0)
    _write_run_metadata(context, run_root)
    return _public_context(context)


def finalize_run_summary(summary: dict[str, Any], *, run_root: str | Path) -> None:
    write_json(summary, Path(run_root) / "run_summary.json")
