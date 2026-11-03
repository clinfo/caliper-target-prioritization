from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo
import json
import shutil
import time

from common.metadata import json_safe as _json_safe, write_json_safe
from val4.config import resolved_config_dict

TOKYO = ZoneInfo("Asia/Tokyo")


def _relative(run_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(run_root))
    except Exception:
        return str(path)


def write_config_snapshot(*, cfg, output_dir: Path) -> Dict[str, str]:
    config_dir = output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}

    config_copy = config_dir / cfg.config_path.name
    try:
        shutil.copy2(cfg.config_path, config_copy)
    except Exception:
        config_copy.write_text("# Original config could not be copied.\n", encoding="utf-8")
    outputs["config"] = _relative(output_dir, config_copy)

    common_path = getattr(cfg, "common_config_path", None)
    if common_path is not None:
        common_copy = config_dir / common_path.name
        try:
            shutil.copy2(common_path, common_copy)
        except Exception:
            common_copy.write_text("# Original common config could not be copied.\n", encoding="utf-8")
        outputs["common_config"] = _relative(output_dir, common_copy)

    resolved_path = config_dir / "resolved_config.json"
    write_json_safe(_json_safe(resolved_config_dict(cfg)), resolved_path)
    outputs["resolved_config"] = _relative(output_dir, resolved_path)
    return outputs


def initialize_run_context(
    *,
    cfg,
    dataset_spec,
    output_dir: Path,
    task: str | None = None,
) -> Dict[str, Any]:
    task_name = task or getattr(cfg, "task", "val4")
    started_at = datetime.now(TOKYO)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_null = int(cfg.analysis.get("n_null", 0)) if hasattr(cfg, "analysis") else 0
    bootstrap_reps = int(cfg.analysis.get("bootstrap_reps", 0)) if hasattr(cfg, "analysis") else 0
    main_null = cfg.main.get("null", {}) if isinstance(getattr(cfg, "main", {}), dict) else {}
    lofo_null = cfg.lofo.get("null", {}) if isinstance(getattr(cfg, "lofo", {}), dict) else {}
    return {
        "run_id": output_dir.name,
        "started_at": started_at,
        "finished_at": None,
        "duration_sec": None,
        "duration_min": None,
        "timezone": "Asia/Tokyo",
        "run_root": str(output_dir),
        "output_dir": str(output_dir),
        "config_path": str(cfg.config_path),
        "common_config_path": None if getattr(cfg, "common_config_path", None) is None else str(cfg.common_config_path),
        "registry_path": str(cfg.registry_path),
        "seed": int(cfg.seed),
        "data": {
            "kg": dataset_spec.kg,
            "perturbation_mode": dataset_spec.perturbation_mode,
            "label_source": dataset_spec.label_source,
        },
        "sections": dict(getattr(cfg, "sections", {})),
        "params": {
            "task": task_name,
            "k_star": int(cfg.analysis["k_star"]),
            "gap": None if "gap" not in cfg.analysis else int(cfg.analysis.get("gap", 0)),
            "path_strategy": str(cfg.analysis.get("path_strategy", "")),
            "condition_universe": str(cfg.analysis.get("condition_universe", "")),
            "epsilon": None if "epsilon" not in cfg.analysis else float(cfg.analysis.get("epsilon", 0.0)),
            "family_selection_mode": str(cfg.family_selection.get("mode", "kg_support")),
            "min_physical_edges": int(cfg.family_selection.get("min_physical_edges", 20)),
            "n_null": n_null,
            "bootstrap_reps": bootstrap_reps,
            "multi_family_edges": str(cfg.analysis.get("multi_family_edges", "")),
            "exclude_multi_family_terminal_edges": bool(cfg.analysis.get("exclude_multi_family_terminal_edges", False)),
            "save_observation_rows": bool(cfg.analysis.get("save_observation_rows", False)),
            "save_condition_counts": bool(cfg.analysis.get("save_condition_counts", False)),
            "terminal_family_removal": str(getattr(cfg, "main", {}).get("terminal_family_removal", "")),
            "main_null_mode": str(main_null.get("mode", "")),
            "lofo_enabled": bool(getattr(cfg, "lofo", {}).get("enabled", False)),
            "lofo_strict_edge_removal": bool(getattr(cfg, "lofo", {}).get("strict_edge_removal", False)),
            "lofo_null_mode": str(lofo_null.get("mode", "")),
            "resolved_start_sign": None,
            "family_definition": dict(cfg.family_definition),
        },
        "resolved_inputs": {
            "ppi_path": str(dataset_spec.ppi_path),
            "parquet_path": str(dataset_spec.parquet_path),
            "registry_path": str(cfg.registry_path),
            "config_path": str(cfg.config_path),
        },
        "data_summary": {},
        "section_outputs": {},
        "section_timings": {},
        "output_manifest": "output_manifest.json",
        "_perf_start": time.perf_counter(),
        "_section_perf_start": {},
    }


def mark_section_start(run_context: Dict[str, Any], section: str) -> None:
    run_context.setdefault("_section_perf_start", {})[str(section)] = time.perf_counter()


def mark_section_end(run_context: Dict[str, Any], section: str) -> None:
    starts = run_context.setdefault("_section_perf_start", {})
    started = starts.pop(str(section), None)
    duration = 0.0 if started is None else float(time.perf_counter() - float(started))
    run_context.setdefault("section_timings", {})[str(section)] = {
        "duration_sec": duration,
        "duration_min": duration / 60.0,
    }


def _manifest_entries(output_dir: Path, section: str, outputs: Dict[str, str]) -> list[dict[str, Any]]:
    entries = []
    for logical_name, filename in outputs.items():
        rel = Path(filename)
        p = output_dir / rel
        entries.append(
            {
                "section": section,
                "logical_name": logical_name,
                "relative_path": rel.as_posix(),
                "exists": p.exists(),
                "size_bytes": int(p.stat().st_size) if p.exists() and p.is_file() else None,
            }
        )
    return entries


def build_output_manifest(*, run_root: Path, section_outputs: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {"sections": {}, "all_outputs": []}
    for section_name, outputs in section_outputs.items():
        entries = _manifest_entries(run_root, section_name, outputs)
        manifest["sections"][section_name] = entries
        manifest["all_outputs"].extend(entries)
    return manifest


def finalize_run_metadata(
    *,
    run_context: Dict[str, Any],
    n_nodes: int,
    n_edges: int,
    n_pairs: int,
    n_perturbations: int,
    n_common_genes: int,
    output_dir: Path,
    resolved_start_sign: int,
    outputs: Dict[str, str] | None = None,
    section_outputs: Dict[str, Dict[str, str]] | None = None,
    n_families: int | None = None,
    n_disease_signatures: int | None = None,
    summary: Dict[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    if section_outputs is None:
        section_outputs = {"outputs": dict(outputs or {})}

    finished_at = datetime.now(TOKYO)
    started_perf = run_context.get("_perf_start")
    duration_sec = float(time.perf_counter() - float(started_perf)) if started_perf is not None else None
    run_context["finished_at"] = finished_at
    run_context["duration_sec"] = duration_sec
    run_context["duration_min"] = None if duration_sec is None else duration_sec / 60.0
    run_context["params"]["resolved_start_sign"] = int(resolved_start_sign)
    run_context["data_summary"] = {
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "n_pairs": int(n_pairs),
        "n_perturbations": int(n_perturbations),
        "n_pert_sigs": int(n_perturbations),
        "n_dis_sigs": None if n_disease_signatures is None else int(n_disease_signatures),
        "n_common_genes": int(n_common_genes),
        "n_families": None if n_families is None else int(n_families),
    }
    run_context["section_outputs"] = section_outputs

    manifest = build_output_manifest(run_root=output_dir, section_outputs=section_outputs)
    manifest_path = write_json_safe(_json_safe(manifest), output_dir / "output_manifest.json")

    clean = {k: v for k, v in run_context.items() if not k.startswith("_")}
    clean["output_manifest"] = "output_manifest.json"
    metadata_path = write_json_safe(_json_safe(clean), output_dir / "metadata.json")
    run_summary = {
        "run_id": clean.get("run_id"),
        "run_root": str(output_dir),
        "started_at": clean.get("started_at"),
        "finished_at": clean.get("finished_at"),
        "duration_sec": clean.get("duration_sec"),
        "duration_min": clean.get("duration_min"),
        "metadata": "metadata.json",
        "output_manifest": "output_manifest.json",
    }
    if summary:
        run_summary.update(_json_safe(summary))
    run_summary_path = write_json_safe(_json_safe(run_summary), output_dir / "run_summary.json")
    return metadata_path, manifest_path, run_summary_path


def append_outputs_to_manifest(run_root: Path, outputs: Dict[str, str], *, section_name: str = "outputs") -> None:
    if not outputs:
        return
    run_root = Path(run_root)
    manifest_path = run_root / "output_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"sections": {}, "all_outputs": []}
    manifest.setdefault("sections", {})
    manifest.setdefault("all_outputs", [])

    new_entries = _manifest_entries(run_root, section_name, outputs)
    manifest["sections"][section_name] = new_entries
    manifest["all_outputs"] = [
        row for row in manifest.get("all_outputs", []) if row.get("section") != section_name
    ] + new_entries
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(manifest), f, indent=2, ensure_ascii=False)

    for metadata_name in ("metadata.json",):
        metadata_path = run_root / metadata_name
        if not metadata_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if not isinstance(metadata, dict):
            continue
        section_outputs = metadata.setdefault("section_outputs", {})
        if isinstance(section_outputs, dict):
            section_outputs[section_name] = dict(outputs)
        metadata["output_manifest"] = "output_manifest.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(metadata), f, indent=2, ensure_ascii=False)
