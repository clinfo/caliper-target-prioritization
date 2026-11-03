from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo

from .config import ResolvedConfig
from .registry import DatasetSpec
from .io import DataBundle
from common.metadata import json_safe as _json_safe, relative_path as _relative, write_json_safe as write_json


TOKYO = ZoneInfo("Asia/Tokyo")



def initialize_run_context(
    *,
    cfg: ResolvedConfig,
    dataset_spec: DatasetSpec,
    output_dir: Path,
) -> Dict[str, Any]:
    started_at = datetime.now(TOKYO)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_id": output_dir.name,
        "started_at": started_at,
        "finished_at": None,
        "duration_sec": None,
        "duration_min": None,
        "timezone": "Asia/Tokyo",
        "seed": int(cfg.seed),
        "data": {
            "kg": dataset_spec.kg,
            "perturbation_mode": dataset_spec.perturbation_mode,
            "label_source": dataset_spec.label_source,
        },
        "sections": dict(cfg.sections),
        "params": {
            "ks": list(cfg.analysis["ks"]),
            "k_valid": list(cfg.analysis.get("k_valid", cfg.analysis["ks"])),
            "K_list": list(cfg.analysis["K_list"]),
            "tie_policy": str(cfg.analysis.get("tie_policy", "expected")),
            "resolved_start_sign": None,
            "ablation": dict(cfg.ablation),
            "permutation": dict(cfg.permutation),
        },
        "resolved_inputs": {
            "ppi_path": str(dataset_spec.ppi_path),
            "labels_path": str(dataset_spec.parquet_path),
            "registry_path": str(cfg.registry_path),
            "config_path": str(cfg.config_path),
        },
        "data_summary": {},
        "output_dir": str(output_dir),
    }



def build_output_manifest(*, run_root: Path, section_outputs: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    manifest = {"sections": {}, "all_outputs": []}
    for section_name, outputs in section_outputs.items():
        manifest["sections"][section_name] = []
        for logical_name, relative_path in outputs.items():
            path = run_root / str(relative_path)
            entry = {
                "section": section_name,
                "logical_name": logical_name,
                "relative_path": str(relative_path),
                "exists": path.exists(),
                "size_bytes": int(path.stat().st_size) if path.exists() and path.is_file() else None,
            }
            manifest["sections"][section_name].append(entry)
            manifest["all_outputs"].append(entry)
    return manifest


def finalize_run_metadata(
    *,
    run_context: Dict[str, Any],
    data_bundle: DataBundle,
    output_dir: Path,
    resolved_start_sign: int,
    section_outputs: Dict[str, Dict[str, str]],
    summary: Dict[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    finished_at = datetime.now(TOKYO)
    duration_sec = (finished_at - run_context["started_at"]).total_seconds()
    run_context["finished_at"] = finished_at
    run_context["duration_sec"] = float(duration_sec)
    run_context["duration_min"] = float(duration_sec) / 60.0
    run_context["params"]["resolved_start_sign"] = int(resolved_start_sign)
    run_context["data_summary"] = {
        "n_nodes": int(data_bundle.graph.number_of_nodes()),
        "n_edges": int(data_bundle.graph.number_of_edges()),
        "n_pairs": int(len(data_bundle.labels_df)),
        "n_pert_sigs": int(len(data_bundle.perturbation_signatures)),
        "n_dis_sigs": int(len(data_bundle.disease_signatures)),
        "n_common_genes": int(len(data_bundle.common_genes)),
    }

    clean_meta = _json_safe(run_context)
    metadata_path = write_json(clean_meta, output_dir / "metadata.json")

    manifest = build_output_manifest(run_root=output_dir, section_outputs=section_outputs)
    manifest_path = write_json(manifest, output_dir / "output_manifest.json")

    run_summary = {
        "run_root": str(output_dir),
        "run_id": clean_meta.get("run_id"),
        "started_at": clean_meta.get("started_at"),
        "finished_at": clean_meta.get("finished_at"),
        "duration_sec": clean_meta.get("duration_sec"),
        "duration_min": clean_meta.get("duration_min"),
        "metadata": _relative(output_dir, metadata_path),
        "output_manifest": _relative(output_dir, manifest_path),
    }
    if summary:
        run_summary.update(_json_safe(summary))
    run_summary_path = write_json(run_summary, output_dir / "run_summary.json")
    return metadata_path, manifest_path, run_summary_path
