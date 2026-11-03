from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import copy
import json


@dataclass(frozen=True)
class ResolvedConfig:
    config_path: Path
    registry_path: Path
    seed: int
    output_root: Path
    output_name: str
    date_tag: str
    sections: Dict[str, bool]
    analysis: Dict[str, Any]
    ablation: Dict[str, Any]
    permutation: Dict[str, Any]
    stats: Dict[str, Any]
    data: Dict[str, Any]
    steps: Dict[str, Any]
    report: Dict[str, Any]
    raw: Dict[str, Any]


def _load_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("PyYAML is required for YAML config files.") from e
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config extension: {suffix}")

    if not isinstance(data, dict):
        raise ValueError("Config must contain a mapping at the top level.")
    return data


def resolve_start_sign(perturbation_mode: str) -> int:
    mode = str(perturbation_mode).strip().lower()
    if mode == "knockdown":
        return -1
    if mode == "overexpression":
        return 1
    raise ValueError(
        "data.perturbation_mode must be either 'knockdown' or 'overexpression'. "
        f"Got: {perturbation_mode!r}"
    )


def _bool_section_map(raw: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "ablation": bool(raw.get("ablation", True)),
        "permutation": bool(raw.get("permutation", True)),
        "make_figures": bool(raw.get("make_figures", True)),
        "make_report": bool(raw.get("make_report", True)),
    }


def _normalize_steps(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "default": str(raw.get("default", "all")).strip().lower(),
    }


def _normalize_report(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "language": str(raw.get("language", "ja")),
        "max_rows": int(raw.get("max_rows", 20)),
    }


def _require_mapping(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' must be a mapping.")
    return value


def _normalize_analysis(raw: Dict[str, Any]) -> Dict[str, Any]:
    import os
    ks = raw.get("ks")
    if not isinstance(ks, list) or len(ks) == 0:
        raise ValueError("analysis.ks must be a non-empty list.")
    k_list = raw.get("K_list")
    if not isinstance(k_list, list) or len(k_list) == 0:
        raise ValueError("analysis.K_list must be a non-empty list.")
    normalized_ks = [int(k) for k in ks]
    k_valid_raw = raw.get("k_valid", raw.get("K_valid"))
    if k_valid_raw is None:
        k_valid = list(normalized_ks)
    elif not isinstance(k_valid_raw, list) or len(k_valid_raw) == 0:
        raise ValueError("analysis.k_valid must be a non-empty list when provided.")
    else:
        k_valid = [int(k) for k in k_valid_raw]
    unknown = sorted(set(k_valid) - set(normalized_ks))
    if unknown:
        raise ValueError(f"analysis.k_valid must be a subset of analysis.ks. unknown={unknown}")
    tie_policy = str(raw.get("tie_policy", "expected")).strip().lower()
    tie_aliases = {
        "default": "expected",
        "tie_aware": "expected",
        "tie-aware": "expected",
        "average": "expected",
        "average_ties": "expected",
        "expected_ties": "expected",
    }
    tie_policy = tie_aliases.get(tie_policy, tie_policy)
    if tie_policy not in {"expected", "legacy"}:
        raise ValueError("analysis.tie_policy must be either 'expected' or 'legacy'.")
    n_jobs_raw = int(raw.get("n_jobs", 1))
    if n_jobs_raw <= 0:
        n_jobs_raw = os.cpu_count() or 1
    return {
        "ks": normalized_ks,
        "k_valid": k_valid,
        "K_list": [int(k) for k in k_list],
        "tie_policy": tie_policy,
        "n_jobs": n_jobs_raw,
    }


_KNOWN_ABLATION_METHODS = {
    "proposed",
    "kg_nosign",
    "kg_sign_noalign",
    "noKG",
    "noKG_sizematch",
    "ncs",
}


def _normalize_ncs(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the ablation.ncs block (required when 'ncs' is selected).

    Parquet paths are kept as-is here and resolved against the config directory
    in load_unified_config (mirroring registry_path/output_root).
    """
    if not isinstance(raw, dict):
        raise ValueError("ablation.ncs block is required when 'ncs' is in ablation.methods.")

    ckd = raw.get("continuous_kd_profile")
    if not isinstance(ckd, dict) or not ckd.get("parquet"):
        raise ValueError("ablation.ncs.continuous_kd_profile.parquet is required.")

    ds = raw.get("disease_signature") or {}
    if not isinstance(ds, dict):
        raise ValueError("ablation.ncs.disease_signature must be a mapping.")
    source = str(ds.get("source", "full_creeds")).strip().lower()
    if source not in {"full_creeds", "val2"}:
        raise ValueError("ablation.ncs.disease_signature.source must be 'full_creeds' or 'val2'.")
    ds_parquet = ds.get("parquet")
    if source == "full_creeds" and not ds_parquet:
        raise ValueError(
            "ablation.ncs.disease_signature.parquet is required when source='full_creeds'."
        )

    policy = str(raw.get("undefined_policy", "floor")).strip().lower()
    if policy not in {"floor", "drop"}:
        raise ValueError("ablation.ncs.undefined_policy must be 'floor' or 'drop'.")

    return {
        "continuous_kd_profile": {
            "parquet": str(ckd["parquet"]),
            "id_col": str(ckd.get("id_col", "ncbigene_id")),
            "cell_col": str(ckd.get("cell_col", "cell_id")),
        },
        "disease_signature": {
            "source": source,
            "parquet": str(ds_parquet) if ds_parquet else None,
            "id_col": str(ds.get("id_col", "doid")),
        },
        "undefined_policy": policy,
    }


def _normalize_ablation(raw: Dict[str, Any]) -> Dict[str, Any]:
    methods = raw.get("methods")
    if not isinstance(methods, list) or len(methods) == 0:
        raise ValueError("ablation.methods must be a non-empty list.")
    methods = [str(x) for x in methods]
    unknown = sorted(set(methods) - _KNOWN_ABLATION_METHODS)
    if unknown:
        raise ValueError(
            f"ablation.methods contains unknown method(s): {unknown}. "
            f"valid={sorted(_KNOWN_ABLATION_METHODS)}"
        )
    ncs = _normalize_ncs(raw.get("ncs")) if "ncs" in methods else None
    return {
        "methods": methods,
        "sizematch_reps": int(raw.get("sizematch_reps", 100)),
        "save_pred_sign_ambiguity": bool(raw.get("save_pred_sign_ambiguity", True)),
        "save_proposed_w": bool(raw.get("save_proposed_w", True)),
        "save_scores": bool(raw.get("save_scores", True)),
        "save_metrics": bool(raw.get("save_metrics", True)),
        "ncs": ncs,
    }


def _normalize_permutation(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "n_perm": int(raw.get("n_perm", 30)),
        "save_by_disease": bool(raw.get("save_by_disease", True)),
    }


def _normalize_stats(raw: Dict[str, Any]) -> Dict[str, Any]:
    # Significance testing is fixed to the k=4 paired t-test with Holm
    # correction; the correction is no longer configurable.
    if "primary_correction" in raw:
        correction = str(raw.get("primary_correction")).strip().lower()
        if correction != "holm":
            raise ValueError(
                "stats.primary_correction is fixed to 'holm'; remove the key or set it to 'holm'."
            )
    metrics_raw = raw.get("metrics") or ["AP", "Lift@10"]
    metrics = [str(m) for m in metrics_raw]
    comparisons_raw = raw.get("comparisons") or ["pairwise", "best_baseline"]
    comparisons = [str(c).strip().lower() for c in comparisons_raw]
    unknown = sorted(set(comparisons) - {"pairwise", "best_baseline"})
    if unknown:
        raise ValueError(f"stats.comparisons must be a subset of ['pairwise','best_baseline']. unknown={unknown}")
    baselines_raw = raw.get("baselines") or ["kg_nosign", "kg_sign_noalign", "noKG_sizematch"]
    baselines = [str(b) for b in baselines_raw]
    return {
        "enabled": bool(raw.get("enabled", True)),
        "n_boot": int(raw.get("n_boot", 10000)),
        "bootstrap_seed": int(raw.get("bootstrap_seed", 0)),
        "alpha": float(raw.get("alpha", 0.05)),
        "metrics": metrics,
        "comparisons": comparisons,
        "baselines": baselines,
        "proposed_method": str(raw.get("proposed_method", "proposed")),
    }


def load_unified_config(config_path: str | Path) -> ResolvedConfig:
    config_path = Path(config_path).resolve()
    raw = _load_mapping(config_path)

    run = _require_mapping(raw, "run")
    data = _require_mapping(raw, "data")
    sections = _bool_section_map(_require_mapping(raw, "sections"))
    analysis = _normalize_analysis(_require_mapping(raw, "analysis"))
    ablation = _normalize_ablation(_require_mapping(raw, "ablation"))
    permutation = _normalize_permutation(_require_mapping(raw, "permutation"))
    stats = _normalize_stats(raw.get("stats") or {})
    steps = _normalize_steps(raw.get("steps") or {})
    report = _normalize_report(raw.get("report") or {})

    registry_rel = run.get("registry_path", "./dataset_registry.yaml")
    registry_path = (config_path.parent / str(registry_rel)).resolve()

    output_root = Path(str(run.get("output_root", "./output"))).expanduser()
    if not output_root.is_absolute():
        output_root = (config_path.parent / output_root).resolve()

    date_tag = str(run.get("date_tag", "auto"))
    if date_tag == "auto":
        date_tag = datetime.now().strftime("%Y%m%d")

    output_name = str(run.get("output_name", "auto"))
    seed = int(run.get("seed", 0))

    # Resolve ablation.ncs parquet paths against the config directory
    # (mirroring registry_path/output_root above).
    if ablation.get("ncs"):
        ncs = ablation["ncs"]

        def _resolve(rel: str) -> str:
            p = Path(str(rel)).expanduser()
            if not p.is_absolute():
                p = (config_path.parent / p).resolve()
            return str(p)

        ncs["continuous_kd_profile"]["parquet"] = _resolve(
            ncs["continuous_kd_profile"]["parquet"]
        )
        if ncs["disease_signature"].get("parquet"):
            ncs["disease_signature"]["parquet"] = _resolve(ncs["disease_signature"]["parquet"])

    return ResolvedConfig(
        config_path=config_path,
        registry_path=registry_path,
        seed=seed,
        output_root=output_root,
        output_name=output_name,
        date_tag=date_tag,
        sections=sections,
        analysis=analysis,
        ablation=ablation,
        permutation=permutation,
        stats=stats,
        data={
            "kg": str(data.get("kg", "")).strip(),
            "perturbation_mode": str(data.get("perturbation_mode", "")).strip(),
            "label_source": str(data.get("label_source", "")).strip(),
        },
        steps=steps,
        report=report,
        raw=copy.deepcopy(raw),
    )


def resolve_output_dir(cfg: ResolvedConfig) -> Path:
    if cfg.output_name == "auto":
        name = (
            f"val2_{cfg.date_tag}_{cfg.data['kg']}_"
            f"{cfg.data['perturbation_mode']}_{cfg.data['label_source']}"
        )
    else:
        name = cfg.output_name
    return cfg.output_root / "val2" / name
