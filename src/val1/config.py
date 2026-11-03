from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import copy
import json

from .selection_rules import (
    DEFAULT_BETAS,
    DEFAULT_PRIMARY_BETA,
    DEFAULT_PRIMARY_TAU,
    DEFAULT_TAUS,
    KNEE_METHODS,
    SENSITIVITY_RULES,
    THRESHOLD_SCALES,
)


@dataclass(frozen=True)
class Val1Config:
    config_path: Path
    registry_path: Path
    seed: int
    output_root: Path
    output_name: str
    date_tag: str
    data: Dict[str, Any]
    sections: Dict[str, bool]
    analysis: Dict[str, Any]
    metrics: Dict[str, Any]
    steps: Dict[str, Any]
    report: Dict[str, Any]
    progress: Dict[str, Any]
    raw: Dict[str, Any]


# Backward-compatible alias for older imports in this package.
ResolvedConfig = Val1Config


def _load_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix in {".yml", ".yaml"}:
        import yaml
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config extension: {suffix}")

    if not isinstance(data, dict):
        raise ValueError("Config must contain a mapping at the top level.")
    return data


def _require_mapping(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' must be a mapping.")
    return value


def _normalize_sections(raw: Dict[str, Any]) -> Dict[str, bool]:
    # New v2.5 section names. Legacy names are intentionally not used for k selection.
    return {
        "prepare_inputs": bool(raw.get("prepare_inputs", True)),
        "compute_kmax": bool(raw.get("compute_kmax", True)),
        "compute_metrics": bool(raw.get("compute_metrics", True)),
        "select_k": bool(raw.get("select_k", True)),
        "compute_td_individual_stats": bool(raw.get("compute_td_individual_stats", True)),
        "run_selection_sensitivity": bool(raw.get("run_selection_sensitivity", True)),
        # Off by default: the path benchmark takes hours, so it is opted into with
        # --steps pathbench (or sections.run_path_benchmark=true).
        "run_path_benchmark": bool(raw.get("run_path_benchmark", False)),
        "make_figures": bool(raw.get("make_figures", True)),
        "make_report": bool(raw.get("make_report", True)),
    }


def _normalize_step_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    default = str(raw.get("default", "all")).strip().lower()
    return {
        "default": default,
        "debug_outputs": bool(raw.get("debug_outputs", False)),
    }


PATH_BENCHMARK_KEYS = (
    "enabled",
    "max_paths",
    "timeout_sec",
    "warmup",
    "repeats",
    "measure_node_score",
    "ks",
)


def _normalize_path_benchmark(raw: Dict[str, Any], *, ks: list[int]) -> Dict[str, Any]:
    """Validate ``analysis.path_benchmark`` with a strict whitelist.

    The path budget ``max_paths`` is the reported truncation mechanism;
    ``timeout_sec`` is only a safety net. ``warmup`` is capped at 1 because the
    benchmark records a single cold execution per source per k.
    """
    if not isinstance(raw, dict):
        raise ValueError("analysis.path_benchmark must be a mapping.")
    unknown = set(map(str, raw.keys())) - set(PATH_BENCHMARK_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown analysis.path_benchmark key(s): {sorted(unknown)}. valid={list(PATH_BENCHMARK_KEYS)}"
        )

    max_paths_raw = raw.get("max_paths", 100000) if "max_paths" in raw else 100000
    if max_paths_raw is None:
        max_paths: int | None = None
    else:
        max_paths = int(max_paths_raw)
        if max_paths < 1:
            raise ValueError("analysis.path_benchmark.max_paths must be >= 1 (or null for unbounded).")

    timeout_sec = float(raw.get("timeout_sec", 600))
    if timeout_sec < 0:
        raise ValueError("analysis.path_benchmark.timeout_sec must be >= 0 (0 disables the safety net).")

    warmup = int(raw.get("warmup", 0))
    if warmup < 0 or warmup > 1:
        raise ValueError(
            "analysis.path_benchmark.warmup must be 0 or 1. The benchmark reports a single cold "
            "execution per source per k, so warmup >= 2 is not supported."
        )

    repeats = int(raw.get("repeats", 1))
    if repeats < 1:
        raise ValueError("analysis.path_benchmark.repeats must be >= 1.")

    ks_raw = raw.get("ks", "inherit")
    if isinstance(ks_raw, str):
        if ks_raw.strip().lower() != "inherit":
            raise ValueError("analysis.path_benchmark.ks must be 'inherit' or a list of int.")
        pb_ks: Any = "inherit"
    elif isinstance(ks_raw, list):
        pb_ks = sorted({int(k) for k in ks_raw})
        if not pb_ks:
            raise ValueError("analysis.path_benchmark.ks must be non-empty when given as a list.")
        # Keep the benchmark x-axis structurally aligned with the Val1 k grid.
        # Under ks_policy='kmax' the grid is resolved at runtime, so there is
        # nothing to check against here.
        if ks:
            outside = sorted(set(pb_ks) - set(int(k) for k in ks))
            if outside:
                raise ValueError(
                    f"analysis.path_benchmark.ks must be a subset of analysis.ks. outside={outside}"
                )
    else:
        raise ValueError("analysis.path_benchmark.ks must be 'inherit' or a list of int.")

    return {
        "enabled": bool(raw.get("enabled", True)),
        "max_paths": max_paths,
        "timeout_sec": timeout_sec,
        "warmup": warmup,
        "repeats": repeats,
        "measure_node_score": bool(raw.get("measure_node_score", True)),
        "ks": pb_ks,
    }


def _normalize_analysis(raw: Dict[str, Any]) -> Dict[str, Any]:
    ks_policy = str(raw.get("ks_policy", "kmax")).strip().lower()
    if ks_policy not in {"config", "kmax"}:
        raise ValueError("analysis.ks_policy must be either 'config' or 'kmax'.")

    ks_raw = raw.get("ks", [])
    if ks_raw in (None, []):
        ks: list[int] = []
    elif isinstance(ks_raw, list):
        ks = [int(k) for k in ks_raw]
    else:
        raise ValueError("analysis.ks must be a list when provided.")

    canonical_requires_kmax = bool(raw.get("canonical_requires_kmax", True))
    if canonical_requires_kmax and ks_policy != "kmax":
        raise ValueError(
            "analysis.canonical_requires_kmax=true requires analysis.ks_policy='kmax'. "
            "Set canonical_requires_kmax=false for sensitivity/debug runs."
        )

    return {
        "ks_policy": ks_policy,
        "ks": ks,
        "canonical_requires_kmax": canonical_requires_kmax,
        "path_benchmark": _normalize_path_benchmark(raw.get("path_benchmark") or {}, ks=ks),
    }


def _float_grid(values: Any, *, key: str, default: list[float]) -> list[float]:
    if values is None:
        return list(default)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{key} must be a non-empty list.")
    grid = sorted({float(v) for v in values})
    return grid


def _normalize_selection_sensitivity(raw: Dict[str, Any]) -> Dict[str, Any]:
    coverage_threshold = dict(raw.get("coverage_threshold") or {})
    ambiguity_ceiling = dict(raw.get("ambiguity_ceiling") or {})
    pareto = dict(raw.get("pareto") or {})
    resampling = dict(raw.get("resampling") or {})

    rules_raw = raw.get("rules", list(SENSITIVITY_RULES))
    if not isinstance(rules_raw, list):
        raise ValueError("metrics.selection_sensitivity.rules must be a list.")
    rules = [str(r).strip() for r in rules_raw]
    unknown = set(rules) - set(SENSITIVITY_RULES)
    if unknown:
        raise ValueError(
            f"Unknown metrics.selection_sensitivity.rules entries: {sorted(unknown)}. "
            f"valid={list(SENSITIVITY_RULES)}"
        )

    scale = str(coverage_threshold.get("scale", "raw")).strip().lower()
    if scale not in THRESHOLD_SCALES:
        raise ValueError(f"metrics.selection_sensitivity.coverage_threshold.scale must be one of {list(THRESHOLD_SCALES)}.")

    taus = _float_grid(coverage_threshold.get("taus"), key="metrics.selection_sensitivity.coverage_threshold.taus", default=DEFAULT_TAUS)
    primary_tau = float(coverage_threshold.get("primary_tau", DEFAULT_PRIMARY_TAU))
    if primary_tau not in taus:
        raise ValueError(
            "metrics.selection_sensitivity.coverage_threshold.primary_tau must be one of the swept taus "
            "so that the prespecified value stays visible in the sweep table."
        )

    betas = _float_grid(ambiguity_ceiling.get("betas"), key="metrics.selection_sensitivity.ambiguity_ceiling.betas", default=DEFAULT_BETAS)
    primary_beta = float(ambiguity_ceiling.get("primary_beta", DEFAULT_PRIMARY_BETA))
    if primary_beta not in betas:
        raise ValueError(
            "metrics.selection_sensitivity.ambiguity_ceiling.primary_beta must be one of the swept betas "
            "so that the prespecified value stays visible in the sweep table."
        )

    knee_method = str(pareto.get("knee_method", "max_chord_distance")).strip()
    if knee_method not in KNEE_METHODS:
        raise ValueError(f"metrics.selection_sensitivity.pareto.knee_method must be one of {list(KNEE_METHODS)}.")

    n_bootstrap = int(resampling.get("bootstrap_targets", 0))
    if n_bootstrap < 0:
        raise ValueError("metrics.selection_sensitivity.resampling.bootstrap_targets must be >= 0.")

    return {
        "enabled": bool(raw.get("enabled", True)),
        "rules": rules,
        "coverage_threshold": {"scale": scale, "taus": taus, "primary_tau": primary_tau},
        "ambiguity_ceiling": {"scale": scale, "betas": betas, "primary_beta": primary_beta},
        "pareto": {"knee_method": knee_method},
        "resampling": {
            "leave_one_target_out": bool(resampling.get("leave_one_target_out", True)),
            "leave_one_disease_out": bool(resampling.get("leave_one_disease_out", True)),
            "bootstrap_targets": n_bootstrap,
        },
    }


def _normalize_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    coverage = dict(raw.get("coverage") or {})
    ambiguity = dict(raw.get("ambiguity") or {})
    integration = dict(raw.get("integration") or {})
    selection = dict(raw.get("selection") or {})

    coverage_primary = str(coverage.get("primary", "cov_ratio")).strip()
    if coverage_primary not in {"cov_nz", "cov_ratio"}:
        raise ValueError("metrics.coverage.primary must be 'cov_nz' or 'cov_ratio'.")

    empty_policy = str(coverage.get("cov_ratio_empty_base_policy", "nan")).strip().lower()
    if empty_policy != "nan":
        raise ValueError("metrics.coverage.cov_ratio_empty_base_policy must be 'nan'.")

    ratio_agg = str(coverage.get("cov_ratio_primary_aggregation", "median")).strip().lower()
    if ratio_agg != "median":
        raise ValueError("metrics.coverage.cov_ratio_primary_aggregation must be 'median'.")

    coverage_compute = coverage.get("compute", ["cov_nz", "cov_ratio"])
    if not isinstance(coverage_compute, list):
        raise ValueError("metrics.coverage.compute must be a list.")
    coverage_compute = [str(x).strip() for x in coverage_compute]
    unknown = set(coverage_compute) - {"cov_nz", "cov_ratio"}
    if unknown:
        raise ValueError(f"Unknown metrics.coverage.compute entries: {sorted(unknown)}")

    ambiguity_primary = str(ambiguity.get("primary", "amb_median")).strip()
    if ambiguity_primary != "amb_median":
        raise ValueError("metrics.ambiguity.primary must be 'amb_median'.")

    ambiguity_compute = ambiguity.get("compute", ["amb_median"])
    if not isinstance(ambiguity_compute, list):
        raise ValueError("metrics.ambiguity.compute must be a list.")
    ambiguity_compute = [str(x).strip() for x in ambiguity_compute]
    unknown = set(ambiguity_compute) - {"amb_median"}
    if unknown:
        raise ValueError(f"Unknown metrics.ambiguity.compute entries: {sorted(unknown)}")

    method = str(integration.get("method", "harmonic_mean")).strip()
    if method != "harmonic_mean":
        raise ValueError("metrics.integration.method must be 'harmonic_mean'.")

    tie_breaker = str(selection.get("tie_breaker", "smaller_k")).strip()
    if tie_breaker != "smaller_k":
        raise ValueError("metrics.selection.tie_breaker must be 'smaller_k'.")

    top_n = int(selection.get("top_n", 3))
    if top_n < 1:
        raise ValueError("metrics.selection.top_n must be >= 1.")

    return {
        "coverage": {
            "primary": coverage_primary,
            "compute": coverage_compute,
            "cov_ratio_empty_base_policy": empty_policy,
            "cov_ratio_primary_aggregation": ratio_agg,
        },
        "ambiguity": {
            "primary": ambiguity_primary,
            "compute": ambiguity_compute,
        },
        "integration": {
            "method": method,
            "normalize": bool(integration.get("normalize", True)),
            "zero_division": float(integration.get("zero_division", 0.0)),
        },
        "selection": {
            "top_n": top_n,
            "tie_breaker": tie_breaker,
        },
        "selection_sensitivity": _normalize_selection_sensitivity(raw.get("selection_sensitivity") or {}),
    }


def load_run_metrics_settings(run_root: str | Path) -> Dict[str, Any]:
    """Recover normalized metrics settings for post-hoc steps run with --run-root only.

    Runs written before ``metrics.selection_sensitivity`` existed simply fall back
    to the defaults.
    """
    resolved_path = Path(run_root) / "config" / "resolved_config.json"
    resolved: Dict[str, Any] = {}
    if resolved_path.exists():
        with resolved_path.open("r", encoding="utf-8") as f:
            resolved = json.load(f)
    metrics = resolved.get("metrics") if isinstance(resolved, dict) else None
    return _normalize_metrics(metrics if isinstance(metrics, dict) else {})


def load_run_seed(run_root: str | Path) -> int:
    resolved_path = Path(run_root) / "config" / "resolved_config.json"
    if not resolved_path.exists():
        return 0
    with resolved_path.open("r", encoding="utf-8") as f:
        resolved = json.load(f)
    try:
        return int(resolved.get("seed", 0))
    except Exception:
        return 0


def _normalize_report(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "language": str(raw.get("language", "ja")),
        "max_targets_table_rows": int(raw.get("max_targets_table_rows", 20)),
        "include_target_depth_diagnostics": bool(raw.get("include_target_depth_diagnostics", True)),
        "include_pair_coverage_diagnostics": bool(raw.get("include_pair_coverage_diagnostics", True)),
        "include_td_distribution": bool(raw.get("include_td_distribution", True)),
        "explicitly_report_cov_ratio_policy": bool(raw.get("explicitly_report_cov_ratio_policy", True)),
    }


def _normalize_progress(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "enabled": bool(raw.get("enabled", True)),
        "use_tqdm": bool(raw.get("use_tqdm", True)),
        "leave": bool(raw.get("leave", False)),
        "mininterval": float(raw.get("mininterval", 1.0)),
    }


def resolve_start_sign(perturbation_mode: str) -> int:
    mode = str(perturbation_mode).strip().lower()
    if mode == "knockdown":
        return -1
    if mode == "overexpression":
        return 1
    raise ValueError(
        f"perturbation_mode must be 'knockdown' or 'overexpression'. got={perturbation_mode!r}"
    )


def load_val1_config(config_path: str | Path) -> Val1Config:
    config_path = Path(config_path).resolve()
    raw = _load_mapping(config_path)

    run = _require_mapping(raw, "run")
    data = _require_mapping(raw, "data")
    sections = _normalize_sections(_require_mapping(raw, "sections"))
    analysis = _normalize_analysis(_require_mapping(raw, "analysis"))
    metrics = _normalize_metrics(_require_mapping(raw, "metrics"))
    steps = _normalize_step_config(raw.get("steps") or {})
    report = _normalize_report(raw.get("report") or {})
    progress = _normalize_progress(raw.get("progress") or {})

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

    return Val1Config(
        config_path=config_path,
        registry_path=registry_path,
        seed=seed,
        output_root=output_root,
        output_name=output_name,
        date_tag=date_tag,
        data={
            "kg": str(data.get("kg", "")).strip(),
            "perturbation_mode": str(data.get("perturbation_mode", "")).strip(),
            "label_source": str(data.get("label_source", "")).strip(),
        },
        sections=sections,
        analysis=analysis,
        metrics=metrics,
        steps=steps,
        report=report,
        progress=progress,
        raw=copy.deepcopy(raw),
    )


def _coverage_token(primary: str) -> str:
    return "covratio" if primary == "cov_ratio" else "covnz"


def resolve_output_dir(cfg: Val1Config) -> Path:
    if cfg.output_name == "auto":
        cov_primary = cfg.metrics["coverage"]["primary"]
        aggregation = "median" if cov_primary == "cov_ratio" else "mean"
        mode = "norm" if cfg.metrics["integration"]["normalize"] else "raw"
        run_name = (
            f"{cfg.date_tag}_{cfg.data['kg']}_{cfg.data['perturbation_mode']}_"
            f"{cfg.data['label_source']}_{_coverage_token(cov_primary)}_{aggregation}_{mode}"
        )
    else:
        run_name = cfg.output_name
    return cfg.output_root / "val1" / run_name


def tqdm_kwargs(cfg: Val1Config) -> Dict[str, Any]:
    if not cfg.progress.get("enabled", True):
        return {"disable": True}
    return {
        "disable": not bool(cfg.progress.get("use_tqdm", True)),
        "leave": bool(cfg.progress.get("leave", False)),
        "mininterval": float(cfg.progress.get("mininterval", 1.0)),
    }


def resolved_config_dict(cfg: Val1Config) -> Dict[str, Any]:
    return {
        "config_path": str(cfg.config_path),
        "registry_path": str(cfg.registry_path),
        "seed": cfg.seed,
        "output_root": str(cfg.output_root),
        "output_name": cfg.output_name,
        "date_tag": cfg.date_tag,
        "data": dict(cfg.data),
        "sections": dict(cfg.sections),
        "analysis": copy.deepcopy(cfg.analysis),
        "metrics": copy.deepcopy(cfg.metrics),
        "steps": dict(cfg.steps),
        "report": dict(cfg.report),
        "progress": dict(cfg.progress),
    }
