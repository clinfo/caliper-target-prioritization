"""Configuration loader for the val2 external-baseline comparison.

The five "司令不足項目" (SI-01..SI-05) from the methodology must be set
explicitly; missing or invalid values raise a ValueError instead of silently
falling back to defaults (methodology §11 validation checks #1, #2).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import copy
import json


# Allowed baseline names (validation) and the per-mode defaults used when
# ``external_baseline.baselines`` is omitted are separate: DADA is a global
# shared-network method and has no target-local counterpart, so it is only a
# default (and only permitted) under primary_analysis='native_shared_network'.
ALLOWED_BASELINE_METHODS = ["NCS", "proximity", "RWR", "DADA"]
DEFAULT_BASELINES_BY_PRIMARY_ANALYSIS = {
    "native_shared_network": ["NCS", "proximity", "RWR", "DADA"],
    "directed_target_reach_matched": ["NCS", "proximity", "RWR"],
}
NATIVE_ONLY_BASELINE_METHODS = ["DADA"]
_BASELINE_ALIASES = {
    "ncs": "NCS",
    "l1000_ncs": "NCS",
    "l1000": "NCS",
    "proximity": "proximity",
    "network_proximity": "proximity",
    "rwr": "RWR",
    "classical_rwr": "RWR",
    "dada": "DADA",
}


@dataclass(frozen=True)
class ResolvedBaselineConfig:
    config_path: Path
    registry_path: Path
    seed: int
    output_root: Path
    output_name: str
    date_tag: str
    data: Dict[str, str]
    primary_analysis: str
    baselines: List[str]
    # external baseline specific
    k_selection_json: Path
    manual_k_star: int | None
    caliper_ks: List[int] | None
    continuous_kd_parquet: Path
    continuous_kd_id_col: str
    continuous_kd_cell_col: str
    val2_label_parquet: Path
    # baseline disease-signature source (WTCS query + proximity/RWR seeds only;
    # CALIPER always keeps the val2 discretized disease profile).
    baseline_disease_signature_source: str
    creeds_disease_parquet: Path | None
    creeds_disease_id_col: str
    rwr_restart_probability: float
    proximity_n_random: int
    proximity_min_bin_size: int
    shared_seed: int
    min_disease_seeds: int
    target_reach_horizon: str
    target_reach_ncs_fallback_score: float
    target_reach_proximity_fallback_score: float
    target_reach_rwr_fallback_score: float
    target_reach_rwr_target_only_seed_score: float
    step4_n_jobs: int
    mp_start_method: str
    K_eff_cap: int
    K_list: List[int]
    tie_policy: str
    lift_eval_min_expected_hits: float
    sections: Dict[str, bool]
    stats: Dict[str, Any]
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


def _require_mapping(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' must be a mapping.")
    return value


def _resolve_path(base_dir: Path, raw: str) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _normalize_sections(raw: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "scores": bool(raw.get("scores", True)),
        "evaluation": bool(raw.get("evaluation", True)),
        "make_figures": bool(raw.get("make_figures", True)),
        "make_report": bool(raw.get("make_report", True)),
        "sensitivity": bool(raw.get("sensitivity", False)),
    }


def _normalize_stats(raw: Dict[str, Any], *, default_seed: int) -> Dict[str, Any]:
    # Significance testing is fixed to the k=4 paired t-test with Holm
    # correction; the correction is no longer configurable.
    if "primary_correction" in raw:
        correction = str(raw.get("primary_correction")).strip().lower()
        if correction != "holm":
            raise ValueError(
                "stats.primary_correction is fixed to 'holm'; remove the key or set it to 'holm'."
            )
    metrics_raw = raw.get("metrics") or ["AP", "Lift@10", "Lift@10_eval"]
    metrics = [str(m) for m in metrics_raw]
    return {
        "enabled": bool(raw.get("enabled", True)),
        "n_boot": int(raw.get("n_boot", 10000)),
        "bootstrap_seed": int(raw.get("bootstrap_seed", default_seed)),
        "alpha": float(raw.get("alpha", 0.05)),
        "metrics": metrics,
    }


def _normalize_primary_analysis(raw: Any) -> str:
    mode = str(raw or "native_shared_network").strip().lower()
    aliases = {
        "current": "native_shared_network",
        "native": "native_shared_network",
        "native_shared_network": "native_shared_network",
        "target_reach_matched": "directed_target_reach_matched",
        "directed_target_reach_matched": "directed_target_reach_matched",
    }
    normalized = aliases.get(mode)
    if normalized is None:
        raise ValueError(
            "external_baseline.primary_analysis must be one of "
            "'native_shared_network' or 'directed_target_reach_matched'. "
            f"got {raw!r}"
        )
    return normalized


def _normalize_baselines(raw: Any, *, primary_analysis: str) -> List[str]:
    explicit = raw is not None
    if raw is None:
        items = list(DEFAULT_BASELINES_BY_PRIMARY_ANALYSIS[primary_analysis])
    elif isinstance(raw, str):
        items = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("external_baseline.baselines must be a non-empty list or comma-separated string.")

    out: List[str] = []
    for item in items:
        key = str(item).strip().lower().replace("-", "_").replace(" ", "_")
        method = _BASELINE_ALIASES.get(key)
        if method is None:
            raise ValueError(
                "external_baseline.baselines entries must be one of "
                f"{ALLOWED_BASELINE_METHODS}. got {item!r}"
            )
        if method not in out:
            out.append(method)
    if not out:
        raise ValueError("external_baseline.baselines must contain at least one baseline.")
    if explicit and primary_analysis != "native_shared_network":
        native_only = [m for m in NATIVE_ONLY_BASELINE_METHODS if m in out]
        if native_only:
            raise ValueError(
                "external_baseline.baselines entries "
                f"{native_only} require primary_analysis='native_shared_network' "
                f"(global shared-network baselines have no target-local variant). "
                f"got primary_analysis={primary_analysis!r}"
            )
    return out


def _normalize_target_reach(raw: Dict[str, Any], *, primary_analysis: str) -> Dict[str, Any]:
    tr = raw.get("target_reach_matched") or {}
    horizon = str(tr.get("horizon", "k_star")).strip().lower()
    if primary_analysis == "directed_target_reach_matched" and horizon != "k_star":
        raise ValueError(
            "external_baseline.target_reach_matched.horizon must be 'k_star' "
            "for the primary directed target-reach matched analysis."
        )
    ncs = tr.get("ncs") or {}
    proximity = tr.get("proximity") or {}
    rwr = tr.get("rwr") or {}
    return {
        "horizon": horizon,
        "ncs_fallback_score": float(ncs.get("fallback_score", 0.0)),
        "proximity_fallback_score": float(proximity.get("fallback_score", 0.0)),
        "rwr_fallback_score": float(rwr.get("fallback_score", 0.0)),
        "rwr_target_only_seed_score": float(rwr.get("target_only_seed_score", 1.0)),
    }


def load_baseline_config(config_path: str | Path) -> ResolvedBaselineConfig:
    config_path = Path(config_path).resolve()
    raw = _load_mapping(config_path)
    base = config_path.parent

    run = _require_mapping(raw, "run")
    data = _require_mapping(raw, "data")
    eb = _require_mapping(raw, "external_baseline")
    analysis = raw.get("analysis") or {}
    report = raw.get("report") or {}
    primary_analysis = _normalize_primary_analysis(eb.get("primary_analysis"))
    baselines = _normalize_baselines(eb.get("baselines"), primary_analysis=primary_analysis)
    target_reach = _normalize_target_reach(eb, primary_analysis=primary_analysis)

    registry_rel = run.get("registry_path", "./dataset_registry.yaml")
    registry_path = _resolve_path(base, str(registry_rel))

    output_root = Path(str(run.get("output_root", "./output"))).expanduser()
    if not output_root.is_absolute():
        output_root = (base / output_root).resolve()

    date_tag = str(run.get("date_tag", "auto"))
    if date_tag == "auto":
        date_tag = datetime.now().strftime("%Y%m%d")
    output_name = str(run.get("output_name", "auto"))
    seed = int(run.get("seed", 42))

    # ---- SI-02: val1 selection output ----
    if "k_selection_json" not in eb:
        raise ValueError("external_baseline.k_selection_json (SI-02) is required but missing.")
    k_selection_json = _resolve_path(base, str(eb["k_selection_json"]))
    manual_k_star = eb.get("k_star")
    manual_k_star = int(manual_k_star) if manual_k_star is not None else None

    caliper_cfg = eb.get("caliper") or {}
    caliper_ks_raw = caliper_cfg.get("ks")
    caliper_ks: List[int] | None = None
    if caliper_ks_raw is not None:
        if not isinstance(caliper_ks_raw, list) or not caliper_ks_raw:
            raise ValueError("external_baseline.caliper.ks must be a non-empty list of positive integers.")
        caliper_ks = sorted(dict.fromkeys(int(k) for k in caliper_ks_raw))
        if any(k <= 0 for k in caliper_ks):
            raise ValueError(f"external_baseline.caliper.ks entries must be positive. got {caliper_ks!r}")

    # ---- NCS reference profile (target-cell averaged BING_knockdown_signature) ----
    # Per user direction the NCS reference uses the target-cell averaged profile
    # (not per-sig_id Level 5 signatures); see wtcs.py for the §6.2.1 deviation.
    ckd = _require_mapping(eb, "continuous_kd_profile")
    if "parquet" not in ckd:
        raise ValueError("external_baseline.continuous_kd_profile.parquet (NCS reference) is required but missing.")
    continuous_kd_parquet = _resolve_path(base, str(ckd["parquet"]))
    continuous_kd_id_col = str(ckd.get("id_col", "ncbigene_id"))
    continuous_kd_cell_col = str(ckd.get("cell_col", "cell_id"))

    # ---- SI-04: val2 label input ----
    if "val2_label_parquet" not in eb:
        raise ValueError("external_baseline.val2_label_parquet (SI-04) is required but missing.")
    val2_label_parquet = _resolve_path(base, str(eb["val2_label_parquet"]))

    # ---- baseline disease-signature source (methodology §3.3) ----
    # 'val2'       : reuse bundle.disease_signatures (pert∩dis∩KG restricted; current default)
    # 'full_creeds': sign of the full CREEDS disease signature (gene universe not
    #                restricted to the perturbation space). CALIPER is unaffected.
    bds = eb.get("baseline_disease_signature") or {}
    baseline_disease_signature_source = str(bds.get("source", "val2")).strip().lower()
    if baseline_disease_signature_source not in {"val2", "full_creeds"}:
        raise ValueError(
            "external_baseline.baseline_disease_signature.source must be 'val2' or 'full_creeds'. "
            f"got {baseline_disease_signature_source!r}"
        )
    creeds_disease_id_col = str(bds.get("id_col", "doid"))
    creeds_disease_parquet: Path | None = None
    if baseline_disease_signature_source == "full_creeds":
        if "parquet" not in bds:
            raise ValueError(
                "external_baseline.baseline_disease_signature.parquet is required when source='full_creeds'."
            )
        creeds_disease_parquet = _resolve_path(base, str(bds["parquet"]))

    # ---- SI-01: RWR restart probability ----
    rwr = _require_mapping(eb, "rwr")
    if "restart_probability" not in rwr:
        raise ValueError("external_baseline.rwr.restart_probability (SI-01) is required but missing.")
    r = float(rwr["restart_probability"])
    if not (0.0 < r < 1.0):
        raise ValueError(f"external_baseline.rwr.restart_probability must satisfy 0<r<1. got {r}")

    # ---- SI-05: shared random seed ----
    if "shared_seed" not in eb:
        raise ValueError(
            "external_baseline.shared_seed (SI-05) is required. The methodology forbids "
            "implicitly adopting the default; set it explicitly (e.g. 42)."
        )
    shared_seed = int(eb["shared_seed"])

    prox = eb.get("proximity") or {}
    proximity_n_random = int(prox.get("n_random", 1000))
    proximity_min_bin_size = int(prox.get("min_bin_size", 100))
    min_disease_seeds = int(eb.get("min_disease_seeds", 20))

    K_eff_cap = int(analysis.get("K_eff_cap", 10))
    k_list_raw = analysis.get("K_list")
    if k_list_raw is not None:
        if not isinstance(k_list_raw, list) or not k_list_raw:
            raise ValueError("analysis.K_list must be a non-empty list of positive integers.")
        K_list = sorted(dict.fromkeys(int(k) for k in k_list_raw))
        if any(k <= 0 for k in K_list):
            raise ValueError(f"analysis.K_list entries must be positive. got {K_list!r}")
    else:
        K_list = [K_eff_cap]
    tie_policy = str(analysis.get("tie_policy", "expected")).strip().lower()
    if tie_policy not in {"expected", "legacy"}:
        raise ValueError("analysis.tie_policy must be either 'expected' or 'legacy'.")
    lift_eval_min_expected_hits = float(analysis.get("lift_eval_min_expected_hits", 1.0))

    step4_n_jobs = int(analysis.get("step4_n_jobs", 1))
    if step4_n_jobs < 1:
        raise ValueError(f"analysis.step4_n_jobs must be >= 1. got {step4_n_jobs}")
    mp_start_method = str(analysis.get("mp_start_method", "fork")).strip().lower()
    if mp_start_method not in {"fork", "spawn", "forkserver"}:
        raise ValueError("analysis.mp_start_method must be one of 'fork', 'spawn', 'forkserver'.")

    return ResolvedBaselineConfig(
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
        primary_analysis=primary_analysis,
        baselines=baselines,
        k_selection_json=k_selection_json,
        manual_k_star=manual_k_star,
        caliper_ks=caliper_ks,
        continuous_kd_parquet=continuous_kd_parquet,
        continuous_kd_id_col=continuous_kd_id_col,
        continuous_kd_cell_col=continuous_kd_cell_col,
        val2_label_parquet=val2_label_parquet,
        baseline_disease_signature_source=baseline_disease_signature_source,
        creeds_disease_parquet=creeds_disease_parquet,
        creeds_disease_id_col=creeds_disease_id_col,
        rwr_restart_probability=r,
        proximity_n_random=proximity_n_random,
        proximity_min_bin_size=proximity_min_bin_size,
        shared_seed=shared_seed,
        min_disease_seeds=min_disease_seeds,
        target_reach_horizon=str(target_reach["horizon"]),
        target_reach_ncs_fallback_score=float(target_reach["ncs_fallback_score"]),
        target_reach_proximity_fallback_score=float(target_reach["proximity_fallback_score"]),
        target_reach_rwr_fallback_score=float(target_reach["rwr_fallback_score"]),
        target_reach_rwr_target_only_seed_score=float(target_reach["rwr_target_only_seed_score"]),
        step4_n_jobs=step4_n_jobs,
        mp_start_method=mp_start_method,
        K_eff_cap=K_eff_cap,
        K_list=K_list,
        tie_policy=tie_policy,
        lift_eval_min_expected_hits=lift_eval_min_expected_hits,
        sections=_normalize_sections(raw.get("sections") or {}),
        stats=_normalize_stats(raw.get("stats") or {}, default_seed=seed),
        report={"language": str(report.get("language", "ja")), "max_rows": int(report.get("max_rows", 20))},
        raw=copy.deepcopy(raw),
    )


def resolve_output_dir(cfg: ResolvedBaselineConfig) -> Path:
    if cfg.output_name == "auto":
        prefix = "val2_baseline"
        if cfg.primary_analysis == "directed_target_reach_matched":
            prefix = "val2_baseline_directed_target_reach_matched"
        name = f"{prefix}_{cfg.date_tag}_{cfg.data['kg']}_{cfg.data['perturbation_mode']}_{cfg.data['label_source']}"
    else:
        name = cfg.output_name
    return cfg.output_root / "val2_baseline" / name
