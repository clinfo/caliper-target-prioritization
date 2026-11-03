from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import json


@dataclass(frozen=True)
class Val4Config:
    config_path: Path
    common_config_path: Path | None
    task: str
    registry_path: Path
    seed: int
    output_root: Path
    output_name: str
    date_tag: str
    data: Dict[str, str]
    analysis: Dict[str, Any]
    family_selection: Dict[str, Any]
    family_definition: Dict[str, list[str]]
    sections: Dict[str, bool]
    terminal: Dict[str, Any]
    lofo: Dict[str, Any]
    sign_shuffle: Dict[str, Any]
    main: Dict[str, Any]
    raw: Dict[str, Any]


# Backward-compatible name used by val4.h1.
ResolvedConfig = Val4Config


_MULTI_FAMILY_EDGES = frozenset({"exclude", "first_family", "split_equal", "include_all"})


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


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _require_mapping(parent: Dict[str, Any], key: str, *, required: bool = True) -> Dict[str, Any]:
    value = parent.get(key)
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' must be a mapping.")
    return value


def _normalize_bool_mapping(raw: Dict[str, Any], defaults: Dict[str, bool] | None = None) -> Dict[str, bool]:
    out = dict(defaults or {})
    for key, value in raw.items():
        out[str(key)] = bool(value)
    return out


def _normalize_analysis(raw: Dict[str, Any], *, task: str) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(raw)
    if "k_star" in out:
        out["k_star"] = int(out["k_star"])
    else:
        out["k_star"] = 0

    if task in {"h1", "h2"}:
        out["n_null"] = int(out.get("n_null", 1000))
        out["bootstrap_reps"] = int(out.get("bootstrap_reps", 1000))

    if "gap" in out:
        out["gap"] = int(out["gap"])
    if "path_strategy" in out:
        out["path_strategy"] = str(out["path_strategy"]).strip().lower()

    mfe = out.get("multi_family_edges")
    legacy_exc = out.get("exclude_multi_family_terminal_edges")
    if mfe is not None:
        mode = str(mfe).strip().lower()
        if mode not in _MULTI_FAMILY_EDGES:
            raise ValueError(
                f"analysis.multi_family_edges must be one of {sorted(_MULTI_FAMILY_EDGES)}, got {mfe!r}"
            )
    elif legacy_exc is not None:
        mode = "exclude" if bool(legacy_exc) else "first_family"
    else:
        mode = "include_all" if task in {"h1", "h3"} else "exclude"
    out["multi_family_edges"] = mode
    out["exclude_multi_family_terminal_edges"] = mode == "exclude"

    if task == "h1":
        out["save_observation_rows"] = bool(out.get("save_observation_rows", True))
    if task == "h2":
        out.setdefault("metrics", ["AP", "Lift@10"])
        out["save_pair_scores"] = bool(out.get("save_pair_scores", False))
        out["save_null_pair_scores"] = bool(out.get("save_null_pair_scores", False))
        out["min_family_removed_path_votes"] = int(out.get("min_family_removed_path_votes", 1))
        out["min_evaluable_diseases"] = int(out.get("min_evaluable_diseases", 1))

        for removed_key in (
            "h2_scoring_backend",
            "h2_fast_equivalence_check",
            "h2_fast_equivalence_max_conditions",
        ):
            out.pop(removed_key, None)

        out["h2_n_jobs"] = int(out.get("h2_n_jobs", 1))
        if out["h2_n_jobs"] < 1:
            raise ValueError("analysis.h2_n_jobs must be >= 1.")

        out["h2_parallel_start_method"] = str(out.get("h2_parallel_start_method", "fork")).strip().lower()
        if out["h2_parallel_start_method"] not in {"fork", "spawn", "forkserver"}:
            raise ValueError("analysis.h2_parallel_start_method must be fork, spawn, or forkserver.")

        out["h2_parallel_chunksize"] = int(out.get("h2_parallel_chunksize", 1))
        if out["h2_parallel_chunksize"] < 1:
            raise ValueError("analysis.h2_parallel_chunksize must be >= 1.")
    if task == "h3":
        out["bootstrap_reps"] = int(out.get("bootstrap_reps", 1000))
        out["epsilon"] = float(out.get("epsilon", 1.0e-12))
        out["condition_universe"] = str(out.get("condition_universe", "positive_only")).strip().lower()
        out["save_condition_counts"] = bool(out.get("save_condition_counts", True))
    return out


def _normalize_family_definition(raw: Dict[str, Any]) -> Dict[str, list[str]]:
    family_definition: Dict[str, list[str]] = {}
    global_seen: dict[str, str] = {}
    for family, mechanisms in raw.items():
        family_name = str(family).strip()
        if not family_name:
            raise ValueError("family_definition keys must be non-empty strings.")
        if not isinstance(mechanisms, list):
            raise ValueError(f"family_definition.{family_name} must be a list of mechanism strings.")
        tokens: list[str] = []
        local_seen: set[str] = set()
        for mech in mechanisms:
            token = str(mech).strip().lower()
            if not token:
                continue
            if token in local_seen:
                raise ValueError(f"Duplicate mechanism token in family_definition.{family_name}: {token}")
            if token in global_seen:
                raise ValueError(
                    f"Mechanism token {token!r} is assigned to multiple families: "
                    f"{global_seen[token]!r}, {family_name!r}."
                )
            tokens.append(token)
            local_seen.add(token)
            global_seen[token] = family_name
        if len(tokens) == 0:
            raise ValueError(f"family_definition.{family_name} must contain at least one mechanism token.")
        family_definition[family_name] = tokens
    if len(family_definition) == 0:
        raise ValueError("family_definition must not be empty.")
    return family_definition


def _normalize_family_selection(raw: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(raw.get("mode", "kg_support")).strip().lower()
    if mode != "kg_support":
        raise ValueError("family_selection.mode currently supports only 'kg_support'.")
    min_physical_edges = int(raw.get("min_physical_edges", 20))
    if min_physical_edges < 1:
        raise ValueError("family_selection.min_physical_edges must be >= 1.")
    return {"mode": mode, "min_physical_edges": min_physical_edges}


def _infer_task(path: Path, raw: Dict[str, Any], explicit_task: str | None) -> str:
    if explicit_task:
        return explicit_task
    stem = path.stem.lower()
    if stem in {"h1", "h2", "h3"}:
        return stem
    if stem == "common":
        return "common"
    if (
        raw.get("main") is not None
        or raw.get("terminal") is not None
        or raw.get("lofo") is not None
        or raw.get("sign_shuffle") is not None
    ):
        return "h2"
    return "h1"


def resolve_start_sign(perturbation_mode: str) -> int:
    mode = str(perturbation_mode).strip().lower()
    if mode == "knockdown":
        return -1
    if mode == "overexpression":
        return 1
    raise ValueError(
        f"perturbation_mode must be 'knockdown' or 'overexpression'. got={perturbation_mode!r}"
    )


def load_val4_config(config_path: str | Path, *, task: str | None = None) -> Val4Config:
    config_path = Path(config_path).expanduser().resolve()
    local_raw = _load_mapping(config_path)
    common_config_path: Path | None = None
    raw = dict(local_raw)
    if "extends" in local_raw:
        common_config_path = (config_path.parent / str(local_raw["extends"])).resolve()
        common_raw = _load_mapping(common_config_path)
        raw = _deep_merge(common_raw, local_raw)

    task_name = _infer_task(config_path, raw, task)

    run = _require_mapping(raw, "run")
    data = _require_mapping(raw, "data")
    analysis = _normalize_analysis(_require_mapping(raw, "analysis"), task=task_name)
    family_selection = _normalize_family_selection(_require_mapping(raw, "family_selection"))
    family_definition = _normalize_family_definition(_require_mapping(raw, "family_definition"))
    sections = _normalize_bool_mapping(
        _require_mapping(raw, "sections", required=False),
        defaults={"compute": True, "figures": True, "report": True},
    )

    registry_rel = run.get("registry_path", "./dataset_registry.yaml")
    registry_path = (config_path.parent / str(registry_rel)).resolve()

    output_root = Path(str(run.get("output_root", "./output"))).expanduser()
    if not output_root.is_absolute():
        output_root = (config_path.parent / output_root).resolve()

    date_tag = str(run.get("date_tag", "auto"))
    if date_tag == "auto":
        date_tag = datetime.now().strftime("%Y%m%d")

    cfg = Val4Config(
        config_path=config_path,
        common_config_path=common_config_path,
        task=task_name,
        registry_path=registry_path,
        seed=int(run.get("seed", 0)),
        output_root=output_root,
        output_name=str(run.get("output_name", "auto")),
        date_tag=date_tag,
        data={
            "kg": str(data.get("kg", "")).strip(),
            "perturbation_mode": str(data.get("perturbation_mode", "")).strip(),
            "label_source": str(data.get("label_source", "")).strip(),
        },
        analysis=analysis,
        family_selection=family_selection,
        family_definition=family_definition,
        sections=sections,
        terminal=_require_mapping(raw, "terminal", required=False),
        lofo=_require_mapping(raw, "lofo", required=False),
        sign_shuffle=_require_mapping(raw, "sign_shuffle", required=False),
        main=_require_mapping(raw, "main", required=False),
        raw=raw,
    )
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: Val4Config) -> None:
    if cfg.data["kg"] != "signor":
        raise ValueError("val4 currently supports only kg='signor'.")
    if int(cfg.analysis["k_star"]) < 1:
        raise ValueError("analysis.k_star must be >= 1.")
    if cfg.task in {"h1", "h2"}:
        if int(cfg.analysis.get("n_null", 0)) < 1:
            raise ValueError("analysis.n_null must be >= 1.")
        if int(cfg.analysis.get("bootstrap_reps", 0)) < 1:
            raise ValueError("analysis.bootstrap_reps must be >= 1.")
    if cfg.task == "h2" and str(cfg.analysis.get("multi_family_edges", "")).strip().lower() == "split_equal":
        terminal_enabled = bool(cfg.terminal.get("enabled", True))
        lofo_enabled = bool(cfg.lofo.get("enabled", True))
        sign_shuffle_enabled = bool(cfg.sign_shuffle.get("enabled", True))
        if terminal_enabled or lofo_enabled or sign_shuffle_enabled:
            raise ValueError("Val4/H2 does not support analysis.multi_family_edges='split_equal' for terminal/LOFO/sign-shuffle analyses.")
    if cfg.task == "h3":
        if "gap" not in cfg.analysis:
            raise ValueError("analysis.gap must be present in merged config for Val4/H3.")
        if int(cfg.analysis.get("gap", -1)) < 0:
            raise ValueError("analysis.gap must be >= 0.")
        if "path_strategy" not in cfg.analysis:
            raise ValueError("analysis.path_strategy must be present in merged config for Val4/H3.")
        if str(cfg.analysis.get("path_strategy", "")).strip().lower() not in {"simple", "shortest"}:
            raise ValueError("analysis.path_strategy must be 'simple' or 'shortest'.")
        if int(cfg.analysis.get("bootstrap_reps", 0)) < 1:
            raise ValueError("analysis.bootstrap_reps must be >= 1.")
        if float(cfg.analysis.get("epsilon", 0.0)) <= 0:
            raise ValueError("analysis.epsilon must be > 0.")
        if str(cfg.analysis.get("condition_universe", "")).strip().lower() != "positive_only":
            raise ValueError("Val4/H3 currently supports only analysis.condition_universe='positive_only'.")
        if "multi_family_edges" not in cfg.analysis:
            raise ValueError("analysis.multi_family_edges must be present in merged config for Val4/H3.")
    resolve_start_sign(cfg.data["perturbation_mode"])


def load_val4_h1_config(config_path: str | Path) -> Val4Config:
    return load_val4_config(config_path, task="h1")


def load_val4_h2_config(config_path: str | Path) -> Val4Config:
    return load_val4_config(config_path, task="h2")


def load_val4_h3_config(config_path: str | Path) -> Val4Config:
    return load_val4_config(config_path, task="h3")


def resolved_config_dict(cfg: Val4Config) -> Dict[str, Any]:
    return {
        "config_path": str(cfg.config_path),
        "common_config_path": None if cfg.common_config_path is None else str(cfg.common_config_path),
        "task": cfg.task,
        "registry_path": str(cfg.registry_path),
        "seed": cfg.seed,
        "output_root": str(cfg.output_root),
        "output_name": cfg.output_name,
        "date_tag": cfg.date_tag,
        "data": dict(cfg.data),
        "analysis": deepcopy(cfg.analysis),
        "family_selection": deepcopy(cfg.family_selection),
        "family_definition": deepcopy(cfg.family_definition),
        "sections": dict(cfg.sections),
        "terminal": deepcopy(cfg.terminal),
        "lofo": deepcopy(cfg.lofo),
        "sign_shuffle": deepcopy(cfg.sign_shuffle),
        "main": deepcopy(cfg.main),
    }


def resolve_val4_output_dir(cfg: Val4Config, task: str | None = None) -> Path:
    task_name = task or cfg.task
    if cfg.output_name == "auto":
        name = (
            f"val4_{task_name}_{cfg.date_tag}_{cfg.data['kg']}_"
            f"{cfg.data['perturbation_mode']}_{cfg.data['label_source']}"
        )
    else:
        name = cfg.output_name
    return cfg.output_root / "val4" / task_name / name


def resolve_output_dir(cfg: Val4Config) -> Path:
    return resolve_val4_output_dir(cfg, task="h1")
