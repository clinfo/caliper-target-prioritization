from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .io import read_yaml_or_json, resolve_path


ALLOWED_METRICS = {"jaccard"}
ALLOWED_BASELINE = {"global_paired_gene_label_permutation"}
DECOY_NULL_TYPE = "observed_target_pool_decoy_control"
LEGACY_DECOY_NULL_TYPE = "evaluable_decoy_anchor_resampling"
TARGET_DECOY_NULL_TYPE = "topology_profile_matched_target_decoy_control"

# Primary specification of the topology- and profile-matched decoy target control.
TARGET_DECOY_PRIMARY_N_DECOYS = 10
TARGET_DECOY_PRIMARY_POSITIVE_EXCLUSION_SCOPE = "same_drug_disease"
TARGET_DECOY_PRIMARY_K_STAR = 4
TARGET_DECOY_SELECTION_MATCHED = "matched"
TARGET_DECOY_SELECTION_RANDOM = "random"
ALLOWED_TARGET_DECOY_SELECTIONS = {TARGET_DECOY_SELECTION_MATCHED, TARGET_DECOY_SELECTION_RANDOM}

# Candidate universe the decoys are drawn from. The primary universe is restricted
# to the profile-supported observed target set, exactly matching the population of
# the legacy observed-target-pool control. The broader universes remain available for
# explicitly exploratory runs.
TARGET_DECOY_CANDIDATE_UNIVERSE_L1000_KD = "l1000_kd_profile_x_signor_nodes"
TARGET_DECOY_CANDIDATE_UNIVERSE_LABELED_TARGETS = "l1000_kd_profile_x_signor_nodes_x_labeled_targets"
TARGET_DECOY_CANDIDATE_UNIVERSE_PROFILE_SUPPORTED_OBSERVED_TARGETS = (
    "l1000_kd_profile_x_signor_nodes_x_profile_supported_observed_targets"
)
ALLOWED_TARGET_DECOY_CANDIDATE_UNIVERSES = {
    TARGET_DECOY_CANDIDATE_UNIVERSE_L1000_KD,
    TARGET_DECOY_CANDIDATE_UNIVERSE_LABELED_TARGETS,
    TARGET_DECOY_CANDIDATE_UNIVERSE_PROFILE_SUPPORTED_OBSERVED_TARGETS,
}
TARGET_DECOY_PRIMARY_CANDIDATE_UNIVERSE = (
    TARGET_DECOY_CANDIDATE_UNIVERSE_PROFILE_SUPPORTED_OBSERVED_TARGETS
)

ALLOWED_NULL_MODELS = {
    "global_paired_gene_label_permutation",
    "path_count_length_matched_structural_resampling",
    "target_khop_restricted_paired_gene_label_permutation",
    DECOY_NULL_TYPE,
    LEGACY_DECOY_NULL_TYPE,
    TARGET_DECOY_NULL_TYPE,
}
ALLOWED_CELL_PAIR_MODES = {"cartesian", "shared_only"}
ALLOWED_EVALUABLE_SET_TYPES = {"fixed_coverage", "path_existing_pairs"}
ALLOWED_EDGE_SCOPES = ("all_edges", "first_edge_excluded")
ALLOWED_PATH_STRATEGIES = {"simple", "shortest"}
ALLOWED_INTERMEDIATE_RETENTION = {"path_exploration", "evaluation", "report", "none"}


@dataclass(frozen=True)
class Val3TripletConfig:
    config_path: Path
    registry_path: Path
    seed: int
    output_root: Path
    output_name: str
    date_tag: str
    data: dict[str, Any]
    sections: dict[str, bool]
    analysis: dict[str, Any]
    infer: dict[str, Any]
    baseline: dict[str, Any]
    evaluation: dict[str, Any]
    progress: dict[str, Any]
    storage: dict[str, Any]
    raw: dict[str, Any]


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' must be a mapping.")
    return value


def _require_optional_mapping(parent: dict[str, Any], key: str, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    value = parent.get(key, default or {})
    if value is None:
        return dict(default or {})
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' must be a mapping.")
    return dict(value)


def _normalize_sections(raw: dict[str, Any]) -> dict[str, bool]:
    return {
        "prepare_inputs": bool(raw.get("prepare_inputs", True)),
        "path_exploration": bool(raw.get("path_exploration", True)),
        "evaluation": bool(raw.get("evaluation", True)),
        "make_visualization_tables": bool(raw.get("make_visualization_tables", True)),
        "make_report": bool(raw.get("make_report", True)),
    }


def _normalize_analysis(raw: dict[str, Any]) -> dict[str, Any]:
    k_star = int(raw.get("k_star", raw.get("cutoff", 3)))
    gap = int(raw.get("gap", 1))
    if k_star < 1:
        raise ValueError("analysis.k_star must be a positive integer.")
    if gap < 0:
        raise ValueError("analysis.gap must be a non-negative integer.")
    cell_pair_mode = str(raw.get("cell_pair_mode", "cartesian"))
    if cell_pair_mode not in ALLOWED_CELL_PAIR_MODES:
        raise ValueError(f"analysis.cell_pair_mode must be one of {sorted(ALLOWED_CELL_PAIR_MODES)}")
    directions = list(raw.get("directions", ["inhibitory"]))
    return {
        "k_star": k_star,
        "gap": gap,
        "directions": [str(d) for d in directions],
        "cell_pair_mode": cell_pair_mode,
    }


def _normalize_target_decoy_null_model(model_cfg: dict[str, Any]) -> None:
    """Normalize and validate the topology- and profile-matched Target-Decoy control.

    ``is_primary_spec`` (default true) enforces the fixed primary values.  Setting it
    to false is the only supported way to run an exploratory configuration, and the
    flag is recorded in run metadata so exploratory output cannot be mistaken for the
    primary run.

    ``decoy_selection`` chooses how the decoys are drawn from the same candidate
    universe.  It is not part of the primary specification: both ``matched`` and
    ``random`` can be run as a primary run.

    ``decoy_candidate_universe`` chooses which candidate universe the decoys are
    drawn from.  Unlike ``decoy_selection`` this *is* part of the primary
    specification: a primary run must use the profile-supported observed-target
    universe, exactly matching the legacy observed-target-pool population. The full
    labeled-target and plain L1000-KD-x-KG universes stay available for exploratory
    (``is_primary_spec=false``) runs.
    """
    model_cfg["enabled"] = bool(model_cfg.get("enabled", False))
    model_cfg["is_primary_spec"] = bool(model_cfg.get("is_primary_spec", True))
    model_cfg["decoy_selection"] = str(model_cfg.get("decoy_selection", TARGET_DECOY_SELECTION_MATCHED))
    if model_cfg["decoy_selection"] not in ALLOWED_TARGET_DECOY_SELECTIONS:
        raise ValueError(
            f"baseline.null_models.{TARGET_DECOY_NULL_TYPE}.decoy_selection must be one of "
            f"{sorted(ALLOWED_TARGET_DECOY_SELECTIONS)}."
        )
    model_cfg["decoy_candidate_universe"] = str(
        model_cfg.get("decoy_candidate_universe", TARGET_DECOY_PRIMARY_CANDIDATE_UNIVERSE)
    )
    if model_cfg["decoy_candidate_universe"] not in ALLOWED_TARGET_DECOY_CANDIDATE_UNIVERSES:
        raise ValueError(
            f"baseline.null_models.{TARGET_DECOY_NULL_TYPE}.decoy_candidate_universe must be one of "
            f"{sorted(ALLOWED_TARGET_DECOY_CANDIDATE_UNIVERSES)}."
        )
    model_cfg["n_decoys_per_triplet"] = int(model_cfg.get("n_decoys_per_triplet", TARGET_DECOY_PRIMARY_N_DECOYS))
    model_cfg["exclude_known_positive_targets"] = bool(model_cfg.get("exclude_known_positive_targets", True))
    model_cfg["positive_exclusion_scope"] = str(
        model_cfg.get("positive_exclusion_scope", TARGET_DECOY_PRIMARY_POSITIVE_EXCLUSION_SCOPE)
    )
    if model_cfg["n_decoys_per_triplet"] < 1:
        raise ValueError(
            f"baseline.null_models.{TARGET_DECOY_NULL_TYPE}.n_decoys_per_triplet must be >= 1."
        )
    if model_cfg["positive_exclusion_scope"] != TARGET_DECOY_PRIMARY_POSITIVE_EXCLUSION_SCOPE:
        raise ValueError(
            f"baseline.null_models.{TARGET_DECOY_NULL_TYPE}.positive_exclusion_scope must be "
            f"'{TARGET_DECOY_PRIMARY_POSITIVE_EXCLUSION_SCOPE}'."
        )
    if not model_cfg["enabled"]:
        return
    if not model_cfg["is_primary_spec"]:
        return
    if model_cfg["n_decoys_per_triplet"] != TARGET_DECOY_PRIMARY_N_DECOYS:
        raise ValueError(
            f"baseline.null_models.{TARGET_DECOY_NULL_TYPE}.n_decoys_per_triplet must be "
            f"{TARGET_DECOY_PRIMARY_N_DECOYS} in a primary run. Set is_primary_spec=false for an "
            "exploratory run."
        )
    if model_cfg["decoy_candidate_universe"] != TARGET_DECOY_PRIMARY_CANDIDATE_UNIVERSE:
        raise ValueError(
            f"baseline.null_models.{TARGET_DECOY_NULL_TYPE}.decoy_candidate_universe must be "
            f"'{TARGET_DECOY_PRIMARY_CANDIDATE_UNIVERSE}' in a primary run. Set is_primary_spec=false "
            "for an exploratory run."
        )
    if not model_cfg["exclude_known_positive_targets"]:
        raise ValueError(
            f"baseline.null_models.{TARGET_DECOY_NULL_TYPE}.exclude_known_positive_targets must be "
            "true in a primary run. Set is_primary_spec=false for an exploratory run."
        )


def _validate_target_decoy_primary_spec(
    baseline: dict[str, Any],
    analysis: dict[str, Any],
    infer: dict[str, Any],
) -> None:
    model_cfg = dict((baseline.get("null_models") or {}).get(TARGET_DECOY_NULL_TYPE) or {})
    if not bool(model_cfg.get("enabled", False)):
        return
    if bool(infer.get("restrict_baseline_to_observed_evaluable", False)):
        raise ValueError(
            f"'{TARGET_DECOY_NULL_TYPE}' fixes path_evaluability_prefilter=false, so "
            "infer.restrict_baseline_to_observed_evaluable must be false."
        )
    if not bool(model_cfg.get("is_primary_spec", True)):
        return
    if int(analysis.get("k_star", 0)) != TARGET_DECOY_PRIMARY_K_STAR:
        raise ValueError(
            f"analysis.k_star must be {TARGET_DECOY_PRIMARY_K_STAR} in a primary "
            f"'{TARGET_DECOY_NULL_TYPE}' run. Set is_primary_spec=false for an exploratory run."
        )


def _store_target_decoy_fixed_values(
    baseline: dict[str, Any],
    *,
    analysis: dict[str, Any],
    seed: int,
) -> None:
    """Persist the fixed model definition into the resolved config.

    These entries are not user-tunable options: they document what the run actually
    used so that metadata never reports hard-coded display values.  The matching
    variables, rank transform and distance only exist for ``decoy_selection: matched``;
    a random draw records them as ``not_applicable`` rather than claiming a matching
    rule it never used.
    """
    model_cfg = (baseline.get("null_models") or {}).get(TARGET_DECOY_NULL_TYPE)
    if not isinstance(model_cfg, dict):
        return
    matched = str(model_cfg.get("decoy_selection", TARGET_DECOY_SELECTION_MATCHED)) == TARGET_DECOY_SELECTION_MATCHED
    not_applicable = "not_applicable"
    model_cfg["candidate_universe_type"] = str(
        model_cfg.get("decoy_candidate_universe", TARGET_DECOY_PRIMARY_CANDIDATE_UNIVERSE)
    )
    model_cfg["matching_variables"] = [
        "directed_out_degree",
        "reachable_unique_nodes_k4",
        "unique_kd_cell_lines",
    ] if matched else not_applicable
    model_cfg["rank_transform"] = "midrank_percentile" if matched else not_applicable
    model_cfg["distance"] = "equal_weight_euclidean" if matched else not_applicable
    # The reach cutoff is needed for the candidate universe in both selection modes.
    model_cfg["reach_cutoff"] = int(analysis.get("k_star", TARGET_DECOY_PRIMARY_K_STAR))
    model_cfg["within_triplet_replacement"] = False
    model_cfg["across_triplet_reuse"] = True
    model_cfg["path_evaluability_prefilter"] = False
    model_cfg["post_inference_backfill"] = False
    model_cfg["non_evaluable_decoy_score"] = "missing"
    model_cfg["tie_break_seed"] = int(seed)


def _normalize_baseline(raw: dict[str, Any]) -> dict[str, Any]:
    raw = dict(raw or {})
    n_shuffle = int(raw.get("n_shuffle", 100))

    legacy_enabled = bool(raw.get("enabled", True))
    legacy_type = str(raw.get("type", "global_paired_gene_label_permutation"))
    if legacy_type not in ALLOWED_BASELINE:
        raise ValueError(f"baseline.type must be one of {sorted(ALLOWED_BASELINE)}")
    legacy_shuffle_per_triplet = bool(raw.get("shuffle_per_triplet", True))

    null_models_raw = raw.get("null_models")
    if null_models_raw is not None and not isinstance(null_models_raw, dict):
        raise ValueError("baseline.null_models must be a mapping when specified.")
    if isinstance(null_models_raw, dict) and LEGACY_DECOY_NULL_TYPE in null_models_raw:
        null_models_raw = dict(null_models_raw)
        legacy_decoy_cfg = dict(null_models_raw.pop(LEGACY_DECOY_NULL_TYPE) or {})
        merged_decoy_cfg = dict(null_models_raw.get(DECOY_NULL_TYPE) or {})
        merged_decoy_cfg.update(legacy_decoy_cfg)
        null_models_raw[DECOY_NULL_TYPE] = merged_decoy_cfg

    def _default_null_models() -> dict[str, Any]:
        return {
            "global_paired_gene_label_permutation": {
                "enabled": False,
                "shuffle_per_triplet": legacy_shuffle_per_triplet,
            },
            "path_count_length_matched_structural_resampling": {
                "enabled": False,
            },
            "target_khop_restricted_paired_gene_label_permutation": {
                "enabled": False,
                "min_local_bg_genes": 0,
            },
            DECOY_NULL_TYPE: {
                "enabled": False,
                "exclude_known_positive_targets": True,
                "max_decoy_anchors_per_triplet": None,
            },
            TARGET_DECOY_NULL_TYPE: {
                "enabled": False,
                "n_decoys_per_triplet": TARGET_DECOY_PRIMARY_N_DECOYS,
                "exclude_known_positive_targets": True,
                "positive_exclusion_scope": TARGET_DECOY_PRIMARY_POSITIVE_EXCLUSION_SCOPE,
                "is_primary_spec": True,
            },
        }

    if null_models_raw is None:
        null_models = _default_null_models()
        null_models["global_paired_gene_label_permutation"]["enabled"] = legacy_enabled
        null_models["global_paired_gene_label_permutation"]["shuffle_per_triplet"] = legacy_shuffle_per_triplet
        null_models["path_count_length_matched_structural_resampling"]["enabled"] = bool(
            (_require_optional_mapping(raw, "structural_null")).get("enabled", False)
        )
    else:
        unknown = sorted(set(null_models_raw) - ALLOWED_NULL_MODELS)
        if unknown:
            raise ValueError(f"Unsupported baseline.null_models entries: {unknown}")
        null_models = _default_null_models()
        for name, model_cfg in null_models_raw.items():
            if not isinstance(model_cfg, dict):
                raise ValueError(f"baseline.null_models.{name} must be a mapping.")
            null_models[name].update(model_cfg)
        null_models["global_paired_gene_label_permutation"]["enabled"] = bool(
            null_models["global_paired_gene_label_permutation"].get("enabled", False)
        )
        null_models["global_paired_gene_label_permutation"]["shuffle_per_triplet"] = bool(
            null_models["global_paired_gene_label_permutation"].get("shuffle_per_triplet", legacy_shuffle_per_triplet)
        )
        null_models["path_count_length_matched_structural_resampling"]["enabled"] = bool(
            null_models["path_count_length_matched_structural_resampling"].get("enabled", False)
        )
        null_models["target_khop_restricted_paired_gene_label_permutation"]["enabled"] = bool(
            null_models["target_khop_restricted_paired_gene_label_permutation"].get("enabled", False)
        )
        null_models["target_khop_restricted_paired_gene_label_permutation"]["min_local_bg_genes"] = int(
            null_models["target_khop_restricted_paired_gene_label_permutation"].get("min_local_bg_genes", 0)
        )
        decoy_nm = null_models[DECOY_NULL_TYPE]
        decoy_nm["enabled"] = bool(decoy_nm.get("enabled", False))
        decoy_nm["exclude_known_positive_targets"] = bool(decoy_nm.get("exclude_known_positive_targets", True))
        # Null 3 uses all final evaluable unique decoy anchors; n_shuffle and replacement sampling are not used.
        decoy_nm.pop("sample_with_replacement", None)
        max_decoy = decoy_nm.get("max_decoy_anchors_per_triplet")
        decoy_nm["max_decoy_anchors_per_triplet"] = int(max_decoy) if max_decoy is not None else None

        _normalize_target_decoy_null_model(null_models[TARGET_DECOY_NULL_TYPE])
        # The legacy control stays readable so an existing run's saved config can be
        # loaded again, but it can no longer be turned on from a config.
        if decoy_nm["enabled"]:
            raise ValueError(
                f"'{DECOY_NULL_TYPE}' / '{LEGACY_DECOY_NULL_TYPE}' は config から有効化できません "
                f"（figure上 '{TARGET_DECOY_NULL_TYPE}' と同じ 'Target-Decoy' ラベルになるため）。"
                "既存 run の再現以外では baseline.null_models からキーを削除してください。"
            )

        legacy_keys_present = any(k in raw for k in ["enabled", "type", "shuffle_per_triplet", "structural_null"])
        if legacy_keys_present:
            expected_gene = legacy_enabled and legacy_type == "global_paired_gene_label_permutation"
            actual_gene = bool(null_models["global_paired_gene_label_permutation"].get("enabled", False))
            if expected_gene != actual_gene:
                raise ValueError(
                    "Conflicting baseline configuration: legacy baseline.enabled/type and "
                    "baseline.null_models.global_paired_gene_label_permutation.enabled disagree."
                )
            actual_shuffle = bool(null_models["global_paired_gene_label_permutation"].get("shuffle_per_triplet", True))
            if legacy_shuffle_per_triplet != actual_shuffle:
                raise ValueError(
                    "Conflicting baseline configuration: legacy baseline.shuffle_per_triplet and "
                    "baseline.null_models.global_paired_gene_label_permutation.shuffle_per_triplet disagree."
                )
            legacy_structural = bool((_require_optional_mapping(raw, "structural_null")).get("enabled", False))
            actual_structural = bool(null_models["path_count_length_matched_structural_resampling"].get("enabled", False))
            if legacy_structural != actual_structural:
                raise ValueError(
                    "Conflicting baseline configuration: legacy baseline.structural_null.enabled and "
                    "baseline.null_models.path_count_length_matched_structural_resampling.enabled disagree."
                )

    enabled_null_models = [name for name, model_cfg in null_models.items() if bool(model_cfg.get("enabled", False))]
    if enabled_null_models and n_shuffle < 1:
        raise ValueError("baseline.n_shuffle must be >= 1 when any null model is enabled.")
    return {
        "enabled": bool(null_models["global_paired_gene_label_permutation"].get("enabled", False)),
        "type": legacy_type,
        "n_shuffle": n_shuffle,
        "is_primary_spec": bool(raw.get("is_primary_spec", True)),
        "shuffle_per_triplet": bool(null_models["global_paired_gene_label_permutation"].get("shuffle_per_triplet", True)),
        "null_models": null_models,
        "enabled_null_models": enabled_null_models,
    }


SHUFFLE_NULL_PRIMARY_N_SHUFFLE = 100
SHUFFLE_NULL_TYPES_REQUIRING_N_SHUFFLE = (
    "global_paired_gene_label_permutation",
    "target_khop_restricted_paired_gene_label_permutation",
    "path_count_length_matched_structural_resampling",
)


def _validate_shuffle_null_primary_spec(baseline: dict[str, Any]) -> None:
    """Pin repeat-driven nulls to the P0-4 primary repeat count.

    Target-Decoy does not consume ``n_shuffle`` (it uses every evaluable decoy
    anchor), so only the gene-node shuffle and structural resampling nulls are
    checked here.
    """
    if not bool(baseline.get("is_primary_spec", True)):
        return
    null_models = baseline.get("null_models") or {}
    active = [
        name
        for name in SHUFFLE_NULL_TYPES_REQUIRING_N_SHUFFLE
        if bool((null_models.get(name) or {}).get("enabled", False))
    ]
    if not active:
        return
    n_shuffle = int(baseline.get("n_shuffle", 0))
    if n_shuffle != SHUFFLE_NULL_PRIMARY_N_SHUFFLE:
        raise ValueError(
            "baseline.n_shuffle must be "
            f"{SHUFFLE_NULL_PRIMARY_N_SHUFFLE} when a repeat-driven null model is enabled "
            f"({', '.join(active)}) and baseline.is_primary_spec=true, but got {n_shuffle}. "
            "P0-4 primary runs are fixed at 100 repeats; set baseline.is_primary_spec=false "
            "to use a different repeat count for exploratory or test runs."
        )


def _normalize_evaluation(raw: dict[str, Any]) -> dict[str, Any]:
    raw = dict(raw or {})
    metrics = [str(m) for m in raw.get("metrics", ["jaccard"])]
    bad = sorted(set(metrics) - ALLOWED_METRICS)
    if bad:
        raise ValueError(f"Unsupported evaluation.metrics: {bad}")
    if metrics != ["jaccard"]:
        raise ValueError("evaluation.metrics must be exactly ['jaccard'].")
    filters = raw.get("filters") or {}
    if not isinstance(filters, dict):
        raise ValueError("evaluation.filters must be a mapping.")

    global_effect_size = raw.get("global_effect_size") or {}
    if not isinstance(global_effect_size, dict):
        raise ValueError("evaluation.global_effect_size must be a mapping.")

    evaluable_sets_raw = raw.get("evaluable_sets") or {}
    if not isinstance(evaluable_sets_raw, dict):
        raise ValueError("evaluation.evaluable_sets must be a mapping.")
    fixed_raw = evaluable_sets_raw.get("fixed_coverage") or {}
    path_existing_raw = evaluable_sets_raw.get("path_existing_pairs") or {}
    if not isinstance(fixed_raw, dict):
        raise ValueError("evaluation.evaluable_sets.fixed_coverage must be a mapping.")
    if not isinstance(path_existing_raw, dict):
        raise ValueError("evaluation.evaluable_sets.path_existing_pairs must be a mapping.")

    min_candidate_nodes = int(fixed_raw.get("min_candidate_nodes", 10))
    if min_candidate_nodes < 1:
        raise ValueError("evaluation.evaluable_sets.fixed_coverage.min_candidate_nodes must be >= 1.")

    fixed_enabled = bool(fixed_raw.get("enabled", False))
    path_existing_enabled = bool(path_existing_raw.get("enabled", False))

    edge_scopes_raw = raw.get("edge_scopes") or {}
    if not isinstance(edge_scopes_raw, dict):
        raise ValueError("evaluation.edge_scopes must be a mapping.")
    unknown_scopes = sorted(set(edge_scopes_raw) - set(ALLOWED_EDGE_SCOPES))
    if unknown_scopes:
        raise ValueError(f"Unsupported evaluation.edge_scopes entries: {unknown_scopes}")
    all_edges_raw = edge_scopes_raw.get("all_edges") or {}
    first_edge_excluded_raw = edge_scopes_raw.get("first_edge_excluded") or {}
    if not isinstance(all_edges_raw, dict):
        raise ValueError("evaluation.edge_scopes.all_edges must be a mapping.")
    if not isinstance(first_edge_excluded_raw, dict):
        raise ValueError("evaluation.edge_scopes.first_edge_excluded must be a mapping.")
    if not bool(all_edges_raw.get("enabled", True)):
        raise ValueError(
            "evaluation.edge_scopes.all_edges.enabled must be true; the all-edges scope is the primary analysis."
        )
    # Configs written before the first-edge sensitivity analysis existed must keep
    # producing exactly the primary outputs.
    first_edge_excluded_enabled = bool(first_edge_excluded_raw.get("enabled", False))

    max_workers = int(raw.get("max_workers", 1))
    if max_workers < 1:
        raise ValueError("evaluation.max_workers must be >= 1.")

    return {
        "metrics": metrics,
        "filters": {
            "require_both_nonempty_inferred_paths": bool(filters.get("require_both_nonempty_inferred_paths", True)),
            "require_both_nonempty_signed_edges": bool(filters.get("require_both_nonempty_signed_edges", False)),
        },
        "evaluable_sets": {
            "fixed_coverage": {
                "enabled": fixed_enabled,
                "min_candidate_nodes": min_candidate_nodes,
            },
            "path_existing_pairs": {
                "enabled": path_existing_enabled,
            },
        },
        "global_effect_size": {
            "enabled": bool(global_effect_size.get("enabled", False)),
        },
        "edge_scopes": {
            "all_edges": {"enabled": True},
            "first_edge_excluded": {"enabled": first_edge_excluded_enabled},
        },
        "max_workers": max_workers,
    }


def _normalize_progress(raw: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(raw.get("enabled", True))
    use_tqdm = bool(raw.get("use_tqdm", True))
    if enabled and use_tqdm:
        try:
            import tqdm  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError("progress.use_tqdm=true but tqdm is not importable.") from e
    return {
        "enabled": enabled,
        "use_tqdm": use_tqdm,
        "leave": bool(raw.get("leave", False)),
        "mininterval": float(raw.get("mininterval", 1.0)),
    }



def _normalize_storage(raw: dict[str, Any]) -> dict[str, Any]:
    raw = dict(raw or {})
    retention = str(raw.get("intermediate_retention", "evaluation"))
    if retention not in ALLOWED_INTERMEDIATE_RETENTION:
        raise ValueError(
            "storage.intermediate_retention must be one of "
            f"{sorted(ALLOWED_INTERMEDIATE_RETENTION)}"
        )
    return {"intermediate_retention": retention}

def _global_summary_enabled(evaluation: dict[str, Any]) -> bool:
    return bool(evaluation["global_effect_size"].get("enabled", False))


def load_val3_triplet_config(config_path: str | Path) -> Val3TripletConfig:
    config_path = Path(config_path).resolve()
    raw = read_yaml_or_json(config_path)
    run = _require_mapping(raw, "run")
    data = _require_mapping(raw, "data")
    sections = _normalize_sections(_require_mapping(raw, "sections"))
    analysis = _normalize_analysis(_require_mapping(raw, "analysis"))
    infer = dict(raw.get("infer") or {})
    path_strategy = str(infer.get("path_strategy", "simple"))
    if path_strategy not in ALLOWED_PATH_STRATEGIES:
        raise ValueError(f"infer.path_strategy must be one of {sorted(ALLOWED_PATH_STRATEGIES)}")
    infer["path_strategy"] = path_strategy
    infer["restrict_baseline_to_observed_evaluable"] = bool(
        infer.get("restrict_baseline_to_observed_evaluable", False)
    )
    baseline = _normalize_baseline(raw.get("baseline") or {})
    evaluation = _normalize_evaluation(raw.get("evaluation") or {})
    progress = _normalize_progress(raw.get("progress") or {})
    storage = _normalize_storage(raw.get("storage") or {})
    _validate_shuffle_null_primary_spec(baseline)
    _validate_target_decoy_primary_spec(baseline, analysis, infer)

    registry_rel = run.get("registry_path", "./dataset_registry.yaml")
    registry_path = resolve_path(config_path.parent, registry_rel)
    assert registry_path is not None

    output_root = Path(str(run.get("output_root", "../output"))).expanduser()
    if not output_root.is_absolute():
        output_root = (config_path.parent / output_root).resolve()

    date_tag = str(run.get("date_tag", "auto"))
    if date_tag == "auto":
        date_tag = datetime.now().strftime("%Y%m%d")

    output_name = str(run.get("output_name", "auto"))
    seed = int(run.get("seed", 0))
    _store_target_decoy_fixed_values(baseline, analysis=analysis, seed=seed)

    needs_global_summary = _global_summary_enabled(evaluation)
    if needs_global_summary:
        if not sections["evaluation"]:
            raise ValueError("At least one global summary option requires sections.evaluation=true.")
        if not (
            evaluation["evaluable_sets"]["fixed_coverage"].get("enabled", False)
            or evaluation["evaluable_sets"]["path_existing_pairs"].get("enabled", False)
        ):
            raise ValueError("At least one evaluation.evaluable_sets.*.enabled must be true when a global summary option is enabled.")
        if not baseline.get("enabled_null_models"):
            raise ValueError("At least one baseline.null_models.*.enabled must be true when a global summary option is enabled.")
        if int(baseline.get("n_shuffle", 0)) < 1:
            raise ValueError("Global summary options require baseline.n_shuffle >= 1.")

    return Val3TripletConfig(
        config_path=config_path,
        registry_path=registry_path,
        seed=seed,
        output_root=output_root,
        output_name=output_name,
        date_tag=date_tag,
        data=dict(data),
        sections=sections,
        analysis=analysis,
        infer=infer,
        baseline=baseline,
        evaluation=evaluation,
        progress=progress,
        storage=storage,
        raw=raw,
    )


def resolve_output_dir(cfg: Val3TripletConfig, *, direction: str | None = None) -> Path:
    if cfg.output_name != "auto":
        return cfg.output_root / "val3" / cfg.output_name
    kg = str(cfg.data.get("kg", "kg"))
    triplet_source = str(cfg.data.get("triplet_source", "triplets"))
    d = direction or "multi"
    return cfg.output_root / "val3" / f"{cfg.date_tag}_{kg}_{triplet_source}_{d}_k{cfg.analysis['k_star']}_g{cfg.analysis['gap']}"


def direction_modes(cfg: Val3TripletConfig, direction: str) -> tuple[str, str]:
    modes = (cfg.infer.get("direction_modes") or {}).get(direction)
    if modes:
        return str(modes["drug_mode"]), str(modes["pert_mode"])
    if direction == "inhibitory":
        return "knockdown", "knockdown"
    if direction == "activatory":
        return "overexpression", "overexpression"
    raise ValueError(f"Unsupported direction: {direction}")


def tqdm_kwargs(cfg: Val3TripletConfig) -> dict[str, Any]:
    return {"leave": cfg.progress["leave"], "mininterval": cfg.progress["mininterval"]}
