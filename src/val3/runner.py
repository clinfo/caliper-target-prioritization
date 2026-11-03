from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import direction_modes, load_val3_triplet_config, resolve_output_dir, tqdm_kwargs
from .infer_runner import (
    infer_output_path,
    iter_pert_id_group_batches,
    run_infer_dataframe_batches,
    run_infer_paired_dataframe_batches,
    run_infer_subprocess,
)
from .io import ensure_dir, read_parquet_any, read_yaml_or_json, write_json, write_parquet
from .metadata import finalize_run_context, finalize_run_summary, initialize_run_context, timed_run_section
from .metrics import (
    build_global_evaluable_triplets,
    build_path_existing_evaluable_triplets,
    evaluate_direction,
    load_infer_signed_edge_df,
)
from .prepare import derive_disease_profiles_from_target, load_disease_profiles, load_target_profiles, prepare_observed_inputs
from .registry import load_dataset_registry, resolve_val3_registry_spec
from .decoy import (
    DECOY_NULL_TYPE,
    DECOY_NULL_TYPES,
    LEGACY_DECOY_NULL_TYPE,
    TARGET_DECOY_CANDIDATE_POOL_TYPE_LABELED,
    TARGET_DECOY_CANDIDATE_POOL_TYPE_PROFILE_SUPPORTED_OBSERVED,
    TARGET_DECOY_NULL_TYPE,
    build_decoy_candidate_table,
    build_positive_targets_by_drug_disease,
    build_target_decoy_assignments,
    build_target_decoy_candidate_universe,
    build_target_decoy_matching_balance,
    build_target_decoy_matching_features,
    decoy_candidate_manifest,
    iter_decoy_paired_input_batches,
    load_directed_kg_graph,
    save_decoy_candidate_table,
    summarize_decoy_candidates,
    summarize_target_decoy_assignments,
    target_decoy_model_metadata,
)
from .shuffle import (
    build_target_khop_restricted_paired_gene_label_permutation_context,
    make_global_paired_gene_label_permutation_specs,
    make_target_khop_restricted_paired_gene_label_permutation_specs_from_context,
    save_baseline_permutation_specs,
    save_null1_permutation_specs,
)
from . import viz
from .report import VAL3_REPORT_FILENAME, build_v3_report_tables, write_val3_triplet_visualization_report
from .cleanup import cleanup_intermediates


VALID_STEPS = {"all", "prepare_inputs", "path_exploration", "evaluation", "visualization_tables", "report"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run revised Val3 triplet target-centered path-overlap evaluation")
    p.add_argument(
        "--config",
        default=None,
        help="Config file (.yaml/.yml/.json). When omitted (except output-only with --run-root), defaults to <repo>/config/val3.yaml.",
    )
    p.add_argument("--run-root", default=None, help="Existing or target output/val3/<run_name> directory.")
    p.add_argument("--direction", choices=["inhibitory", "activatory"], default=None, help="Override direction")
    p.add_argument(
        "--steps",
        action="append",
        default=None,
        help="Comma-separated Val3 steps to run: all,prepare_inputs,path_exploration,evaluation,visualization_tables,report.",
    )
    return p.parse_args(argv)


def _parse_steps(raw_steps: list[str] | None) -> set[str]:
    if not raw_steps:
        return {"all"}
    steps: set[str] = set()
    for chunk in raw_steps:
        for item in str(chunk).split(","):
            item = item.strip().lower()
            if item == "make_visualization_tables":
                item = "visualization_tables"
            if item:
                steps.add(item)
    unknown = steps - VALID_STEPS
    if unknown:
        raise ValueError(f"Unknown Val3 step(s): {sorted(unknown)}. valid={sorted(VALID_STEPS)}")
    if "all" in steps and len(steps) > 1:
        raise ValueError("Do not combine step 'all' with other steps.")
    return steps or {"all"}


def _progress_iter(iterable, *, cfg=None, desc: str, unit: str | None = None, total: int | None = None):
    if cfg is not None and cfg.progress.get("enabled", True) and cfg.progress.get("use_tqdm", True):
        return tqdm(iterable, desc=desc, unit=unit, total=total, **tqdm_kwargs(cfg))
    return iterable


def _prepare_kg_for_infer(spec, run_root: Path) -> tuple[Path, set[str]]:
    kg = pd.read_csv(spec.kg_path, sep=spec.kg_sep)
    required = {spec.kg_source_col, spec.kg_target_col, spec.kg_sign_col}
    missing = required - set(kg.columns)
    if missing:
        raise ValueError(f"KG file missing columns: {sorted(missing)}")
    out = pd.DataFrame({
        "source": kg[spec.kg_source_col].astype(str),
        "target": kg[spec.kg_target_col].astype(str),
        "polarity": pd.to_numeric(kg[spec.kg_sign_col], errors="raise").astype(int),
    })
    kg_dir = ensure_dir(run_root / "prepared_inputs" / "kg")
    out_path = kg_dir / "kg_for_infer.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    nodes = set(out["source"]).union(set(out["target"]))
    return out_path, nodes


def _load_manifest_payload(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"baseline_manifest": list(data)}
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unsupported baseline manifest format: {path}")


def _load_baseline_manifest(path: str | Path) -> list[dict[str, Any]]:
    data = _load_manifest_payload(path)
    if "baseline_manifest" in data:
        return list(data["baseline_manifest"])
    raise ValueError(f"Unsupported baseline manifest format: {path}")


def _baseline_pairs_from_manifest(manifest: list[dict[str, Any]]) -> list[tuple[int, str, str, str]]:
    pairs = []
    for item in manifest:
        if bool(item.get("inprocess", False)):
            continue
        pairs.append((
            int(item["iteration"]),
            item["baseline_drug_infer"],
            item["baseline_pert_infer"],
            str(item.get("null_type", "global_paired_gene_label_permutation")),
        ))
    return pairs


def _baseline_specs_from_manifest(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for item in manifest:
        if not bool(item.get("inprocess", False)):
            continue
        specs.append({
            "iteration": int(item["iteration"]),
            "null_type": str(item.get("null_type", "target_khop_restricted_paired_gene_label_permutation")),
            "permutation_specs": item.get("permutation_specs"),
        })
    return specs


def _null_model_enabled(cfg, name: str, default: bool = False) -> bool:
    null_models = cfg.baseline.get("null_models") or {}
    null_cfg = null_models.get(name) or {}
    if not null_cfg and name == LEGACY_DECOY_NULL_TYPE:
        null_cfg = null_models.get(DECOY_NULL_TYPE) or {}
    if not null_cfg and name == DECOY_NULL_TYPE:
        null_cfg = null_models.get(LEGACY_DECOY_NULL_TYPE) or {}
    return bool(null_cfg.get("enabled", default))


def _decoy_null_enabled_cfg(cfg) -> bool:
    return any(_null_model_enabled(cfg, name, False) for name in sorted(DECOY_NULL_TYPES))


def _decoy_null_type_cfg(cfg) -> str:
    """Which decoy control this run uses; both cannot be enabled at once."""
    if _null_model_enabled(cfg, TARGET_DECOY_NULL_TYPE, False):
        return TARGET_DECOY_NULL_TYPE
    return DECOY_NULL_TYPE


def _decoy_null_cfg(cfg) -> dict[str, Any]:
    null_models = cfg.baseline.get("null_models") or {}
    if _null_model_enabled(cfg, TARGET_DECOY_NULL_TYPE, False):
        return dict(null_models.get(TARGET_DECOY_NULL_TYPE) or {})
    return dict(null_models.get(DECOY_NULL_TYPE) or null_models.get(LEGACY_DECOY_NULL_TYPE) or {})


def _gene_label_null_name(cfg) -> str | None:
    if _null_model_enabled(cfg, "target_khop_restricted_paired_gene_label_permutation", False):
        return "target_khop_restricted_paired_gene_label_permutation"
    if _null_model_enabled(cfg, "global_paired_gene_label_permutation", False):
        return "global_paired_gene_label_permutation"
    if cfg.baseline.get("enabled", False):
        return "global_paired_gene_label_permutation"
    return None


def _infer_requested_workers(cfg) -> int:
    infer_cfg = cfg.infer or {}
    return max(1, int(infer_cfg.get("max_workers", infer_cfg.get("path_exploration_max_workers", 2))))


def _infer_max_workers(cfg, n_tasks: int) -> int:
    return max(1, min(int(n_tasks), _infer_requested_workers(cfg)))


def _run_infer_tasks_with_progress(tasks, *, cfg, run_one, desc: str):
    if not tasks:
        return
    max_workers = _infer_max_workers(cfg, len(tasks))
    if max_workers == 1:
        completed_iter = (run_one(task) for task in tasks)
        yield from _progress_iter(completed_iter, cfg=cfg, desc=desc, unit="run", total=len(tasks))
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        completed_iter = (future.result() for future in as_completed(futures))
        yield from _progress_iter(completed_iter, cfg=cfg, desc=desc, unit="run", total=len(tasks))


def _make_visualization_tables(run_root: Path, direction: str, cfg=None) -> dict[str, str]:
    vis_dir = ensure_dir(run_root / "visualization_tables")
    dfs = viz.load_val3_triplet_outputs(run_root, direction)
    outputs: dict[str, str] = {}
    table_specs = {
        "edge_count_diagnostics.tsv": viz.summarize_edge_count_diagnostics(dfs["triplet_true"], dfs["triplet_baseline"]),
        "evaluable_pair_diagnostics.tsv": viz.summarize_evaluable_pair_diagnostics(dfs["diag_true"], dfs["diag_baseline"]),
    }
    for name, df in _progress_iter(list(table_specs.items()), cfg=cfg, desc="visualization tables", unit="table"):
        path = vis_dir / name
        df.to_csv(path, sep="\t", index=False)
        outputs[name] = str(path)
    return outputs


def _existing_infer_output(output_dir: Path, *, mode: str, cutoff: int, gap: int) -> Path | None:
    path = infer_output_path(output_dir, mode=mode, cutoff=cutoff, gap=gap)
    if path.exists():
        return path
    dataset_path = path.with_suffix("")
    if dataset_path.exists():
        return dataset_path
    return None


def _load_manifest_if_exists(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    if not path.exists():
        return None
    payload = read_yaml_or_json(path)
    return list(payload.get("baseline_manifest", [])), dict(payload.get("decoy_candidate_manifest", {}))


def _observed_input_paths(run_root: Path, direction: str) -> tuple[Path, Path]:
    base = run_root / "prepared_inputs" / "observed" / direction
    return base / "drug.parquet", base / "pert.parquet"


def _output_step_exists(run_root: Path, direction: str, step: str, cfg=None) -> bool:
    if step == "evaluation":
        root = run_root / "evaluation" / direction
        required = [root / "summary.json", root / "triplet_metrics_true.parquet"]
        if cfg is not None:
            evaluation_cfg = cfg.evaluation or {}
            global_enabled = bool((evaluation_cfg.get("global_effect_size") or {}).get("enabled", False))
            if global_enabled:
                required.extend([
                    root / "global_evaluable_triplets.tsv",
                    root / "global_joint_triplet_metrics_true.parquet",
                    root / "global_joint_triplet_metrics_baseline.parquet",
                    root / "global_joint_triplet_comparison_sets.tsv",
                    root / "global_joint_triplet_comparison.tsv",
                    root / "global_effect_size.tsv",
                ])
        return all(p.exists() for p in required)
    if step == "visualization_tables":
        root = run_root / "visualization_tables"
        return (root / "edge_count_diagnostics.tsv").exists() and (root / "evaluable_pair_diagnostics.tsv").exists()
    if step == "report":
        return (run_root / "report" / VAL3_REPORT_FILENAME).exists()
    return False


def _path_exists(path_like: str | Path | None) -> bool:
    return bool(path_like) and Path(str(path_like)).exists()


def _manifest_infer_outputs_complete(baseline_manifest: list[dict[str, Any]], decoy_manifest: dict[str, Any]) -> bool:
    for item in baseline_manifest:
        if bool(item.get("inprocess", False)):
            if not _path_exists(item.get("permutation_specs")):
                return False
            continue
        if not _path_exists(item.get("baseline_drug_infer")) or not _path_exists(item.get("baseline_pert_infer")):
            return False
    if decoy_manifest:
        if not _path_exists(decoy_manifest.get("candidate_drug_infer")) or not _path_exists(decoy_manifest.get("candidate_pert_infer")):
            return False
    return True


def _load_prepared_frames_for_decoy(spec, *, observed_drug_input: str | Path) -> dict[str, Any]:
    drug_input = read_parquet_any(observed_drug_input)
    target_df = load_target_profiles(spec)
    disease_df = load_disease_profiles(spec)
    if disease_df is None:
        disease_df = derive_disease_profiles_from_target(target_df)
    return {"drug_df": drug_input, "target_df": target_df, "disease_df": disease_df}


def _baseline_restriction_enabled(cfg) -> bool:
    return bool((cfg.infer or {}).get("restrict_baseline_to_observed_evaluable", False))


def _enabled_evaluable_set_types(evaluation_cfg: dict[str, Any]) -> list[str]:
    ev = (evaluation_cfg or {}).get("evaluable_sets") or {}
    out: list[str] = []
    if bool((ev.get("fixed_coverage") or {}).get("enabled", False)):
        out.append("fixed_coverage")
    if bool((ev.get("path_existing_pairs") or {}).get("enabled", False)):
        out.append("path_existing_pairs")
    return out


def _baseline_restriction_from_manifest(baseline_manifest: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in baseline_manifest:
        info = item.get("baseline_input_restriction")
        if isinstance(info, dict) and bool(info.get("enabled", False)):
            return dict(info)
    return None


def _manifest_baseline_restriction_complete(baseline_manifest: list[dict[str, Any]]) -> bool:
    if not baseline_manifest:
        return True
    info = _baseline_restriction_from_manifest(baseline_manifest)
    if not info:
        return False
    return _path_exists(info.get("baseline_drug_input")) and _path_exists(info.get("baseline_pert_input"))


def _build_observed_evaluable_triplets_for_restriction(
    *,
    observed_drug_infer: str | Path,
    observed_pert_infer: str | Path,
    drug_mode: str,
    pert_mode: str,
    cell_pair_mode: str,
    evaluation_cfg: dict[str, Any],
) -> pd.DataFrame:
    eval_set_types = _enabled_evaluable_set_types(evaluation_cfg)
    if not eval_set_types:
        raise ValueError(
            "infer.restrict_baseline_to_observed_evaluable=true requires at least one "
            "evaluation.evaluable_sets.*.enabled entry."
        )
    true_drug_df = load_infer_signed_edge_df(observed_drug_infer, mode=drug_mode)
    true_pert_df = load_infer_signed_edge_df(observed_pert_infer, mode=pert_mode)
    ev_cfg = dict((evaluation_cfg or {}).get("evaluable_sets") or {})
    parts: list[pd.DataFrame] = []
    if "fixed_coverage" in eval_set_types:
        fixed_cfg = dict(ev_cfg.get("fixed_coverage") or {})
        parts.append(
            build_global_evaluable_triplets(
                true_drug_df,
                true_pert_df,
                cell_pair_mode=cell_pair_mode,
                min_candidate_nodes=int(fixed_cfg.get("min_candidate_nodes", 10)),
            )
        )
    if "path_existing_pairs" in eval_set_types:
        parts.append(
            build_path_existing_evaluable_triplets(
                true_drug_df,
                true_pert_df,
                cell_pair_mode=cell_pair_mode,
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _write_restricted_baseline_inputs(
    *,
    observed_drug_input: str | Path,
    observed_pert_input: str | Path,
    evaluable_triplets: pd.DataFrame,
    out_dir: str | Path,
) -> dict[str, Any]:
    if evaluable_triplets.empty or "triplet_id" not in evaluable_triplets.columns:
        raise ValueError("Cannot restrict baseline inputs: observed evaluable triplet table is empty.")
    evaluable = evaluable_triplets.copy()
    evaluable["triplet_id"] = evaluable["triplet_id"].astype(str)
    evaluable_ids = sorted(
        set(
            evaluable.loc[
                evaluable.get("is_global_evaluable", pd.Series(False, index=evaluable.index)).astype(bool),
                "triplet_id",
            ].astype(str)
        )
    )
    if not evaluable_ids:
        raise ValueError("Cannot restrict baseline inputs: no observed evaluable triplets found.")
    id_set = set(evaluable_ids)

    out_dir = ensure_dir(out_dir)
    drug_df = read_parquet_any(observed_drug_input)
    pert_df = read_parquet_any(observed_pert_input)
    if "triplet_id" not in drug_df.columns or "triplet_id" not in pert_df.columns:
        raise ValueError("Cannot restrict baseline inputs: observed input parquet files must contain triplet_id.")
    drug_restricted = drug_df.loc[drug_df["triplet_id"].astype(str).isin(id_set)].copy()
    pert_restricted = pert_df.loc[pert_df["triplet_id"].astype(str).isin(id_set)].copy()
    missing_drug = sorted(id_set - set(drug_restricted["triplet_id"].astype(str)))
    missing_pert = sorted(id_set - set(pert_restricted["triplet_id"].astype(str)))
    if missing_drug or missing_pert:
        raise ValueError(
            "Cannot restrict baseline inputs: evaluable triplet IDs are missing from observed inputs. "
            f"missing_drug={missing_drug[:10]}, missing_pert={missing_pert[:10]}"
        )

    drug_path = out_dir / "drug.parquet"
    pert_path = out_dir / "pert.parquet"
    ids_path = out_dir / "evaluable_triplet_ids.tsv"
    table_path = out_dir / "evaluable_triplets.tsv"
    summary_path = out_dir / "summary.json"
    write_parquet(drug_restricted, drug_path)
    write_parquet(pert_restricted, pert_path)
    pd.DataFrame({"triplet_id": evaluable_ids}).to_csv(ids_path, sep="\t", index=False)
    evaluable.to_csv(table_path, sep="\t", index=False)
    summary = {
        "enabled": True,
        "mode": "observed_evaluable_union",
        "baseline_drug_input": str(drug_path),
        "baseline_pert_input": str(pert_path),
        "evaluable_triplet_ids": str(ids_path),
        "evaluable_triplets": str(table_path),
        "n_evaluable_triplets": int(len(evaluable_ids)),
        "n_drug_rows_before": int(len(drug_df)),
        "n_drug_rows_after": int(len(drug_restricted)),
        "n_pert_rows_before": int(len(pert_df)),
        "n_pert_rows_after": int(len(pert_restricted)),
        "evaluable_set_types": sorted(set(evaluable.loc[evaluable["is_global_evaluable"].astype(bool), "evaluable_set_type"].astype(str))),
    }
    write_json(summary, summary_path)
    summary["summary_json"] = str(summary_path)
    return summary


def _restrict_decoy_candidate_manifest_to_triplets(
    *,
    decoy_manifest: dict[str, Any],
    allowed_triplet_ids: set[str],
    out_dir: str | Path,
) -> dict[str, Any]:
    if not decoy_manifest or not decoy_manifest.get("candidate_table"):
        return decoy_manifest
    table_path = Path(str(decoy_manifest["candidate_table"]))
    if not table_path.exists():
        return decoy_manifest
    decoy_candidates = pd.read_csv(table_path, sep="	")
    if decoy_candidates.empty or "original_triplet_id" not in decoy_candidates.columns:
        return decoy_manifest
    before = int(len(decoy_candidates))
    filtered = decoy_candidates.loc[
        decoy_candidates["original_triplet_id"].astype(str).isin(set(map(str, allowed_triplet_ids)))
    ].copy()
    out_dir = ensure_dir(out_dir)
    summary = {
        "n_decoy_candidates": int(len(filtered)),
        "n_decoy_candidates_before_restrict_baseline_to_observed_evaluable": before,
        "n_original_triplets": int(filtered["original_triplet_id"].astype(str).nunique()) if not filtered.empty else 0,
        "restricted_by": "infer.restrict_baseline_to_observed_evaluable",
        "materialized_prepared_inputs": False,
    }
    saved = save_decoy_candidate_table(
        out_dir=out_dir,
        decoy_candidates=filtered,
        summary=summary,
    )
    updated = dict(decoy_manifest)
    updated.update(decoy_candidate_manifest(saved))
    return updated


def _prepare_target_decoy_assignments(
    cfg,
    *,
    direction: str,
    prepared: dict[str, Any],
    kg_infer_path: str | Path,
    run_root: Path,
    null_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the Target-Decoy candidate universe, feature table and decoy assignments.

    Selection runs on the profile-supported observed triplets before path inference,
    and the same-drug-disease positive set is built from the full label input rather
    than from the path-evaluable subset.

    ``decoy_selection`` picks between 1:N percentile-rank matching and a uniform random
    draw from the same eligible pool.  The pre-inference matching balance table only
    describes a matching, so it is written for the matched selection only.
    """
    graph = load_directed_kg_graph(kg_infer_path)
    reach_cutoff = int(cfg.analysis["k_star"])
    candidate_pool_type = str(
        null_cfg.get(
            "decoy_candidate_universe",
            TARGET_DECOY_CANDIDATE_POOL_TYPE_PROFILE_SUPPORTED_OBSERVED,
        )
    )
    profile_supported_target_ids = set(
        prepared["profile_supported_observed_triplets"]["target_id"].astype(str)
    )
    if candidate_pool_type == TARGET_DECOY_CANDIDATE_POOL_TYPE_PROFILE_SUPPORTED_OBSERVED:
        labeled_target_ids = profile_supported_target_ids
    elif candidate_pool_type == TARGET_DECOY_CANDIDATE_POOL_TYPE_LABELED:
        labeled_target_ids = set(prepared["all_triplets"]["target_id"].astype(str))
    else:
        labeled_target_ids = None
    candidate_universe = build_target_decoy_candidate_universe(
        prepared["target_df"], kg_nodes=set(graph.nodes()), labeled_target_ids=labeled_target_ids,
    )
    if candidate_pool_type == TARGET_DECOY_CANDIDATE_POOL_TYPE_PROFILE_SUPPORTED_OBSERVED:
        candidate_set = set(candidate_universe)
        if candidate_set != profile_supported_target_ids:
            missing = sorted(profile_supported_target_ids - candidate_set)
            extra = sorted(candidate_set - profile_supported_target_ids)
            raise ValueError(
                "Primary Target-Decoy candidate universe must exactly match the "
                "profile-supported observed target population; "
                f"missing={missing[:10]}, extra={extra[:10]}."
            )
    feature_table = build_target_decoy_matching_features(
        candidate_universe,
        graph=graph,
        target_df=prepared["target_df"],
        reach_cutoff=reach_cutoff,
        candidate_pool_type=candidate_pool_type,
    )
    exclude_positives = bool(null_cfg.get("exclude_known_positive_targets", True))
    positives = (
        build_positive_targets_by_drug_disease(prepared["all_triplets"])
        if exclude_positives else {}
    )
    n_decoys = int(null_cfg.get("n_decoys_per_triplet", 10))
    decoy_selection = str(null_cfg.get("decoy_selection", "matched"))
    assignments = build_target_decoy_assignments(
        prepared["profile_supported_observed_triplets"],
        feature_table=feature_table,
        positive_targets_by_drug_disease=positives,
        direction=direction,
        n_decoys_per_triplet=n_decoys,
        seed=int(cfg.seed),
        exclude_known_positive_targets=exclude_positives,
        decoy_selection=decoy_selection,
    )

    metadata = target_decoy_model_metadata(
        n_candidate_universe=len(candidate_universe),
        reach_cutoff=reach_cutoff,
        n_decoys_per_triplet=n_decoys,
        tie_break_seed=int(cfg.seed),
        is_primary_spec=bool(null_cfg.get("is_primary_spec", True)),
        decoy_selection=decoy_selection,
        candidate_pool_type=candidate_pool_type,
    )
    n_profile_supported_observed_triplets = int(
        prepared["profile_supported_observed_triplets"]["triplet_id"].astype(str).nunique()
    )
    summary = {
        **summarize_target_decoy_assignments(
            assignments, n_input_triplets=n_profile_supported_observed_triplets,
        ),
        **metadata,
    }
    summary["n_profile_supported_observed_triplets"] = n_profile_supported_observed_triplets
    summary["n_profile_supported_observed_targets"] = len(profile_supported_target_ids)
    summary["candidate_universe_matches_profile_supported_observed_targets"] = (
        set(candidate_universe) == profile_supported_target_ids
    )
    summary["n_decoy_candidates"] = int(len(assignments))

    # Diagnostics live under evaluation/ so that they survive intermediate cleanup.
    eval_dir = ensure_dir(run_root / "evaluation" / direction)
    universe_path = eval_dir / "decoy_candidate_universe.tsv"
    assignments_path = eval_dir / "decoy_matching_assignments.tsv"
    feature_table.to_csv(universe_path, sep="\t", index=False)
    assignments.to_csv(assignments_path, sep="\t", index=False)
    write_json(summary, eval_dir / "decoy_matching_summary.json")
    summary["decoy_candidate_universe"] = str(universe_path)
    summary["decoy_matching_assignments"] = str(assignments_path)
    if decoy_selection == "matched":
        balance_path = eval_dir / "decoy_matching_balance.tsv"
        build_target_decoy_matching_balance(
            assignments,
            analysis_scope="all_profile_supported",
            stage="pre_inference_selected",
        ).assign(evaluable_set_type=pd.NA).to_csv(balance_path, sep="\t", index=False)
        summary["decoy_matching_balance_pre_inference"] = str(balance_path)
    print(
        f"[INFO] Target-Decoy selection={decoy_selection} n_candidate_universe={len(candidate_universe)} "
        f"n_triplets={summary['n_original_triplets']} n_assignments={len(assignments)}"
    )
    return assignments, summary


def _record_target_decoy_model_metadata(context: dict[str, Any], decoy_manifest: dict[str, Any]) -> None:
    """Copy the Target-Decoy model metadata into the run context.

    The fixed model definition and the actual run values (decoy selection rule,
    candidate-universe size, tie-break seed) belong in run metadata whether the
    selection was built in this run or loaded from an existing manifest.  The report
    reads ``decoy_selection`` from here to decide the figure label.
    """
    model_metadata = (decoy_manifest or {}).get("model_metadata")
    if model_metadata:
        context["target_decoy_model"] = dict(model_metadata)


def _run_output_steps(run_root: str | Path, direction: str, steps: set[str], cfg=None) -> dict[str, Any]:
    run_root = Path(run_root).expanduser().resolve()
    out: dict[str, Any] = {"run_root": str(run_root), "direction": direction, "steps": sorted(steps)}
    if "visualization_tables" in steps:
        if _output_step_exists(run_root, direction, "visualization_tables", cfg):
            print(f"[SKIP] make_visualization_tables existing outputs found run_root={run_root}")
            out["visualization_tables"] = {"skipped_existing": True}
        else:
            print(f"[RUN] make_visualization_tables run_root={run_root} direction={direction}")
            out["visualization_tables"] = _make_visualization_tables(run_root, direction, cfg)
    if "report" in steps:
        if _output_step_exists(run_root, direction, "report", cfg):
            print(f"[SKIP] report existing output found run_root={run_root}")
            out["report"] = {"skipped_existing": True}
        else:
            print(f"[RUN] report run_root={run_root} direction={direction}")
            out["report"] = write_val3_triplet_visualization_report(
                run_root=run_root,
                direction=direction,
                config=cfg.raw if cfg is not None else None,
                evaluation_summary=None,
            )
    return out


def _run_direction(
    cfg,
    *,
    direction: str,
    run_root_override: str | Path | None = None,
    steps: set[str] | None = None,
) -> dict[str, Any]:
    requested = steps or {"all"}
    is_all = "all" in requested
    if requested <= {"visualization_tables", "report"}:
        run_root = Path(run_root_override) if run_root_override is not None else resolve_output_dir(cfg, direction=direction)
        return _run_output_steps(run_root, direction, requested, cfg)

    repo_root = _repo_root()
    registry = load_dataset_registry(cfg.registry_path)
    spec = resolve_val3_registry_spec(cfg.data, registry, cfg.registry_path.parent, direction=direction)
    run_root = ensure_dir(Path(run_root_override) if run_root_override is not None else resolve_output_dir(cfg, direction=direction))
    context = initialize_run_context(cfg=cfg, spec=spec, run_root=run_root, repo_root=repo_root)
    with timed_run_section(context, run_root=run_root, section="setup"):
        kg_infer_path, kg_nodes = _prepare_kg_for_infer(spec, run_root)
        drug_mode, pert_mode = direction_modes(cfg, direction)

    print(f"[CONFIG] {cfg.config_path}")
    print(f"[DIRECTION] {direction} drug_mode={drug_mode} pert_mode={pert_mode}")
    print(f"[OUTPUT] {run_root}")

    observed_drug_input: Path | None = None
    observed_pert_input: Path | None = None
    observed_drug_infer: Path | str | None = None
    observed_pert_infer: Path | str | None = None
    baseline_drug_input_for_null: Path | str | None = None
    baseline_pert_input_for_null: Path | str | None = None
    baseline_manifest: list[dict[str, Any]] = []
    decoy_manifest: dict[str, Any] = {}

    prepared: dict[str, Any] | None = None
    run_prepare_inputs = is_all and cfg.sections.get("prepare_inputs", True) or "prepare_inputs" in requested
    run_path_exploration = is_all and cfg.sections.get("path_exploration", True) or "path_exploration" in requested
    run_evaluation = is_all and cfg.sections.get("evaluation", True) or "evaluation" in requested
    run_visualization_tables = (
        is_all and cfg.sections.get("make_visualization_tables", True) and run_evaluation
    ) or "visualization_tables" in requested
    run_report = is_all and cfg.sections.get("make_report", False) or "report" in requested

    if run_prepare_inputs:
        with timed_run_section(context, run_root=run_root, section="prepare_inputs"):
            obs_dir = run_root / "prepared_inputs" / "observed" / direction
            existing_drug_input, existing_pert_input = _observed_input_paths(run_root, direction)
            manifest_path = run_root / "prepared_inputs" / "baseline_manifest_inputs.json"
            null_inputs_requested = _gene_label_null_name(cfg) is not None or _decoy_null_enabled_cfg(cfg)
            if existing_drug_input.exists() and existing_pert_input.exists() and (not null_inputs_requested or manifest_path.exists()):
                print("[SKIP] prepare_inputs existing outputs found")
                observed_drug_input = existing_drug_input
                observed_pert_input = existing_pert_input
                loaded_manifest = _load_manifest_if_exists(manifest_path)
                if loaded_manifest is not None:
                    baseline_manifest, decoy_manifest = loaded_manifest
            else:
                print("[RUN] prepare_inputs")
                prepared = prepare_observed_inputs(spec=spec, direction=direction, out_dir=obs_dir, kg_nodes=kg_nodes)
                observed_drug_input = prepared["drug_path"]
                observed_pert_input = prepared["pert_path"]
                drug_df = prepared["drug_df"]
                pert_df = prepared["pert_df"]

                gene_label_null_type = _gene_label_null_name(cfg)
                if gene_label_null_type is not None:
                    rng = np.random.default_rng(cfg.seed)
                    base_input_root = run_root / "prepared_inputs" / "baseline" / direction
                    n_shuffle = int(cfg.baseline["n_shuffle"])
                    null_diag_parts: list[pd.DataFrame] = []
                    null_cfg = (cfg.baseline.get("null_models") or {}).get(gene_label_null_type) or {}
                    target_khop_context = None
                    if gene_label_null_type == "target_khop_restricted_paired_gene_label_permutation":
                        target_khop_context = build_target_khop_restricted_paired_gene_label_permutation_context(
                            drug_df,
                            pert_df,
                            kg_tsv=kg_infer_path,
                            cutoff=int(cfg.analysis["k_star"]),
                            kg_nodes=kg_nodes,
                        )
                    for iteration in _progress_iter(range(1, n_shuffle + 1), cfg=cfg, desc="baseline permutation", unit="iter"):
                        if gene_label_null_type == "target_khop_restricted_paired_gene_label_permutation":
                            perm_specs, diag = make_target_khop_restricted_paired_gene_label_permutation_specs_from_context(
                                target_khop_context,
                                rng=rng,
                            )
                            diag = diag.copy()
                            diag["iteration"] = int(iteration)
                            diag["min_local_bg_genes"] = int(null_cfg.get("min_local_bg_genes", 0))
                            null_diag_parts.append(diag)
                            saved = save_null1_permutation_specs(
                                out_dir=base_input_root,
                                iteration=iteration,
                                permutation_specs=perm_specs,
                            )
                            saved["inprocess"] = True
                        else:
                            shuffle_per_triplet = bool(null_cfg.get("shuffle_per_triplet", cfg.baseline.get("shuffle_per_triplet", True)))
                            perm_specs = make_global_paired_gene_label_permutation_specs(
                                drug_df,
                                pert_df,
                                rng=rng,
                                kg_nodes=kg_nodes,
                                shuffle_per_triplet=shuffle_per_triplet,
                            )
                            saved = save_baseline_permutation_specs(
                                out_dir=base_input_root,
                                iteration=iteration,
                                permutation_specs=perm_specs,
                            )
                        baseline_manifest.append({"iteration": int(iteration), "null_type": gene_label_null_type, **saved})
                    if null_diag_parts:
                        diag_path = ensure_dir(run_root / "evaluation" / direction) / "null1_local_shuffle_diagnostics.tsv"
                        pd.concat(null_diag_parts, ignore_index=True).to_csv(diag_path, sep="\t", index=False)
                    write_json({"baseline_manifest": baseline_manifest, "decoy_candidate_manifest": decoy_manifest}, run_root / "prepared_inputs" / "baseline_manifest_inputs.json")

                if _decoy_null_enabled_cfg(cfg):
                    if prepared is None:
                        raise ValueError("Decoy control candidate selection requires prepare_inputs=true.")
                    null_cfg = _decoy_null_cfg(cfg)
                    decoy_null_type = _decoy_null_type_cfg(cfg)
                    if decoy_null_type == TARGET_DECOY_NULL_TYPE:
                        decoy_candidates, decoy_summary = _prepare_target_decoy_assignments(
                            cfg,
                            direction=direction,
                            prepared=prepared,
                            kg_infer_path=kg_infer_path,
                            run_root=run_root,
                            null_cfg=null_cfg,
                        )
                    else:
                        decoy_rng = np.random.default_rng(int(cfg.seed) + 3000003)
                        max_decoy = null_cfg.get("max_decoy_anchors_per_triplet")
                        decoy_candidates = build_decoy_candidate_table(
                            prepared["triplets"],
                            prepared["target_df"],
                            kg_nodes=kg_nodes,
                            direction=direction,
                            exclude_known_positive_targets=bool(null_cfg.get("exclude_known_positive_targets", True)),
                            max_decoy_anchors_per_triplet=None if max_decoy is None else int(max_decoy),
                            rng=decoy_rng,
                            anchor_universe_triplets=prepared.get("profile_supported_observed_triplets", prepared["triplets"]),
                        )
                        decoy_summary = {
                            "n_decoy_candidates": int(len(decoy_candidates)),
                            "materialized_prepared_inputs": False,
                        }
                    decoy_root = run_root / "prepared_inputs" / "decoy_candidate" / direction
                    saved_decoy = save_decoy_candidate_table(
                        out_dir=decoy_root,
                        decoy_candidates=decoy_candidates,
                        summary=decoy_summary,
                    )
                    decoy_diag_path = ensure_dir(run_root / "evaluation" / direction) / "decoy_candidate_diagnostics.tsv"
                    summarize_decoy_candidates(decoy_candidates, null_type=decoy_null_type).to_csv(
                        decoy_diag_path, sep="\t", index=False
                    )
                    decoy_manifest = decoy_candidate_manifest(saved_decoy, null_type=decoy_null_type)
                    decoy_manifest["decoy_candidate_diagnostics"] = str(decoy_diag_path)
                    if decoy_null_type == TARGET_DECOY_NULL_TYPE:
                        decoy_manifest["decoy_candidate_universe"] = decoy_summary["decoy_candidate_universe"]
                        decoy_manifest["decoy_matching_assignments"] = decoy_summary["decoy_matching_assignments"]
                        decoy_manifest["model_metadata"] = dict(decoy_summary)
                    write_json({"baseline_manifest": baseline_manifest, "decoy_candidate_manifest": decoy_manifest}, run_root / "prepared_inputs" / "baseline_manifest_inputs.json")
    else:
        print("[SKIP] prepare_inputs")
        pre = cfg.infer.get("precomputed") or {}
        existing_drug_input, existing_pert_input = _observed_input_paths(run_root, direction)
        observed_drug_input = Path(pre.get("observed_drug_input") or existing_drug_input)
        observed_pert_input = Path(pre.get("observed_pert_input") or existing_pert_input)
        prepared_manifest_path = Path(pre.get("baseline_manifest_json") or (run_root / "prepared_inputs" / "baseline_manifest_inputs.json"))
        loaded_manifest = _load_manifest_if_exists(prepared_manifest_path)
        if loaded_manifest is not None:
            baseline_manifest, decoy_manifest = loaded_manifest

    _record_target_decoy_model_metadata(context, decoy_manifest)

    baseline_drug_input_for_null = observed_drug_input
    baseline_pert_input_for_null = observed_pert_input
    existing_restriction = _baseline_restriction_from_manifest(baseline_manifest)
    if existing_restriction is not None:
        baseline_drug_input_for_null = existing_restriction.get("baseline_drug_input") or baseline_drug_input_for_null
        baseline_pert_input_for_null = existing_restriction.get("baseline_pert_input") or baseline_pert_input_for_null

    if run_path_exploration:
        with timed_run_section(context, run_root=run_root, section="path_exploration"):
            pre = cfg.infer.get("precomputed") or {}
            existing_observed_drug = Path(pre["observed_drug_infer"]) if pre.get("observed_drug_infer") else _existing_infer_output(
                run_root / "infer_outputs" / "observed" / direction / "drug",
                mode=drug_mode,
                cutoff=int(cfg.analysis["k_star"]),
                gap=int(cfg.analysis["gap"]),
            )
            existing_observed_pert = Path(pre["observed_pert_infer"]) if pre.get("observed_pert_infer") else _existing_infer_output(
                run_root / "infer_outputs" / "observed" / direction / "pert",
                mode=pert_mode,
                cutoff=int(cfg.analysis["k_star"]),
                gap=int(cfg.analysis["gap"]),
            )
            infer_manifest_path = Path(pre.get("baseline_manifest_json") or (run_root / "infer_outputs" / "baseline_manifest.json"))
            loaded_infer_manifest = _load_manifest_if_exists(infer_manifest_path)
            if loaded_infer_manifest is not None:
                baseline_manifest, decoy_manifest = loaded_infer_manifest
                existing_restriction = _baseline_restriction_from_manifest(baseline_manifest)
                if existing_restriction is not None:
                    baseline_drug_input_for_null = existing_restriction.get("baseline_drug_input") or baseline_drug_input_for_null
                    baseline_pert_input_for_null = existing_restriction.get("baseline_pert_input") or baseline_pert_input_for_null
            if (
                existing_observed_drug is not None
                and existing_observed_pert is not None
                and _manifest_infer_outputs_complete(baseline_manifest, decoy_manifest)
                and (
                    not _baseline_restriction_enabled(cfg)
                    or _manifest_baseline_restriction_complete(baseline_manifest)
                )
            ):
                print("[SKIP] path_exploration existing outputs found")
                observed_drug_infer = existing_observed_drug
                observed_pert_infer = existing_observed_pert
                write_json({"baseline_manifest": baseline_manifest, "decoy_candidate_manifest": decoy_manifest}, run_root / "infer_outputs" / "baseline_manifest.json")
                run_path_exploration_body = False
            else:
                run_path_exploration_body = True

            if not run_path_exploration_body:
                pass
            elif observed_drug_input is None or observed_pert_input is None or not Path(observed_drug_input).exists() or not Path(observed_pert_input).exists():
                raise ValueError("path_exploration=true requires prepared observed inputs or infer.precomputed observed inputs.")
            else:
                pass

            infer_tasks: list[tuple[str, str, Path, Path, int | None, str | None]] = []
            if run_path_exploration_body:
                print("[RUN] path_exploration")
                if decoy_manifest:
                    print("[INFO] decoy candidate infer will run in batches without materialized wide inputs.")

            def _run_infer_task(task: tuple[str, str, Path, Path, int | None, str | None]) -> tuple[str, str, int | None, Path]:
                source, side, input_parquet, output_dir, iteration, permutation_specs = task
                mode = drug_mode if side == "drug" else pert_mode
                path_cache_dir = run_root / "infer_outputs" / "path_cache" / direction
                log_dir = run_root / "logs" / "infer"
                out = run_infer_subprocess(
                    repo_root=repo_root,
                    mode=mode,
                    cutoff=int(cfg.analysis["k_star"]),
                    gap=int(cfg.analysis["gap"]),
                    input_parquet=input_parquet,
                    kg_tsv=kg_infer_path,
                    output_dir=output_dir,
                    permutation_specs=permutation_specs,
                    path_cache_dir=path_cache_dir,
                    path_strategy=str(cfg.infer.get("path_strategy", "simple")),
                    log_dir=log_dir,
                    log_label=f"{source}-{side}" + (f"-iter={iteration}" if iteration is not None else ""),
                )
                return source, side, iteration, out

            def _record_infer_output(source: str, side: str, iteration: int | None, out: Path) -> None:
                nonlocal observed_drug_infer, observed_pert_infer
                if source == "observed" and side == "drug":
                    observed_drug_infer = out
                elif source == "observed" and side == "pert":
                    observed_pert_infer = out
                elif source == "baseline":
                    # Update manifest in-place.
                    for m in baseline_manifest:
                        if int(m["iteration"]) == int(iteration):
                            m[f"baseline_{side}_infer"] = str(out)
                            break
                elif source == "decoy_candidate":
                    decoy_manifest[f"candidate_{side}_infer"] = str(out)

            # Build the observed path cache first. Baseline gene-label nulls use the
            # same source/mode/cutoff path skeletons, so warming this cache avoids
            # duplicated baseline path enumeration without changing any infer input,
            # permutation, or downstream evaluation definition.
            #
            # One observed side is a single wide input, so running it as one task left
            # all but two workers idle for the whole observed pass. Its pert_id groups
            # are independent, so it is inferred as a batch stream over contiguous runs
            # of those groups instead; the parts concatenate back to the same
            # group-major rows the single-task route produced.
            if run_path_exploration_body:
                observed_workers = _infer_requested_workers(cfg)
                for side, input_parquet in (
                    ("drug", Path(observed_drug_input)),
                    ("pert", Path(observed_pert_input)),
                ):
                    observed_out = run_infer_dataframe_batches(
                        repo_root=repo_root,
                        mode=drug_mode if side == "drug" else pert_mode,
                        cutoff=int(cfg.analysis["k_star"]),
                        gap=int(cfg.analysis["gap"]),
                        input_batches=iter_pert_id_group_batches(
                            input_parquet,
                            target_batches=2 * observed_workers,
                            kg_nodes=kg_nodes,
                        ),
                        kg_tsv=kg_infer_path,
                        output_dir=run_root / "infer_outputs" / "observed" / direction / side,
                        path_cache_dir=run_root / "infer_outputs" / "path_cache" / direction,
                        path_strategy=str(cfg.infer.get("path_strategy", "simple")),
                        log_dir=run_root / "logs" / "infer",
                        log_label=f"observed-{side}",
                        max_workers=observed_workers,
                    )
                    n_parts = len(list(Path(observed_out).glob("part-*.parquet")))
                    print(f"[INFO] observed {side} infer: {n_parts} batch part(s), max_workers={observed_workers}")
                    _record_infer_output("observed", side, None, observed_out)

            if run_path_exploration_body and _baseline_restriction_enabled(cfg) and baseline_manifest:
                restriction = _write_restricted_baseline_inputs(
                    observed_drug_input=Path(observed_drug_input),
                    observed_pert_input=Path(observed_pert_input),
                    evaluable_triplets=_build_observed_evaluable_triplets_for_restriction(
                        observed_drug_infer=observed_drug_infer,
                        observed_pert_infer=observed_pert_infer,
                        drug_mode=drug_mode,
                        pert_mode=pert_mode,
                        cell_pair_mode=str(cfg.analysis["cell_pair_mode"]),
                        evaluation_cfg=cfg.evaluation,
                    ),
                    out_dir=run_root / "prepared_inputs" / "baseline_restricted" / direction,
                )
                baseline_drug_input_for_null = restriction["baseline_drug_input"]
                baseline_pert_input_for_null = restriction["baseline_pert_input"]
                for item in baseline_manifest:
                    item["baseline_input_restriction"] = restriction
                print(
                    "[INFO] baseline inputs restricted to observed evaluable triplets "
                    f"n={restriction['n_evaluable_triplets']} "
                    f"drug_rows={restriction['n_drug_rows_before']}->{restriction['n_drug_rows_after']} "
                    f"pert_rows={restriction['n_pert_rows_before']}->{restriction['n_pert_rows_after']}"
                )
                if decoy_manifest:
                    allowed_ids = set(pd.read_csv(restriction["evaluable_triplet_ids"], sep="	")["triplet_id"].astype(str))
                    decoy_manifest = _restrict_decoy_candidate_manifest_to_triplets(
                        decoy_manifest=decoy_manifest,
                        allowed_triplet_ids=allowed_ids,
                        out_dir=run_root / "prepared_inputs" / "decoy_candidate" / direction,
                    )

            if run_path_exploration_body:
                baseline_drug_input_for_infer = Path(baseline_drug_input_for_null or observed_drug_input)
                baseline_pert_input_for_infer = Path(baseline_pert_input_for_null or observed_pert_input)
                for item in baseline_manifest:
                    if bool(item.get("inprocess", False)):
                        item["baseline_drug_infer"] = None
                        item["baseline_pert_infer"] = None
                        continue
                    iteration = int(item["iteration"])
                    infer_tasks.extend([
                        (
                            "baseline",
                            "drug",
                            baseline_drug_input_for_infer,
                            run_root / "infer_outputs" / "baseline" / direction / f"iter={iteration}" / "drug",
                            iteration,
                            str(item["permutation_specs"]),
                        ),
                        (
                            "baseline",
                            "pert",
                            baseline_pert_input_for_infer,
                            run_root / "infer_outputs" / "baseline" / direction / f"iter={iteration}" / "pert",
                            iteration,
                            str(item["permutation_specs"]),
                        ),
                    ])

            if infer_tasks:
                for source, side, iteration, out in _run_infer_tasks_with_progress(
                    infer_tasks,
                    cfg=cfg,
                    run_one=_run_infer_task,
                    desc="infer subprocess",
                ):
                    _record_infer_output(source, side, iteration, out)
            if run_path_exploration_body and decoy_manifest:
                if prepared is None:
                    print("[INFO] loading existing prepared inputs for decoy batch infer")
                    prepared = _load_prepared_frames_for_decoy(spec, observed_drug_input=observed_drug_input)
                decoy_table = Path(decoy_manifest["candidate_table"])
                decoy_candidates = pd.read_csv(decoy_table, sep="\t")
                path_cache_dir = run_root / "infer_outputs" / "path_cache" / direction
                log_dir = run_root / "logs" / "infer"
                batch_size = 250
                decoy_batch_workers = int((cfg.infer or {}).get("max_workers", 1) or 1)
                # Both sides of a candidate batch are built by the same call, and the
                # two sides no longer take turns: one pool spans them, so the barrier
                # that used to idle every worker at the end of the drug pass is gone.
                decoy_drug_out, decoy_pert_out = run_infer_paired_dataframe_batches(
                    repo_root=repo_root,
                    cutoff=int(cfg.analysis["k_star"]),
                    gap=int(cfg.analysis["gap"]),
                    input_batches=iter_decoy_paired_input_batches(
                        decoy_candidates=decoy_candidates,
                        drug_df=prepared["drug_df"],
                        target_df=prepared["target_df"],
                        disease_df=prepared["disease_df"],
                        batch_size_candidates=batch_size,
                    ),
                    drug_mode=drug_mode,
                    pert_mode=pert_mode,
                    drug_output_dir=run_root / "infer_outputs" / "decoy_candidate" / direction / "drug",
                    pert_output_dir=run_root / "infer_outputs" / "decoy_candidate" / direction / "pert",
                    kg_tsv=kg_infer_path,
                    path_cache_dir=path_cache_dir,
                    path_strategy=str(cfg.infer.get("path_strategy", "simple")),
                    log_dir=log_dir,
                    log_label="decoy_candidate",
                    max_workers=decoy_batch_workers,
                )
                decoy_manifest["candidate_drug_infer"] = str(decoy_drug_out)
                decoy_manifest["candidate_pert_infer"] = str(decoy_pert_out)
            write_json({"baseline_manifest": baseline_manifest, "decoy_candidate_manifest": decoy_manifest}, run_root / "infer_outputs" / "baseline_manifest.json")
    else:
        print("[SKIP] path_exploration")
        pre = cfg.infer.get("precomputed") or {}
        observed_drug_infer = pre.get("observed_drug_infer") or _existing_infer_output(
            run_root / "infer_outputs" / "observed" / direction / "drug",
            mode=drug_mode,
            cutoff=int(cfg.analysis["k_star"]),
            gap=int(cfg.analysis["gap"]),
        )
        observed_pert_infer = pre.get("observed_pert_infer") or _existing_infer_output(
            run_root / "infer_outputs" / "observed" / direction / "pert",
            mode=pert_mode,
            cutoff=int(cfg.analysis["k_star"]),
            gap=int(cfg.analysis["gap"]),
        )
        existing_drug_input, existing_pert_input = _observed_input_paths(run_root, direction)
        observed_drug_input = pre.get("observed_drug_input") or str(existing_drug_input)
        observed_pert_input = pre.get("observed_pert_input") or str(existing_pert_input)
        manifest_path = Path(pre.get("baseline_manifest_json") or (run_root / "infer_outputs" / "baseline_manifest.json"))
        loaded_manifest = _load_manifest_if_exists(manifest_path)
        if loaded_manifest is not None:
            baseline_manifest, decoy_manifest = loaded_manifest
            existing_restriction = _baseline_restriction_from_manifest(baseline_manifest)
            if existing_restriction is not None:
                baseline_drug_input_for_null = existing_restriction.get("baseline_drug_input") or baseline_drug_input_for_null
                baseline_pert_input_for_null = existing_restriction.get("baseline_pert_input") or baseline_pert_input_for_null

    evaluation_summary: dict[str, Any] = {}
    if run_evaluation:
        with timed_run_section(context, run_root=run_root, section="evaluation"):
            if (
                _baseline_restriction_enabled(cfg)
                and baseline_manifest
                and not _manifest_baseline_restriction_complete(baseline_manifest)
            ):
                raise ValueError(
                    "infer.restrict_baseline_to_observed_evaluable=true requires restricted baseline inputs. "
                    "Run the path_exploration step first, or provide an infer_outputs/baseline_manifest.json "
                    "that records baseline_input_restriction."
                )
            if _output_step_exists(run_root, direction, "evaluation", cfg):
                print("[SKIP] evaluation existing outputs found")
                evaluation_summary = read_yaml_or_json(run_root / "evaluation" / direction / "summary.json")
            elif observed_drug_infer is None or observed_pert_infer is None:
                raise ValueError("evaluation=true requires observed infer outputs.")
            else:
                print("[RUN] evaluation")
                baseline_pairs = _baseline_pairs_from_manifest(baseline_manifest) if baseline_manifest else []
                evaluation_summary = evaluate_direction(
                    true_drug_infer_path=observed_drug_infer,
                    true_pert_infer_path=observed_pert_infer,
                    true_drug_input_path=observed_drug_input,
                    true_pert_input_path=observed_pert_input,
                    baseline_drug_input_path=baseline_drug_input_for_null,
                    baseline_pert_input_path=baseline_pert_input_for_null,
                    kg_tsv=kg_infer_path,
                    cutoff=int(cfg.analysis["k_star"]),
                    gap=int(cfg.analysis["gap"]),
                    drug_mode=drug_mode,
                    pert_mode=pert_mode,
                    cell_pair_mode=str(cfg.analysis["cell_pair_mode"]),
                    evaluation_cfg=cfg.evaluation,
                    baseline_pairs=baseline_pairs,
                    out_dir=run_root / "evaluation" / direction,
                    baseline_specs=_baseline_specs_from_manifest(baseline_manifest) if baseline_manifest else [],
                    path_cache_dir=run_root / "infer_outputs" / "path_cache" / direction,
                    path_strategy=str(cfg.infer.get("path_strategy", "simple")),
                    decoy_candidate_drug_infer_path=decoy_manifest.get("candidate_drug_infer"),
                    decoy_candidate_pert_infer_path=decoy_manifest.get("candidate_pert_infer"),
                    decoy_candidate_table_path=decoy_manifest.get("candidate_table"),
                    decoy_null_type=decoy_manifest.get("null_type") or _decoy_null_type_cfg(cfg),
                    progress=bool(cfg.progress.get("enabled", True) and cfg.progress.get("use_tqdm", True)),
                    tqdm_kwargs=tqdm_kwargs(cfg),
                    baseline_cfg=cfg.baseline,
                    seed=int(cfg.seed),
                    summary_extra={"direction": direction, "drug_mode": drug_mode, "pert_mode": pert_mode},
                )
                v3_tables = build_v3_report_tables(
                    run_root=run_root,
                    direction=direction,
                    config=cfg.raw,
                    evaluation_summary=evaluation_summary,
                )
                evaluation_summary["v3_report_tables"] = v3_tables
                cleanup_df = cleanup_intermediates(
                    run_root=run_root,
                    retention=str(cfg.storage.get("intermediate_retention", "evaluation")),
                    after_stage="evaluation",
                )
                if not cleanup_df.empty:
                    evaluation_summary["cleanup_after_evaluation"] = cleanup_df.to_dict(orient="records")
    else:
        print("[SKIP] evaluation")

    visualization_outputs: dict[str, str] = {}
    if run_visualization_tables:
        with timed_run_section(context, run_root=run_root, section="visualization_tables"):
            if _output_step_exists(run_root, direction, "visualization_tables", cfg):
                print("[SKIP] make_visualization_tables existing outputs found")
                visualization_outputs = {"skipped_existing": True}
            else:
                print("[RUN] make_visualization_tables")
                visualization_outputs = _make_visualization_tables(run_root, direction, cfg)

    report_outputs: dict[str, Any] = {}
    if run_report:
        with timed_run_section(context, run_root=run_root, section="report"):
            if _output_step_exists(run_root, direction, "report", cfg):
                print("[SKIP] report existing output found")
                report_outputs = {"skipped_existing": True}
            else:
                print("[RUN] report")
                report_outputs = write_val3_triplet_visualization_report(
                    run_root=run_root,
                    direction=direction,
                    config=cfg.raw if cfg is not None else None,
                    evaluation_summary=evaluation_summary,
                )
                cleanup_df = cleanup_intermediates(
                    run_root=run_root,
                    retention=str(cfg.storage.get("intermediate_retention", "evaluation")),
                    after_stage="report",
                )
                if not cleanup_df.empty:
                    report_outputs["cleanup_after_report"] = cleanup_df.to_dict(orient="records")

    public_context = finalize_run_context(context, run_root=run_root)
    run_summary = {
        **public_context,
        "direction": direction,
        "drug_mode": drug_mode,
        "pert_mode": pert_mode,
        "observed_drug_infer": str(observed_drug_infer) if observed_drug_infer is not None else None,
        "observed_pert_infer": str(observed_pert_infer) if observed_pert_infer is not None else None,
        "baseline_manifest": baseline_manifest,
        "decoy_candidate_manifest": decoy_manifest,
        "evaluation_summary": evaluation_summary,
        "visualization_outputs": visualization_outputs,
        "report_outputs": report_outputs,
    }
    finalize_run_summary(run_summary, run_root=run_root)
    return run_summary


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = parse_args(argv)
    steps = _parse_steps(args.steps)
    if args.config is None:
        if args.run_root is not None and steps <= {"visualization_tables", "report"}:
            direction = args.direction or "inhibitory"
            return [_run_output_steps(args.run_root, direction, steps)]
        args.config = str(_repo_root() / "config" / "val3.yaml")
    cfg = load_val3_triplet_config(args.config)
    if args.steps is None:
        default_steps = str((cfg.raw.get("steps") or {}).get("default", "all"))
        steps = _parse_steps([default_steps])
    directions = [args.direction] if args.direction else list(cfg.analysis.get("directions", ["inhibitory"]))
    summaries = []
    for direction in directions:
        summaries.append(_run_direction(cfg, direction=direction, run_root_override=args.run_root, steps=steps))
    if len(summaries) > 1:
        print(json.dumps({"runs": [s.get("run_root") for s in summaries]}, indent=2, ensure_ascii=False))
    return summaries


if __name__ == "__main__":
    main()
