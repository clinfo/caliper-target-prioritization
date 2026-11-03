"""Orchestrator for the val2 external-baseline comparison (methodology §7).

Fixed execution order Step1-8:
  1. input validation
  2. shared network H
  3. disease inputs (Q_up/Q_down/S_d/S_d^H)
  4. label-free score computation (CALIPER / NCS / proximity / RWR)
  5. availability table + common sets
  6. re-aggregate CALIPER and NCS over common cells
  7. join labels, build same-ordered evaluation table
  8. statistics and diagnostics
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common.metadata import write_json_safe as write_json
from val2.config import resolve_start_sign

from . import availability as availability_mod
from . import caliper as caliper_mod
from . import dada as dada_mod
from . import evaluate as evaluate_mod
from . import metadata as metadata_mod
from . import network as network_mod
from . import parallel as parallel_mod
from . import proximity as proximity_mod
from . import reach as reach_mod
from . import rwr as rwr_mod
from . import step4_worker
from . import wtcs as wtcs_mod
from .config import ResolvedBaselineConfig, load_baseline_config, resolve_output_dir
from .io import (
    ContinuousKD,
    file_manifest_entry,
    load_bundle,
    load_continuous_kd,
    load_creeds_disease_signatures,
    load_k_star,
    resolve_caliper_ks,
    resolve_spec,
)

TOKYO = ZoneInfo("Asia/Tokyo")
OUTPUT_ONLY_STEPS = {"stats", "figures", "report"}

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it


def _write_tsv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return path


def _rel(run_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(run_root))
    except Exception:
        return str(path)


def _compute_primary_ttest(
    deltas_df: pd.DataFrame,
    *,
    baselines: List[str],
    metrics: List[str],
    caliper_methods: List[str] | None = None,
    k_list: List[int],
    k_star: int,
    alpha: float,
) -> pd.DataFrame:
    """The run's only significance table: k=4 paired t-test + Holm.

    CALIPER series scored at other hop radii stay in the metric / effect-size
    tables but are not tested, so a run whose CALIPER radii exclude the primary
    distance produces an empty table rather than a test at the wrong k.
    """
    test_methods = evaluate_mod.resolve_test_caliper_methods(
        deltas_df, caliper_methods=caliper_methods, k_star=k_star
    )
    if not test_methods:
        print(
            "[STEP8] no CALIPER series at the primary distance "
            f"k={caliper_mod.PRIMARY_K} (k_star={k_star}); skipping the paired t-test."
        )
        return pd.DataFrame()
    return evaluate_mod.compute_ttest_table(
        deltas_df,
        baselines=baselines,
        metrics=metrics,
        caliper_methods=test_methods,
        k_list=k_list,
        alpha=alpha,
    )


def _make_dirs(run_root: Path) -> Dict[str, Path]:
    dirs = {
        "config": run_root / "config",
        "network": run_root / "network",
        "disease_inputs": run_root / "disease_inputs",
        "scores": run_root / "scores",
        "common_sets": run_root / "common_sets",
        "evaluation": run_root / "evaluation",
        "sensitivity": run_root / "sensitivity",
        "report": run_root / "report",
        "figures": run_root / "report" / "figures",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _selected_methods(cfg: ResolvedBaselineConfig) -> List[str]:
    if (
        cfg.primary_analysis == "native_shared_network"
        and cfg.caliper_ks is not None
        and len(cfg.caliper_ks) > 1
    ):
        return caliper_mod.caliper_method_keys(cfg.caliper_ks) + list(cfg.baselines)
    return ["CALIPER"] + list(cfg.baselines)


def _uses_network_baseline(cfg: ResolvedBaselineConfig) -> bool:
    # DADA is a shared-network baseline and shares the min_disease_seeds gate.
    return any(b in cfg.baselines for b in ("proximity", "RWR", "DADA"))


def _passes_native_disease_gate(disease_info: Dict[str, Any], cfg: ResolvedBaselineConfig) -> bool:
    if "NCS" in cfg.baselines and not bool(disease_info.get("ncs_query_valid", False)):
        return False
    if _uses_network_baseline(cfg) and int(disease_info.get("s_d_h_size", 0)) < int(cfg.min_disease_seeds):
        return False
    return True


def _passes_directed_disease_gate(disease_info: Dict[str, Any], cfg: ResolvedBaselineConfig) -> bool:
    if "NCS" in cfg.baselines and not bool(disease_info.get("ncs_query_valid", False)):
        return False
    if _uses_network_baseline(cfg) and int(disease_info.get("s_d_h_size", 0)) < int(cfg.min_disease_seeds):
        return False
    return True


def _build_disease_inputs(
    *,
    disease_ids: Set[str],
    disease_signatures: Dict[str, Dict[str, int]],
    continuous_kd: ContinuousKD,
    H_nodes: Set[str],
    min_disease_seeds: int,
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    feature_set = set(continuous_kd.feature_genes)
    info: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    for d in sorted(disease_ids):
        sig = disease_signatures.get(d, {})
        q_up = {g for g, s in sig.items() if int(s) == 1}
        q_down = {g for g, s in sig.items() if int(s) == -1}
        if q_up & q_down:
            raise ValueError(f"[disease {d}] up/down gene sets overlap (validation #5).")
        up_in_f = sorted(q_up & feature_set)
        down_in_f = sorted(q_down & feature_set)
        ncs_query_valid = bool(up_in_f) and bool(down_in_f)
        up_positions = np.array([continuous_kd.feature_index[g] for g in up_in_f], dtype=np.int64)
        down_positions = np.array([continuous_kd.feature_index[g] for g in down_in_f], dtype=np.int64)
        s_d = q_up | q_down
        seeds = sorted(s_d & H_nodes)
        info[d] = {
            "q_up": q_up,
            "q_down": q_down,
            "up_positions": up_positions,
            "down_positions": down_positions,
            "ncs_query_valid": ncs_query_valid,
            "seeds": seeds,
            "s_d_h_size": len(seeds),
        }
        rows.append(
            {
                "disease_id": d,
                "n_up": len(q_up),
                "n_down": len(q_down),
                "n_up_in_F": len(up_in_f),
                "n_down_in_F": len(down_in_f),
                "ncs_query_valid": ncs_query_valid,
                "n_S_d": len(s_d),
                "n_S_d_H": len(seeds),
                "meets_seed_threshold": len(seeds) >= int(min_disease_seeds),
            }
        )
    return info, pd.DataFrame(rows).sort_values("disease_id").reset_index(drop=True)


def _run_directed_target_reach_matched(
    cfg: ResolvedBaselineConfig,
    *,
    run_root: Path,
    dirs: Dict[str, Path],
    started_at: datetime,
) -> Dict[str, Any]:
    section_outputs: Dict[str, Dict[str, str]] = {}
    selected_baselines = list(cfg.baselines)
    selected_methods = _selected_methods(cfg)
    use_ncs = "NCS" in selected_baselines
    use_proximity = "proximity" in selected_baselines
    use_rwr = "RWR" in selected_baselines

    # ---------------- Step 1: input validation ----------------
    print("[STEP1] input validation (directed target-reach matched)")
    spec = resolve_spec(cfg)
    if spec.parquet_path.resolve() != cfg.val2_label_parquet.resolve():
        raise ValueError(
            "[SI-04] val2_label_parquet does not match the registry-resolved label parquet:\n"
            f"  registry: {spec.parquet_path}\n  SI-04   : {cfg.val2_label_parquet}"
        )
    bundle = load_bundle(spec)
    start_sign = resolve_start_sign(spec.perturbation_mode)
    k_star, k_selection_payload = load_k_star(cfg.k_selection_json, cfg.manual_k_star)
    h = int(k_star)
    print(f"[STEP1] h=k_star={h}, start_sign={start_sign}")

    labels_df = bundle.labels_df.copy()
    labels_df["disease_id"] = labels_df["disease_id"].astype(str)
    labels_df["target_gene"] = labels_df["target_gene"].astype(str)
    labels_df["cell_id"] = labels_df["cell_id"].astype(str)
    G_nodes = set(map(str, bundle.graph.nodes()))
    disease_ids = set(labels_df["disease_id"].unique())
    candidate_targets = set(labels_df["target_gene"].unique()) & G_nodes

    print("[STEP1] loading NCS reference profiles (full population)")
    continuous_kd = load_continuous_kd(
        cfg.continuous_kd_parquet,
        id_col=cfg.continuous_kd_id_col,
        cell_col=cfg.continuous_kd_cell_col,
        needed_genes=None,
    )
    sample = continuous_kd.matrix[: min(5, continuous_kd.matrix.shape[0])]
    if sample.size and np.all(np.isin(sample, np.array([-1.0, 0.0, 1.0], dtype=continuous_kd.matrix.dtype))):
        raise ValueError(
            "[NCS] reference profile appears to be discretized (only -1/0/1 values). "
            "The discretized profile must not be used for WTCS/NCS (validation #4)."
        )

    input_manifest = {
        "kg": file_manifest_entry(spec.ppi_path),
        "val2_label_parquet": file_manifest_entry(cfg.val2_label_parquet),
        "continuous_kd_profile": file_manifest_entry(cfg.continuous_kd_parquet),
        "k_selection_json": file_manifest_entry(cfg.k_selection_json),
    }

    if cfg.baseline_disease_signature_source == "full_creeds":
        print("[STEP1] loading full CREEDS disease signatures for baselines")
        baseline_disease_signatures = load_creeds_disease_signatures(
            cfg.creeds_disease_parquet, id_col=cfg.creeds_disease_id_col
        )
        input_manifest["creeds_disease_signature"] = file_manifest_entry(cfg.creeds_disease_parquet)
    else:
        baseline_disease_signatures = bundle.disease_signatures
    print(
        f"[STEP1] baseline disease-signature source: {cfg.baseline_disease_signature_source} "
        f"({len(baseline_disease_signatures)} diseases)"
    )

    # ---------------- Step 3: disease inputs ----------------
    print("[STEP3] building disease inputs")
    disease_info, disease_inputs_df = _build_disease_inputs(
        disease_ids=disease_ids,
        disease_signatures=baseline_disease_signatures,
        continuous_kd=continuous_kd,
        H_nodes=G_nodes,
        min_disease_seeds=cfg.min_disease_seeds,
    )
    disease_inputs_df = disease_inputs_df.rename(columns={"n_S_d_H": "n_S_d_in_G"})
    disease_inputs_df["primary_analysis_seed_gate"] = (
        disease_inputs_df["meets_seed_threshold"] if _uses_network_baseline(cfg) else True
    )
    _write_tsv(disease_inputs_df, dirs["disease_inputs"] / "disease_inputs.tsv")
    section_outputs["disease_inputs"] = {
        "disease_inputs": _rel(run_root, dirs["disease_inputs"] / "disease_inputs.tsv")
    }

    valid_diseases = {d for d, di in disease_info.items() if _passes_directed_disease_gate(di, cfg)}
    print(f"[STEP3] diseases passing selected-baseline gates: {len(valid_diseases)}/{len(disease_ids)}")
    cand_labels = labels_df[
        labels_df["disease_id"].isin(valid_diseases) & labels_df["target_gene"].isin(G_nodes)
    ].reset_index(drop=True)
    pair_cells: Dict[Tuple[str, str], List[str]] = {}
    for r in cand_labels.itertuples(index=False):
        pair_cells.setdefault((r.disease_id, r.target_gene), []).append(r.cell_id)
    continuous_cells: Set[Tuple[str, str]] = set(continuous_kd.keys)

    # ---------------- Step 2+4: per-target reach + label-free scores ----------------
    # Reach construction (STEP2) and CALIPER / NCS-local / proximity-RWR
    # (STEP4) are each independent per target t (k-hop neighborhood + sign
    # votes, local feature universe, local graph + distance/transition
    # matrices are all keyed by t alone), so both are fused into one
    # per-target unit of work and mapped over all candidate targets --
    # sequentially in-process when step4_n_jobs==1, or via a fork-based
    # process pool otherwise (see step4_worker.py / parallel.py). Reach is
    # still computed for every candidate target (diagnostic coverage,
    # matching the pre-parallel STEP2 scope); the scoring parts are skipped
    # for targets with no disease-gated pair (empty pair_cells_by_target[t]).
    pair_cells_by_target: Dict[str, Dict[str, List[str]]] = {}
    for (d, t), cells in pair_cells.items():
        pair_cells_by_target.setdefault(t, {})[d] = cells
    print(
        f"[STEP2/4] target reach + label-free scores for {len(candidate_targets)} candidate targets "
        f"(n_jobs={cfg.step4_n_jobs}, mp_start_method={cfg.mp_start_method})"
    )
    adjacency = reach_mod.build_undirected_adjacency(bundle.graph)
    gene_id_int = wtcs_mod.gene_ids_to_int(continuous_kd.feature_genes)
    cell_to_rows = continuous_kd.rows_by_cell()
    worker_state = step4_worker.build_worker_state(
        graph=bundle.graph,
        adjacency=adjacency,
        h=h,
        start_sign=start_sign,
        pert_sigs=bundle.perturbation_signatures,
        dis_sigs=bundle.disease_signatures,
        disease_info=disease_info,
        pair_cells_by_target=pair_cells_by_target,
        use_ncs=use_ncs,
        use_proximity=use_proximity,
        use_rwr=use_rwr,
        continuous_cells=continuous_cells,
        continuous_kd_feature_index=continuous_kd.feature_index,
        continuous_kd_matrix=continuous_kd.matrix,
        continuous_kd_key_to_row=continuous_kd.key_to_row,
        cell_to_rows=cell_to_rows,
        gene_id_int=gene_id_int,
        ncs_fallback_score=cfg.target_reach_ncs_fallback_score,
        proximity_n_random=cfg.proximity_n_random,
        proximity_min_bin_size=cfg.proximity_min_bin_size,
        proximity_fallback_score=cfg.target_reach_proximity_fallback_score,
        shared_seed=cfg.shared_seed,
        rwr_restart_probability=cfg.rwr_restart_probability,
        rwr_fallback_score=cfg.target_reach_rwr_fallback_score,
        rwr_target_only_seed_score=cfg.target_reach_rwr_target_only_seed_score,
    )
    target_results = parallel_mod.parallel_map_ordered(
        step4_worker.process_target,
        sorted(candidate_targets),
        n_jobs=cfg.step4_n_jobs,
        start_method=cfg.mp_start_method,
        initializer=step4_worker.configure_worker_state,
        initargs=(worker_state,),
        progress=tqdm,
        desc="target-reach+scores",
    )

    reach_rows: List[Dict[str, Any]] = []
    reach_summary_by_target: Dict[str, Dict[str, Any]] = {}
    caliper_cell_rows: List[Dict[str, Any]] = []
    ncs_cell_rows: List[Dict[str, Any]] = []
    ncs_norm_rows: List[Dict[str, Any]] = []
    ncs_fallback_by_pair: Dict[Tuple[str, str], int] = {}
    proximity_rows: List[Dict[str, Any]] = []
    rwr_rows: List[Dict[str, Any]] = []
    # Results are already in sorted-target order (parallel_map_ordered
    # preserves task order regardless of worker completion order), and each
    # per-target worker is internally deterministic given its inputs, so the
    # merged row lists below do not depend on n_jobs or scheduling order.
    for res in target_results:
        t = str(res["target"])
        reach_rows.append(res["reach_row"])
        reach_summary_by_target[t] = res["reach_row"]
        caliper_cell_rows.extend(res["caliper_rows"])
        ncs_cell_rows.extend(res["ncs_cell_rows"])
        ncs_norm_rows.extend(res["ncs_norm_rows"])
        for d, n in res["ncs_fallback_counts"].items():
            ncs_fallback_by_pair[(d, t)] = ncs_fallback_by_pair.get((d, t), 0) + int(n)
        proximity_rows.extend(res["proximity_rows"])
        rwr_rows.extend(res["rwr_rows"])

    target_reach_df = pd.DataFrame(
        reach_rows,
        columns=[
            "target_gene",
            "h",
            "target_in_G",
            "n_reach_nodes",
            "n_anchor_nodes",
            "n_local_nodes",
            "n_local_edges",
        ],
    ).sort_values("target_gene").reset_index(drop=True)
    _write_tsv(target_reach_df, dirs["network"] / "target_reach_by_target.tsv")
    section_outputs["network"] = {
        "target_reach_by_target": _rel(run_root, dirs["network"] / "target_reach_by_target.tsv")
    }

    caliper_cell_df = pd.DataFrame(
        caliper_cell_rows,
        columns=["disease_id", "target_gene", "cell_id", "score", "n_rev", "n_meas", "covAlign", "support"],
    )

    ncs_cell_value: Dict[Tuple[str, str, str], float] = {}
    ncs_cell_fallback: Dict[Tuple[str, str, str], bool] = {}
    for row in ncs_cell_rows:
        key = (str(row["disease_id"]), str(row["target_gene"]), str(row["cell_id"]))
        ncs_cell_value[key] = float(row["ncs"])
        ncs_cell_fallback[key] = bool(row["fallback"])
    ncs_cell_df = pd.DataFrame(
        ncs_cell_rows,
        columns=[
            "disease_id", "target_gene", "cell_id", "wtcs", "ncs", "score",
            "valid", "fallback", "reason", "n_F_h", "n_q_up_h", "n_q_down_h",
        ],
    )
    ncs_norm_df = pd.DataFrame(
        ncs_norm_rows,
        columns=[
            "disease_id", "target_gene", "cell_id", "n_F_h", "n_q_up_h", "n_q_down_h",
            "n_ref_total", "n_ref_defined", "n_pos_wtcs", "n_neg_wtcs",
            "n_zero_wtcs", "mu_pos", "mu_neg", "valid", "fallback", "reason",
        ],
    )

    proximity_info: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in proximity_rows:
        key = (str(row["disease_id"]), str(row["target_gene"]))
        proximity_info[key] = {k: v for k, v in row.items() if k not in ("disease_id", "target_gene")}
    rwr_info: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in rwr_rows:
        key = (str(row["disease_id"]), str(row["target_gene"]))
        rwr_info[key] = {k: v for k, v in row.items() if k not in ("disease_id", "target_gene")}

    proximity_df = pd.DataFrame(proximity_rows)
    rwr_df = pd.DataFrame(rwr_rows)

    _write_tsv(caliper_cell_df, dirs["scores"] / "caliper_cell_level.tsv")
    _write_tsv(ncs_cell_df, dirs["scores"] / "ncs_cell_level.tsv")
    _write_tsv(ncs_norm_df, dirs["scores"] / "ncs_norm_diagnostics.tsv")
    _write_tsv(proximity_df, dirs["scores"] / "proximity_pair.tsv")
    _write_tsv(rwr_df, dirs["scores"] / "rwr_pair.tsv")

    # ---------------- Step 5: availability + common sets ----------------
    print("[STEP5] availability and common sets")
    common = availability_mod.build_common_sets_directed_target_reach(
        labels_df=labels_df,
        G_nodes=G_nodes,
        disease_info=disease_info,
        continuous_cells=continuous_cells,
        reach_info=reach_summary_by_target,
        ncs_fallback_by_pair=ncs_fallback_by_pair,
        proximity_info=proximity_info,
        rwr_info=rwr_info,
        selected_baselines=selected_baselines,
        min_disease_seeds=cfg.min_disease_seeds,
    )
    _write_tsv(common["availability_df"], dirs["common_sets"] / "availability_by_pair.tsv")
    _write_tsv(common["common_diseases_df"], dirs["common_sets"] / "common_diseases.tsv")
    _write_tsv(common["common_candidates_df"], dirs["common_sets"] / "common_candidates_by_disease.tsv")
    _write_tsv(common["common_cells_df"], dirs["common_sets"] / "common_cells_by_pair.tsv")
    _write_tsv(common["exclusion_summary_df"], dirs["common_sets"] / "exclusion_summary.tsv")
    section_outputs["common_sets"] = {
        "availability_by_pair": _rel(run_root, dirs["common_sets"] / "availability_by_pair.tsv"),
        "common_diseases": _rel(run_root, dirs["common_sets"] / "common_diseases.tsv"),
        "common_candidates_by_disease": _rel(run_root, dirs["common_sets"] / "common_candidates_by_disease.tsv"),
        "common_cells_by_pair": _rel(run_root, dirs["common_sets"] / "common_cells_by_pair.tsv"),
        "exclusion_summary": _rel(run_root, dirs["common_sets"] / "exclusion_summary.tsv"),
    }
    flow = common["flow"]
    print(f"[STEP5] flow: {flow}")

    common_cells: Dict[Tuple[str, str], List[str]] = common["common_cells"]
    pair_label: Dict[Tuple[str, str], int] = common["pair_label"]

    # ---------------- Step 6: re-aggregate CALIPER & NCS over common cells ----------------
    print("[STEP6] re-aggregating CALIPER and NCS over common cells")
    caliper_target_df = caliper_mod.aggregate_target_level(caliper_cell_df, common_cells)

    ncs_target_rows: List[Dict[str, Any]] = []
    ncs_fallback_pairs: Set[Tuple[str, str]] = set()
    if use_ncs:
        for (d, t), cells in common_cells.items():
            missing = [c for c in cells if (d, t, c) not in ncs_cell_value]
            if missing:
                raise ValueError(f"[directed NCS] missing cell score for pair {(d, t)} cells={missing[:5]}")
            # Aggregate over non-fallback cells only (mirrors the native
            # analysis, which never mixes a fallback placeholder into the
            # mean). A pair with zero valid cells is itself a fallback and is
            # floored below the disease's valid NCS scores in STEP7.
            valid_cells = [c for c in cells if not ncs_cell_fallback.get((d, t, c), False)]
            if valid_cells:
                vals = [float(ncs_cell_value[(d, t, c)]) for c in valid_cells]
            else:
                ncs_fallback_pairs.add((d, t))
                vals = [float(ncs_cell_value[(d, t, c)]) for c in cells]
            mean_ncs = float(sum(vals) / len(vals))
            ncs_target_rows.append(
                {
                    "disease_id": d,
                    "target_gene": t,
                    "mean_ncs": mean_ncs,
                    "score": float(-mean_ncs),
                    "n_common_cells": int(len(cells)),
                    "n_valid_common_cells": int(len(valid_cells)),
                }
            )
    ncs_target_df = pd.DataFrame(
        ncs_target_rows,
        columns=["disease_id", "target_gene", "mean_ncs", "score", "n_common_cells", "n_valid_common_cells"],
    )
    _write_tsv(caliper_target_df, dirs["scores"] / "caliper_target_level.tsv")
    _write_tsv(ncs_target_df, dirs["scores"] / "ncs_target_level.tsv")
    section_outputs["scores"] = {
        "caliper_cell_level": _rel(run_root, dirs["scores"] / "caliper_cell_level.tsv"),
        "caliper_target_level": _rel(run_root, dirs["scores"] / "caliper_target_level.tsv"),
        "ncs_cell_level": _rel(run_root, dirs["scores"] / "ncs_cell_level.tsv"),
        "ncs_norm_diagnostics": _rel(run_root, dirs["scores"] / "ncs_norm_diagnostics.tsv"),
        "ncs_target_level": _rel(run_root, dirs["scores"] / "ncs_target_level.tsv"),
        "proximity_pair": _rel(run_root, dirs["scores"] / "proximity_pair.tsv"),
        "rwr_pair": _rel(run_root, dirs["scores"] / "rwr_pair.tsv"),
    }

    # ---------------- Step 7: join labels, build same-ordered eval table ----------------
    print("[STEP7] building scores_common_long")
    cal_map = {(r.disease_id, r.target_gene): r.score for r in caliper_target_df.itertuples(index=False)}
    ncs_map = {(r.disease_id, r.target_gene): r.score for r in ncs_target_df.itertuples(index=False)}
    prox_map = {(r.disease_id, r.target_gene): r.score for r in proximity_df.itertuples(index=False)}
    rwr_map = {(r.disease_id, r.target_gene): r.score for r in rwr_df.itertuples(index=False)}
    method_maps = {"CALIPER": cal_map}
    if use_ncs:
        method_maps["NCS"] = ncs_map
    if use_proximity:
        method_maps["proximity"] = prox_map
    if use_rwr:
        method_maps["RWR"] = rwr_map
    common_candidates_df = common["common_candidates_df"]

    # "No local evidence" fallback scores must not sit at an arbitrary neutral
    # point in a method's own score distribution: floor them below every
    # valid score within the same disease and method (methodology fairness
    # fix). CALIPER has no fallback concept and is excluded.
    fallback_keys_by_method: Dict[str, Set[Tuple[str, str]]] = {}
    if use_ncs:
        fallback_keys_by_method["NCS"] = set(ncs_fallback_pairs)
    if use_proximity:
        fallback_keys_by_method["proximity"] = {
            key for key, info in proximity_info.items() if bool(info.get("fallback", False))
        }
    if use_rwr:
        fallback_keys_by_method["RWR"] = {
            key for key, info in rwr_info.items() if bool(info.get("fallback", False))
        }
    evaluate_mod.apply_fallback_floor(method_maps, fallback_keys_by_method, common_candidates_df)

    long_rows: List[Dict[str, Any]] = []
    for r in common_candidates_df.itertuples(index=False):
        key = (r.disease_id, r.target_gene)
        label = int(pair_label.get(key, 0))
        for method, mp in method_maps.items():
            if key not in mp:
                raise ValueError(f"[STEP7] common pair {key} missing score for method '{method}'.")
            long_rows.append(
                {
                    "disease_id": r.disease_id,
                    "target_gene": r.target_gene,
                    "method": method,
                    "score": float(mp[key]),
                    "label": label,
                }
            )
    scores_common_long = pd.DataFrame(
        long_rows, columns=["disease_id", "target_gene", "method", "score", "label"]
    ).sort_values(["disease_id", "method", "target_gene"]).reset_index(drop=True)
    evaluate_mod.assert_method_consistency(scores_common_long, methods=selected_methods)
    _write_tsv(scores_common_long, dirs["evaluation"] / "scores_common_long.tsv")

    # ---------------- Step 8: statistics and diagnostics ----------------
    print("[STEP8] statistics and diagnostics")
    stats_cfg = cfg.stats
    k_list = list(cfg.K_list)
    metric_list = list(stats_cfg.get("metrics") or evaluate_mod.default_stats_metrics(k_list))
    n_boot = int(stats_cfg.get("n_boot", 10000))
    boot_seed = int(stats_cfg.get("bootstrap_seed", cfg.seed))
    alpha = float(stats_cfg.get("alpha", 0.05))

    metrics_df = evaluate_mod.compute_metrics_by_disease(
        scores_common_long,
        methods=selected_methods,
        K_list=k_list,
        K_eff_cap=cfg.K_eff_cap,
        tie_policy=cfg.tie_policy,
    )
    lift_eval_cohort_df = evaluate_mod.build_lift_eval_cohorts(
        metrics_df,
        k_list=k_list,
        min_expected_hits=cfg.lift_eval_min_expected_hits,
    )
    metrics_df = evaluate_mod.annotate_lift_eval_flags(metrics_df, lift_eval_cohort_df)
    method_summary = evaluate_mod.compute_method_summary(metrics_df, methods=selected_methods, k_list=k_list)
    deltas_df = evaluate_mod.compute_paired_deltas(metrics_df, baselines=selected_baselines, k_list=k_list)
    bootstrap_df = evaluate_mod.compute_bootstrap_summary(
        deltas_df, baselines=selected_baselines, metrics=metric_list, k_list=k_list, n_boot=n_boot, seed=boot_seed
    )
    effect_sizes_df = evaluate_mod.compute_effect_sizes(
        deltas_df, baselines=selected_baselines, metrics=metric_list, k_list=k_list
    )
    ttest_df = _compute_primary_ttest(
        deltas_df,
        baselines=selected_baselines,
        metrics=metric_list,
        k_list=k_list,
        k_star=int(k_star),
        alpha=alpha,
    )
    tie_diag_df = evaluate_mod.build_tie_diagnostics(metrics_df, k_list=k_list)

    _write_tsv(lift_eval_cohort_df, dirs["evaluation"] / "lift_eval_cohort.tsv")
    _write_tsv(metrics_df, dirs["evaluation"] / "metrics_by_disease_method.tsv")
    _write_tsv(method_summary, dirs["evaluation"] / "method_summary.tsv")
    _write_tsv(deltas_df, dirs["evaluation"] / "paired_deltas.tsv")
    _write_tsv(bootstrap_df, dirs["evaluation"] / "bootstrap_summary.tsv")
    _write_tsv(ttest_df, dirs["evaluation"] / "ttest_fdr.tsv")
    _write_tsv(effect_sizes_df, dirs["evaluation"] / "effect_sizes.tsv")
    _write_tsv(tie_diag_df, dirs["evaluation"] / "tie_diagnostics.tsv")
    section_outputs["evaluation"] = {
        "lift_eval_cohort": _rel(run_root, dirs["evaluation"] / "lift_eval_cohort.tsv"),
        "scores_common_long": _rel(run_root, dirs["evaluation"] / "scores_common_long.tsv"),
        "metrics_by_disease_method": _rel(run_root, dirs["evaluation"] / "metrics_by_disease_method.tsv"),
        "method_summary": _rel(run_root, dirs["evaluation"] / "method_summary.tsv"),
        "paired_deltas": _rel(run_root, dirs["evaluation"] / "paired_deltas.tsv"),
        "bootstrap_summary": _rel(run_root, dirs["evaluation"] / "bootstrap_summary.tsv"),
        "ttest_fdr": _rel(run_root, dirs["evaluation"] / "ttest_fdr.tsv"),
        "effect_sizes": _rel(run_root, dirs["evaluation"] / "effect_sizes.tsv"),
        "tie_diagnostics": _rel(run_root, dirs["evaluation"] / "tie_diagnostics.tsv"),
    }

    if cfg.sections.get("sensitivity", False):
        raise ValueError(
            "sections.sensitivity is currently supported only for primary_analysis='native_shared_network'."
        )

    _write_config_snapshot(cfg, dirs, run_root, section_outputs)

    finished_at = datetime.now(TOKYO)
    common_manifest = {
        "run_id": run_root.name,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_sec": (finished_at - started_at).total_seconds(),
        "timezone": "Asia/Tokyo",
        "git_or_source": "caliper-td",
        "primary_analysis": cfg.primary_analysis,
        "selected_baselines": selected_baselines,
        "methods": selected_methods,
        "data": dict(cfg.data),
        "global_random_seed": cfg.shared_seed,
        "k_star": int(k_star),
        "k_selection": k_selection_payload,
        "rwr_restart_probability": cfg.rwr_restart_probability,
        "ncs_feature_space_size": len(continuous_kd.feature_genes),
        "ncs_reference_profile_unit": "target_cell_averaged",
        "ncs_reference_n_profiles": int(continuous_kd.matrix.shape[0]),
        "ncs_reference_n_cells": int(len(continuous_kd.rows_by_cell())),
        "baseline_disease_signature_source": cfg.baseline_disease_signature_source,
        "min_disease_seeds": cfg.min_disease_seeds,
        "min_disease_seeds_applies_to_primary_analysis": _uses_network_baseline(cfg),
        "proximity": {"n_random": cfg.proximity_n_random, "min_bin_size": cfg.proximity_min_bin_size},
        "target_reach_matched": {
            "horizon": cfg.target_reach_horizon,
            "h": int(h),
            "reach_definition": "directed downstream nodes on signed causal KG",
            "local_graph_rules": [
                "node_set_is_target_plus_directed_reach",
                "induced_edges_within_anchor_nodes",
                "drop_direction",
                "drop_polarity",
                "collapse_parallel_edges",
                "remove_self_loops",
                "no_lcc_filtering",
            ],
            "fallback_scores": {
                "NCS": cfg.target_reach_ncs_fallback_score,
                "proximity": cfg.target_reach_proximity_fallback_score,
                "RWR": cfg.target_reach_rwr_fallback_score,
            },
            "fallback_floor_applied": True,
            "fallback_floor_note": (
                "Per-disease, per-method: fallback pairs (no local evidence) are "
                "overwritten with min(valid_scores) - 1.0 so they always rank "
                "below every valid score for that method in that disease; a "
                "disease with no valid score for a method ties all its entries "
                "at a constant (0.0). CALIPER is excluded (no fallback concept)."
            ),
            "proximity_rwr_seed_excludes_target": True,
            "rwr_target_only_seed_score": cfg.target_reach_rwr_target_only_seed_score,
            "rwr_target_only_seed_score_note": (
                "No longer consulted: disease seeds passed to proximity/RWR "
                "exclude the candidate target, so a target-only local graph "
                "never has a usable seed and always falls back."
            ),
        },
        "tie_policy": cfg.tie_policy,
        "K_eff_cap": cfg.K_eff_cap,
        "K_list": list(cfg.K_list),
        "lift_eval_min_expected_hits": cfg.lift_eval_min_expected_hits,
        "inputs": input_manifest,
        "si_resolution": {
            "SI-01_rwr_restart_probability": cfg.rwr_restart_probability,
            "SI-02_k_selection_json": str(cfg.k_selection_json),
            "SI-03_continuous_kd_profile": str(cfg.continuous_kd_parquet),
            "SI-04_val2_label_parquet": str(cfg.val2_label_parquet),
            "SI-05_shared_seed": cfg.shared_seed,
        },
    }
    run_summary = {
        "run_root": str(run_root),
        "run_id": run_root.name,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "primary_analysis": cfg.primary_analysis,
        "selected_baselines": selected_baselines,
        "methods": selected_methods,
        "k_star": int(k_star),
        "rwr_restart_probability": cfg.rwr_restart_probability,
        "flow": flow,
        "method_summary": method_summary.to_dict(orient="records"),
        "lift_eval": {
            "min_expected_hits": float(cfg.lift_eval_min_expected_hits),
            "K_list": list(k_list),
            "n_diseases_common": int(method_summary["n_diseases"].iloc[0]) if len(method_summary) else 0,
            "n_diseases_eval_by_K": {
                str(k): int(method_summary[f"n_diseases_lift_eval_{k}"].iloc[0])
                if len(method_summary) and f"n_diseases_lift_eval_{k}" in method_summary.columns
                else 0
                for k in k_list
            },
            "n_diseases_eval": int(method_summary.get("n_diseases_lift_eval", pd.Series([0])).iloc[0])
            if len(method_summary)
            else 0,
        },
        "n_valid_diseases_cheap_gate": len(valid_diseases),
        "sections": cfg.sections,
    }
    output_manifest = metadata_mod.build_output_manifest(section_outputs, run_root=run_root)
    metadata_mod.write_run_metadata(common_manifest, output_manifest, run_summary, run_root=run_root)

    if cfg.sections.get("make_figures", True):
        print("[FIGURES]")
        try:
            from .viz import make_all_figures

            fig_outputs = make_all_figures(run_root)
            section_outputs["figures"] = fig_outputs
        except Exception as e:  # pragma: no cover
            print(f"[FIGURES] skipped due to error: {e}")
    if cfg.sections.get("make_report", True):
        print("[REPORT]")
        try:
            from .report import write_report

            report_path = write_report(run_root)
            section_outputs["report"] = {"report": _rel(run_root, report_path)}
        except Exception as e:  # pragma: no cover
            print(f"[REPORT] skipped due to error: {e}")

    output_manifest = metadata_mod.build_output_manifest(section_outputs, run_root=run_root)
    metadata_mod.write_run_metadata(common_manifest, output_manifest, run_summary, run_root=run_root)

    print("[DONE]")
    return run_summary


def run(cfg: ResolvedBaselineConfig, *, run_root_override: str | Path | None = None) -> Dict[str, Any]:
    started_at = datetime.now(TOKYO)
    run_root = Path(run_root_override) if run_root_override is not None else resolve_output_dir(cfg)
    dirs = _make_dirs(run_root)
    if cfg.primary_analysis == "directed_target_reach_matched":
        return _run_directed_target_reach_matched(
            cfg,
            run_root=run_root,
            dirs=dirs,
            started_at=started_at,
        )
    section_outputs: Dict[str, Dict[str, str]] = {}
    selected_baselines = list(cfg.baselines)
    selected_methods = _selected_methods(cfg)
    use_ncs = "NCS" in selected_baselines
    use_proximity = "proximity" in selected_baselines
    use_rwr = "RWR" in selected_baselines
    use_dada = "DADA" in selected_baselines

    # ---------------- Step 1: input validation ----------------
    print("[STEP1] input validation")
    spec = resolve_spec(cfg)
    if spec.parquet_path.resolve() != cfg.val2_label_parquet.resolve():
        raise ValueError(
            "[SI-04] val2_label_parquet does not match the registry-resolved label parquet:\n"
            f"  registry: {spec.parquet_path}\n  SI-04   : {cfg.val2_label_parquet}"
        )
    bundle = load_bundle(spec)
    start_sign = resolve_start_sign(spec.perturbation_mode)
    caliper_k_list, k_star, k_selection_payload = resolve_caliper_ks(
        cfg.k_selection_json,
        cfg.caliper_ks,
        cfg.manual_k_star,
    )
    caliper_methods = caliper_mod.caliper_method_keys(caliper_k_list)
    print(f"[STEP1] caliper_ks={caliper_k_list}, k_star={k_star}, start_sign={start_sign}")

    labels_df = bundle.labels_df.copy()
    labels_df["disease_id"] = labels_df["disease_id"].astype(str)
    labels_df["target_gene"] = labels_df["target_gene"].astype(str)
    labels_df["cell_id"] = labels_df["cell_id"].astype(str)
    H_nodes_full = set(map(str, bundle.graph.nodes()))

    # Candidate targets restricted to those usable for the common universe.
    disease_ids = set(labels_df["disease_id"].unique())
    candidate_targets = set(labels_df["target_gene"].unique()) & H_nodes_full

    print("[STEP1] loading NCS reference profiles (full population)")
    # NCS normalization must use the full reference population per cell line,
    # independent of the val2 candidate set (§6.2.2, validation #13). Load every
    # target-cell profile (needed_genes=None), not only candidate targets.
    continuous_kd = load_continuous_kd(
        cfg.continuous_kd_parquet,
        id_col=cfg.continuous_kd_id_col,
        cell_col=cfg.continuous_kd_cell_col,
        needed_genes=None,
    )
    # Profile unit deviation (user direction): reference is the target-cell
    # averaged BING_knockdown_signature, not per-sig_id Level 5 signatures
    # (§6.2.1 / validation #4). The continuous profile must still not be the
    # discretized one.
    sample = continuous_kd.matrix[: min(5, continuous_kd.matrix.shape[0])]
    if sample.size and np.all(np.isin(sample, np.array([-1.0, 0.0, 1.0], dtype=continuous_kd.matrix.dtype))):
        raise ValueError(
            "[NCS] reference profile appears to be discretized (only -1/0/1 values). "
            "The discretized profile must not be used for WTCS/NCS (validation #4)."
        )

    input_manifest = {
        "kg": file_manifest_entry(spec.ppi_path),
        "val2_label_parquet": file_manifest_entry(cfg.val2_label_parquet),
        "continuous_kd_profile": file_manifest_entry(cfg.continuous_kd_parquet),
        "k_selection_json": file_manifest_entry(cfg.k_selection_json),
    }

    # Baseline disease signatures (WTCS query + proximity/RWR seeds only).
    # CALIPER always uses bundle.disease_signatures (val2 discretized profile).
    if cfg.baseline_disease_signature_source == "full_creeds":
        print("[STEP1] loading full CREEDS disease signatures for baselines")
        baseline_disease_signatures = load_creeds_disease_signatures(
            cfg.creeds_disease_parquet, id_col=cfg.creeds_disease_id_col
        )
        input_manifest["creeds_disease_signature"] = file_manifest_entry(cfg.creeds_disease_parquet)
    else:
        baseline_disease_signatures = bundle.disease_signatures
    print(
        f"[STEP1] baseline disease-signature source: {cfg.baseline_disease_signature_source} "
        f"({len(baseline_disease_signatures)} diseases)"
    )

    # ---------------- Step 2: shared network H ----------------
    print("[STEP2] building shared network H")
    H, net_stats = network_mod.build_shared_network(bundle.graph)
    H_nodes = set(H.nodes())
    node_list, node_index, dist_matrix = proximity_mod.build_distance_matrix(H)
    matcher = proximity_mod.DegreeMatcher(H, node_index, cfg.proximity_min_bin_size)
    write_json(net_stats, dirs["network"] / "shared_network_stats.json")
    _write_tsv(pd.DataFrame(matcher.export_bins()), dirs["network"] / "degree_bins.tsv")
    section_outputs["network"] = {
        "shared_network_stats": _rel(run_root, dirs["network"] / "shared_network_stats.json"),
        "degree_bins": _rel(run_root, dirs["network"] / "degree_bins.tsv"),
    }
    print(f"[STEP2] H nodes={net_stats['n_nodes']:,} edges={net_stats['n_edges']:,}")

    dada_network: Dict[str, Any] = {}
    if use_dada:
        dada_network = dada_mod.build_network_context(H)
        _write_tsv(
            dada_mod.node_centrality_table(dada_network),
            dirs["network"] / "dada_node_centrality.tsv",
        )
        section_outputs["network"]["dada_node_centrality"] = _rel(
            run_root, dirs["network"] / "dada_node_centrality.tsv"
        )
        print(f"[STEP2] DADA network mean degree lambda={dada_network['network_mean_degree']:.4f}")

    # ---------------- Step 3: disease inputs ----------------
    print("[STEP3] building disease inputs")
    disease_info, disease_inputs_df = _build_disease_inputs(
        disease_ids=disease_ids,
        disease_signatures=baseline_disease_signatures,
        continuous_kd=continuous_kd,
        H_nodes=H_nodes,
        min_disease_seeds=cfg.min_disease_seeds,
    )
    _write_tsv(disease_inputs_df, dirs["disease_inputs"] / "disease_inputs.tsv")
    section_outputs["disease_inputs"] = {
        "disease_inputs": _rel(run_root, dirs["disease_inputs"] / "disease_inputs.tsv")
    }

    # Cheap gates that any common-universe pair must pass; restrict scoring to them.
    valid_diseases = {d for d, di in disease_info.items() if _passes_native_disease_gate(di, cfg)}
    print(f"[STEP3] diseases passing selected-baseline gates: {len(valid_diseases)}/{len(disease_ids)}")

    cand_mask = (
        labels_df["disease_id"].isin(valid_diseases) & labels_df["target_gene"].isin(H_nodes)
    )
    cand_labels = labels_df[cand_mask].reset_index(drop=True)
    pair_cells: Dict[Tuple[str, str], List[str]] = {}
    for r in cand_labels.itertuples(index=False):
        pair_cells.setdefault((r.disease_id, r.target_gene), []).append(r.cell_id)

    continuous_cells: Set[Tuple[str, str]] = set(continuous_kd.keys)

    # ---------------- Step 4: label-free scores ----------------
    print("[STEP4] CALIPER cell-level scores")
    caliper_triples = [
        (r.disease_id, r.target_gene, r.cell_id) for r in cand_labels.itertuples(index=False)
    ]
    caliper_cell_frames: List[pd.DataFrame] = []
    for k in caliper_k_list:
        cell_df = caliper_mod.compute_caliper_cell_scores(
            graph=bundle.graph,
            pert_sigs=bundle.perturbation_signatures,
            dis_sigs=bundle.disease_signatures,
            cell_triples=caliper_triples,
            k_star=int(k),
            start_sign=start_sign,
            progress=tqdm,
        )
        cell_df = cell_df.copy()
        cell_df["k"] = int(k)
        caliper_cell_frames.append(cell_df)
    caliper_cell_df = (
        pd.concat(caliper_cell_frames, ignore_index=True)
        if caliper_cell_frames
        else pd.DataFrame()
    )

    # NCS cell-level scoring is only needed when NCS is a selected baseline.
    # When it is not, skip the (expensive) per-cell WTCS/NCS computation
    # entirely; the empty accumulators below keep the common-set builder and
    # the NCS-gated STEP6/STEP7 branches unaffected.
    ncs_cell_rows: List[Dict[str, Any]] = []
    ncs_norm_rows: List[Dict[str, Any]] = []
    ncs_valid_cells: Set[Tuple[str, str, str]] = set()
    ncs_cell_value: Dict[Tuple[str, str, str], float] = {}
    if use_ncs:
        print("[STEP4] NCS cell-level scores (per-cell normalization over full reference)")
        gene_id_int = wtcs_mod.gene_ids_to_int(continuous_kd.feature_genes)
        cell_to_rows = continuous_kd.rows_by_cell()

        # Candidate (disease, cell) -> target list, and cell -> diseases, restricted
        # to candidate pairs whose (target, cell) profile exists.
        cand_targets_by_dc: Dict[Tuple[str, str], List[str]] = {}
        cell_to_diseases: Dict[str, Set[str]] = {}
        for (d, t), cells in pair_cells.items():
            for c in cells:
                if (t, c) in continuous_cells:
                    cand_targets_by_dc.setdefault((d, c), []).append(t)
                    cell_to_diseases.setdefault(c, set()).add(d)

        for c in tqdm(sorted(cell_to_diseases.keys()), desc="ncs-cells"):
            rows = cell_to_rows.get(c, [])
            if not rows:
                continue
            ref_matrix = continuous_kd.matrix[rows]
            ranks_2d = wtcs_mod.ranks_for_matrix(ref_matrix, gene_id_int)
            local_of_global = {grow: i for i, grow in enumerate(rows)}
            for d in sorted(cell_to_diseases[c]):
                di = disease_info[d]
                res = wtcs_mod.compute_cell_ncs(
                    ref_matrix, ranks_2d, di["up_positions"], di["down_positions"]
                )
                ncs_norm_rows.append(
                    {
                        "disease_id": d, "cell_id": c,
                        "n_ref_total": int(res["n_ref_total"]),
                        "n_ref_defined": int(res["n_ref_defined"]),
                        "n_pos_wtcs": int(res["n_pos"]),
                        "n_neg_wtcs": int(res["n_neg"]),
                        "n_zero_wtcs": int(res["n_zero"]),
                        "mu_pos": float(res["mu_pos"]),
                        "mu_neg": float(res["mu_neg"]),
                        "valid": bool(res["valid"]),
                    }
                )
                wtcs_arr = res["wtcs"]
                ncs_arr = res["ncs"]
                for t in cand_targets_by_dc.get((d, c), []):
                    grow = continuous_kd.key_to_row.get((t, c))
                    if grow is None or grow not in local_of_global:
                        continue
                    li = local_of_global[grow]
                    w = float(wtcs_arr[li])
                    ncs_v = float(ncs_arr[li])
                    valid = bool(np.isfinite(ncs_v))
                    reason = ""
                    if not valid:
                        if not res["valid"]:
                            reason = "norm_undefined"
                        elif not np.isfinite(w):
                            reason = "wtcs_undefined"
                        else:
                            reason = "ncs_undefined"
                    ncs_cell_rows.append(
                        {
                            "disease_id": d, "target_gene": t, "cell_id": c,
                            "wtcs": w if np.isfinite(w) else np.nan,
                            "ncs": ncs_v if valid else np.nan,
                            "valid": valid, "reason": reason,
                        }
                    )
                    if valid:
                        ncs_valid_cells.add((d, t, c))
                        ncs_cell_value[(d, t, c)] = ncs_v
    ncs_cell_df = pd.DataFrame(
        ncs_cell_rows,
        columns=["disease_id", "target_gene", "cell_id", "wtcs", "ncs", "valid", "reason"],
    )
    ncs_norm_df = pd.DataFrame(
        ncs_norm_rows,
        columns=[
            "disease_id", "cell_id", "n_ref_total", "n_ref_defined",
            "n_pos_wtcs", "n_neg_wtcs", "n_zero_wtcs", "mu_pos", "mu_neg", "valid",
        ],
    )
    # (disease, cell) -> (n_pos_wtcs, n_neg_wtcs) for availability diagnostics.
    ncs_ref_counts: Dict[Tuple[str, str], Tuple[int, int]] = {
        (r["disease_id"], r["cell_id"]): (int(r["n_pos_wtcs"]), int(r["n_neg_wtcs"]))
        for r in ncs_norm_rows
    }

    n_nodes = len(node_index)
    rwr_disease_vec: Dict[str, np.ndarray] = {}
    rwr_disease_meta: Dict[str, Dict[str, Any]] = {}
    if use_rwr or use_dada:
        # DADA's raw component is the Classical RWR steady-state vector, so the
        # power iteration runs once even when both baselines are selected.
        print("[STEP4] RWR steady-state vectors")
        W = rwr_mod.build_transition_matrix(H, node_index)
        for d in tqdm(sorted(valid_diseases), desc="rwr"):
            seeds = disease_info[d]["seeds"]
            seed_idx = [node_index[s] for s in seeds if s in node_index]
            p, n_iter, final_l1 = rwr_mod.rwr_steady_state(W, n_nodes, seed_idx, cfg.rwr_restart_probability)
            rwr_disease_vec[d] = p
            rwr_disease_meta[d] = {"n_iter": n_iter, "final_l1": final_l1, "n_seed_in_H": len(seed_idx)}

    disease_seed_null: Dict[str, Tuple[Any, str]] = {}
    if use_proximity:
        print("[STEP4] proximity pair-level scores")
        # Precompute the degree-matched seed null once per disease (seed set is
        # identical across all candidate targets of a disease).
        for d in sorted({d for (d, _t) in pair_cells.keys()}):
            seeds = disease_info[d]["seeds"]
            ds_seed = proximity_mod.derive_seed(cfg.shared_seed, d, "__SEEDS__")
            disease_seed_null[d] = proximity_mod.build_disease_seed_null(
                matcher=matcher,
                node_index=node_index,
                seeds=seeds,
                n_random=cfg.proximity_n_random,
                derived_seed_value=ds_seed,
            )

    proximity_rows: List[Dict[str, Any]] = []
    proximity_valid: Dict[Tuple[str, str], bool] = {}
    rwr_rows: List[Dict[str, Any]] = []
    rwr_valid: Dict[Tuple[str, str], bool] = {}
    if use_proximity or use_rwr:
        for (d, t) in tqdm(sorted(pair_cells.keys()), desc="proximity/rwr"):
            di = disease_info[d]
            seeds = di["seeds"]
            seed_target_overlap = bool(t in set(seeds))
            if use_proximity:
                ds = proximity_mod.derive_seed(cfg.shared_seed, d, t)
                seed_null, seed_null_reason = disease_seed_null[d]
                prox = proximity_mod.compute_proximity_pair(
                    matrix=dist_matrix,
                    node_index=node_index,
                    matcher=matcher,
                    target_gene=t,
                    seeds=seeds,
                    seed_null=seed_null,
                    seed_null_reason=seed_null_reason,
                    n_random=cfg.proximity_n_random,
                    target_seed_value=ds,
                )
                proximity_rows.append(
                    {"disease_id": d, "target_gene": t, **prox, "seed_target_overlap": seed_target_overlap}
                )
                proximity_valid[(d, t)] = bool(prox["valid"])

            if use_rwr:
                p = rwr_disease_vec.get(d)
                if p is not None and t in node_index:
                    score = float(p[node_index[t]])
                    meta = rwr_disease_meta[d]
                    rwr_rows.append(
                        {
                            "disease_id": d, "target_gene": t, "score": score,
                            "restart_probability": cfg.rwr_restart_probability,
                            "n_iter": meta["n_iter"], "final_l1": meta["final_l1"],
                            "n_seed_in_H": meta["n_seed_in_H"],
                            "seed_target_overlap": seed_target_overlap, "valid": True,
                        }
                    )
                    rwr_valid[(d, t)] = True
                else:
                    rwr_rows.append(
                        {
                            "disease_id": d, "target_gene": t, "score": np.nan,
                            "restart_probability": cfg.rwr_restart_probability,
                            "n_iter": np.nan, "final_l1": np.nan, "n_seed_in_H": 0,
                            "seed_target_overlap": seed_target_overlap, "valid": False,
                        }
                    )
                    rwr_valid[(d, t)] = False

    proximity_df = pd.DataFrame(proximity_rows)
    rwr_df = pd.DataFrame(rwr_rows)

    dada_strategy: Dict[str, Dict[str, Any]] = {}
    dada_strategy_df = pd.DataFrame(columns=dada_mod.DISEASE_STRATEGY_COLUMNS)
    dada_pair_df = pd.DataFrame(columns=dada_mod.PAIR_COLUMNS)
    dada_valid: Dict[Tuple[str, str], bool] = {}
    dada_selected: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if use_dada:
        print("[STEP4] DADA centrality adjustment and seed-degree component selection")
        dada_strategy, dada_strategy_df = dada_mod.build_disease_strategy(
            disease_ids=disease_ids,
            disease_info=disease_info,
            network_context=dada_network,
            restart_probability=cfg.rwr_restart_probability,
        )
        dada_pair_df, dada_valid, dada_selected = dada_mod.build_pair_components(
            pairs=pair_cells.keys(),
            disease_strategy=dada_strategy,
            network_context=dada_network,
            rwr_disease_vec=rwr_disease_vec,
            node_index=node_index,
        )
        _write_tsv(dada_strategy_df, dirs["disease_inputs"] / "dada_disease_strategy.tsv")
        section_outputs["disease_inputs"]["dada_disease_strategy"] = _rel(
            run_root, dirs["disease_inputs"] / "dada_disease_strategy.tsv"
        )
        _write_tsv(dada_pair_df, dirs["scores"] / "dada_pair.tsv")
        print(
            "[STEP4] DADA selected components by disease: "
            f"{dada_mod.selected_component_counts(dada_strategy_df)}"
        )

    _write_tsv(caliper_cell_df, dirs["scores"] / "caliper_cell_level.tsv")
    _write_tsv(ncs_cell_df, dirs["scores"] / "ncs_cell_level.tsv")
    _write_tsv(ncs_norm_df, dirs["scores"] / "ncs_norm_diagnostics.tsv")
    _write_tsv(proximity_df, dirs["scores"] / "proximity_pair.tsv")
    _write_tsv(rwr_df, dirs["scores"] / "rwr_pair.tsv")

    # ---------------- Step 5: availability + common sets ----------------
    print("[STEP5] availability and common sets")
    common = availability_mod.build_common_sets(
        labels_df=labels_df,
        H_nodes=H_nodes,
        disease_info=disease_info,
        continuous_cells=continuous_cells,
        ncs_valid_cells=ncs_valid_cells,
        ncs_ref_counts=ncs_ref_counts,
        proximity_valid=proximity_valid,
        rwr_valid=rwr_valid,
        min_disease_seeds=cfg.min_disease_seeds,
        selected_baselines=selected_baselines,
        dada_valid=dada_valid,
    )
    _write_tsv(common["availability_df"], dirs["common_sets"] / "availability_by_pair.tsv")
    _write_tsv(common["common_diseases_df"], dirs["common_sets"] / "common_diseases.tsv")
    _write_tsv(common["common_candidates_df"], dirs["common_sets"] / "common_candidates_by_disease.tsv")
    _write_tsv(common["common_cells_df"], dirs["common_sets"] / "common_cells_by_pair.tsv")
    _write_tsv(common["exclusion_summary_df"], dirs["common_sets"] / "exclusion_summary.tsv")
    section_outputs["common_sets"] = {
        "availability_by_pair": _rel(run_root, dirs["common_sets"] / "availability_by_pair.tsv"),
        "common_diseases": _rel(run_root, dirs["common_sets"] / "common_diseases.tsv"),
        "common_candidates_by_disease": _rel(run_root, dirs["common_sets"] / "common_candidates_by_disease.tsv"),
        "common_cells_by_pair": _rel(run_root, dirs["common_sets"] / "common_cells_by_pair.tsv"),
        "exclusion_summary": _rel(run_root, dirs["common_sets"] / "exclusion_summary.tsv"),
    }
    flow = common["flow"]
    print(f"[STEP5] flow: {flow}")

    common_cells: Dict[Tuple[str, str], List[str]] = common["common_cells"]
    pair_label: Dict[Tuple[str, str], int] = common["pair_label"]

    # ---------------- Step 6: re-aggregate CALIPER & NCS over common cells ----------------
    print("[STEP6] re-aggregating CALIPER and NCS over common cells")
    caliper_target_frames: List[pd.DataFrame] = []
    for k, method_key in zip(caliper_k_list, caliper_methods):
        cell_sub = caliper_cell_df[caliper_cell_df["k"].astype(int) == int(k)].drop(columns=["k"])
        target_df = caliper_mod.aggregate_target_level(cell_sub, common_cells)
        if not target_df.empty:
            target_df = target_df.copy()
            target_df["k"] = int(k)
            target_df["method"] = method_key
        caliper_target_frames.append(target_df)
    caliper_target_df = (
        pd.concat(caliper_target_frames, ignore_index=True)
        if caliper_target_frames
        else pd.DataFrame()
    )

    ncs_target_rows: List[Dict[str, Any]] = []
    if use_ncs:
        for (d, t), cells in common_cells.items():
            vals = [ncs_cell_value[(d, t, c)] for c in cells if (d, t, c) in ncs_cell_value]
            if not vals:
                continue
            mean_ncs = float(sum(vals) / len(vals))
            ncs_target_rows.append(
                {
                    "disease_id": d, "target_gene": t,
                    "mean_ncs": mean_ncs, "score": float(-mean_ncs),
                    "n_common_cells": int(len(vals)),
                }
            )
    ncs_target_df = pd.DataFrame(
        ncs_target_rows, columns=["disease_id", "target_gene", "mean_ncs", "score", "n_common_cells"]
    )
    _write_tsv(caliper_target_df, dirs["scores"] / "caliper_target_level.tsv")
    _write_tsv(ncs_target_df, dirs["scores"] / "ncs_target_level.tsv")
    section_outputs["scores"] = {
        "caliper_cell_level": _rel(run_root, dirs["scores"] / "caliper_cell_level.tsv"),
        "caliper_target_level": _rel(run_root, dirs["scores"] / "caliper_target_level.tsv"),
        "ncs_cell_level": _rel(run_root, dirs["scores"] / "ncs_cell_level.tsv"),
        "ncs_norm_diagnostics": _rel(run_root, dirs["scores"] / "ncs_norm_diagnostics.tsv"),
        "ncs_target_level": _rel(run_root, dirs["scores"] / "ncs_target_level.tsv"),
        "proximity_pair": _rel(run_root, dirs["scores"] / "proximity_pair.tsv"),
        "rwr_pair": _rel(run_root, dirs["scores"] / "rwr_pair.tsv"),
    }

    # DADA ranks within each disease over the fixed common candidate set, so its
    # rank population is exactly the universe evaluated for every other method.
    dada_ranking_df = pd.DataFrame(columns=dada_mod.COMMON_RANKING_COLUMNS)
    dada_map: Dict[Tuple[str, str], float] = {}
    if use_dada:
        print("[STEP6] DADA common-set ranking")
        dada_ranking_df, dada_map = dada_mod.build_common_ranking(
            common_pairs=[
                (r.disease_id, r.target_gene)
                for r in common["common_candidates_df"].itertuples(index=False)
            ],
            selected_components=dada_selected,
        )
        _write_tsv(dada_ranking_df, dirs["scores"] / "dada_common_ranking.tsv")
        section_outputs["scores"]["dada_pair"] = _rel(run_root, dirs["scores"] / "dada_pair.tsv")
        section_outputs["scores"]["dada_common_ranking"] = _rel(
            run_root, dirs["scores"] / "dada_common_ranking.tsv"
        )

    # ---------------- Step 7: join labels, build same-ordered eval table ----------------
    print("[STEP7] building scores_common_long")
    method_maps: Dict[str, Dict[Tuple[str, str], float]] = {}
    for k, method_key in zip(caliper_k_list, caliper_methods):
        sub = caliper_target_df[caliper_target_df["k"].astype(int) == int(k)]
        method_maps[method_key] = {
            (r.disease_id, r.target_gene): r.score for r in sub.itertuples(index=False)
        }
    ncs_map = {(r.disease_id, r.target_gene): r.score for r in ncs_target_df.itertuples(index=False)}
    prox_map = {
        (r.disease_id, r.target_gene): r.score
        for r in proximity_df.itertuples(index=False)
        if bool(r.valid)
    }
    rwr_map = {
        (r.disease_id, r.target_gene): r.score
        for r in rwr_df.itertuples(index=False)
        if bool(r.valid)
    }

    if use_ncs:
        method_maps["NCS"] = ncs_map
    if use_proximity:
        method_maps["proximity"] = prox_map
    if use_rwr:
        method_maps["RWR"] = rwr_map
    if use_dada:
        method_maps["DADA"] = dada_map
    long_rows: List[Dict[str, Any]] = []
    common_candidates_df = common["common_candidates_df"]
    for r in common_candidates_df.itertuples(index=False):
        key = (r.disease_id, r.target_gene)
        label = int(pair_label.get(key, 0))
        for method, mp in method_maps.items():
            if key not in mp:
                raise ValueError(
                    f"[STEP7] common pair {key} missing score for method '{method}' "
                    "(validation #6: missing score after common-set fix)."
                )
            long_rows.append(
                {
                    "disease_id": r.disease_id, "target_gene": r.target_gene,
                    "method": method, "score": float(mp[key]), "label": label,
                }
            )
    scores_common_long = pd.DataFrame(
        long_rows, columns=["disease_id", "target_gene", "method", "score", "label"]
    ).sort_values(["disease_id", "method", "target_gene"]).reset_index(drop=True)
    evaluate_mod.assert_method_consistency(scores_common_long, methods=selected_methods)
    _write_tsv(scores_common_long, dirs["evaluation"] / "scores_common_long.tsv")

    # ---------------- Step 8: statistics and diagnostics ----------------
    print("[STEP8] statistics and diagnostics")
    stats_cfg = cfg.stats
    k_list = list(cfg.K_list)
    metric_list = list(stats_cfg.get("metrics") or evaluate_mod.default_stats_metrics(k_list))
    n_boot = int(stats_cfg.get("n_boot", 10000))
    boot_seed = int(stats_cfg.get("bootstrap_seed", cfg.seed))
    alpha = float(stats_cfg.get("alpha", 0.05))

    metrics_df = evaluate_mod.compute_metrics_by_disease(
        scores_common_long,
        methods=selected_methods,
        K_list=k_list,
        K_eff_cap=cfg.K_eff_cap,
        tie_policy=cfg.tie_policy,
    )
    lift_eval_cohort_df = evaluate_mod.build_lift_eval_cohorts(
        metrics_df,
        k_list=k_list,
        min_expected_hits=cfg.lift_eval_min_expected_hits,
    )
    metrics_df = evaluate_mod.annotate_lift_eval_flags(metrics_df, lift_eval_cohort_df)
    method_summary = evaluate_mod.compute_method_summary(metrics_df, methods=selected_methods, k_list=k_list)
    deltas_df = evaluate_mod.compute_paired_deltas(
        metrics_df,
        baselines=selected_baselines,
        caliper_methods=caliper_methods,
        k_list=k_list,
    )
    bootstrap_df = evaluate_mod.compute_bootstrap_summary(
        deltas_df,
        baselines=selected_baselines,
        metrics=metric_list,
        caliper_methods=caliper_methods,
        k_list=k_list,
        n_boot=n_boot,
        seed=boot_seed,
    )
    effect_sizes_df = evaluate_mod.compute_effect_sizes(
        deltas_df,
        baselines=selected_baselines,
        metrics=metric_list,
        caliper_methods=caliper_methods,
        k_list=k_list,
    )
    ttest_df = _compute_primary_ttest(
        deltas_df,
        baselines=selected_baselines,
        metrics=metric_list,
        caliper_methods=caliper_methods,
        k_list=k_list,
        k_star=int(k_star),
        alpha=alpha,
    )
    tie_diag_df = evaluate_mod.build_tie_diagnostics(metrics_df, k_list=k_list)

    _write_tsv(lift_eval_cohort_df, dirs["evaluation"] / "lift_eval_cohort.tsv")
    _write_tsv(metrics_df, dirs["evaluation"] / "metrics_by_disease_method.tsv")
    _write_tsv(method_summary, dirs["evaluation"] / "method_summary.tsv")
    _write_tsv(deltas_df, dirs["evaluation"] / "paired_deltas.tsv")
    _write_tsv(bootstrap_df, dirs["evaluation"] / "bootstrap_summary.tsv")
    _write_tsv(ttest_df, dirs["evaluation"] / "ttest_fdr.tsv")
    _write_tsv(effect_sizes_df, dirs["evaluation"] / "effect_sizes.tsv")
    _write_tsv(tie_diag_df, dirs["evaluation"] / "tie_diagnostics.tsv")
    section_outputs["evaluation"] = {
        "lift_eval_cohort": _rel(run_root, dirs["evaluation"] / "lift_eval_cohort.tsv"),
        "scores_common_long": _rel(run_root, dirs["evaluation"] / "scores_common_long.tsv"),
        "metrics_by_disease_method": _rel(run_root, dirs["evaluation"] / "metrics_by_disease_method.tsv"),
        "method_summary": _rel(run_root, dirs["evaluation"] / "method_summary.tsv"),
        "paired_deltas": _rel(run_root, dirs["evaluation"] / "paired_deltas.tsv"),
        "bootstrap_summary": _rel(run_root, dirs["evaluation"] / "bootstrap_summary.tsv"),
        "ttest_fdr": _rel(run_root, dirs["evaluation"] / "ttest_fdr.tsv"),
        "effect_sizes": _rel(run_root, dirs["evaluation"] / "effect_sizes.tsv"),
        "tie_diagnostics": _rel(run_root, dirs["evaluation"] / "tie_diagnostics.tsv"),
    }

    # ---------------- Step 9 (optional): candidate-excluded sensitivity ----------------
    if cfg.sections.get("sensitivity", False):
        if not (use_proximity and use_rwr):
            raise ValueError("sections.sensitivity requires both 'proximity' and 'RWR' in external_baseline.baselines.")
        print("[STEP9] candidate-excluded sensitivity (§9.1)")
        from . import sensitivity as sensitivity_mod

        sens = sensitivity_mod.run_candidate_excluded(
            common_candidates_df=common_candidates_df,
            disease_info=disease_info,
            proximity_main=prox_map,
            rwr_main=rwr_map,
            dist_matrix=dist_matrix,
            node_index=node_index,
            matcher=matcher,
            W=W,
            n_nodes=n_nodes,
            n_random=cfg.proximity_n_random,
            rwr_restart_probability=cfg.rwr_restart_probability,
            shared_seed=cfg.shared_seed,
            K_eff_cap=cfg.K_eff_cap,
            K_list=k_list,
            tie_policy=cfg.tie_policy,
            lift_eval_min_expected_hits=cfg.lift_eval_min_expected_hits,
            progress=tqdm,
        )
        _write_tsv(sens["scores"], dirs["sensitivity"] / "candidate_excluded_scores.tsv")
        _write_tsv(sens["metrics"], dirs["sensitivity"] / "candidate_excluded_metrics.tsv")
        _write_tsv(sens["summary"], dirs["sensitivity"] / "candidate_excluded_summary.tsv")
        section_outputs["sensitivity"] = {
            "candidate_excluded_scores": _rel(run_root, dirs["sensitivity"] / "candidate_excluded_scores.tsv"),
            "candidate_excluded_metrics": _rel(run_root, dirs["sensitivity"] / "candidate_excluded_metrics.tsv"),
            "candidate_excluded_summary": _rel(run_root, dirs["sensitivity"] / "candidate_excluded_summary.tsv"),
        }

    # ---------------- config snapshot + manifests ----------------
    _write_config_snapshot(cfg, dirs, run_root, section_outputs)

    finished_at = datetime.now(TOKYO)
    common_manifest = {
        "run_id": run_root.name,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_sec": (finished_at - started_at).total_seconds(),
        "timezone": "Asia/Tokyo",
        "git_or_source": "caliper-td",
        "primary_analysis": cfg.primary_analysis,
        "selected_baselines": selected_baselines,
        "methods": selected_methods,
        "data": dict(cfg.data),
        "global_random_seed": cfg.shared_seed,
        "k_star": int(k_star),
        "caliper_ks": list(caliper_k_list),
        "k_selection": k_selection_payload,
        "rwr_restart_probability": cfg.rwr_restart_probability,
        "ncs_feature_space_size": len(continuous_kd.feature_genes),
        "ncs_reference_profile_unit": "target_cell_averaged",
        "ncs_reference_n_profiles": int(continuous_kd.matrix.shape[0]),
        "ncs_reference_n_cells": int(len(continuous_kd.rows_by_cell())),
        "ncs_profile_unit_deviation": (
            "Reference is the target-cell averaged BING_knockdown_signature, not "
            "per-sig_id Level 5 signatures (methodology §6.2.1 / validation #4). "
            "Score is a target-cell-averaged adapted NCS, not official Touchstone "
            "NCS nor sig_id-level adapted NCS. pert_type==trt_sh.cgs holds by "
            "construction (validation #14)."
        ),
        "baseline_disease_signature_source": cfg.baseline_disease_signature_source,
        "shared_network_rules": net_stats["rule"],
        "shared_network_stats": net_stats,
        "min_disease_seeds": cfg.min_disease_seeds,
        "proximity": {"n_random": cfg.proximity_n_random, "min_bin_size": cfg.proximity_min_bin_size},
        "target_reach_matched": {
            "horizon": cfg.target_reach_horizon,
            "fallback_scores": {
                "NCS": cfg.target_reach_ncs_fallback_score,
                "proximity": cfg.target_reach_proximity_fallback_score,
                "RWR": cfg.target_reach_rwr_fallback_score,
            },
            "rwr_target_only_seed_score": cfg.target_reach_rwr_target_only_seed_score,
            "applies_to_primary_analysis": False,
        },
        **(
            {
                "dada": {
                    **dada_mod.metadata_block(restart_probability=cfg.rwr_restart_probability),
                    "network_mean_degree": float(dada_network["network_mean_degree"]),
                    "network_degree_sum": float(dada_network["degree_sum"]),
                    "selected_component_n_diseases": dada_mod.selected_component_counts(dada_strategy_df),
                    "task_adaptation": (
                        "SIGNOR has no PPI confidence, so every collapsed edge of H has weight 1; "
                        "disease seeds are uniformly weighted; the restart probability is shared with "
                        "Classical RWR (external_baseline.rwr.restart_probability); and candidate "
                        "targets remain in the disease seed set. This is a val2 task adaptation, not a "
                        "reproduction of the paper's OMIM/PPI experiments."
                    ),
                }
            }
            if use_dada
            else {}
        ),
        "tie_policy": cfg.tie_policy,
        "K_eff_cap": cfg.K_eff_cap,
        "K_list": list(cfg.K_list),
        "caliper_ks": list(caliper_k_list),
        "lift_eval_min_expected_hits": cfg.lift_eval_min_expected_hits,
        "inputs": input_manifest,
        "si_resolution": {
            "SI-01_rwr_restart_probability": cfg.rwr_restart_probability,
            "SI-02_k_selection_json": str(cfg.k_selection_json),
            "SI-03_continuous_kd_profile": str(cfg.continuous_kd_parquet),
            "SI-04_val2_label_parquet": str(cfg.val2_label_parquet),
            "SI-05_shared_seed": cfg.shared_seed,
        },
    }
    run_summary = {
        "run_root": str(run_root),
        "run_id": run_root.name,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "primary_analysis": cfg.primary_analysis,
        "selected_baselines": selected_baselines,
        "methods": selected_methods,
        "k_star": int(k_star),
        "caliper_ks": list(caliper_k_list),
        "rwr_restart_probability": cfg.rwr_restart_probability,
        "flow": flow,
        "method_summary": method_summary.to_dict(orient="records"),
        "lift_eval": {
            "min_expected_hits": float(cfg.lift_eval_min_expected_hits),
            "K_list": list(k_list),
            "n_diseases_common": int(method_summary["n_diseases"].iloc[0]) if len(method_summary) else 0,
            "n_diseases_eval_by_K": {
                str(k): int(method_summary[f"n_diseases_lift_eval_{k}"].iloc[0])
                if len(method_summary) and f"n_diseases_lift_eval_{k}" in method_summary.columns
                else 0
                for k in k_list
            },
            "n_diseases_eval": int(method_summary.get("n_diseases_lift_eval", pd.Series([0])).iloc[0])
            if len(method_summary)
            else 0,
        },
        "n_valid_diseases_cheap_gate": len(valid_diseases),
        "sections": cfg.sections,
    }
    output_manifest = metadata_mod.build_output_manifest(section_outputs, run_root=run_root)
    metadata_mod.write_run_metadata(common_manifest, output_manifest, run_summary, run_root=run_root)

    # ---------------- figures + report ----------------
    if cfg.sections.get("make_figures", True):
        print("[FIGURES]")
        try:
            from .viz import make_all_figures

            fig_outputs = make_all_figures(run_root)
            section_outputs["figures"] = fig_outputs
        except Exception as e:  # pragma: no cover
            print(f"[FIGURES] skipped due to error: {e}")
    if cfg.sections.get("make_report", True):
        print("[REPORT]")
        try:
            from .report import write_report

            report_path = write_report(run_root)
            section_outputs["report"] = {"report": _rel(run_root, report_path)}
        except Exception as e:  # pragma: no cover
            print(f"[REPORT] skipped due to error: {e}")

    output_manifest = metadata_mod.build_output_manifest(section_outputs, run_root=run_root)
    metadata_mod.write_run_metadata(common_manifest, output_manifest, run_summary, run_root=run_root)

    print("[DONE]")
    return run_summary


def _write_config_snapshot(
    cfg: ResolvedBaselineConfig,
    dirs: Dict[str, Path],
    run_root: Path,
    section_outputs: Dict[str, Dict[str, str]],
) -> None:
    import shutil

    config_copy = dirs["config"] / "val2_baseline.yaml"
    try:
        shutil.copy2(cfg.config_path, config_copy)
    except Exception:
        config_copy.write_text("# Original config could not be copied.\n", encoding="utf-8")
    resolved = {
        "config_path": str(cfg.config_path),
        "registry_path": str(cfg.registry_path),
        "seed": cfg.seed,
        "data": cfg.data,
        "primary_analysis": cfg.primary_analysis,
        "baselines": list(cfg.baselines),
        "k_selection_json": str(cfg.k_selection_json),
        "manual_k_star": cfg.manual_k_star,
        "caliper_ks": list(cfg.caliper_ks) if cfg.caliper_ks is not None else None,
        "continuous_kd_parquet": str(cfg.continuous_kd_parquet),
        "val2_label_parquet": str(cfg.val2_label_parquet),
        "baseline_disease_signature_source": cfg.baseline_disease_signature_source,
        "creeds_disease_parquet": str(cfg.creeds_disease_parquet) if cfg.creeds_disease_parquet else None,
        "rwr_restart_probability": cfg.rwr_restart_probability,
        "proximity_n_random": cfg.proximity_n_random,
        "proximity_min_bin_size": cfg.proximity_min_bin_size,
        "shared_seed": cfg.shared_seed,
        "min_disease_seeds": cfg.min_disease_seeds,
        "target_reach_horizon": cfg.target_reach_horizon,
        "target_reach_ncs_fallback_score": cfg.target_reach_ncs_fallback_score,
        "target_reach_proximity_fallback_score": cfg.target_reach_proximity_fallback_score,
        "target_reach_rwr_fallback_score": cfg.target_reach_rwr_fallback_score,
        "target_reach_rwr_target_only_seed_score": cfg.target_reach_rwr_target_only_seed_score,
        "step4_n_jobs": cfg.step4_n_jobs,
        "mp_start_method": cfg.mp_start_method,
        "K_eff_cap": cfg.K_eff_cap,
        "K_list": list(cfg.K_list),
        "tie_policy": cfg.tie_policy,
        "lift_eval_min_expected_hits": cfg.lift_eval_min_expected_hits,
        "sections": cfg.sections,
        "stats": cfg.stats,
    }
    write_json(resolved, dirs["config"] / "resolved_config.json")
    section_outputs["config"] = {
        "config": _rel(run_root, config_copy),
        "resolved_config": _rel(run_root, dirs["config"] / "resolved_config.json"),
    }


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run val2 external-baseline comparison.")
    parser.add_argument("--config", default=None, help="val2_baseline config YAML/JSON.")
    parser.add_argument("--run-root", default=None, help="Override output run directory.")
    parser.add_argument(
        "--steps",
        action="append",
        default=None,
        help="Output-only steps for an existing run: stats,figures,report.",
    )
    return parser.parse_args(argv)


def _parse_output_steps(raw_steps: List[str] | None) -> set[str]:
    if not raw_steps:
        return set()
    steps = {
        token.strip().lower()
        for raw in raw_steps
        for token in str(raw).split(",")
        if token.strip()
    }
    unknown = steps - OUTPUT_ONLY_STEPS
    if unknown:
        raise ValueError(
            f"Invalid output-only step(s): {sorted(unknown)}. "
            f"Valid steps: {sorted(OUTPUT_ONLY_STEPS)}"
        )
    return steps


def _rebuild_stats(run_root: Path) -> Dict[str, str]:
    """Recompute the significance table of an existing run from its paired deltas.

    Scores and per-disease metrics are untouched: only ``evaluation/ttest_fdr.tsv``
    is rewritten, from ``evaluation/paired_deltas.tsv`` plus the run's own config
    snapshot and metadata. This is how a finished run adopts a change to the
    statistical procedure without re-scoring.
    """
    eval_dir = run_root / "evaluation"
    deltas_path = eval_dir / "paired_deltas.tsv"
    if not deltas_path.is_file():
        raise FileNotFoundError(deltas_path)
    deltas_df = pd.read_csv(deltas_path, sep="\t")

    resolved_path = run_root / "config" / "resolved_config.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8")) if resolved_path.is_file() else {}
    meta_path = run_root / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    stats_cfg = resolved.get("stats") or {}
    k_list = [int(k) for k in (meta.get("K_list") or resolved.get("K_list") or [10])]
    metric_list = list(stats_cfg.get("metrics") or evaluate_mod.default_stats_metrics(k_list))
    alpha = float(stats_cfg.get("alpha", 0.05))
    baselines = [str(b) for b in (meta.get("selected_baselines") or resolved.get("baselines") or evaluate_mod.BASELINES)]
    k_star = int(meta.get("k_star") or resolved.get("manual_k_star") or caliper_mod.PRIMARY_K)

    ttest_df = _compute_primary_ttest(
        deltas_df,
        baselines=baselines,
        metrics=metric_list,
        k_list=k_list,
        k_star=k_star,
        alpha=alpha,
    )
    ttest_path = _write_tsv(ttest_df, eval_dir / "ttest_fdr.tsv")
    print(f"[stats] rewrote {_rel(run_root, ttest_path)} ({len(ttest_df)} rows)")
    return {"ttest_fdr": _rel(run_root, ttest_path)}


def _run_output_steps(run_root: str | Path, steps: set[str]) -> Dict[str, Any]:
    run_root = Path(run_root)
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)

    manifest_path = run_root / "output_manifest.json"
    existing = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    sections = existing.get("sections", {}) if isinstance(existing, dict) else {}
    if not isinstance(sections, dict):
        raise ValueError(f"Invalid manifest sections in {manifest_path}")

    result: Dict[str, Any] = {
        "run_root": str(run_root),
        "steps": sorted(steps),
    }
    if "stats" in steps:
        stats_outputs = _rebuild_stats(run_root)
        evaluation = sections.get("evaluation")
        if isinstance(evaluation, dict):
            # ttest_fdr.tsv is rewritten in place; wilcoxon_signed_rank.tsv is no
            # longer produced, so drop its stale manifest entry.
            evaluation.pop("wilcoxon_signed_rank", None)
            evaluation.update(stats_outputs)
        else:
            sections["evaluation"] = stats_outputs
        result["stats"] = stats_outputs
    if "figures" in steps:
        from .viz import make_all_figures

        figure_outputs = make_all_figures(run_root)
        sections["figures"] = figure_outputs
        result["figures"] = figure_outputs
    if "report" in steps:
        from .report import write_report

        report_path = write_report(run_root)
        sections["report"] = {"report": _rel(run_root, report_path)}
        result["report"] = str(report_path)

    refreshed = metadata_mod.build_output_manifest(sections, run_root=run_root)
    write_json(refreshed, manifest_path)
    return result


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    args = parse_args(argv)
    output_steps = _parse_output_steps(args.steps)
    if output_steps:
        if args.run_root is None:
            raise ValueError("--run-root is required for stats/figures/report-only steps.")
        return _run_output_steps(args.run_root, output_steps)
    if args.config is None:
        args.config = str(Path(__file__).resolve().parents[2] / "config" / "val2_baseline.yaml")
    cfg = load_baseline_config(args.config)
    return run(cfg, run_root_override=args.run_root)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False, default=str))
