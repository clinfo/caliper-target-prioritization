"""Per-target worker for the directed target-reach matched STEP2+STEP4 pipeline.

Reach construction (methodology §4/§7 STEP2) and label-free scoring (STEP4:
CALIPER / horizon-masked NCS / local proximity / local RWR) are independent
per candidate target ``t`` -- CALIPER's k-hop neighborhood + sign votes, the
NCS local feature universe, and the proximity/RWR local graph and its
distance/transition matrices are all keyed by ``t`` alone. ``process_target``
bundles all of this into a single per-target unit of work so it can be run
in parallel (see :mod:`val2_baseline.parallel`).

Large read-only inputs are stored in the module-level ``_STATE`` dict by
:func:`configure_worker_state`, called once in the parent process before the
pool starts (and, for the default ``fork`` start method, inherited by worker
processes via copy-on-write memory with no pickling).
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import numpy as np

from . import caliper as caliper_mod
from . import proximity as proximity_mod
from . import reach as reach_mod
from . import rwr as rwr_mod
from . import wtcs as wtcs_mod

_STATE: Dict[str, Any] = {}


def configure_worker_state(state: Dict[str, Any]) -> None:
    """Replace the module-level worker state (called once per worker process).

    Under ``fork`` this simply re-affirms the state the worker already
    inherited from the parent at fork time; under ``spawn``/``forkserver`` it
    is what actually populates the freshly imported module in the child.
    """
    _STATE.clear()
    _STATE.update(state)


def build_worker_state(
    *,
    graph,
    adjacency: Dict[str, Set[str]],
    h: int,
    start_sign: int,
    pert_sigs: Dict[Tuple[str, str], Dict[str, int]],
    dis_sigs: Dict[str, Dict[str, int]],
    disease_info: Dict[str, Dict[str, Any]],
    pair_cells_by_target: Dict[str, Dict[str, List[str]]],
    use_ncs: bool,
    use_proximity: bool,
    use_rwr: bool,
    continuous_cells: Set[Tuple[str, str]],
    continuous_kd_feature_index: Dict[str, int],
    continuous_kd_matrix: np.ndarray,
    continuous_kd_key_to_row: Dict[Tuple[str, str], int],
    cell_to_rows: Dict[str, List[int]],
    gene_id_int: np.ndarray,
    ncs_fallback_score: float,
    proximity_n_random: int,
    proximity_min_bin_size: int,
    proximity_fallback_score: float,
    shared_seed: int,
    rwr_restart_probability: float,
    rwr_fallback_score: float,
    rwr_target_only_seed_score: float,
) -> Dict[str, Any]:
    """Assemble the state dict passed to :func:`configure_worker_state`."""
    return {
        "graph": graph,
        "adjacency": adjacency,
        "h": int(h),
        "start_sign": int(start_sign),
        "pert_sigs": pert_sigs,
        "dis_sigs": dis_sigs,
        "disease_info": disease_info,
        "pair_cells_by_target": pair_cells_by_target,
        "use_ncs": bool(use_ncs),
        "use_proximity": bool(use_proximity),
        "use_rwr": bool(use_rwr),
        "continuous_cells": continuous_cells,
        "continuous_kd_feature_index": continuous_kd_feature_index,
        "continuous_kd_matrix": continuous_kd_matrix,
        "continuous_kd_key_to_row": continuous_kd_key_to_row,
        "cell_to_rows": cell_to_rows,
        "gene_id_int": gene_id_int,
        "ncs_fallback_score": float(ncs_fallback_score),
        "proximity_n_random": int(proximity_n_random),
        "proximity_min_bin_size": int(proximity_min_bin_size),
        "proximity_fallback_score": float(proximity_fallback_score),
        "shared_seed": int(shared_seed),
        "rwr_restart_probability": float(rwr_restart_probability),
        "rwr_fallback_score": float(rwr_fallback_score),
        "rwr_target_only_seed_score": float(rwr_target_only_seed_score),
    }


def _empty_result(t: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "target": t,
        "reach_row": summary,
        "caliper_rows": [],
        "ncs_cell_rows": [],
        "ncs_norm_rows": [],
        "ncs_fallback_counts": {},
        "proximity_rows": [],
        "rwr_rows": [],
    }


def process_target(t: str) -> Dict[str, Any]:
    """Compute reach + all label-free scores for one candidate target ``t``."""
    st = _STATE
    graph = st["graph"]
    h = st["h"]

    reach_nodes = reach_mod.directed_reach_nodes(graph, t, h)
    anchors = reach_mod.anchor_nodes(t, reach_nodes)
    H_local = reach_mod.build_local_undirected_graph_from_adjacency(st["adjacency"], anchors)
    summary = reach_mod.summarize_target_reach(
        t, h, reach_nodes, H_local, target_in_graph=(t in graph)
    )

    dc_map = st["pair_cells_by_target"].get(t)
    if not dc_map:
        return _empty_result(t, summary)

    result = _empty_result(t, summary)
    use_ncs = st["use_ncs"]
    use_proximity = st["use_proximity"]
    use_rwr = st["use_rwr"]
    continuous_cells = st["continuous_cells"]
    disease_info = st["disease_info"]

    # ---- CALIPER cell-level scores (methodology §6.1) ----
    caliper_triples = [
        (d, t, c)
        for d, cells in dc_map.items()
        for c in cells
        if (not use_ncs) or (t, c) in continuous_cells
    ]
    if caliper_triples:
        caliper_df = caliper_mod.compute_caliper_cell_scores(
            graph=graph,
            pert_sigs=st["pert_sigs"],
            dis_sigs=st["dis_sigs"],
            cell_triples=caliper_triples,
            k_star=h,
            start_sign=st["start_sign"],
            progress=None,
        )
        result["caliper_rows"] = caliper_df.to_dict("records")

    # ---- horizon-masked NCS cell-level scores (methodology §6.2) ----
    if use_ncs:
        cell_to_diseases: Dict[str, Set[str]] = {}
        for d, cells in dc_map.items():
            for c in cells:
                if (t, c) in continuous_cells:
                    cell_to_diseases.setdefault(c, set()).add(d)

        if cell_to_diseases:
            feature_index = st["continuous_kd_feature_index"]
            gene_id_int = st["gene_id_int"]
            matrix = st["continuous_kd_matrix"]
            key_to_row = st["continuous_kd_key_to_row"]
            cell_to_rows = st["cell_to_rows"]

            local_genes = sorted(
                (g for g in reach_nodes if g in feature_index),
                key=lambda g: feature_index[g],
            )
            local_cols = np.array([feature_index[g] for g in local_genes], dtype=np.int64)
            local_gene_id_int = gene_id_int[local_cols] if len(local_cols) else np.array([], dtype=np.int64)
            local_index = {g: i for i, g in enumerate(local_genes)}
            n_F_h = int(len(local_cols))

            for c in sorted(cell_to_diseases.keys()):
                rows = cell_to_rows.get(c, [])
                if not rows:
                    continue
                rows_arr = np.asarray(rows, dtype=np.int64)
                local_of_global = {int(grow): i for i, grow in enumerate(rows)}
                grow = key_to_row.get((t, c))
                if grow is None or grow not in local_of_global:
                    continue
                li = local_of_global[grow]

                local_ref_matrix = None
                local_ranks = None
                if n_F_h:
                    local_ref_matrix = matrix[np.ix_(rows_arr, local_cols)]
                    local_ranks = wtcs_mod.ranks_for_matrix(local_ref_matrix, local_gene_id_int)

                for d in sorted(cell_to_diseases[c]):
                    di = disease_info[d]
                    q_up_h = sorted(set(di["q_up"]) & set(local_index))
                    q_down_h = sorted(set(di["q_down"]) & set(local_index))
                    up_pos = np.array([local_index[g] for g in q_up_h], dtype=np.int64)
                    down_pos = np.array([local_index[g] for g in q_down_h], dtype=np.int64)
                    reason = ""
                    fallback = False
                    wtcs_v = np.nan
                    ncs_v = 0.0
                    res: Dict[str, Any] = {
                        "n_ref_total": len(rows),
                        "n_ref_defined": 0,
                        "n_pos": 0,
                        "n_neg": 0,
                        "n_zero": 0,
                        "mu_pos": np.nan,
                        "mu_neg": np.nan,
                        "valid": False,
                    }
                    if n_F_h == 0:
                        reason = "empty_local_feature_universe"
                        fallback = True
                    elif len(up_pos) == 0:
                        reason = "empty_local_up_query"
                        fallback = True
                    elif len(down_pos) == 0:
                        reason = "empty_local_down_query"
                        fallback = True
                    else:
                        assert local_ref_matrix is not None and local_ranks is not None
                        res = wtcs_mod.compute_cell_ncs(local_ref_matrix, local_ranks, up_pos, down_pos)
                        wtcs_arr = res["wtcs"]
                        ncs_arr = res["ncs"]
                        wtcs_v = float(wtcs_arr[li]) if np.isfinite(wtcs_arr[li]) else np.nan
                        raw_ncs = float(ncs_arr[li]) if np.isfinite(ncs_arr[li]) else np.nan
                        if bool(res["valid"]) and np.isfinite(raw_ncs):
                            ncs_v = raw_ncs
                        else:
                            reason = "norm_undefined" if not bool(res["valid"]) else "ncs_undefined"
                            fallback = True
                    if fallback:
                        result["ncs_fallback_counts"][d] = result["ncs_fallback_counts"].get(d, 0) + 1
                        ncs_v = float(-st["ncs_fallback_score"])
                    result["ncs_norm_rows"].append(
                        {
                            "disease_id": d,
                            "target_gene": t,
                            "cell_id": c,
                            "n_F_h": n_F_h,
                            "n_q_up_h": int(len(up_pos)),
                            "n_q_down_h": int(len(down_pos)),
                            "n_ref_total": int(res["n_ref_total"]),
                            "n_ref_defined": int(res["n_ref_defined"]),
                            "n_pos_wtcs": int(res["n_pos"]),
                            "n_neg_wtcs": int(res["n_neg"]),
                            "n_zero_wtcs": int(res["n_zero"]),
                            "mu_pos": float(res["mu_pos"]),
                            "mu_neg": float(res["mu_neg"]),
                            "valid": bool(res["valid"]),
                            "fallback": bool(fallback),
                            "reason": reason,
                        }
                    )
                    result["ncs_cell_rows"].append(
                        {
                            "disease_id": d,
                            "target_gene": t,
                            "cell_id": c,
                            "wtcs": wtcs_v,
                            "ncs": float(ncs_v),
                            "score": float(-ncs_v),
                            "valid": bool(not fallback),
                            "fallback": bool(fallback),
                            "reason": reason,
                            "n_F_h": n_F_h,
                            "n_q_up_h": int(len(up_pos)),
                            "n_q_down_h": int(len(down_pos)),
                        }
                    )

    # ---- local proximity / local RWR (methodology §6.3/§6.4) ----
    if use_proximity or use_rwr:
        prox_ctx = None
        rwr_ctx = None
        for d in sorted(dc_map.keys()):
            # t is kept as the query node (proximity distance target / RWR
            # restart-probability readout node) but excluded from the disease
            # seed set to avoid a self-seed effect dominating the score.
            seeds = set(disease_info[d]["seeds"]) - {t}
            has_local_seed = bool(seeds & anchors)
            if use_proximity:
                if prox_ctx is None and has_local_seed:
                    prox_ctx = proximity_mod.build_local_proximity_context(
                        H_local, st["proximity_min_bin_size"]
                    )
                prox = proximity_mod.compute_proximity_pair_local(
                    H_local=H_local,
                    target_gene=t,
                    disease_seeds=seeds,
                    n_random=st["proximity_n_random"],
                    min_bin_size=st["proximity_min_bin_size"],
                    shared_seed=st["shared_seed"],
                    disease_id=d,
                    fallback_score=st["proximity_fallback_score"],
                    context=prox_ctx,
                )
                result["proximity_rows"].append({"disease_id": d, "target_gene": t, **prox})
            if use_rwr:
                if rwr_ctx is None and has_local_seed:
                    rwr_ctx = rwr_mod.build_local_rwr_context(H_local)
                rwr = rwr_mod.compute_rwr_pair_local(
                    H_local=H_local,
                    target_gene=t,
                    disease_seeds=seeds,
                    restart_probability=st["rwr_restart_probability"],
                    fallback_score=st["rwr_fallback_score"],
                    target_only_seed_score=st["rwr_target_only_seed_score"],
                    context=rwr_ctx,
                )
                result["rwr_rows"].append({"disease_id": d, "target_gene": t, **rwr})

    return result
