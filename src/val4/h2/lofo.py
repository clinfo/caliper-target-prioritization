from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from val4.h2.parallel import parallel_map_completed
from val4.h2.scoring import score_val2_proposed_pairs
from val4.ranking import evaluate_disease_metrics
from val4.stats import bh_fdr, bootstrap_median_ci, empirical_p_greater_equal


def build_lofo_graph(
    G: nx.MultiDiGraph,
    family: str,
    edge_id_to_families: dict[int, tuple[str, ...]],
) -> tuple[nx.MultiDiGraph, int]:
    H = G.copy()
    to_remove: list[tuple[str, str, Any]] = []
    for u, v, key, attrs in H.edges(keys=True, data=True):
        edge_id = int(attrs["edge_id"])
        if family in edge_id_to_families.get(edge_id, tuple()):
            to_remove.append((u, v, key))
    H.remove_edges_from(to_remove)
    return H, len(to_remove)


def build_random_edge_drop_graph(
    G: nx.MultiDiGraph,
    n_remove: int,
    seed: int,
) -> tuple[nx.MultiDiGraph, int]:
    H = G.copy()
    edges = list(H.edges(keys=True))
    if len(edges) == 0 or int(n_remove) <= 0:
        return H, 0
    rng = np.random.default_rng(int(seed))
    n = min(int(n_remove), len(edges))
    idx = rng.choice(len(edges), size=n, replace=False)
    to_remove = [edges[int(i)] for i in idx]
    H.remove_edges_from(to_remove)
    return H, len(to_remove)


def _support_loss_fraction(full_scores: pd.DataFrame, alt_scores: pd.DataFrame) -> float:
    if full_scores is None or alt_scores is None or len(full_scores) == 0 or len(alt_scores) == 0:
        return np.nan
    key_cols = ["disease_id", "target_gene", "cell_id"]
    full_support = full_scores.groupby(key_cols)["w_sum"].max().rename("w_sum_full")
    alt_support = alt_scores.groupby(key_cols)["w_sum"].max().rename("w_sum_alt")
    merged = pd.concat([full_support, alt_support], axis=1, join="inner")
    if len(merged) == 0:
        return np.nan
    loss = (merged["w_sum_full"] > 0) & ~(merged["w_sum_alt"] > 0)
    return float(loss.mean())


def _merge_lofo_metrics(
    *,
    family: str,
    full_metrics: pd.DataFrame,
    alt_metrics: pd.DataFrame,
    removed_edges: int,
    condition_name: str,
    null_rep: int | None = None,
) -> pd.DataFrame:
    merged = full_metrics.merge(alt_metrics, on="disease_id", suffixes=("_full", "_alt"), how="inner")
    if len(merged) == 0:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "family": family,
            "disease_id": merged["disease_id"],
            "condition": condition_name,
            "null_rep": np.nan if null_rep is None else int(null_rep),
            "removed_edges": int(removed_edges),
            "AP_full": merged["AP_full"],
            "AP_alt": merged["AP_alt"],
            "delta_AP": merged["AP_full"] - merged["AP_alt"],
            "Lift10_full": merged["Lift@10_full"],
            "Lift10_alt": merged["Lift@10_alt"],
            "delta_Lift10": merged["Lift@10_full"] - merged["Lift@10_alt"],
            "n_pos": merged["n_pos_full"],
            "n_targets": merged["n_targets_full"],
        }
    )


def _diseasewise_adjusted_values(
    obs: pd.DataFrame,
    null: pd.DataFrame,
    *,
    obs_col: str,
    null_col: str | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    if len(obs) == 0 or len(null) == 0:
        return np.array([], dtype=float), np.nan, np.array([], dtype=float)
    null_col = str(obs_col if null_col is None else null_col)
    null_mean_name = f"{obs_col}_null_mean"
    null_mean = null.groupby("disease_id")[null_col].mean().rename(null_mean_name)
    merged = obs[["disease_id", obs_col]].merge(null_mean.reset_index(), on="disease_id", how="inner")
    if len(merged) == 0:
        return np.array([], dtype=float), np.nan, np.array([], dtype=float)
    adjusted = merged[obs_col].to_numpy(dtype=float) - merged[null_mean_name].to_numpy(dtype=float)
    null_mean_median = float(np.nanmedian(merged[null_mean_name].to_numpy(dtype=float)))
    return adjusted, null_mean_median, merged[null_mean_name].to_numpy(dtype=float)


def _summarize_lofo(
    disease_metrics: pd.DataFrame,
    null_metrics: pd.DataFrame,
    support_qc: pd.DataFrame,
    family_order: list[str],
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, family in enumerate(family_order):
        qc = support_qc[support_qc["family"] == family]
        qc_row = qc.iloc[0].to_dict() if len(qc) else {}
        is_degenerate = bool(qc_row.get("lofo_degenerate_flag", False))
        obs = disease_metrics[disease_metrics["family"] == family]
        null = null_metrics[null_metrics["family"] == family]
        delta_ap = obs["delta_AP"].to_numpy(dtype=float) if len(obs) else np.array([], dtype=float)
        obs_med = float(np.nanmedian(delta_ap)) if delta_ap.size else np.nan
        delta_lift = obs["delta_Lift10"].to_numpy(dtype=float) if len(obs) else np.array([], dtype=float)
        lift_med = float(np.nanmedian(delta_lift)) if delta_lift.size else np.nan
        null_by_rep = (
            null.groupby("null_rep")["delta_AP"].median().to_numpy(dtype=float)
            if len(null)
            else np.array([], dtype=float)
        )
        ci_low, ci_high = bootstrap_median_ci(delta_ap, int(bootstrap_reps), int(seed) + i)
        lift_ci_low, lift_ci_high = bootstrap_median_ci(delta_lift, int(bootstrap_reps), int(seed) + 20000 + i)
        adj_ap, null_ap_median, _ = _diseasewise_adjusted_values(obs, null, obs_col="delta_AP")
        adj_lift, null_lift_median, _ = _diseasewise_adjusted_values(obs, null, obs_col="delta_Lift10")
        adj_ci_low, adj_ci_high = bootstrap_median_ci(adj_ap, int(bootstrap_reps), int(seed) + 40000 + i)
        adj_lift_ci_low, adj_lift_ci_high = bootstrap_median_ci(adj_lift, int(bootstrap_reps), int(seed) + 50000 + i)
        adj_ap_med = float(np.nanmedian(adj_ap)) if adj_ap.size else np.nan
        adj_lift_med = float(np.nanmedian(adj_lift)) if adj_lift.size else np.nan
        p_ap = empirical_p_greater_equal(obs_med, null_by_rep)
        if is_degenerate:
            obs_med = ci_low = ci_high = null_ap_median = adj_ap_med = adj_ci_low = adj_ci_high = p_ap = np.nan
            lift_med = lift_ci_low = lift_ci_high = null_lift_median = adj_lift_med = adj_lift_ci_low = adj_lift_ci_high = np.nan
        rows.append(
            {
                "family": family,
                "n_evaluable_diseases": int(len(obs)) if not is_degenerate else 0,
                "removed_edges": int(qc_row.get("removed_edges", 0)),
                "lofo_no_support_flag": bool(qc_row.get("lofo_no_support_flag", False)),
                "lofo_degenerate_flag": is_degenerate,
                "lofo_score_support_loss_row_fraction": qc_row.get("lofo_score_support_loss_row_fraction", np.nan),
                "delta_AP_median": obs_med,
                "delta_AP_ci_low": ci_low,
                "delta_AP_ci_high": ci_high,
                "delta_AP_null_mean_median": null_ap_median,
                "delta_AP_null_adjusted_median": adj_ap_med,
                "delta_AP_null_adjusted_ci_low": adj_ci_low,
                "delta_AP_null_adjusted_ci_high": adj_ci_high,
                "p_empirical_AP": p_ap,
                "delta_Lift10_median": lift_med,
                "delta_Lift10_ci_low": lift_ci_low,
                "delta_Lift10_ci_high": lift_ci_high,
                "delta_Lift10_null_mean_median": null_lift_median,
                "delta_Lift10_null_adjusted_median": adj_lift_med,
                "delta_Lift10_null_adjusted_ci_low": adj_lift_ci_low,
                "delta_Lift10_null_adjusted_ci_high": adj_lift_ci_high,
            }
        )
    summary = pd.DataFrame(rows)
    if len(summary):
        summary["q_empirical_AP_BH"] = bh_fdr(summary["p_empirical_AP"])
    return summary


_LOFO_WORKER_CTX: dict[str, Any] | None = None


def _init_lofo_worker(ctx: dict[str, Any]) -> None:
    global _LOFO_WORKER_CTX
    _LOFO_WORKER_CTX = ctx


def _lofo_observed_worker(task: dict[str, Any]) -> dict[str, Any]:
    if _LOFO_WORKER_CTX is None:
        raise RuntimeError("LOFO worker context is not initialized.")
    ctx = _LOFO_WORKER_CTX
    family = str(task["family"])
    f_idx = int(task["f_idx"])
    total_edges = int(ctx["total_edges"])
    lofo_graph, removed_edges = build_lofo_graph(ctx["G"], family, ctx["edge_id_to_families"])
    lofo_no_support = int(removed_edges) == 0
    lofo_degenerate = int(removed_edges) > 0 and int(removed_edges) >= total_edges
    support_loss = np.nan
    disease_df = pd.DataFrame()

    if lofo_no_support:
        alt_scores = ctx["full_pair_scores"].copy()
        alt_metrics = ctx["full_metrics"].copy()
        support_loss = 0.0
        disease_df = _merge_lofo_metrics(
            family=family,
            full_metrics=ctx["full_metrics"],
            alt_metrics=alt_metrics,
            removed_edges=removed_edges,
            condition_name="lofo",
        )
    elif not lofo_degenerate:
        alt_scores = score_val2_proposed_pairs(
            lofo_graph,
            ctx["labels_df"],
            ctx["pert_sigs"],
            ctx["dis_sigs"],
            int(ctx["k_star"]),
            int(ctx["start_sign"]),
        )
        support_loss = _support_loss_fraction(ctx["full_pair_scores"], alt_scores)
        alt_metrics = evaluate_disease_metrics(alt_scores)
        disease_df = _merge_lofo_metrics(
            family=family,
            full_metrics=ctx["full_metrics"],
            alt_metrics=alt_metrics,
            removed_edges=removed_edges,
            condition_name="lofo",
        )

    qc_row = {
        "family": family,
        "removed_edges": int(removed_edges),
        "total_edges": int(total_edges),
        "lofo_no_support_flag": bool(lofo_no_support),
        "lofo_degenerate_flag": bool(lofo_degenerate),
        "lofo_score_support_loss_row_fraction": float(support_loss) if np.isfinite(support_loss) else np.nan,
    }
    return {"f_idx": f_idx, "family": family, "disease_metrics": disease_df, "support_qc": qc_row}


def _lofo_null_worker(task: dict[str, Any]) -> dict[str, Any]:
    if _LOFO_WORKER_CTX is None:
        raise RuntimeError("LOFO worker context is not initialized.")
    ctx = _LOFO_WORKER_CTX
    family = str(task["family"])
    f_idx = int(task["f_idx"])
    rep = int(task["rep"])
    removed_edges = int(task["removed_edges"])
    seed_rep = int(task["seed"])
    rand_graph, rand_removed = build_random_edge_drop_graph(ctx["G"], removed_edges, seed_rep)
    if rand_removed == 0:
        rand_metrics = ctx["full_metrics"].copy()
    else:
        rand_scores = score_val2_proposed_pairs(
            rand_graph,
            ctx["labels_df"],
            ctx["pert_sigs"],
            ctx["dis_sigs"],
            int(ctx["k_star"]),
            int(ctx["start_sign"]),
        )
        rand_metrics = evaluate_disease_metrics(rand_scores)
    out = _merge_lofo_metrics(
        family=family,
        full_metrics=ctx["full_metrics"],
        alt_metrics=rand_metrics,
        removed_edges=rand_removed,
        condition_name="random_physical_edge_drop",
        null_rep=rep,
    )
    return {"f_idx": f_idx, "family": family, "rep": rep, "null_metrics": out}


def run_lofo_analysis(
    *,
    G: nx.MultiDiGraph,
    labels_df: pd.DataFrame,
    pert_sigs: dict,
    dis_sigs: dict,
    k_star: int,
    start_sign: int,
    edge_id_to_families: dict[int, tuple[str, ...]],
    family_order: list[str],
    n_null: int,
    bootstrap_reps: int,
    seed: int,
    full_pair_scores: pd.DataFrame | None = None,
    progress: bool = True,
    n_jobs: int = 1,
    parallel_start_method: str = "fork",
    parallel_chunksize: int = 1,
    progress_recorder=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_scores = (
        score_val2_proposed_pairs(G, labels_df, pert_sigs, dis_sigs, int(k_star), int(start_sign))
        if full_pair_scores is None
        else full_pair_scores
    )
    full_metrics = evaluate_disease_metrics(full_scores)
    ctx = {
        "G": G,
        "labels_df": labels_df,
        "pert_sigs": pert_sigs,
        "dis_sigs": dis_sigs,
        "k_star": int(k_star),
        "start_sign": int(start_sign),
        "edge_id_to_families": edge_id_to_families,
        "full_pair_scores": full_scores,
        "full_metrics": full_metrics,
        "total_edges": int(G.number_of_edges()),
    }
    observed_tasks = []
    for f_idx, family in enumerate(family_order):
        removed_edges = sum(
            1
            for _, _, _, attrs in G.edges(keys=True, data=True)
            if str(family) in edge_id_to_families.get(int(attrs["edge_id"]), tuple())
        )
        observed_tasks.append(
            {
                "f_idx": int(f_idx),
                "family": str(family),
                "estimated_cost": int(removed_edges),
            }
        )
    observed_tasks_for_run = sorted(observed_tasks, key=lambda x: int(x.get("estimated_cost", 0)), reverse=True)

    observed_section = "lofo.observed"
    if progress_recorder is not None:
        progress_recorder.start(observed_section, total=len(observed_tasks_for_run), unit="family", desc="H2 LOFO families")
    observed_results = parallel_map_completed(
        _lofo_observed_worker,
        observed_tasks_for_run,
        n_jobs=int(n_jobs),
        start_method=str(parallel_start_method),
        chunksize=int(parallel_chunksize),
        initializer=_init_lofo_worker,
        initargs=(ctx,),
        progress=progress,
        desc="H2 LOFO families",
        unit="family",
        on_result=None if progress_recorder is None else progress_recorder.callback(observed_section),
        on_error=None if progress_recorder is None else progress_recorder.error_callback(observed_section),
    )
    if progress_recorder is not None:
        progress_recorder.complete(observed_section)
    observed_results = sorted(observed_results, key=lambda x: int(x["f_idx"]))
    disease_rows = [x["disease_metrics"] for x in observed_results]
    support_qc_rows = [x["support_qc"] for x in observed_results]

    null_tasks: list[dict[str, Any]] = []
    for result in observed_results:
        qc = result["support_qc"]
        if bool(qc["lofo_degenerate_flag"]):
            continue
        f_idx = int(result["f_idx"])
        family = str(result["family"])
        removed_edges = int(qc["removed_edges"])
        for rep in range(int(n_null)):
            null_tasks.append(
                {
                    "f_idx": f_idx,
                    "family": family,
                    "removed_edges": removed_edges,
                    "rep": int(rep),
                    "seed": int(seed) + 300000 * (f_idx + 1) + int(rep),
                }
            )

    null_tasks_for_run = sorted(null_tasks, key=lambda x: int(x.get("removed_edges", 0)), reverse=True)
    null_section = "lofo.null"
    if progress_recorder is not None:
        progress_recorder.start(null_section, total=len(null_tasks_for_run), unit="condition", desc="H2 LOFO nulls")
    null_results = parallel_map_completed(
        _lofo_null_worker,
        null_tasks_for_run,
        n_jobs=int(n_jobs),
        start_method=str(parallel_start_method),
        chunksize=int(parallel_chunksize),
        initializer=_init_lofo_worker,
        initargs=(ctx,),
        progress=progress,
        desc="H2 LOFO nulls",
        unit="condition",
        on_result=None if progress_recorder is None else progress_recorder.callback(null_section),
        on_error=None if progress_recorder is None else progress_recorder.error_callback(null_section),
    )
    if progress_recorder is not None:
        progress_recorder.complete(null_section)
    null_results = sorted(null_results, key=lambda x: (int(x["f_idx"]), int(x["rep"])))
    null_rows = [x["null_metrics"] for x in null_results]

    disease_metrics = pd.concat([x for x in disease_rows if x is not None and len(x)], ignore_index=True) if disease_rows else pd.DataFrame()
    null_metrics = pd.concat([x for x in null_rows if x is not None and len(x)], ignore_index=True) if null_rows else pd.DataFrame()
    support_qc = pd.DataFrame(support_qc_rows)
    summary = _summarize_lofo(
        disease_metrics=disease_metrics,
        null_metrics=null_metrics,
        support_qc=support_qc,
        family_order=family_order,
        bootstrap_reps=int(bootstrap_reps),
        seed=int(seed),
    )
    return disease_metrics, null_metrics, summary
