from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from val4.h2.parallel import parallel_map_completed
from val4.h2.scoring import score_val2_proposed_pairs
from val4.ranking import evaluate_disease_metrics
from val4.stats import bh_fdr, bootstrap_median_ci


def _family_edge_refs(
    G: nx.MultiDiGraph,
    family: str,
    edge_id_to_families: dict[int, tuple[str, ...]],
) -> list[tuple[str, str, Any]]:
    refs: list[tuple[str, str, Any]] = []
    for u, v, key, attrs in G.edges(keys=True, data=True):
        edge_id = int(attrs["edge_id"])
        if family in edge_id_to_families.get(edge_id, tuple()):
            refs.append((u, v, key))
    return refs


def _edge_signs(G: nx.MultiDiGraph, refs: list[tuple[str, str, Any]], *, sign_attr: str = "polarity") -> list[int]:
    signs: list[int] = []
    for u, v, key in refs:
        sign = int(G.edges[u, v, key][sign_attr])
        if sign not in {-1, 1}:
            raise ValueError(f"Edge polarity must be -1 or +1. got={sign!r} on {u!r}->{v!r}.")
        signs.append(sign)
    return signs


def build_family_sign_shuffle_graph(
    G: nx.MultiDiGraph,
    edge_refs: list[tuple[str, str, Any]],
    seed: int,
    *,
    sign_attr: str = "polarity",
) -> tuple[nx.MultiDiGraph, bool]:
    H = G.copy()
    original = np.asarray(_edge_signs(G, edge_refs, sign_attr=sign_attr), dtype=np.int64)
    if original.size == 0:
        return H, True
    rng = np.random.default_rng(int(seed))
    shuffled = rng.permutation(original)
    no_op = bool(np.array_equal(original, shuffled))
    for (u, v, key), sign in zip(edge_refs, shuffled.tolist(), strict=True):
        H.edges[u, v, key][sign_attr] = int(sign)
    return H, no_op


def _merge_sign_shuffle_metrics(
    *,
    family: str,
    full_metrics: pd.DataFrame,
    shuffled_metrics: pd.DataFrame,
    null_rep: int,
    shuffled_edges: int,
    no_op_iteration: bool,
) -> pd.DataFrame:
    merged = full_metrics.merge(shuffled_metrics, on="disease_id", suffixes=("_full", "_shuffled"), how="inner")
    if len(merged) == 0:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "family": family,
            "null_rep": int(null_rep),
            "disease_id": merged["disease_id"],
            "shuffled_edges": int(shuffled_edges),
            "no_op_iteration": bool(no_op_iteration),
            "AP_full": merged["AP_full"],
            "AP_sign": merged["AP_shuffled"],
            "delta_AP": merged["AP_full"] - merged["AP_shuffled"],
            "Lift10_full": merged["Lift@10_full"],
            "Lift10_sign": merged["Lift@10_shuffled"],
            "delta_Lift10": merged["Lift@10_full"] - merged["Lift@10_shuffled"],
            "n_pos": merged["n_pos_full"],
            "n_targets": merged["n_targets_full"],
        }
    )


def _empirical_positive_delta_p(rep_medians: np.ndarray) -> float:
    vals = np.asarray(rep_medians, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float((1 + np.sum(vals <= 0.0)) / (1 + vals.size))


def _summarize_sign_shuffle(
    disease_metrics: pd.DataFrame,
    qc_df: pd.DataFrame,
    family_order: list[str],
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, family in enumerate(family_order):
        fam_rows = disease_metrics[disease_metrics["family"] == family] if disease_metrics is not None and len(disease_metrics) and "family" in disease_metrics.columns else pd.DataFrame()
        qc_rows = qc_df[qc_df["family"] == family]
        qc = qc_rows.iloc[0].to_dict() if len(qc_rows) else {}
        is_degenerate = bool(qc.get("sign_shuffle_family_level_degenerate", False))
        if is_degenerate or len(fam_rows) == 0:
            rows.append(
                {
                    "family": family,
                    "n_evaluable_diseases": 0,
                    "sign_shuffle_eligible_edge_count": int(qc.get("sign_shuffle_eligible_edge_count", 0)),
                    "sign_shuffle_family_level_degenerate": bool(is_degenerate),
                    "sign_shuffle_no_op_iteration_fraction": qc.get("sign_shuffle_no_op_iteration_fraction", np.nan),
                    "delta_AP_median": np.nan,
                    "delta_AP_ci_low": np.nan,
                    "delta_AP_ci_high": np.nan,
                    "p_empirical_AP": np.nan,
                    "delta_Lift10_median": np.nan,
                    "delta_Lift10_ci_low": np.nan,
                    "delta_Lift10_ci_high": np.nan,
                    "p_empirical_Lift10": np.nan,
                }
            )
            continue
        per_disease = fam_rows.groupby("disease_id", as_index=False).agg(
            delta_AP=("delta_AP", "mean"),
            delta_Lift10=("delta_Lift10", "mean"),
        )
        delta_ap = per_disease["delta_AP"].to_numpy(dtype=float)
        delta_lift = per_disease["delta_Lift10"].to_numpy(dtype=float)
        rep_ap = fam_rows.groupby("null_rep")["delta_AP"].median().to_numpy(dtype=float)
        rep_lift = fam_rows.groupby("null_rep")["delta_Lift10"].median().to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_median_ci(delta_ap, int(bootstrap_reps), int(seed) + i)
        lift_ci_low, lift_ci_high = bootstrap_median_ci(delta_lift, int(bootstrap_reps), int(seed) + 30000 + i)
        rows.append(
            {
                "family": family,
                "n_evaluable_diseases": int(len(per_disease)),
                "sign_shuffle_eligible_edge_count": int(qc.get("sign_shuffle_eligible_edge_count", 0)),
                "sign_shuffle_family_level_degenerate": bool(is_degenerate),
                "sign_shuffle_no_op_iteration_fraction": qc.get("sign_shuffle_no_op_iteration_fraction", np.nan),
                "delta_AP_median": float(np.nanmedian(delta_ap)) if delta_ap.size else np.nan,
                "delta_AP_ci_low": ci_low,
                "delta_AP_ci_high": ci_high,
                "p_empirical_AP": _empirical_positive_delta_p(rep_ap),
                "delta_Lift10_median": float(np.nanmedian(delta_lift)) if delta_lift.size else np.nan,
                "delta_Lift10_ci_low": lift_ci_low,
                "delta_Lift10_ci_high": lift_ci_high,
                "p_empirical_Lift10": _empirical_positive_delta_p(rep_lift),
            }
        )
    summary = pd.DataFrame(rows)
    if len(summary):
        summary["q_empirical_AP_BH"] = bh_fdr(summary["p_empirical_AP"])
    return summary


_SIGN_WORKER_CTX: dict[str, Any] | None = None


def _init_sign_worker(ctx: dict[str, Any]) -> None:
    global _SIGN_WORKER_CTX
    _SIGN_WORKER_CTX = ctx


def _sign_shuffle_null_worker(task: dict[str, Any]) -> dict[str, Any]:
    if _SIGN_WORKER_CTX is None:
        raise RuntimeError("sign-shuffle worker context is not initialized.")
    ctx = _SIGN_WORKER_CTX
    family = str(task["family"])
    f_idx = int(task["f_idx"])
    rep = int(task["rep"])
    edge_refs = _family_edge_refs(ctx["G"], family, ctx["edge_id_to_families"])
    seed_rep = int(task["seed"])

    shuffled_graph, no_op = build_family_sign_shuffle_graph(ctx["G"], edge_refs, seed_rep)
    if bool(no_op):
        shuffled_metrics = ctx["full_metrics"].copy()
    else:
        shuffled_scores = score_val2_proposed_pairs(
            shuffled_graph,
            ctx["labels_df"],
            ctx["pert_sigs"],
            ctx["dis_sigs"],
            int(ctx["k_star"]),
            int(ctx["start_sign"]),
        )
        shuffled_metrics = evaluate_disease_metrics(shuffled_scores)
    out = _merge_sign_shuffle_metrics(
        family=family,
        full_metrics=ctx["full_metrics"],
        shuffled_metrics=shuffled_metrics,
        null_rep=rep,
        shuffled_edges=int(len(edge_refs)),
        no_op_iteration=bool(no_op),
    )
    return {"f_idx": f_idx, "family": family, "rep": rep, "no_op": bool(no_op), "disease_metrics": out}


def run_sign_shuffle_analysis(
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        "full_metrics": full_metrics,
        "edge_id_to_families": edge_id_to_families,
    }

    qc_rows: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for f_idx, family in enumerate(family_order):
        edge_refs = _family_edge_refs(G, family, edge_id_to_families)
        signs = _edge_signs(G, edge_refs) if edge_refs else []
        is_degenerate = len(edge_refs) < 2 or len(set(signs)) < 2
        qc_rows.append(
            {
                "family": str(family),
                "sign_shuffle_eligible_edge_count": int(len(edge_refs)),
                "sign_shuffle_family_level_degenerate": bool(is_degenerate),
                "sign_shuffle_no_op_iteration_fraction": np.nan,
            }
        )
        if is_degenerate:
            continue
        for rep in range(int(n_null)):
            tasks.append(
                {
                    "f_idx": int(f_idx),
                    "family": str(family),
                    "estimated_cost": int(len(edge_refs)),
                    "rep": int(rep),
                    "seed": int(seed) + 500000 * (f_idx + 1) + int(rep),
                }
            )

    tasks_for_run = sorted(tasks, key=lambda x: int(x.get("estimated_cost", 0)), reverse=True)
    sign_section = "sign_shuffle.null"
    if progress_recorder is not None:
        progress_recorder.start(sign_section, total=len(tasks_for_run), unit="condition", desc="H2 sign-shuffle nulls")
    results = parallel_map_completed(
        _sign_shuffle_null_worker,
        tasks_for_run,
        n_jobs=int(n_jobs),
        start_method=str(parallel_start_method),
        chunksize=int(parallel_chunksize),
        initializer=_init_sign_worker,
        initargs=(ctx,),
        progress=progress,
        desc="H2 sign-shuffle nulls",
        unit="condition",
        on_result=None if progress_recorder is None else progress_recorder.callback(sign_section),
        on_error=None if progress_recorder is None else progress_recorder.error_callback(sign_section),
    )
    if progress_recorder is not None:
        progress_recorder.complete(sign_section)
    results = sorted(results, key=lambda x: (int(x["f_idx"]), int(x["rep"])))

    no_op_by_family: dict[str, list[bool]] = {str(f): [] for f in family_order}
    disease_rows: list[pd.DataFrame] = []
    for result in results:
        no_op_by_family[str(result["family"])].append(bool(result["no_op"]))
        disease_rows.append(result["disease_metrics"])

    for row in qc_rows:
        flags = no_op_by_family.get(str(row["family"]), [])
        row["sign_shuffle_no_op_iteration_fraction"] = float(np.mean(flags)) if flags else np.nan

    disease_metrics = pd.concat([x for x in disease_rows if x is not None and len(x)], ignore_index=True) if disease_rows else pd.DataFrame()
    qc_df = pd.DataFrame(qc_rows)
    summary = _summarize_sign_shuffle(
        disease_metrics=disease_metrics,
        qc_df=qc_df,
        family_order=family_order,
        bootstrap_reps=int(bootstrap_reps),
        seed=int(seed),
    )
    return disease_metrics, summary
