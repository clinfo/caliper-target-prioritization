"""Candidate-excluded sensitivity analysis (methodology §9.1).

When a candidate target ``t`` is itself a network-mapped disease seed
(``t in S_d^H``), Network proximity gets distance 0 and RWR places restart mass
directly on ``t``. This self-seed membership can dominate the network baseline
score. This analysis recomputes proximity and RWR with the seed set

    S_{d,-t}^H = S_d^H \\ {t}

for the overlapping pairs, reuses the main score for non-overlapping pairs
(S_{d,-t}^H == S_d^H), and reports the change in AP / Lift@10 against the main
analysis plus seed-target overlap rates (overall / positive / negative). The
common universe D^common / C_d^common is NOT changed; CALIPER and NCS are not
recomputed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from . import proximity as proximity_mod
from . import rwr as rwr_mod
from .evaluate import (
    annotate_lift_eval_flags,
    build_lift_eval_cohorts,
    compute_metrics_by_disease,
    in_lift_eval_col,
    lift_col,
)

SENS_METHODS = ["proximity", "RWR"]


def run_candidate_excluded(
    *,
    common_candidates_df: pd.DataFrame,
    disease_info: Dict[str, Dict[str, Any]],
    proximity_main: Dict[Tuple[str, str], float],
    rwr_main: Dict[Tuple[str, str], float],
    dist_matrix: np.ndarray,
    node_index: Dict[str, int],
    matcher: proximity_mod.DegreeMatcher,
    W,
    n_nodes: int,
    n_random: int,
    rwr_restart_probability: float,
    shared_seed: int,
    K_eff_cap: int = 10,
    K_list: List[int] | None = None,
    tie_policy: str = "expected",
    lift_eval_min_expected_hits: float = 1.0,
    progress=None,
) -> Dict[str, pd.DataFrame]:
    """Recompute proximity/RWR with S_{d,-t}^H for self-seed pairs.

    Returns a dict with ``scores``, ``metrics`` and ``summary`` DataFrames.
    """
    rows: List[Dict[str, Any]] = []
    pairs = list(common_candidates_df.itertuples(index=False))
    iterator = progress(pairs, desc="sensitivity", total=len(pairs)) if progress else pairs
    for r in iterator:
        d = str(r.disease_id)
        t = str(r.target_gene)
        label = int(r.label)
        seeds = disease_info[d]["seeds"]
        seed_set = set(seeds)
        overlap = bool(t in seed_set)

        prox_main = float(proximity_main[(d, t)])
        rwr_main_score = float(rwr_main[(d, t)])

        if not overlap:
            prox_excl = prox_main
            rwr_excl = rwr_main_score
        else:
            seeds_minus_t = [s for s in seeds if s != t]
            # proximity recompute with a fresh per-pair degree-matched seed null.
            seed_null, seed_null_reason = proximity_mod.build_disease_seed_null(
                matcher=matcher,
                node_index=node_index,
                seeds=seeds_minus_t,
                n_random=int(n_random),
                derived_seed_value=proximity_mod.derive_seed(shared_seed, d, t + "__EXCL_SEEDS__"),
            )
            prox = proximity_mod.compute_proximity_pair(
                matrix=dist_matrix,
                node_index=node_index,
                matcher=matcher,
                target_gene=t,
                seeds=seeds_minus_t,
                seed_null=seed_null,
                seed_null_reason=seed_null_reason,
                n_random=int(n_random),
                target_seed_value=proximity_mod.derive_seed(shared_seed, d, t + "__EXCL__"),
            )
            prox_excl = float(prox["score"]) if prox["valid"] else float("nan")

            # RWR recompute with p0 over S_{d,-t}^H.
            seed_idx = [node_index[s] for s in seeds_minus_t if s in node_index]
            p, _n_iter, _l1 = rwr_mod.rwr_steady_state(
                W, n_nodes, seed_idx, float(rwr_restart_probability)
            )
            rwr_excl = float(p[node_index[t]]) if t in node_index else float("nan")

        rows.append(
            {
                "disease_id": d,
                "target_gene": t,
                "label": label,
                "seed_target_overlap": overlap,
                "proximity_main": prox_main,
                "proximity_excl": prox_excl,
                "rwr_main": rwr_main_score,
                "rwr_excl": rwr_excl,
            }
        )

    scores_df = pd.DataFrame(
        rows,
        columns=[
            "disease_id", "target_gene", "label", "seed_target_overlap",
            "proximity_main", "proximity_excl", "rwr_main", "rwr_excl",
        ],
    ).sort_values(["disease_id", "target_gene"]).reset_index(drop=True)

    k_list = list(K_list) if K_list else [int(K_eff_cap)]
    primary_k = k_list[0]
    primary_lift = lift_col(primary_k)
    primary_flag = in_lift_eval_col(primary_k)
    long_main = _to_long(scores_df, "main")
    long_excl = _to_long(scores_df, "excl")
    metrics_main = compute_metrics_by_disease(
        long_main, methods=SENS_METHODS, K_list=k_list, K_eff_cap=K_eff_cap, tie_policy=tie_policy
    )
    metrics_excl = compute_metrics_by_disease(
        long_excl, methods=SENS_METHODS, K_list=k_list, K_eff_cap=K_eff_cap, tie_policy=tie_policy
    )
    cohort_df = build_lift_eval_cohorts(
        metrics_main, k_list=k_list, min_expected_hits=lift_eval_min_expected_hits
    )
    metrics_main = annotate_lift_eval_flags(metrics_main, cohort_df)
    metrics_excl = annotate_lift_eval_flags(metrics_excl, cohort_df)

    m = metrics_main[
        ["disease_id", "method", "AP", primary_lift, "n_pos", "n_candidate", primary_flag]
    ].rename(columns={"AP": "AP_main", primary_lift: "Lift_main", primary_flag: "in_lift_eval"})
    e = metrics_excl[["disease_id", "method", "AP", primary_lift]].rename(
        columns={"AP": "AP_excl", primary_lift: "Lift_excl"}
    )
    metrics_df = m.merge(e, on=["disease_id", "method"], how="inner")
    metrics_df["delta_AP"] = metrics_df["AP_excl"] - metrics_df["AP_main"]
    metrics_df["delta_Lift@10"] = metrics_df["Lift_excl"] - metrics_df["Lift_main"]
    metrics_df = metrics_df[
        [
            "disease_id", "method", "AP_main", "AP_excl", "delta_AP",
            "Lift_main", "Lift_excl", "delta_Lift@10", "n_pos", "n_candidate",
        ]
    ].sort_values(["method", "disease_id"]).reset_index(drop=True)

    # ---- method-level summary + overlap rates ----
    overlap_overall = float(scores_df["seed_target_overlap"].mean()) if len(scores_df) else float("nan")
    pos = scores_df[scores_df["label"] == 1]
    neg = scores_df[scores_df["label"] == 0]
    overlap_pos = float(pos["seed_target_overlap"].mean()) if len(pos) else float("nan")
    overlap_neg = float(neg["seed_target_overlap"].mean()) if len(neg) else float("nan")

    summary_rows: List[Dict[str, Any]] = []
    for method in SENS_METHODS:
        sub = metrics_df[metrics_df["method"] == method]
        sub_eval = sub[sub["in_lift_eval"].astype(bool)] if "in_lift_eval" in sub.columns else sub.iloc[0:0]
        summary_rows.append(
            {
                "method": method,
                "n_diseases": int(len(sub)),
                "mAP_main": float(sub["AP_main"].mean()) if len(sub) else float("nan"),
                "mAP_excl": float(sub["AP_excl"].mean()) if len(sub) else float("nan"),
                "mean_delta_AP": float(sub["delta_AP"].mean()) if len(sub) else float("nan"),
                "mean_Lift_main": float(sub["Lift_main"].mean()) if len(sub) else float("nan"),
                "mean_Lift_excl": float(sub["Lift_excl"].mean()) if len(sub) else float("nan"),
                "mean_delta_Lift@10": float(sub["delta_Lift@10"].mean()) if len(sub) else float("nan"),
                "n_diseases_lift_eval": int(len(sub_eval)),
                "mean_Lift_main_eval": float(sub_eval["Lift_main"].mean()) if len(sub_eval) else float("nan"),
                "mean_Lift_excl_eval": float(sub_eval["Lift_excl"].mean()) if len(sub_eval) else float("nan"),
                "mean_delta_Lift@10_eval": float(sub_eval["delta_Lift@10"].mean()) if len(sub_eval) else float("nan"),
                "overlap_rate_overall": overlap_overall,
                "overlap_rate_positive": overlap_pos,
                "overlap_rate_negative": overlap_neg,
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    return {"scores": scores_df, "metrics": metrics_df, "summary": summary_df}


def _to_long(scores_df: pd.DataFrame, which: str) -> pd.DataFrame:
    """Build a 4-column long table (disease/target/method/score+label) for one
    analysis variant ('main' or 'excl') and the two network baselines."""
    col = {"proximity": f"proximity_{which}", "RWR": f"rwr_{which}"}
    rows: List[Dict[str, Any]] = []
    for r in scores_df.itertuples(index=False):
        for method in SENS_METHODS:
            rows.append(
                {
                    "disease_id": r.disease_id,
                    "target_gene": r.target_gene,
                    "method": method,
                    "score": float(getattr(r, col[method])),
                    "label": int(r.label),
                }
            )
    return pd.DataFrame(rows, columns=["disease_id", "target_gene", "method", "score", "label"])
