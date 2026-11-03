"""DADA baseline: centrality-adjusted RWR + seed-degree uniform prioritization.

Implements the final recommended method of Erten et al. 2011 (degree-aware
disease-gene prioritization) as a val2 baseline on the shared undirected graph
H. The implementation details are summarized in the public method overview.

The raw association component is the existing Classical RWR steady-state
vector (``rwr.rwr_steady_state``); DADA never re-runs the power iteration. The
centrality component is the r=0 stationary distribution, which on a
unit-weighted undirected connected graph is exactly degree-proportional, so it
is computed in closed form. Per disease, the ranking component is chosen by the
paper's seed-degree uniform rule (raw when the disease seed set is on average
higher-degree than the graph, centrality-adjusted otherwise), and the final
evaluation score is the negated within-disease descending average rank.

Task adaptations relative to the paper's OMIM/PPI experiments (recorded in the
run metadata and report): SIGNOR has no PPI confidence, so every collapsed edge
has weight 1; disease seeds are uniformly weighted; the restart probability is
shared with Classical RWR; and candidate targets are not removed from the
disease seed set.

Labels are never consulted here (methodology §3.2).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd


RAW_COMPONENT = "raw_rwr"
ADJUSTED_COMPONENT = "centrality_adjusted"

# Pair-level invalid reasons (spec §3.3), checked in this order.
REASON_TARGET_NOT_IN_H = "target_not_in_H"
REASON_NO_DISEASE_RWR_VECTOR = "no_disease_rwr_vector"
REASON_NONPOSITIVE_LOG_RATIO_INPUT = "nonpositive_log_ratio_input"
REASON_NONFINITE_COMPONENT = "nonfinite_dada_component"
REASON_INVALID_DISEASE_STRATEGY = "invalid_disease_strategy"

# Disease-level invalid reason (spec §3.4).
REASON_EMPTY_SEED_SET_IN_H = "empty_seed_set_in_H"

PAIR_COLUMNS = [
    "disease_id",
    "target_gene",
    "raw_rwr_score",
    "centrality_score",
    "adjusted_log_ratio",
    "selected_component",
    "selected_component_score",
    "target_degree_H",
    "seed_target_overlap",
    "valid",
    "reason",
]
DISEASE_STRATEGY_COLUMNS = [
    "disease_id",
    "n_seed_in_H",
    "mean_seed_degree",
    "network_mean_degree",
    "selected_component",
    "restart_probability",
    "valid",
    "reason",
]
COMMON_RANKING_COLUMNS = [
    "disease_id",
    "target_gene",
    "selected_component",
    "selected_component_score",
    "dada_rank",
    "dada_score",
]

METADATA_BLOCK = {
    "variant": "centrality_adjustment_seed_degree_uniform",
    "source": "Erten et al. (2011); see docs/method_overview.md",
    "network": "native_shared_network",
    "edge_weight": "unit",
    "seed_weighting": "uniform",
    "candidate_seed_policy": "include",
    "centrality": "r0_stationary_degree_proportional",
    "adjustment": "natural_log_ratio",
    "degree_threshold": "network_mean_degree",
    "uniform_rule": "raw_if_mean_seed_degree_gt_network_mean_else_adjusted",
    "evaluation_score": "negative_descending_average_rank",
}


def build_network_context(H: nx.Graph) -> Dict[str, Any]:
    """Unit-weighted degree W(v), centrality and the graph mean degree lambda.

    ``centrality`` is the r=0 association score of the paper's Equation 3,
    which for a unit-weighted undirected connected graph equals the
    degree-proportional stationary distribution. ``network_mean_degree`` is
    always taken over every node of H (never over a candidate or seed subset).
    """
    n_nodes = int(H.number_of_nodes())
    if n_nodes == 0:
        raise ValueError("[DADA] shared network H has no nodes.")
    degree = {str(v): float(d) for v, d in H.degree()}
    degree_sum = float(sum(degree.values()))
    if not (degree_sum > 0.0) or not math.isfinite(degree_sum):
        raise ValueError(f"[DADA] total weighted degree of H must be positive and finite. got {degree_sum!r}")
    centrality = {v: d / degree_sum for v, d in degree.items()}
    network_mean_degree = degree_sum / float(n_nodes)
    return {
        "degree": degree,
        "centrality": centrality,
        "degree_sum": degree_sum,
        "network_mean_degree": float(network_mean_degree),
        "n_nodes": n_nodes,
        "n_edges": int(H.number_of_edges()),
    }


def node_centrality_table(network_context: Dict[str, Any]) -> pd.DataFrame:
    degree: Dict[str, float] = network_context["degree"]
    centrality: Dict[str, float] = network_context["centrality"]
    rows = [
        {"gene_id": v, "degree_H": float(degree[v]), "centrality": float(centrality[v])}
        for v in sorted(degree)
    ]
    return pd.DataFrame(rows, columns=["gene_id", "degree_H", "centrality"])


def select_component(mean_seed_degree: float, network_mean_degree: float) -> str:
    """Seed-degree uniform rule (spec §3.4); the boundary goes to adjusted."""
    return RAW_COMPONENT if float(mean_seed_degree) > float(network_mean_degree) else ADJUSTED_COMPONENT


def build_disease_strategy(
    *,
    disease_ids: Iterable[str],
    disease_info: Dict[str, Dict[str, Any]],
    network_context: Dict[str, Any],
    restart_probability: float,
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """Per-disease mean seed degree and selected ranking component.

    Diseases with an empty ``S_d ∩ V(H)`` have no defined mean seed degree; they
    are recorded as invalid with an empty ``selected_component`` and no fallback.
    """
    degree: Dict[str, float] = network_context["degree"]
    network_mean_degree = float(network_context["network_mean_degree"])

    strategy: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    for d in sorted(map(str, disease_ids)):
        seeds = [str(s) for s in (disease_info.get(d, {}) or {}).get("seeds", [])]
        seeds_in_H = [s for s in seeds if s in degree]
        n_seed = len(seeds_in_H)
        if n_seed == 0:
            entry = {
                "n_seed_in_H": 0,
                "mean_seed_degree": float("nan"),
                "selected_component": "",
                "valid": False,
                "reason": REASON_EMPTY_SEED_SET_IN_H,
                "seed_set": set(),
            }
        else:
            mean_seed_degree = float(sum(degree[s] for s in seeds_in_H)) / float(n_seed)
            entry = {
                "n_seed_in_H": n_seed,
                "mean_seed_degree": mean_seed_degree,
                "selected_component": select_component(mean_seed_degree, network_mean_degree),
                "valid": True,
                "reason": "",
                "seed_set": set(seeds_in_H),
            }
        strategy[d] = entry
        rows.append(
            {
                "disease_id": d,
                "n_seed_in_H": int(entry["n_seed_in_H"]),
                "mean_seed_degree": float(entry["mean_seed_degree"]),
                "network_mean_degree": network_mean_degree,
                "selected_component": str(entry["selected_component"]),
                "restart_probability": float(restart_probability),
                "valid": bool(entry["valid"]),
                "reason": str(entry["reason"]),
            }
        )
    strategy_df = pd.DataFrame(rows, columns=DISEASE_STRATEGY_COLUMNS)
    return strategy, strategy_df


def build_pair_components(
    *,
    pairs: Iterable[Tuple[str, str]],
    disease_strategy: Dict[str, Dict[str, Any]],
    network_context: Dict[str, Any],
    rwr_disease_vec: Dict[str, np.ndarray],
    node_index: Dict[str, int],
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], bool], Dict[Tuple[str, str], Dict[str, Any]]]:
    """Pair-level raw / centrality / adjusted components and their validity.

    ``rwr_disease_vec`` is the Classical RWR steady-state vector per disease
    (reused verbatim, never recomputed). Returns the diagnostics table, the
    ``(disease, target) -> valid`` mapping consumed by the common-set builder,
    and the per-pair selected component score used for the final ranking.
    """
    degree: Dict[str, float] = network_context["degree"]
    centrality: Dict[str, float] = network_context["centrality"]

    rows: List[Dict[str, Any]] = []
    valid_map: Dict[Tuple[str, str], bool] = {}
    selected: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for d, t in sorted({(str(d), str(t)) for d, t in pairs}):
        strat = disease_strategy.get(d, {})
        seed_set = strat.get("seed_set", set())
        selected_component = str(strat.get("selected_component", ""))
        seed_target_overlap = bool(t in seed_set)

        raw_score = float("nan")
        centrality_score = float(centrality[t]) if t in centrality else float("nan")
        adjusted = float("nan")
        reason = ""

        p = rwr_disease_vec.get(d)
        if t not in degree:
            reason = REASON_TARGET_NOT_IN_H
        elif p is None or t not in node_index:
            reason = REASON_NO_DISEASE_RWR_VECTOR
        else:
            raw_score = float(p[node_index[t]])
            if not (raw_score > 0.0) or not (centrality_score > 0.0):
                reason = REASON_NONPOSITIVE_LOG_RATIO_INPUT
            else:
                adjusted = float(np.log(raw_score / centrality_score))
                if not (
                    math.isfinite(raw_score)
                    and math.isfinite(centrality_score)
                    and math.isfinite(adjusted)
                ):
                    reason = REASON_NONFINITE_COMPONENT
        if not reason and not bool(strat.get("valid", False)):
            reason = REASON_INVALID_DISEASE_STRATEGY

        valid = not reason
        if valid:
            component_score = raw_score if selected_component == RAW_COMPONENT else adjusted
        else:
            component_score = float("nan")

        rows.append(
            {
                "disease_id": d,
                "target_gene": t,
                "raw_rwr_score": raw_score,
                "centrality_score": centrality_score,
                "adjusted_log_ratio": adjusted,
                "selected_component": selected_component,
                "selected_component_score": component_score,
                "target_degree_H": float(degree.get(t, float("nan"))),
                "seed_target_overlap": seed_target_overlap,
                "valid": bool(valid),
                "reason": reason,
            }
        )
        valid_map[(d, t)] = bool(valid)
        if valid:
            selected[(d, t)] = {
                "selected_component": selected_component,
                "selected_component_score": float(component_score),
            }

    pair_df = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    return pair_df, valid_map, selected


def build_common_ranking(
    *,
    common_pairs: Sequence[Tuple[str, str]],
    selected_components: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], float]]:
    """Within-disease descending average rank over the common candidate set.

    Ties share an average rank (consistent with the repository's tie-aware
    evaluation) and the evaluation score is ``-rank``, so the top candidate
    scores -1 and the score decreases with rank. The ranking population is the
    fixed common candidate set shared by every selected method; labels are not
    used.
    """
    by_disease: Dict[str, List[Tuple[str, float, str]]] = {}
    for d, t in common_pairs:
        d, t = str(d), str(t)
        entry = selected_components.get((d, t))
        if entry is None:
            raise ValueError(
                f"[DADA] common pair {(d, t)} has no valid selected component score "
                "(the common-set builder must have excluded it)."
            )
        score = float(entry["selected_component_score"])
        if not math.isfinite(score):
            raise ValueError(f"[DADA] common pair {(d, t)} has a non-finite selected component score.")
        by_disease.setdefault(d, []).append((t, score, str(entry["selected_component"])))

    rows: List[Dict[str, Any]] = []
    score_map: Dict[Tuple[str, str], float] = {}
    for d in sorted(by_disease):
        items = by_disease[d]
        scores = pd.Series([s for _t, s, _c in items], dtype="float64")
        ranks = scores.rank(ascending=False, method="average").tolist()
        for (t, score, component), rank in zip(items, ranks):
            dada_score = -float(rank)
            rows.append(
                {
                    "disease_id": d,
                    "target_gene": t,
                    "selected_component": component,
                    "selected_component_score": float(score),
                    "dada_rank": float(rank),
                    "dada_score": dada_score,
                }
            )
            score_map[(d, t)] = dada_score

    ranking_df = (
        pd.DataFrame(rows, columns=COMMON_RANKING_COLUMNS)
        .sort_values(["disease_id", "target_gene"])
        .reset_index(drop=True)
    )
    return ranking_df, score_map


def metadata_block(*, restart_probability: float) -> Dict[str, Any]:
    block = dict(METADATA_BLOCK)
    block["restart_probability"] = float(restart_probability)
    return block


def selected_component_counts(strategy_df: pd.DataFrame) -> Dict[str, int]:
    """Disease counts per selected component (report / diagnostics)."""
    if strategy_df.empty:
        return {RAW_COMPONENT: 0, ADJUSTED_COMPONENT: 0, "invalid": 0}
    valid = strategy_df[strategy_df["valid"].astype(bool)]
    counts = valid["selected_component"].astype(str).value_counts().to_dict()
    return {
        RAW_COMPONENT: int(counts.get(RAW_COMPONENT, 0)),
        ADJUSTED_COMPONENT: int(counts.get(ADJUSTED_COMPONENT, 0)),
        "invalid": int(len(strategy_df) - len(valid)),
    }
