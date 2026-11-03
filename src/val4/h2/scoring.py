from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd

from inference.signature_reversal_nodes import (
    _score_proposed,
    khop_nodes,
    predicted_sign_sets_from_votes,
    propagate_sign_votes_shortest_paths,
)


def _build_score_cache(
    G: nx.MultiDiGraph,
    pert_sigs: dict[tuple[str, str], dict[str, int]],
    dis_sigs: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Build the per-condition cache used by the Val2 proposed scorer."""
    return {
        "kg_nodes": set(G.nodes()),
        "dis_keys": {str(d): set(sig.keys()) for d, sig in dis_sigs.items()},
        "pert_keys": {tuple(k): set(sig.keys()) for k, sig in pert_sigs.items()},
        "nodes_k": {},
        "votes": {},
        "pred": {},
    }


def _empty_score_row(
    *,
    disease_id: str,
    target_gene: str,
    cell_id: str,
    pair_label: int,
    n_meas: int = 0,
) -> dict[str, Any]:
    return {
        "disease_id": str(disease_id),
        "target_gene": str(target_gene),
        "cell_id": str(cell_id),
        "score": 0.0,
        "w_sum": 0.0,
        "w_rev": 0.0,
        "n_meas": int(n_meas),
        "zero_denominator": True,
        "pair_label": int(pair_label),
    }


def _score_one_val2_proposed_pair(
    *,
    G: nx.MultiDiGraph,
    cache: dict[str, Any],
    pert_sigs: dict[tuple[str, str], dict[str, int]],
    dis_sigs: dict[str, dict[str, int]],
    disease_id: str,
    target_gene: str,
    cell_id: str,
    pair_label: int,
    k_star: int,
    start_sign: int,
) -> dict[str, Any]:
    disease_id = str(disease_id)
    target_gene = str(target_gene)
    cell_id = str(cell_id)
    pert = pert_sigs.get((target_gene, cell_id), {})
    dis = dis_sigs.get(disease_id, {})
    if len(dis) == 0:
        return _empty_score_row(
            disease_id=disease_id,
            target_gene=target_gene,
            cell_id=cell_id,
            pair_label=pair_label,
        )

    dis_keys = cache.get("dis_keys", {}).get(disease_id, set(dis.keys()))
    pert_keys = cache.get("pert_keys", {}).get((target_gene, cell_id), set(pert.keys()))

    if target_gene in G:
        nk_store = cache.setdefault("nodes_k", {})
        nk_key = (target_gene, int(k_star))
        nodes_k = nk_store.get(nk_key)
        if nodes_k is None:
            nodes_k = khop_nodes(G, target_gene, int(k_star))
            nk_store[nk_key] = nodes_k
        U = set(nodes_k) & dis_keys & pert_keys
    else:
        U = set()

    if target_gene not in G:
        return _empty_score_row(
            disease_id=disease_id,
            target_gene=target_gene,
            cell_id=cell_id,
            pair_label=pair_label,
            n_meas=len(U),
        )

    cache_key = (target_gene, int(k_star), int(start_sign))
    votes_cache = cache.setdefault("votes", {})
    pred_cache = cache.setdefault("pred", {})
    votes = votes_cache.get(cache_key)
    if votes is None:
        votes = propagate_sign_votes_shortest_paths(
            G,
            target_gene,
            int(k_star),
            start_sign=int(start_sign),
            sign_attr="polarity",
            max_neighbors=None,
            seed=0,
        )
        votes_cache[cache_key] = votes
        pred_cache[cache_key] = predicted_sign_sets_from_votes(votes, eps=0.0)

    score, w_rev, n_meas, _covK, _covAlign, w_sum = _score_proposed(U, votes, pert, dis)
    return {
        "disease_id": disease_id,
        "target_gene": target_gene,
        "cell_id": cell_id,
        "score": float(score),
        "w_sum": float(w_sum),
        "w_rev": float(w_rev),
        "n_meas": int(n_meas),
        "zero_denominator": bool(float(w_sum) <= 0.0),
        "pair_label": int(pair_label),
    }


def score_val2_proposed_pairs(
    G: nx.MultiDiGraph,
    labels_df: pd.DataFrame,
    pert_sigs: dict[tuple[str, str], dict[str, int]],
    dis_sigs: dict[str, dict[str, int]],
    k_star: int,
    start_sign: int,
) -> pd.DataFrame:
    """Score all label-defined rows with the Val2 proposed scoring core.

    This is the only H2 pair-scoring entry point. It delegates shortest-path
    vote propagation and the weighted reversal score to
    ``src/inference/signature_reversal_nodes.py`` and preserves the input
    ``labels_df`` universe by returning one score row per input row. Rows that
    lose score support remain present with ``score=0``.
    """
    if not isinstance(G, nx.MultiDiGraph):
        raise TypeError("score_val2_proposed_pairs: G must be nx.MultiDiGraph.")
    if int(start_sign) not in {-1, 1}:
        raise ValueError(f"score_val2_proposed_pairs: start_sign must be -1 or +1. got={start_sign!r}")
    required = {"disease_id", "target_gene", "cell_id", "label"}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"labels_df is missing required columns: {sorted(missing)}")

    cache = _build_score_cache(G, pert_sigs, dis_sigs)
    rows: list[dict[str, Any]] = []
    for row in labels_df[["disease_id", "target_gene", "cell_id", "label"]].itertuples(index=False):
        rows.append(
            _score_one_val2_proposed_pair(
                G=G,
                cache=cache,
                pert_sigs=pert_sigs,
                dis_sigs=dis_sigs,
                disease_id=str(row.disease_id),
                target_gene=str(row.target_gene),
                cell_id=str(row.cell_id),
                pair_label=int(row.label),
                k_star=int(k_star),
                start_sign=int(start_sign),
            )
        )
    out = pd.DataFrame(rows)
    if len(out) != len(labels_df):
        raise RuntimeError(
            f"score_val2_proposed_pairs must preserve the label universe. got {len(out)} rows, expected {len(labels_df)}."
        )
    return out
