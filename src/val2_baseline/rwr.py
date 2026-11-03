"""Classical Random Walk with Restart baseline (methodology §6.4).

Column-normalized adjacency of the shared graph H, equal-probability disease
seeds, power iteration to L1 convergence. Candidate score is the steady-state
probability at the target node.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

try:  # scipy is optional but strongly preferred for large H
    import scipy.sparse as sp

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


L1_TOL = 1e-6
PROB_SUM_TOL = 1e-10
MAX_ITER = 100000


def build_transition_matrix(H: nx.Graph, node_index: Dict[str, int]):
    """Column-normalized adjacency W (each column sums to 1)."""
    n = len(node_index)
    rows: List[int] = []
    cols: List[int] = []
    for u, v in H.edges():
        iu, iv = node_index[u], node_index[v]
        rows.append(iu)
        cols.append(iv)
        rows.append(iv)
        cols.append(iu)  # undirected: symmetric adjacency
    data = np.ones(len(rows), dtype=np.float64)

    if _HAVE_SCIPY:
        A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        col_sums = np.asarray(A.sum(axis=0)).ravel()
        col_sums[col_sums == 0] = 1.0
        d_inv = sp.diags(1.0 / col_sums)
        W = (A @ d_inv).tocsr()
        return W
    A = np.zeros((n, n), dtype=np.float64)
    for r_, c_ in zip(rows, cols):
        A[r_, c_] = 1.0
    col_sums = A.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    return A / col_sums[np.newaxis, :]


def rwr_steady_state(
    W,
    n: int,
    seed_indices: List[int],
    restart_probability: float,
) -> Tuple[np.ndarray, int, float]:
    """Run RWR to convergence. Returns (p, n_iter, final_l1)."""
    if len(seed_indices) == 0:
        raise ValueError("[RWR] empty seed set.")
    p0 = np.zeros(n, dtype=np.float64)
    p0[np.asarray(seed_indices, dtype=np.int64)] = 1.0 / len(seed_indices)
    r = float(restart_probability)
    p = p0.copy()
    final_l1 = float("inf")
    n_iter = 0
    for n_iter in range(1, MAX_ITER + 1):
        p_next = (1.0 - r) * (W @ p) + r * p0
        final_l1 = float(np.abs(p_next - p).sum())
        p = p_next
        if final_l1 < L1_TOL:
            break
    else:
        raise RuntimeError(
            f"[RWR] did not converge within max_iter={MAX_ITER} (final L1={final_l1:.3e})."
        )

    prob_sum = float(p.sum())
    if abs(prob_sum - 1.0) >= PROB_SUM_TOL:
        raise ValueError(f"[RWR] probability vector sum deviates from 1: sum={prob_sum!r}")
    return p, int(n_iter), float(final_l1)


def build_local_rwr_context(H_local: nx.Graph) -> Dict[str, object]:
    """Per-target RWR context reusable across all diseases.

    ``H_h(t)`` depends only on the target (methodology §4), so the node
    ordering and the column-normalized transition matrix are identical for
    every disease sharing that target. Building them once per target avoids
    redundant matrix construction without changing any per-pair result.
    """
    node_list = sorted(H_local.nodes())
    node_index = {n: i for i, n in enumerate(node_list)}
    W = build_transition_matrix(H_local, node_index)
    return {"node_list": node_list, "node_set": set(node_list), "node_index": node_index, "W": W}


def compute_rwr_pair_local(
    *,
    H_local: nx.Graph,
    target_gene: str,
    disease_seeds: set[str],
    restart_probability: float,
    fallback_score: float = 0.0,
    target_only_seed_score: float = 1.0,
    context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Finite-fallback local RWR score on ``H_h(t)``.

    Empty local seed sets use the neutral fallback score. Non-convergence is
    intentionally not swallowed and propagates as an error. ``context`` may
    carry a precomputed per-target transition matrix (see
    :func:`build_local_rwr_context`); when omitted it is built on the fly.

    ``disease_seeds`` is expected to already exclude ``target_gene`` (the
    disease seed set is fixed to ``S_{d,-t}`` upstream) so a target-only local
    graph never has a usable seed and always falls back;
    ``target_only_seed_score`` is retained only for backward-compatible config
    parsing and is no longer consulted.
    """
    target_gene = str(target_gene)
    n_local_nodes = int(H_local.number_of_nodes())
    n_local_edges = int(H_local.number_of_edges())
    node_set = context["node_set"] if context is not None else set(H_local.nodes())  # type: ignore[assignment]
    seed_in_local = sorted(set(map(str, disease_seeds)) & node_set)
    seed_target_overlap = target_gene in set(seed_in_local)
    if target_gene not in H_local:
        return _fallback_local(
            "target_not_in_local_graph",
            fallback_score,
            restart_probability,
            n_local_nodes,
            n_local_edges,
            len(seed_in_local),
            seed_target_overlap,
        )
    if not seed_in_local:
        return _fallback_local(
            "no_seed_in_local",
            fallback_score,
            restart_probability,
            n_local_nodes,
            n_local_edges,
            0,
            False,
        )
    if n_local_nodes == 1:
        return _fallback_local(
            "target_only_no_seed",
            fallback_score,
            restart_probability,
            n_local_nodes,
            n_local_edges,
            len(seed_in_local),
            seed_target_overlap,
        )

    if context is None:
        context = build_local_rwr_context(H_local)
    node_index = context["node_index"]  # type: ignore[assignment]
    W = context["W"]
    seed_idx = [node_index[s] for s in seed_in_local]
    p, n_iter, final_l1 = rwr_steady_state(
        W,
        len(node_index),
        seed_idx,
        float(restart_probability),
    )
    return {
        "score": float(p[node_index[target_gene]]),
        "restart_probability": float(restart_probability),
        "n_iter": int(n_iter),
        "final_l1": float(final_l1),
        "n_seed_in_H": int(len(seed_in_local)),
        "n_seed_in_local": int(len(seed_in_local)),
        "n_local_nodes": n_local_nodes,
        "n_local_edges": n_local_edges,
        "seed_target_overlap": bool(seed_target_overlap),
        "valid": True,
        "fallback": False,
        "reason": "",
    }


def _fallback_local(
    reason: str,
    fallback_score: float,
    restart_probability: float,
    n_local_nodes: int,
    n_local_edges: int,
    n_seed_in_local: int,
    seed_target_overlap: bool,
) -> Dict[str, object]:
    return {
        "score": float(fallback_score),
        "restart_probability": float(restart_probability),
        "n_iter": np.nan,
        "final_l1": np.nan,
        "n_seed_in_H": int(n_seed_in_local),
        "n_seed_in_local": int(n_seed_in_local),
        "n_local_nodes": int(n_local_nodes),
        "n_local_edges": int(n_local_edges),
        "seed_target_overlap": bool(seed_target_overlap),
        "valid": True,
        "fallback": True,
        "reason": reason,
    }
