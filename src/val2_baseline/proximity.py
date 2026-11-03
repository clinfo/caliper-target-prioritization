"""Network proximity to an expression-derived disease module (methodology §6.3).

Closest distance from a singleton candidate target to the disease seed set on
the shared graph H, with a degree-matched null of 1000 randomizations following
the Guney 2016 toolbox (closest distance, degree binning min_bin_size=100,
z-score with sigma=0 -> z=0). Score = -z.

The degree-matched null is drawn per degree bin without replacement (exact
cardinality, distinct nodes), which is statistically equivalent to the
toolbox's per-node draw with the 20-retry distinctness rule but vectorizes
cleanly. A bin whose pool is too small to supply the required distinct count
makes the pair unscorable (methodology §6.3.3 / validation #8).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import hashlib

import numpy as np
import networkx as nx

from .network import get_degree_binning, degree_to_bin_index


def derive_seed(shared_seed: int, disease_id: str, target_gene: str) -> int:
    h = hashlib.sha256(f"{int(shared_seed)}|{disease_id}|{target_gene}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def build_distance_matrix(H: nx.Graph) -> Tuple[List[str], Dict[str, int], np.ndarray]:
    """All-pairs shortest hop distances on H via per-node BFS.

    Returns (node_list, node_index, matrix) where matrix[i, j] is the hop
    distance (int32); -1 marks unreachable (does not occur within one connected
    component but kept for safety).
    """
    node_list = sorted(H.nodes())
    node_index = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)
    matrix = np.full((n, n), -1, dtype=np.int32)
    for i, node in enumerate(node_list):
        lengths = nx.single_source_shortest_path_length(H, node)
        for tgt, d in lengths.items():
            matrix[i, node_index[tgt]] = int(d)
    return node_list, node_index, matrix


class DegreeMatcher:
    """Degree-bin pools for degree-matched sampling (Guney degree binning)."""

    def __init__(self, H: nx.Graph, node_index: Dict[str, int], min_bin_size: int):
        self.bins = get_degree_binning(H, int(min_bin_size))
        self.deg_to_bin = degree_to_bin_index(self.bins)
        self.degree = {n: int(d) for n, d in H.degree()}
        self.node_index = node_index
        self._bin_pool: List[np.ndarray] = [
            np.array([node_index[n] for n in nodes], dtype=np.int64) for (_l, _h, nodes) in self.bins
        ]

    def bin_of_node(self, node: str) -> int:
        return self.deg_to_bin[self.degree[node]]

    def pool(self, bin_index: int) -> np.ndarray:
        return self._bin_pool[bin_index]

    def export_bins(self) -> List[Dict[str, object]]:
        return [
            {"bin_index": i, "low_degree": int(l), "high_degree": int(h), "n_nodes": int(len(nodes))}
            for i, (l, h, nodes) in enumerate(self.bins)
        ]


def _sample_without_replacement(rng: np.random.Generator, pool: np.ndarray, count: int, n_draws: int) -> Optional[np.ndarray]:
    """Return (n_draws, count) of distinct indices sampled from ``pool`` per row.

    None if the pool is too small to supply ``count`` distinct nodes.
    """
    L = len(pool)
    if L < count:
        return None
    if count == 0:
        return np.empty((n_draws, 0), dtype=np.int64)
    rand = rng.random((n_draws, L))
    take = np.argpartition(rand, count - 1, axis=1)[:, :count]
    return pool[take]


def build_disease_seed_null(
    *,
    matcher: DegreeMatcher,
    node_index: Dict[str, int],
    seeds: List[str],
    n_random: int,
    derived_seed_value: int,
) -> Tuple[Optional[np.ndarray], str]:
    """Degree-matched null disease seed sets, computed once per disease.

    Returns (seed_null, reason). seed_null is (n_random, n_seed_in_H) of node
    indices; None when a degree bin cannot supply the required distinct count
    (validation #8). The disease seed set is identical across all candidate
    targets, so reusing the same degree-preserving null across those targets is
    both faithful and the minimal-cost choice; the target null below remains
    per-pair (methodology §6.3.3).
    """
    seed_in_H = [s for s in seeds if s in node_index]
    if len(seed_in_H) == 0:
        return None, "no_seed_in_H"
    rng = np.random.default_rng(int(derived_seed_value))
    seed_bins: Dict[int, List[int]] = {}
    for s in seed_in_H:
        seed_bins.setdefault(matcher.bin_of_node(s), []).append(node_index[s])
    cols: List[np.ndarray] = []
    for b, members in seed_bins.items():
        pool = matcher.pool(b)
        observed = np.array(members, dtype=np.int64)
        pool = pool[~np.isin(pool, observed)]
        block = _sample_without_replacement(rng, pool, len(members), int(n_random))
        if block is None:
            return None, "null_cardinality_not_maintained"
        cols.append(block)
    return np.concatenate(cols, axis=1), ""


def compute_proximity_pair(
    *,
    matrix: np.ndarray,
    node_index: Dict[str, int],
    matcher: DegreeMatcher,
    target_gene: str,
    seeds: List[str],
    seed_null: Optional[np.ndarray],
    seed_null_reason: str,
    n_random: int,
    target_seed_value: int,
) -> Dict[str, object]:
    """Proximity score for one (target, disease) pair using a precomputed,
    per-disease degree-matched seed null and a per-pair target null."""
    if target_gene not in node_index:
        return _unscorable("target_not_in_H", target_seed_value)
    seed_in_H = [s for s in seeds if s in node_index]
    if len(seed_in_H) == 0:
        return _unscorable("no_seed_in_H", target_seed_value)
    if seed_null is None:
        return _unscorable(seed_null_reason or "seed_null_undefined", target_seed_value)

    t_idx = node_index[target_gene]
    seed_idx = np.array([node_index[s] for s in seed_in_H], dtype=np.int64)
    obs_row = matrix[t_idx, seed_idx]
    obs_row = obs_row[obs_row >= 0]
    if obs_row.size == 0:
        return _unscorable("infinite_observed_distance", target_seed_value)
    d_c = float(obs_row.min())

    rng = np.random.default_rng(int(target_seed_value))
    R = int(n_random)

    # target-null: one degree-matched node per replicate (exclude target itself)
    t_bin = matcher.bin_of_node(target_gene)
    t_pool = matcher.pool(t_bin)
    t_pool = t_pool[t_pool != t_idx]
    if len(t_pool) == 0:
        return _unscorable("target_bin_pool_empty", target_seed_value)
    target_null = t_pool[rng.integers(0, len(t_pool), size=R)]

    dists = matrix[target_null[:, None], seed_null]  # (R, n_seed)
    dists = np.where(dists >= 0, dists, np.iinfo(np.int32).max)
    null_vals = dists.min(axis=1).astype(np.float64)
    if not np.all(null_vals < np.iinfo(np.int32).max):
        return _unscorable("infinite_null_distance", target_seed_value)

    mu = float(null_vals.mean())
    sigma = float(null_vals.std())  # population std (ddof=0), matches numpy.std in toolbox
    z = 0.0 if sigma == 0.0 else (d_c - mu) / sigma
    return {
        "closest_distance": float(d_c),
        "null_mean": mu,
        "null_std": sigma,
        "z": float(z),
        "score": float(-z),
        "n_seed_in_H": int(len(seed_in_H)),
        "valid": True,
        "reason": "",
        "random_seed": int(target_seed_value),
        "n_random": int(R),
    }


def build_local_proximity_context(H_local: nx.Graph, min_bin_size: int) -> Dict[str, object]:
    """Per-target proximity context reusable across all diseases.

    ``H_h(t)`` depends only on the target (methodology §4), so the all-pairs
    distance matrix and the degree-binning matcher are identical for every
    disease sharing that target. Building them once per target avoids the
    dominant redundant cost without changing any per-pair result.
    """
    node_list, node_index, matrix = build_distance_matrix(H_local)
    matcher = DegreeMatcher(H_local, node_index, int(min_bin_size))
    return {
        "node_list": node_list,
        "node_set": set(node_list),
        "node_index": node_index,
        "matrix": matrix,
        "matcher": matcher,
    }


def compute_proximity_pair_local(
    *,
    H_local: nx.Graph,
    target_gene: str,
    disease_seeds: set[str],
    n_random: int,
    min_bin_size: int,
    shared_seed: int,
    disease_id: str,
    fallback_score: float = 0.0,
    context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Finite-fallback proximity score on ``H_h(t)`` for directed reach mode.

    ``context`` may carry a precomputed per-target distance matrix and matcher
    (see :func:`build_local_proximity_context`). When omitted they are built
    on the fly, preserving the original behaviour.
    """
    target_gene = str(target_gene)
    n_local_nodes = int(H_local.number_of_nodes())
    n_local_edges = int(H_local.number_of_edges())
    node_set = context["node_set"] if context is not None else set(H_local.nodes())  # type: ignore[assignment]
    seed_in_local = sorted(set(map(str, disease_seeds)) & node_set)
    seed_target_overlap = target_gene in set(seed_in_local)
    random_seed = derive_seed(shared_seed, disease_id, target_gene)
    if target_gene not in H_local:
        return _fallback_local(
            "target_not_in_local_graph",
            random_seed,
            fallback_score,
            n_local_nodes,
            n_local_edges,
            len(seed_in_local),
            seed_target_overlap,
        )
    if not seed_in_local:
        return _fallback_local(
            "no_seed_in_local",
            random_seed,
            fallback_score,
            n_local_nodes,
            n_local_edges,
            0,
            False,
        )

    if context is None:
        context = build_local_proximity_context(H_local, int(min_bin_size))
    node_index = context["node_index"]  # type: ignore[assignment]
    matrix = context["matrix"]  # type: ignore[assignment]
    matcher = context["matcher"]  # type: ignore[assignment]
    seed_null, seed_null_reason = build_disease_seed_null(
        matcher=matcher,
        node_index=node_index,
        seeds=seed_in_local,
        n_random=int(n_random),
        derived_seed_value=derive_seed(shared_seed, disease_id, target_gene + "__LOCAL_SEEDS__"),
    )
    prox = compute_proximity_pair(
        matrix=matrix,
        node_index=node_index,
        matcher=matcher,
        target_gene=target_gene,
        seeds=seed_in_local,
        seed_null=seed_null,
        seed_null_reason=seed_null_reason,
        n_random=int(n_random),
        target_seed_value=random_seed,
    )
    if not bool(prox.get("valid", False)):
        return _fallback_local(
            str(prox.get("reason") or "local_proximity_undefined"),
            random_seed,
            fallback_score,
            n_local_nodes,
            n_local_edges,
            len(seed_in_local),
            seed_target_overlap,
        )
    prox.update(
        {
            "n_seed_in_local": int(len(seed_in_local)),
            "n_local_nodes": n_local_nodes,
            "n_local_edges": n_local_edges,
            "fallback": False,
            "seed_target_overlap": seed_target_overlap,
        }
    )
    return prox


def _fallback_local(
    reason: str,
    derived_seed_value: int,
    fallback_score: float,
    n_local_nodes: int,
    n_local_edges: int,
    n_seed_in_local: int,
    seed_target_overlap: bool,
) -> Dict[str, object]:
    return {
        "closest_distance": np.nan,
        "null_mean": np.nan,
        "null_std": np.nan,
        "z": np.nan,
        "score": float(fallback_score),
        "n_seed_in_H": int(n_seed_in_local),
        "n_seed_in_local": int(n_seed_in_local),
        "n_local_nodes": int(n_local_nodes),
        "n_local_edges": int(n_local_edges),
        "valid": True,
        "fallback": True,
        "reason": reason,
        "random_seed": int(derived_seed_value),
        "n_random": 0,
        "seed_target_overlap": bool(seed_target_overlap),
    }


def _unscorable(reason: str, derived_seed_value: int) -> Dict[str, object]:
    return {
        "closest_distance": np.nan,
        "null_mean": np.nan,
        "null_std": np.nan,
        "z": np.nan,
        "score": np.nan,
        "n_seed_in_H": 0,
        "valid": False,
        "reason": reason,
        "random_seed": int(derived_seed_value),
        "n_random": 0,
    }
