from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Set
import random

import networkx as nx

def khop_nodes(G: nx.MultiDiGraph, source: str, k: int) -> Set[str]:
    """Nodes reachable within <=k hops via directed edges. Excludes source node."""
    if source not in G:
        return set()
    lengths = nx.single_source_shortest_path_length(G, source, cutoff=k)
    nodes = set(lengths.keys())
    nodes.discard(source)
    return nodes

def propagate_sign_votes_shortest_paths(
    G: nx.MultiDiGraph,
    source: str,
    k: int,
    start_sign: int = -1,
    sign_attr: str = "polarity",
    max_neighbors: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, Dict[int, float]]:
    """
    Shortest-path sign voting up to distance k.

    For each node v (distance d<=k), we enumerate only paths of length d
    (i.e., shortest paths in hop count), compute the sign product along the path,
    and vote by the number of such shortest paths (MultiDiGraph: parallel edges count).

    Note:
      - This function always uses edge signs.
      - Each edge attribute `sign_attr` is expected to be exactly -1 or +1.

    Returns:
      votes[node][sign] = weight
    """
    if source not in G:
        return {}

    rng = random.Random(seed)

    # shortest hop distance (ignore edge signs)
    dist = nx.single_source_shortest_path_length(G, source, cutoff=k)

    # frontier at exact distance (step-1): node -> {sign: weight}
    frontier: Dict[str, Dict[int, float]] = {source: {int(start_sign): 1.0}}
    votes: Dict[str, Dict[int, float]] = defaultdict(lambda: {1: 0.0, -1: 0.0})

    for step in range(1, k + 1):
        new_frontier: Dict[str, Dict[int, float]] = defaultdict(lambda: {1: 0.0, -1: 0.0})

        for u, sign_counts in frontier.items():
            if u not in G:
                continue

            # keep only nodes that are exactly at distance step-1
            if dist.get(u, None) != step - 1:
                continue

            nbrs = list(G.successors(u))
            if max_neighbors is not None and len(nbrs) > max_neighbors:
                nbrs = rng.sample(nbrs, max_neighbors)

            for sgn_u, w in sign_counts.items():
                if w == 0:
                    continue

                for v in nbrs:
                    # only extend along shortest paths: dist[v] must be exactly 'step'
                    if dist.get(v, None) != step:
                        continue

                    keydict = G.get_edge_data(u, v, default={})
                    if not keydict:
                        continue

                    for _key, attr in keydict.items():
                        if sign_attr not in attr:
                            raise KeyError(
                                f"Edge attribute '{sign_attr}' is missing: ({u!r} -> {v!r}) key={_key!r} attrs={list(attr.keys())}"
                            )
                        raw = attr[sign_attr]
                        try:
                            e_sign = int(raw)
                        except Exception as e:
                            raise ValueError(
                                f"Edge attribute '{sign_attr}' must be int-like (-1/+1). got={raw!r} on ({u!r} -> {v!r}) key={_key!r}"
                            ) from e

                        if e_sign not in (-1, 1):
                            raise ValueError(
                                f"Edge attribute '{sign_attr}' must be exactly -1 or +1. got={e_sign!r} on ({u!r} -> {v!r}) key={_key!r}"
                            )

                        sgn_v = int(sgn_u * e_sign)
                        new_frontier[v][sgn_v] += w
                        votes[v][sgn_v] += w

        frontier = new_frontier

    return votes

def predicted_sign_sets_from_votes(
    votes_map: Dict[str, Dict[int, float]],
    eps: float = 0.0,
) -> Dict[str, Set[int]]:
    """
    Convert votes to a *set* of possible signs per node.
    If both +1 and -1 have (weight > eps), keep both (no majority vote).
    """
    out: Dict[str, Set[int]] = {}
    for node, vv in votes_map.items():
        sset: Set[int] = set()
        if vv.get(1, 0.0) > eps:
            sset.add(1)
        if vv.get(-1, 0.0) > eps:
            sset.add(-1)
        if sset:
            out[node] = sset
    return out

def confidence_weight_from_votes(
    votes_map: Dict[str, Dict[int, float]],
    node: str,
    sign: int,
) -> float:
    """Soft weight w = C(node, sign) from shortest-path sign votes.

    votes_map[node][+1] and votes_map[node][-1] are (possibly weighted) counts
    of shortest paths that imply each sign.
    """
    vv = votes_map.get(node)
    if not vv:
        return 0.0
    n_pos = float(vv.get(1, 0.0))
    n_neg = float(vv.get(-1, 0.0))
    denom = n_pos + n_neg
    if denom <= 0.0:
        return 0.0
    num = float(vv.get(int(sign), 0.0))
    return float(num / denom)

def _score_proposed(U, votes, pert, dis):
    n_meas = len(U)
    if n_meas == 0:
        return 0.0, 0, 0, 0.0, 0.0, 0

    # Soft-weight alignment by confidence C(g, p(g)):
    #   w(g) = (n_{p(g)} + alpha) / (n_+ + n_- + 2alpha)
    # where n_+, n_- are shortest-path sign votes from `votes`.
    w_sum = 0.0
    w_rev = 0.0
    for g in U:
        ps = pert.get(g, 0)
        w = confidence_weight_from_votes(votes, g, int(ps))
        if w <= 0.0:
            continue

        w_sum += w
        if pert.get(g, 0) * dis.get(g, 0) == -1:
            w_rev += w

    covAlign = (w_sum / n_meas) if n_meas > 0 else 0.0
    if w_sum == 0.0:
        return 0.0, 0, n_meas, 0.0, covAlign, 0

    score = w_rev / w_sum
    return score, w_rev, n_meas, 0.0, covAlign, w_sum
