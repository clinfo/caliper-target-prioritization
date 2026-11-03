from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Optional, Set, Tuple
import random

import networkx as nx
import pandas as pd


def shortest_path_lengths_up_to_k(
    G: nx.MultiDiGraph,
    source: str,
    k: int,
) -> Dict[str, int]:
    if source not in G:
        return {}
    return nx.single_source_shortest_path_length(G, source, cutoff=int(k))


def khop_nodes(G: nx.MultiDiGraph, source: str, k: int) -> Set[str]:
    lengths = shortest_path_lengths_up_to_k(G, source, k)
    nodes = set(lengths.keys())
    nodes.discard(source)
    return nodes


def reachable_hop_depth(G: nx.MultiDiGraph, source: str) -> int:
    """Return D_t = max_v dist(source, v) over nodes reachable from source.

    The source itself has distance 0, so isolated or missing sources yield D_t = 0.
    """
    if source not in G:
        return 0
    lengths = nx.single_source_shortest_path_length(G, source)
    if not lengths:
        return 0
    return int(max(lengths.values()))


def summarize_target_hop_depths(
    G: nx.MultiDiGraph,
    targets: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    n_missing = 0
    for target in sorted({str(t) for t in targets}):
        in_kg = target in G
        if not in_kg:
            n_missing += 1
        d_t = reachable_hop_depth(G, target)
        rows.append({
            "target_gene": target,
            "in_kg": bool(in_kg),
            "D_t": int(d_t),
        })

    df = pd.DataFrame(rows).sort_values(["D_t", "target_gene"], ascending=[False, True]).reset_index(drop=True)
    if len(df) == 0:
        k_max = 0
    else:
        k_max = int(df["D_t"].max())

    summary = {
        "K_max": int(k_max),
        "n_targets": int(len(df)),
        "n_targets_in_kg": int(len(df) - n_missing),
        "n_targets_missing_from_kg": int(n_missing),
    }
    return df, summary


def propagate_sign_votes_shortest_paths(
    G: nx.MultiDiGraph,
    source: str,
    k: int,
    start_sign: int = -1,
    sign_attr: str = "polarity",
    max_neighbors: Optional[int] = None,
    seed: int = 0,
    dist: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[int, float]]:
    if source not in G:
        return {}

    rng = random.Random(int(seed))
    dist = dist if dist is not None else shortest_path_lengths_up_to_k(G, source, k)
    votes, _ = propagate_sign_votes_with_dist(
        G,
        source,
        k,
        dist,
        start_sign=start_sign,
        sign_attr=sign_attr,
        max_neighbors=max_neighbors,
        seed=seed,
        rng=rng,
    )
    return votes


def propagate_sign_votes_with_dist(
    G: nx.MultiDiGraph,
    source: str,
    k: int,
    dist: Dict[str, int],
    start_sign: int = -1,
    sign_attr: str = "polarity",
    max_neighbors: Optional[int] = None,
    seed: int = 0,
    rng: Optional[random.Random] = None,
) -> Tuple[Dict[str, Dict[int, float]], int]:
    if source not in G:
        return {}, 0

    rng = rng or random.Random(int(seed))
    frontier: Dict[str, Dict[int, float]] = {source: {int(start_sign): 1.0}}
    votes: Dict[str, Dict[int, float]] = defaultdict(lambda: {1: 0.0, -1: 0.0})
    scanned_edges = 0

    for step in range(1, int(k) + 1):
        new_frontier: Dict[str, Dict[int, float]] = defaultdict(lambda: {1: 0.0, -1: 0.0})

        for u, sign_counts in frontier.items():
            if u not in G:
                continue
            if dist.get(u, None) != step - 1:
                continue

            nbrs = list(G.successors(u))
            if max_neighbors is not None and len(nbrs) > int(max_neighbors):
                nbrs = rng.sample(nbrs, int(max_neighbors))

            for sgn_u, w in sign_counts.items():
                if w == 0:
                    continue
                for v in nbrs:
                    if dist.get(v, None) != step:
                        continue
                    keydict = G.get_edge_data(u, v, default={})
                    if not keydict:
                        continue
                    for _key, attr in keydict.items():
                        if sign_attr not in attr:
                            raise KeyError(f"Missing edge attr '{sign_attr}' on ({u}->{v})")
                        e_sign = int(attr[sign_attr])
                        if e_sign not in (-1, 1):
                            raise ValueError(f"Edge sign must be ±1. got={e_sign} on ({u}->{v})")
                        scanned_edges += 1
                        sgn_v = int(sgn_u * e_sign)
                        new_frontier[v][sgn_v] += w
                        votes[v][sgn_v] += w

        frontier = new_frontier

    return votes, scanned_edges
