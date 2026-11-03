from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class TerminalVoteTable:
    source: str
    k_star: int
    genes: list[str]
    gene_to_idx: dict[str, int]
    n_pos: np.ndarray
    n_neg: np.ndarray
    assigned_pos: np.ndarray
    assigned_neg: np.ndarray
    family_pos: dict[str, np.ndarray]
    family_neg: dict[str, np.ndarray]
    dist: dict[str, int]


def _empty_table(source: str, k_star: int, family_order: list[str]) -> TerminalVoteTable:
    return TerminalVoteTable(
        source=str(source),
        k_star=int(k_star),
        genes=[],
        gene_to_idx={},
        n_pos=np.zeros(0, dtype=np.float64),
        n_neg=np.zeros(0, dtype=np.float64),
        assigned_pos=np.zeros(0, dtype=np.float64),
        assigned_neg=np.zeros(0, dtype=np.float64),
        family_pos={f: np.zeros(0, dtype=np.float64) for f in family_order},
        family_neg={f: np.zeros(0, dtype=np.float64) for f in family_order},
        dist={},
    )


def collect_terminal_vote_table(
    G: nx.MultiDiGraph,
    source: str,
    k_star: int,
    start_sign: int,
    edge_id_to_families: dict[int, tuple[str, ...]],
    family_order: list[str],
    sign_attr: str = "polarity",
) -> TerminalVoteTable:
    source = str(source)
    if int(start_sign) not in {-1, 1}:
        raise ValueError("start_sign must be -1 or +1")
    if source not in G:
        return _empty_table(source, int(k_star), family_order)

    raw_dist = nx.single_source_shortest_path_length(G, source, cutoff=int(k_star))
    dist = {str(node): int(d) for node, d in raw_dist.items()}
    genes = sorted([node for node, d in dist.items() if node != source and 1 <= int(d) <= int(k_star)])
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    n = len(genes)

    n_pos = np.zeros(n, dtype=np.float64)
    n_neg = np.zeros(n, dtype=np.float64)
    assigned_pos = np.zeros(n, dtype=np.float64)
    assigned_neg = np.zeros(n, dtype=np.float64)
    family_pos = {f: np.zeros(n, dtype=np.float64) for f in family_order}
    family_neg = {f: np.zeros(n, dtype=np.float64) for f in family_order}
    family_set = set(family_order)

    nodes_by_dist: dict[int, list[str]] = defaultdict(list)
    for node, d in dist.items():
        nodes_by_dist[int(d)].append(node)

    state_pos: dict[str, float] = defaultdict(float)
    state_neg: dict[str, float] = defaultdict(float)
    if int(start_sign) == 1:
        state_pos[source] = 1.0
    else:
        state_neg[source] = 1.0

    for d in range(0, int(k_star)):
        next_pos_updates: dict[str, float] = defaultdict(float)
        next_neg_updates: dict[str, float] = defaultdict(float)
        # State is cumulative per node at its shortest layer. Only nodes at d are extended.
        for u in nodes_by_dist.get(d, []):
            pos_u = float(state_pos.get(u, 0.0))
            neg_u = float(state_neg.get(u, 0.0))
            if pos_u == 0.0 and neg_u == 0.0:
                continue
            for _, v, _key, attrs in G.out_edges(u, keys=True, data=True):
                v = str(v)
                if dist.get(v) != d + 1:
                    continue
                if sign_attr not in attrs:
                    raise KeyError(f"Edge sign attribute {sign_attr!r} is missing on {u!r}->{v!r}")
                pol = int(attrs[sign_attr])
                if pol not in {-1, 1}:
                    raise ValueError(f"Edge polarity must be -1 or +1. got={pol!r} on {u!r}->{v!r}")
                if pol == 1:
                    add_pos, add_neg = pos_u, neg_u
                else:
                    add_pos, add_neg = neg_u, pos_u

                next_pos_updates[v] += add_pos
                next_neg_updates[v] += add_neg

                idx = gene_to_idx.get(v)
                if idx is not None:
                    n_pos[idx] += add_pos
                    n_neg[idx] += add_neg
                    edge_id = int(attrs["edge_id"])
                    fams = tuple(f for f in edge_id_to_families.get(edge_id, tuple()) if f in family_set)
                    if fams:
                        assigned_pos[idx] += add_pos
                        assigned_neg[idx] += add_neg
                        for f in fams:
                            family_pos[f][idx] += add_pos
                            family_neg[f][idx] += add_neg
        for v, value in next_pos_updates.items():
            state_pos[v] += value
        for v, value in next_neg_updates.items():
            state_neg[v] += value

    return TerminalVoteTable(
        source=source,
        k_star=int(k_star),
        genes=genes,
        gene_to_idx=gene_to_idx,
        n_pos=n_pos,
        n_neg=n_neg,
        assigned_pos=assigned_pos,
        assigned_neg=assigned_neg,
        family_pos=family_pos,
        family_neg=family_neg,
        dist=dist,
    )
