"""Directed target-reach helpers for val2 external-baseline matching."""

from __future__ import annotations

from typing import Dict, Set

import networkx as nx

from inference.signature_reversal_nodes import khop_nodes


def directed_reach_nodes(graph: nx.MultiDiGraph, target: str, h: int) -> Set[str]:
    """Return nodes reachable from ``target`` within ``h`` directed hops.

    The target itself is excluded, matching ``R_h_dir(t)`` in the methodology
    and the existing CALIPER ``khop_nodes`` helper.
    """
    return set(khop_nodes(graph, str(target), int(h)))


def anchor_nodes(target: str, reach_nodes: Set[str]) -> Set[str]:
    """Return ``A_h_dir(t) = {t} union R_h_dir(t)``."""
    out = set(map(str, reach_nodes))
    out.add(str(target))
    return out


def build_local_undirected_graph(graph: nx.MultiDiGraph, nodes: Set[str]) -> nx.Graph:
    """Project ``G[nodes]`` to an undirected unsigned simple local graph.

    Direction and polarity are dropped, parallel edges are collapsed by
    ``nx.Graph``, self-loops are removed, and no LCC filtering is applied.

    This scans every edge of ``graph`` (O(|E|)); when building many local
    graphs (one per candidate target) prefer
    :func:`build_local_undirected_graph_from_adjacency` with a precomputed
    :func:`build_undirected_adjacency`, which is O(local size) per call.
    """
    node_set = set(map(str, nodes))
    local = nx.Graph()
    local.add_nodes_from(node_set)
    for u, v in graph.edges():
        if u == v:
            continue
        if u in node_set and v in node_set:
            local.add_edge(u, v)
    return local


def build_undirected_adjacency(graph: nx.MultiDiGraph) -> Dict[str, Set[str]]:
    """Undirected, unsigned, self-loop-free adjacency built once from ``graph``.

    Reused across every candidate target's local-graph construction so the
    per-target cost is proportional to the local neighborhood size instead of
    a full O(|E|) edge scan (methodology-preserving speed-up: identical edge
    set as :func:`build_local_undirected_graph`, just precomputed once).
    """
    adjacency: Dict[str, Set[str]] = {str(n): set() for n in graph.nodes()}
    for u, v in graph.edges():
        if u == v:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def build_local_undirected_graph_from_adjacency(
    adjacency: Dict[str, Set[str]], nodes: Set[str]
) -> nx.Graph:
    """Same output as :func:`build_local_undirected_graph`, using a
    precomputed :func:`build_undirected_adjacency` instead of scanning every
    edge of the full graph."""
    node_set = set(map(str, nodes))
    local = nx.Graph()
    local.add_nodes_from(node_set)
    for n in node_set:
        for m in adjacency.get(n, ()):
            if m in node_set and m != n:
                local.add_edge(n, m)
    return local


def summarize_target_reach(
    target: str,
    h: int,
    reach: Set[str],
    local_graph: nx.Graph,
    *,
    target_in_graph: bool = True,
) -> Dict[str, object]:
    """Diagnostics row for ``network/target_reach_by_target.tsv``."""
    return {
        "target_gene": str(target),
        "h": int(h),
        "target_in_G": bool(target_in_graph),
        "n_reach_nodes": int(len(reach)),
        "n_anchor_nodes": int(len(set(reach) | {str(target)})),
        "n_local_nodes": int(local_graph.number_of_nodes()),
        "n_local_edges": int(local_graph.number_of_edges()),
    }
