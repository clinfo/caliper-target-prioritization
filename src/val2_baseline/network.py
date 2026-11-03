"""Shared network H construction and degree binning (methodology §4, §6.3).

Network proximity and RWR use the same undirected simple graph H derived from
CALIPER's signed directed SIGNOR MultiDiGraph. Construction rules (§4):
  1. drop edge direction
  2. drop polarity
  3. collapse parallel edges between a node pair into one
  4. remove self-loops
  5. keep only the largest connected component
  6. node IDs use the same NCBI Gene ID representation as CALIPER
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx


def build_shared_network(directed: nx.MultiDiGraph) -> Tuple[nx.Graph, Dict[str, object]]:
    """Build the undirected simple graph H and return (H, stats)."""
    g = nx.Graph()
    g.add_nodes_from(directed.nodes())  # preserve full node set before LCC restriction
    for u, v in directed.edges():
        if u == v:
            continue  # rule 4: drop self-loops
        g.add_edge(u, v)  # rules 1-3: undirected, no polarity, parallel edges collapse

    n_nodes_full = g.number_of_nodes()
    n_edges_full = g.number_of_edges()
    components = list(nx.connected_components(g))
    n_components = len(components)
    if n_components == 0:
        raise ValueError("[network] Graph has no nodes; cannot build shared network H.")
    largest = max(components, key=len)
    H = g.subgraph(largest).copy()  # rule 5: largest connected component

    stats = {
        "rule": [
            "drop_direction",
            "drop_polarity",
            "collapse_parallel_edges",
            "remove_self_loops",
            "largest_connected_component",
            "ncbi_gene_id_nodes",
        ],
        "n_nodes_before_lcc": int(n_nodes_full),
        "n_edges_before_lcc": int(n_edges_full),
        "n_connected_components": int(n_components),
        "n_nodes": int(H.number_of_nodes()),
        "n_edges": int(H.number_of_edges()),
        "largest_component_node_fraction": float(H.number_of_nodes() / n_nodes_full) if n_nodes_full else 0.0,
    }
    return H, stats


def get_degree_binning(H: nx.Graph, bin_size: int) -> List[Tuple[int, int, List[str]]]:
    """Degree binning following Guney toolbox network_utilities.get_degree_binning.

    Nodes are grouped by degree in ascending order; degrees are merged into a
    bin until it reaches ``bin_size`` members. A trailing under-sized bin is
    merged into the previous bin. Returns a list of (low_degree, high_degree,
    node_list).
    """
    degree_to_nodes: Dict[int, List[str]] = {}
    for node, degree in H.degree():
        degree_to_nodes.setdefault(int(degree), []).append(node)

    values = sorted(degree_to_nodes.keys())
    bins: List[Tuple[int, int, List[str]]] = []
    i = 0
    n = len(values)
    while i < n:
        low = values[i]
        val = list(degree_to_nodes[values[i]])
        while len(val) < bin_size:
            i += 1
            if i == n:
                break
            val.extend(degree_to_nodes[values[i]])
        if i == n:
            i -= 1
        high = values[i]
        i += 1
        if len(val) < bin_size and bins:
            low_, _high_, val_ = bins[-1]
            bins[-1] = (low_, high, val_ + val)
        else:
            bins.append((low, high, val))
    return bins


def degree_to_bin_index(bins: List[Tuple[int, int, List[str]]]) -> Dict[int, int]:
    """Map every degree value covered by ``bins`` to its bin index."""
    out: Dict[int, int] = {}
    for idx, (low, high, _nodes) in enumerate(bins):
        for d in range(int(low), int(high) + 1):
            out[d] = idx
    return out
