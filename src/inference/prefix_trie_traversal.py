"""Prefix-trie traversal of cached signature-reversal path skeletons.

``_evaluate_single_row_arrays`` re-walks every candidate path of a source for every
profile row. Because the acceptance predicate is decided node by node and its two
gap counters are monotone, unsatisfiable prefixes can be cut once instead of being
re-expanded for each path that shares them.

The trie is folded from the cached ``path_polarities`` list *in its existing order*
and every trie terminal keeps the indices of the original paths that end there, so
the accepted-path identities, their order, and therefore the result of
``prune_prefix_paths_with_idx`` are the same as in the reference engine. No KG
adjacency is re-walked, so nothing depends on NetworkX's enumeration order.

Acceptance predicate for a path with nodes g_0..g_{L-1}, cumulative polarities p_j
and (permuted) perturbation / disease values x_j, y_j:

  1. for all j: x_j == 0 or x_j == p_j
  2. x_{L-1} != 0 and y_{L-1} != 0
  3. for all j: x_j == 0 or y_j == 0 or x_j * y_j == -1
  4. #{j < L-1 : x_j == 0} <= gap  and  #{j < L-1 : x_j == 0 or y_j == 0} <= gap
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


class PathTrie:
    """Prefix trie over ``(gene, cumulative polarity)`` pairs for one source."""

    __slots__ = (
        "source",
        "initial_polarity",
        "genes",
        "node_gene",
        "node_polarity",
        "node_parent",
        "children_start",
        "children",
        "terminal_paths",
        "n_paths",
        "num_nodes_in_paths",
        "_names_cache",
        "_edges_cache",
    )

    def __init__(
        self,
        *,
        source: str,
        initial_polarity: int,
        genes: List[str],
        node_gene: np.ndarray,
        node_polarity: np.ndarray,
        node_parent: np.ndarray,
        children_start: np.ndarray,
        children: np.ndarray,
        terminal_paths: Dict[int, List[int]],
        n_paths: int,
        num_nodes_in_paths: int,
    ) -> None:
        self.source = str(source)
        self.initial_polarity = int(initial_polarity)
        self.genes = genes
        self.node_gene = node_gene
        self.node_polarity = node_polarity
        self.node_parent = node_parent
        self.children_start = children_start
        self.children = children
        self.terminal_paths = terminal_paths
        self.n_paths = int(n_paths)
        self.num_nodes_in_paths = int(num_nodes_in_paths)
        self._names_cache: Dict[int, List[str]] = {}
        self._edges_cache: Dict[int, Tuple[Tuple[str, str, int], ...]] = {}

    @property
    def n_nodes(self) -> int:
        return int(len(self.node_gene))

    def node_names(self, node: int) -> List[str]:
        """Gene sequence from the root to ``node``, i.e. ``list(pol_dict.keys())``."""
        cached = self._names_cache.get(node)
        if cached is not None:
            return cached
        names: List[str] = []
        cur = int(node)
        while cur > 0:
            names.append(self.genes[int(self.node_gene[cur])])
            cur = int(self.node_parent[cur])
        names.reverse()
        self._names_cache[node] = names
        return names

    def node_edges(self, node: int) -> Tuple[Tuple[str, str, int], ...]:
        """Signed edges along the root-to-``node`` path, first hop first.

        ``p_j = p_{j-1} * q_j`` for the KG edge polarity ``q_j``, so
        ``_edge_sign_from_polarities(p_{j-1}, p_j)`` equals ``q_j``; recording the
        cumulative polarities is enough to rebuild the signed edges.
        """
        cached = self._edges_cache.get(node)
        if cached is not None:
            return cached
        chain: List[Tuple[str, str, int]] = []
        cur = int(node)
        while cur > 0:
            parent = int(self.node_parent[cur])
            prev_pol = self.initial_polarity if parent == 0 else int(self.node_polarity[parent])
            prev_name = self.source if parent == 0 else self.genes[int(self.node_gene[parent])]
            curr_pol = int(self.node_polarity[cur])
            chain.append((prev_name, self.genes[int(self.node_gene[cur])], 1 if prev_pol == curr_pol else -1))
            cur = parent
        chain.reverse()
        out = tuple(chain)
        self._edges_cache[node] = out
        return out

    def release_row_caches(self) -> None:
        self._names_cache.clear()
        self._edges_cache.clear()


def build_path_trie(
    path_polarities: Sequence[Mapping[str, int]],
    *,
    source: str,
    initial_polarity: int,
) -> PathTrie:
    """Fold the cached path-polarity dicts into a prefix trie, order preserved."""
    genes: List[str] = []
    gene_to_local: Dict[str, int] = {}
    node_gene: List[int] = [-1]
    node_polarity: List[int] = [0]
    node_parent: List[int] = [-1]
    child_maps: List[Dict[Tuple[int, int], int]] = [{}]
    terminal_paths: Dict[int, List[int]] = {}
    all_gene_ids: set[int] = set()

    for path_index, pol_dict in enumerate(path_polarities):
        cur = 0
        for gene, polarity in pol_dict.items():
            gene_s = str(gene)
            local = gene_to_local.get(gene_s)
            if local is None:
                local = len(genes)
                genes.append(gene_s)
                gene_to_local[gene_s] = local
            all_gene_ids.add(local)
            key = (local, int(polarity))
            nxt = child_maps[cur].get(key)
            if nxt is None:
                nxt = len(node_gene)
                node_gene.append(local)
                node_polarity.append(int(polarity))
                node_parent.append(cur)
                child_maps.append({})
                child_maps[cur][key] = nxt
            cur = nxt
        if cur != 0:
            terminal_paths.setdefault(cur, []).append(path_index)

    n_nodes = len(node_gene)
    counts = np.zeros(n_nodes + 1, dtype=np.int64)
    for node, child_map in enumerate(child_maps):
        counts[node + 1] = len(child_map)
    children_start = np.cumsum(counts, dtype=np.int64)
    children = np.empty(int(children_start[-1]), dtype=np.int64)
    for node, child_map in enumerate(child_maps):
        pos = int(children_start[node])
        for child in child_map.values():
            children[pos] = child
            pos += 1
    child_maps.clear()

    return PathTrie(
        source=str(source),
        initial_polarity=int(initial_polarity),
        genes=genes,
        node_gene=np.asarray(node_gene, dtype=np.int64),
        node_polarity=np.asarray(node_polarity, dtype=np.int64),
        node_parent=np.asarray(node_parent, dtype=np.int64),
        children_start=children_start,
        children=children,
        terminal_paths=terminal_paths,
        n_paths=int(len(path_polarities)),
        num_nodes_in_paths=int(len(all_gene_ids)),
    )


def expand_trie_paths(trie: PathTrie) -> List[Tuple[int, List[str], List[int]]]:
    """Every stored path as ``(path_index, gene sequence, polarity sequence)``.

    Only used by the equivalence tests to prove the fold is lossless.
    """
    out: List[Tuple[int, List[str], List[int]]] = []
    for node, indices in trie.terminal_paths.items():
        names = trie.node_names(node)
        polarities: List[int] = []
        cur = int(node)
        while cur > 0:
            polarities.append(int(trie.node_polarity[cur]))
            cur = int(trie.node_parent[cur])
        polarities.reverse()
        for path_index in indices:
            out.append((int(path_index), list(names), list(polarities)))
    out.sort(key=lambda item: item[0])
    return out


def evaluate_trie_row(
    trie: PathTrie,
    pert_values: Sequence[int],
    disease_values: Sequence[int],
    *,
    gap: int,
) -> List[Tuple[int, int]]:
    """Accepted ``(path_index, trie node)`` pairs for one profile row.

    ``pert_values`` / ``disease_values`` are indexed by the trie's local gene table.
    """
    node_gene = trie.node_gene
    node_polarity = trie.node_polarity
    children_start = trie.children_start
    children = trie.children
    terminal_paths = trie.terminal_paths
    gap_local = int(gap)

    accepted: List[Tuple[int, int]] = []
    stack: List[Tuple[int, int, int]] = []
    for k in range(int(children_start[0]), int(children_start[1])):
        stack.append((int(children[k]), 0, 0))

    while stack:
        node, zero_corr, zero_anti = stack.pop()
        local = node_gene[node]
        x = pert_values[local]
        y = disease_values[local]
        if x != 0:
            # A correlation or anti-correlation mismatch here holds for every
            # extension of this prefix, so the whole subtree can be dropped.
            if x != node_polarity[node]:
                continue
            if y != 0:
                if x * y != -1:
                    continue
                if zero_corr <= gap_local and zero_anti <= gap_local:
                    terminals = terminal_paths.get(node)
                    if terminals is not None:
                        for path_index in terminals:
                            accepted.append((path_index, node))
        next_zero_corr = zero_corr + 1 if x == 0 else zero_corr
        next_zero_anti = zero_anti + 1 if (x == 0 or y == 0) else zero_anti
        # Both counters only grow with depth, so exceeding the gap here rules out
        # every descendant as well.
        if next_zero_corr <= gap_local and next_zero_anti <= gap_local:
            for k in range(int(children_start[node]), int(children_start[node + 1])):
                stack.append((int(children[k]), next_zero_corr, next_zero_anti))
    return accepted


def accepted_paths_in_reference_order(accepted: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort accepted terminals by original path index.

    The reference engine appends accepted paths while iterating ``path_polarities``,
    so restoring index order reproduces the exact list that the stable, length-first
    prefix pruning consumes.
    """
    return sorted(accepted, key=lambda item: item[0])
