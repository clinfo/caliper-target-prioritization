"""Fused Null 1 engine: prepared inputs, sparse permutations, trie traversal.

The reference route for one Null 1 iteration is
``read parquet -> signature_reversal_paths_from_graph -> DataFrame of path lists
-> re-parse the path lists into signed edges``. Everything before the signed-edge
sets is per-iteration work on data that does not change between iterations.

This module keeps three things out of the iteration loop:

* the parsed profile input (``prepare_infer_input``), read once per side;
* the gene-node shuffle, expressed as a sparse position remap instead of a dense
  ``n_genes`` gather array per group (``build_sparse_permutation_lookups``);
* the candidate-path skeleton, folded once per source into a prefix trie that is
  reused by every row and every iteration (``val3_path_trie``).

The accepted-path set, its order, the prefix pruning and the resulting signed-edge
sets are identical to the reference route; only the intermediate Python path-list
DataFrame is skipped. ``tests/test_val3_null1_engine_equivalence.py`` pins this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from inference.prefix_trie_traversal import (
    PathTrie,
    accepted_paths_in_reference_order,
    build_path_trie,
    evaluate_trie_row,
)
from inference.signature_reversal_paths import (
    META_COLUMNS_TO_KEEP,
    _collect_path_polarities_cached,
    prune_prefix_paths_with_idx,
)

from .io import read_parquet_any

GLOBAL_PERMUTATION_GROUP = "__global__"


def _downcast_profile_matrix(mat: np.ndarray) -> np.ndarray:
    """Shrink an integer profile matrix to int8 when that is value-preserving.

    Profile entries are in ``{-1, 0, 1}``; the range check keeps this a pure
    representation change so comparisons, signs and products are untouched.
    """
    if mat.dtype == np.int8 or mat.size == 0:
        return mat
    if mat.dtype.kind not in "iu":
        return mat
    if int(mat.min()) < -128 or int(mat.max()) > 127:
        return mat
    return mat.astype(np.int8, copy=False)


class SparsePermutationLookup:
    """Position remap for one permutation group, stored only where it moves.

    ``_build_permutation_source_positions`` materializes a dense ``arange(n_genes)``
    per group and overwrites the permuted background positions. The permutation is
    the identity everywhere outside that background, so only the moved entries need
    to be kept.
    """

    __slots__ = ("_mapping", "_n_genes")

    def __init__(self, mapping: Mapping[int, int], *, n_genes: int) -> None:
        self._mapping = dict(mapping)
        self._n_genes = int(n_genes)

    def __len__(self) -> int:
        return len(self._mapping)

    def items(self) -> List[Tuple[int, int]]:
        """``(target position, source position)`` pairs for the moved entries."""
        return [(int(target), int(source)) for target, source in self._mapping.items()]

    def gather(self, positions: Sequence[int]) -> List[int]:
        mapping = self._mapping
        return [mapping.get(int(pos), int(pos)) for pos in positions]

    def as_dense(self) -> np.ndarray:
        """Dense equivalent, used by the equivalence tests and audit digests."""
        dense = np.arange(self._n_genes, dtype=np.intp)
        for target, source in self._mapping.items():
            dense[int(target)] = int(source)
        return dense


def build_sparse_permutation_lookups(
    genes: Sequence[str],
    permutation_specs: Mapping[str, Any] | None,
) -> Dict[str, SparsePermutationLookup]:
    """Sparse counterpart of ``_build_permutation_source_positions``."""
    if permutation_specs is None:
        return {}
    gene_order = [str(g) for g in permutation_specs["gene_order"]]
    gene_to_pos = {str(g): i for i, g in enumerate(genes)}
    missing = [g for g in gene_order if g not in gene_to_pos]
    if missing:
        raise ValueError(f"Permutation specs reference genes absent from infer input: {missing[:5]}")
    target_pos = [gene_to_pos[g] for g in gene_order]
    per_triplet = permutation_specs.get("per_triplet") or {}
    out: Dict[str, SparsePermutationLookup] = {}
    for group, perm_list in per_triplet.items():
        perm = [int(v) for v in perm_list]
        if len(perm) != len(gene_order):
            raise ValueError(f"Permutation length mismatch for group {group!r}: {len(perm)} != {len(gene_order)}")
        mapping = {}
        for k, target in enumerate(target_pos):
            source = target_pos[perm[k]]
            if source != target:
                mapping[target] = source
        out[str(group)] = SparsePermutationLookup(mapping, n_genes=len(genes))
    return out


@dataclass
class PreparedInferInput:
    """Profile input parsed once and reused by every Null 1 iteration."""

    mode: str
    initial_polarity: int
    disease_genes: List[str]
    pert_genes: List[str]
    disease_gene_to_pos: Dict[str, int]
    pert_gene_to_pos: Dict[str, int]
    disease_mat: np.ndarray
    pert_mat: np.ndarray
    valid_genes: Set[str]
    valid_genes_digest: str
    meta_arrays: Dict[str, np.ndarray]
    do_ids: np.ndarray
    cell_ids: np.ndarray
    labels: np.ndarray
    pert_id_values: np.ndarray
    triplet_ids: np.ndarray | None
    groups: List[Tuple[str, np.ndarray]]
    n_rows: int
    has_triplet_id: bool

    @property
    def group_major_order(self) -> np.ndarray:
        if not self.groups:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([rows for _, rows in self.groups]).astype(np.int64, copy=False)

    def nbytes(self) -> int:
        return int(self.disease_mat.nbytes + self.pert_mat.nbytes)


def prepare_infer_input(
    input_path: str | Path,
    *,
    graph: nx.MultiDiGraph,
    mode: str,
) -> PreparedInferInput:
    """Read one Val3 profile input and cache everything reusable across iterations.

    Mirrors the setup block of ``signature_reversal_paths_from_graph``: same valid
    gene set, same ``pert_id`` filtering against the KG, and the same
    ``groupby(..., sort=False)`` group order.
    """
    from .metrics import _initial_polarity_for_mode, _valid_gene_digest

    df = read_parquet_any(input_path)
    disease_cols = [c for c in df.columns if c.endswith("_disease")]
    pert_cols = [c for c in df.columns if c.endswith("_gene")]
    disease_genes = [c[: -len("_disease")] for c in disease_cols]
    pert_genes = [c[: -len("_gene")] for c in pert_cols]

    kg_nodes = set(graph.nodes)
    valid_genes = set(disease_genes) & set(pert_genes) & kg_nodes

    missing_pert_ids = set(df["pert_id"].unique()) - kg_nodes
    if missing_pert_ids:
        df = df[~df["pert_id"].isin(missing_pert_ids)].copy()
    df = df.reset_index(drop=True)

    groups: List[Tuple[str, np.ndarray]] = [
        (str(pert_id), sub.index.to_numpy())
        for pert_id, sub in df.groupby("pert_id", sort=False)
    ]

    return PreparedInferInput(
        mode=str(mode),
        initial_polarity=_initial_polarity_for_mode(str(mode)),
        disease_genes=disease_genes,
        pert_genes=pert_genes,
        disease_gene_to_pos={g: i for i, g in enumerate(disease_genes)},
        pert_gene_to_pos={g: i for i, g in enumerate(pert_genes)},
        disease_mat=_downcast_profile_matrix(df[disease_cols].to_numpy()),
        pert_mat=_downcast_profile_matrix(df[pert_cols].to_numpy()),
        valid_genes=valid_genes,
        valid_genes_digest=_valid_gene_digest(valid_genes),
        meta_arrays={c: df[c].to_numpy() for c in META_COLUMNS_TO_KEEP if c in df.columns},
        do_ids=df["do_id"].to_numpy(),
        cell_ids=df["cell_id"].to_numpy(),
        labels=df["label"].to_numpy(),
        pert_id_values=df["pert_id"].to_numpy(),
        triplet_ids=df["triplet_id"].astype(str).to_numpy() if "triplet_id" in df.columns else None,
        groups=groups,
        n_rows=int(len(df)),
        has_triplet_id="triplet_id" in df.columns,
    )


@dataclass
class TrieCache:
    """Per-side cache of folded path skeletons, keyed by source gene."""

    mode: str
    path_strategy: str
    cutoff: int
    valid_genes_digest: str
    tries: Dict[str, PathTrie] = field(default_factory=dict)
    n_built: int = 0
    n_reused: int = 0

    def stats(self) -> Dict[str, int]:
        return {
            "n_sources_cached": int(len(self.tries)),
            "n_tries_built": int(self.n_built),
            "n_tries_reused": int(self.n_reused),
            "n_trie_nodes": int(sum(t.n_nodes for t in self.tries.values())),
            "n_trie_paths": int(sum(t.n_paths for t in self.tries.values())),
        }

    def release_row_caches(self) -> None:
        for trie in self.tries.values():
            trie.release_row_caches()


def get_or_build_trie(
    cache: TrieCache,
    *,
    source: str,
    graph: nx.MultiDiGraph,
    initial_polarity: int,
    valid_genes: Set[str],
    path_cache_dir: str | Path | None,
) -> PathTrie:
    """Return the folded, cutoff/valid-gene filtered path trie for ``source``."""
    source_s = str(source)
    cached = cache.tries.get(source_s)
    if cached is not None:
        cache.n_reused += 1
        return cached

    if source_s in graph.nodes:
        path_polarities = _collect_path_polarities_cached(
            source=source_s,
            graph=graph,
            initial_polarity=int(initial_polarity),
            cutoff=int(cache.cutoff),
            mode=str(cache.mode),
            cache_dir=Path(path_cache_dir) if path_cache_dir is not None else None,
            path_strategy=str(cache.path_strategy),
            memory_cache=None,
        )
    else:
        path_polarities = []

    if path_polarities:
        cutoff = int(cache.cutoff)
        path_polarities = [
            pdict
            for pdict in path_polarities
            if len(pdict) <= cutoff and all((g in valid_genes) for g in pdict.keys())
        ]

    trie = build_path_trie(path_polarities, source=source_s, initial_polarity=int(initial_polarity))
    del path_polarities
    cache.tries[source_s] = trie
    cache.n_built += 1
    return trie


def _row_signed_edges(
    trie: PathTrie,
    accepted: List[Tuple[int, int]],
    *,
    include_first_edge_excluded: bool,
) -> Tuple[int, Set[Tuple[str, str, int]], Set[Tuple[str, str, int]]]:
    """Prune prefix-dominated accepted paths and collect their signed edges."""
    if not accepted:
        return 0, set(), set()
    ordered = accepted_paths_in_reference_order(accepted)
    node_by_index = {path_index: node for path_index, node in ordered}
    pruned = prune_prefix_paths_with_idx([(trie.node_names(node), path_index) for path_index, node in ordered])
    signed_edges: Set[Tuple[str, str, int]] = set()
    excluded_edges: Set[Tuple[str, str, int]] = set()
    for _, path_index in pruned:
        chain = trie.node_edges(node_by_index[path_index])
        signed_edges.update(chain)
        if include_first_edge_excluded and len(chain) > 1:
            excluded_edges.update(chain[1:])
    return len(pruned), signed_edges, excluded_edges


def iteration_permutation_blocks(
    prepared: PreparedInferInput,
    permutation_specs: Mapping[str, Any] | None,
) -> Dict[str, List[Tuple[SparsePermutationLookup | None, SparsePermutationLookup | None, List[int]]]]:
    """Per-source ``(pert lookup, disease lookup, rows)`` blocks for one iteration.

    A gene-node shuffle is either global or per triplet, so a source's rows split
    into at most one block per triplet group. Splitting here keeps the traversal
    itself free of any permutation bookkeeping.
    """
    pert_lookups = build_sparse_permutation_lookups(prepared.pert_genes, permutation_specs)
    disease_lookups = build_sparse_permutation_lookups(prepared.disease_genes, permutation_specs)
    global_pert = pert_lookups.get(GLOBAL_PERMUTATION_GROUP)
    global_disease = disease_lookups.get(GLOBAL_PERMUTATION_GROUP)
    per_triplet_mode = (
        permutation_specs is not None
        and prepared.triplet_ids is not None
        and global_pert is None
        and global_disease is None
    )

    blocks_by_source: Dict[str, List[Tuple[SparsePermutationLookup | None, SparsePermutationLookup | None, List[int]]]] = {}
    for source, row_index in prepared.groups:
        if per_triplet_mode:
            triplet_ids = prepared.triplet_ids
            subgroups: Dict[str, List[int]] = {}
            for row in row_index:
                subgroups.setdefault(str(triplet_ids[row]), []).append(int(row))
            blocks = [
                (pert_lookups.get(group), disease_lookups.get(group), rows)
                for group, rows in subgroups.items()
            ]
        else:
            blocks = [(global_pert, global_disease, [int(r) for r in row_index])]
        blocks_by_source[str(source)] = blocks
    return blocks_by_source


def assemble_signed_edge_frame(
    prepared: PreparedInferInput,
    *,
    n_paths_out: np.ndarray,
    num_nodes_out: np.ndarray,
    edges_out: Sequence[Set[Tuple[str, str, int]]],
    excluded_out: Sequence[Set[Tuple[str, str, int]]],
    include_first_edge_excluded: bool,
) -> pd.DataFrame:
    """Emit the per-row traversal results as a Val3 signed-edge frame.

    ``signature_reversal_paths_from_graph`` appends results group by group, so the
    emitted frame is group-major rather than input-row-major. Keep that order.
    """
    order = prepared.group_major_order
    out = pd.DataFrame(
        {
            "pert_id": prepared.pert_id_values[order],
            "pert_cell": prepared.cell_ids[order],
            "disease": prepared.do_ids[order],
            "label": prepared.labels[order],
            "num_nodes_in_paths": num_nodes_out[order],
        }
    )
    for column, values in prepared.meta_arrays.items():
        out[column] = values[order]
    out["signed_edges"] = [edges_out[i] or set() for i in order]
    out["n_paths"] = n_paths_out[order]
    if include_first_edge_excluded:
        from .metrics import FIRST_EDGE_EXCLUDED_SUFFIX

        out["signed_edges" + FIRST_EDGE_EXCLUDED_SUFFIX] = [excluded_out[i] or set() for i in order]
    if "cell_id" not in out.columns:
        out["cell_id"] = out["pert_cell"]
    return out


def null1_signed_edge_df(
    prepared: PreparedInferInput,
    *,
    graph: nx.MultiDiGraph,
    gap: int,
    permutation_specs: Mapping[str, Any] | None,
    trie_cache: TrieCache,
    path_cache_dir: str | Path | None,
    include_first_edge_excluded: bool = False,
) -> pd.DataFrame:
    """Signed-edge rows for one Null 1 iteration (or the observed run).

    Returns the same columns that ``signed_edge_df_from_infer_df`` produces for
    Val3, minus the intermediate path-list columns that nothing downstream reads.
    """
    blocks_by_source = iteration_permutation_blocks(prepared, permutation_specs)

    disease_mat = prepared.disease_mat
    pert_mat = prepared.pert_mat
    n_paths_out = np.zeros(prepared.n_rows, dtype=np.int64)
    edges_out: List[Set[Tuple[str, str, int]]] = [set()] * prepared.n_rows
    excluded_out: List[Set[Tuple[str, str, int]]] = [set()] * prepared.n_rows
    num_nodes_out = np.zeros(prepared.n_rows, dtype=np.int64)

    for source, row_index in prepared.groups:
        trie = get_or_build_trie(
            trie_cache,
            source=source,
            graph=graph,
            initial_polarity=prepared.initial_polarity,
            valid_genes=prepared.valid_genes,
            path_cache_dir=path_cache_dir,
        )
        num_nodes_out[row_index] = trie.num_nodes_in_paths
        if trie.n_paths == 0 or trie.n_nodes <= 1:
            continue

        base_pert_pos = [prepared.pert_gene_to_pos[g] for g in trie.genes]
        base_disease_pos = [prepared.disease_gene_to_pos[g] for g in trie.genes]

        for pert_lookup, disease_lookup, rows in blocks_by_source[str(source)]:
            pert_pos = pert_lookup.gather(base_pert_pos) if pert_lookup is not None else base_pert_pos
            disease_pos = disease_lookup.gather(base_disease_pos) if disease_lookup is not None else base_disease_pos
            pert_block = pert_mat[np.ix_(rows, pert_pos)]
            disease_block = disease_mat[np.ix_(rows, disease_pos)]
            for k, row in enumerate(rows):
                accepted = evaluate_trie_row(
                    trie,
                    pert_block[k].tolist(),
                    disease_block[k].tolist(),
                    gap=int(gap),
                )
                n_paths, signed_edges, excluded_edges = _row_signed_edges(
                    trie, accepted, include_first_edge_excluded=include_first_edge_excluded,
                )
                n_paths_out[row] = n_paths
                edges_out[row] = signed_edges
                excluded_out[row] = excluded_edges

    return assemble_signed_edge_frame(
        prepared,
        n_paths_out=n_paths_out,
        num_nodes_out=num_nodes_out,
        edges_out=edges_out,
        excluded_out=excluded_out,
        include_first_edge_excluded=include_first_edge_excluded,
    )
