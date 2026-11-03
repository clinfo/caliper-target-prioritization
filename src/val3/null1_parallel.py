"""Source-parallel Null 1 traversal (spec §6.9, C8).

``null1_signed_edge_df`` walks ``PreparedInferInput.groups`` one source at a time
and the per-row trie traversal is pure Python, so one iteration is bound by a
single core. The groups are independent: each reads its own rows of the profile
matrices, builds or reuses the trie of its own source, and writes only to its own
row slots. This module runs those groups in worker processes.

What the workers must not do (spec §6.9):

* consume RNG -- permutation specs are built in the parent and shipped as position
  remaps;
* hold a private copy of the wide profile input -- the int8 matrices are published
  once into read-only mmap files that every worker attaches to;
* write any artifact -- workers return row payloads and counters only, and the
  parent assembles the frame and writes the debug shards exactly as before.

Completion order therefore never reaches an output: results are scattered back by
absolute row index and emitted through ``PreparedInferInput.group_major_order``,
so the frame is identical to the serial route for any worker count.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, Tuple

import numpy as np
import pandas as pd

from inference.prefix_trie_traversal import evaluate_trie_row

from .null1_engine import (
    PreparedInferInput,
    SparsePermutationLookup,
    TrieCache,
    _row_signed_edges,
    assemble_signed_edge_frame,
    get_or_build_trie,
    iteration_permutation_blocks,
)

SignedEdgeSet = Set[Tuple[str, str, int]]


# ---------------------------------------------------------------------------
# read-only profile matrices
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _MatrixRef:
    """Locator for one profile matrix published as a read-only mmap."""

    path: str
    dtype: str
    shape: Tuple[int, int]

    def attach(self) -> np.ndarray:
        if self.shape[0] == 0 or self.shape[1] == 0:
            return np.zeros(self.shape, dtype=np.dtype(self.dtype))
        return np.memmap(self.path, dtype=np.dtype(self.dtype), mode="r", shape=self.shape)


def _publish_matrix(mat: np.ndarray, path: Path) -> _MatrixRef:
    arr = np.ascontiguousarray(mat)
    if arr.ndim != 2:
        raise ValueError(f"Profile matrix must be 2-D, got shape {arr.shape}.")
    if arr.size:
        with path.open("wb") as handle:
            arr.tofile(handle)
    return _MatrixRef(path=str(path), dtype=arr.dtype.str, shape=(int(arr.shape[0]), int(arr.shape[1])))


def default_scratch_root() -> str | None:
    """Prefer tmpfs so the published matrices never hit a real disk."""
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        return str(shm)
    return None


def _worker_context() -> Any:
    """Start workers from a single-threaded ancestor.

    The parent has already run thread pools by the time the baseline loop starts,
    and forking a multi-threaded process can deadlock a child on a lock that no
    longer has an owner. ``forkserver`` forks from a clean helper instead, which
    is also why the profile matrices have to be published rather than inherited.
    """
    for method in ("forkserver", "spawn"):
        try:
            return get_context(method)
        except ValueError:
            continue
    return get_context()


# ---------------------------------------------------------------------------
# wire formats
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PermutationWire:
    """A ``SparsePermutationLookup`` as two int32 arrays.

    The moved-entry dict pickles roughly an order of magnitude larger than the
    equivalent arrays, and a gene-node shuffle ships one lookup per triplet group
    per iteration.
    """

    targets: np.ndarray
    sources: np.ndarray

    @classmethod
    def of(cls, lookup: SparsePermutationLookup | None) -> "_PermutationWire | None":
        if lookup is None:
            return None
        pairs = lookup.items()
        if not pairs:
            return cls(targets=np.zeros(0, dtype=np.int32), sources=np.zeros(0, dtype=np.int32))
        targets, sources = zip(*pairs)
        return cls(
            targets=np.asarray(targets, dtype=np.int32),
            sources=np.asarray(sources, dtype=np.int32),
        )

    def to_lookup(self, n_genes: int) -> SparsePermutationLookup:
        mapping = dict(zip(self.targets.tolist(), self.sources.tolist()))
        return SparsePermutationLookup(mapping, n_genes=int(n_genes))


@dataclass(frozen=True)
class _SideSpec:
    """Everything a worker needs to rebuild one side's traversal state."""

    mode: str
    initial_polarity: int
    cutoff: int
    gap: int
    path_strategy: str
    path_cache_dir: str | None
    valid_genes: Set[str]
    valid_genes_digest: str
    pert_gene_to_pos: Dict[str, int]
    disease_gene_to_pos: Dict[str, int]
    n_pert_genes: int
    n_disease_genes: int
    pert_matrix: _MatrixRef
    disease_matrix: _MatrixRef


@dataclass(frozen=True)
class _SourceTask:
    side: str
    source: str
    blocks: Tuple[Tuple["_PermutationWire | None", "_PermutationWire | None", np.ndarray], ...]
    include_first_edge_excluded: bool


@dataclass
class _SourceResult:
    side: str
    source: str
    num_nodes_in_paths: int
    rows: np.ndarray
    n_paths: np.ndarray
    signed_edges: List[SignedEdgeSet]
    excluded_edges: List[SignedEdgeSet]
    elapsed_sec: float


# ---------------------------------------------------------------------------
# worker side
# ---------------------------------------------------------------------------

class _WorkerSide:
    __slots__ = ("spec", "pert_mat", "disease_mat", "trie_cache")

    def __init__(self, spec: _SideSpec) -> None:
        self.spec = spec
        self.pert_mat = spec.pert_matrix.attach()
        self.disease_mat = spec.disease_matrix.attach()
        self.trie_cache = TrieCache(
            mode=spec.mode,
            path_strategy=spec.path_strategy,
            cutoff=spec.cutoff,
            valid_genes_digest=spec.valid_genes_digest,
        )


@dataclass
class _WorkerState:
    graph: Any
    sides: Dict[str, _WorkerSide] = field(default_factory=dict)


_WORKER: _WorkerState | None = None


def _init_worker(graph: Any, side_specs: Mapping[str, _SideSpec]) -> None:
    global _WORKER
    _WORKER = _WorkerState(graph=graph, sides={side: _WorkerSide(spec) for side, spec in side_specs.items()})


def _empty_result(task: _SourceTask, *, num_nodes: int, elapsed: float) -> _SourceResult:
    return _SourceResult(
        side=task.side,
        source=task.source,
        num_nodes_in_paths=int(num_nodes),
        rows=np.zeros(0, dtype=np.int64),
        n_paths=np.zeros(0, dtype=np.int64),
        signed_edges=[],
        excluded_edges=[],
        elapsed_sec=float(elapsed),
    )


def _run_source_task(task: _SourceTask) -> _SourceResult:
    state = _WORKER
    if state is None:
        raise RuntimeError("Null 1 worker was not initialized.")
    side = state.sides[task.side]
    spec = side.spec
    started = time.perf_counter()

    trie = get_or_build_trie(
        side.trie_cache,
        source=task.source,
        graph=state.graph,
        initial_polarity=spec.initial_polarity,
        valid_genes=spec.valid_genes,
        path_cache_dir=spec.path_cache_dir,
    )
    if trie.n_paths == 0 or trie.n_nodes <= 1:
        return _empty_result(task, num_nodes=trie.num_nodes_in_paths, elapsed=time.perf_counter() - started)

    base_pert_pos = [spec.pert_gene_to_pos[g] for g in trie.genes]
    base_disease_pos = [spec.disease_gene_to_pos[g] for g in trie.genes]

    out_rows: List[int] = []
    out_n_paths: List[int] = []
    out_edges: List[SignedEdgeSet] = []
    out_excluded: List[SignedEdgeSet] = []
    for pert_wire, disease_wire, rows in task.blocks:
        pert_pos = (
            pert_wire.to_lookup(spec.n_pert_genes).gather(base_pert_pos)
            if pert_wire is not None
            else base_pert_pos
        )
        disease_pos = (
            disease_wire.to_lookup(spec.n_disease_genes).gather(base_disease_pos)
            if disease_wire is not None
            else base_disease_pos
        )
        row_list = rows.tolist()
        pert_block = side.pert_mat[np.ix_(row_list, pert_pos)]
        disease_block = side.disease_mat[np.ix_(row_list, disease_pos)]
        for k, row in enumerate(row_list):
            accepted = evaluate_trie_row(
                trie,
                pert_block[k].tolist(),
                disease_block[k].tolist(),
                gap=spec.gap,
            )
            if not accepted:
                continue
            n_paths, signed_edges, excluded_edges = _row_signed_edges(
                trie, accepted, include_first_edge_excluded=task.include_first_edge_excluded,
            )
            out_rows.append(int(row))
            out_n_paths.append(int(n_paths))
            out_edges.append(signed_edges)
            out_excluded.append(excluded_edges)

    # Only the fold survives between iterations; the per-node name/edge caches are
    # row-driven and would otherwise grow without bound in a long-lived worker.
    trie.release_row_caches()
    return _SourceResult(
        side=task.side,
        source=task.source,
        num_nodes_in_paths=int(trie.num_nodes_in_paths),
        rows=np.asarray(out_rows, dtype=np.int64),
        n_paths=np.asarray(out_n_paths, dtype=np.int64),
        signed_edges=out_edges,
        excluded_edges=out_excluded,
        elapsed_sec=float(time.perf_counter() - started),
    )


# ---------------------------------------------------------------------------
# parent side
# ---------------------------------------------------------------------------

class _RowBuffers:
    __slots__ = ("n_paths", "num_nodes", "edges", "excluded")

    def __init__(self, n_rows: int) -> None:
        self.n_paths = np.zeros(n_rows, dtype=np.int64)
        self.num_nodes = np.zeros(n_rows, dtype=np.int64)
        self.edges: List[SignedEdgeSet] = [set()] * n_rows
        self.excluded: List[SignedEdgeSet] = [set()] * n_rows


class Null1ParallelEvaluator:
    """Persistent worker pool that evaluates Null 1 source groups off the parent.

    One instance covers every iteration of a direction: the pool, the published
    matrices and the per-worker trie caches are built once and reused, so repeated
    iterations pay only for the permutation remap and the traversal itself.
    """

    def __init__(
        self,
        *,
        sides: Mapping[str, PreparedInferInput],
        graph: Any,
        gap: int,
        cutoff: int,
        path_strategy: str,
        path_cache_dir: str | Path | None,
        max_workers: int,
        scratch_root: str | Path | None = None,
    ) -> None:
        if int(max_workers) < 2:
            raise ValueError("Null1ParallelEvaluator requires max_workers >= 2.")
        self._prepared = dict(sides)
        self._max_workers = int(max_workers)
        root = scratch_root if scratch_root is not None else default_scratch_root()
        self._scratch = Path(tempfile.mkdtemp(prefix="val3_null1_", dir=str(root) if root is not None else None))
        self._group_rows: Dict[str, Dict[str, np.ndarray]] = {}
        self._cost: Dict[Tuple[str, str], float] = {}

        side_specs: Dict[str, _SideSpec] = {}
        for side, prepared in self._prepared.items():
            side_specs[side] = _SideSpec(
                mode=prepared.mode,
                initial_polarity=int(prepared.initial_polarity),
                cutoff=int(cutoff),
                gap=int(gap),
                path_strategy=str(path_strategy),
                path_cache_dir=str(path_cache_dir) if path_cache_dir is not None else None,
                valid_genes=prepared.valid_genes,
                valid_genes_digest=prepared.valid_genes_digest,
                pert_gene_to_pos=prepared.pert_gene_to_pos,
                disease_gene_to_pos=prepared.disease_gene_to_pos,
                n_pert_genes=len(prepared.pert_genes),
                n_disease_genes=len(prepared.disease_genes),
                pert_matrix=_publish_matrix(prepared.pert_mat, self._scratch / f"{side}.pert.bin"),
                disease_matrix=_publish_matrix(prepared.disease_mat, self._scratch / f"{side}.disease.bin"),
            )
            self._group_rows[side] = {
                str(source): np.asarray(rows, dtype=np.int64) for source, rows in prepared.groups
            }

        try:
            self._executor = ProcessPoolExecutor(
                max_workers=self._max_workers,
                mp_context=_worker_context(),
                initializer=_init_worker,
                initargs=(graph, side_specs),
            )
        except BaseException:
            shutil.rmtree(self._scratch, ignore_errors=True)
            raise

    def signed_edge_frames(
        self,
        *,
        permutation_specs: Mapping[str, Any] | None,
        include_first_edge_excluded: bool,
    ) -> Dict[str, pd.DataFrame]:
        """Run one iteration across all sides and return one frame per side."""
        tasks: List[_SourceTask] = []
        for side, prepared in self._prepared.items():
            blocks_by_source = iteration_permutation_blocks(prepared, permutation_specs)
            for source, blocks in blocks_by_source.items():
                tasks.append(
                    _SourceTask(
                        side=side,
                        source=source,
                        blocks=tuple(
                            (
                                _PermutationWire.of(pert_lookup),
                                _PermutationWire.of(disease_lookup),
                                np.asarray(rows, dtype=np.int64),
                            )
                            for pert_lookup, disease_lookup, rows in blocks
                        ),
                        include_first_edge_excluded=bool(include_first_edge_excluded),
                    )
                )

        # Longest-processing-time-first from the previous iteration's measurements.
        # Unmeasured sources keep their submission order on the first iteration.
        tasks.sort(key=lambda task: -self._cost.get((task.side, task.source), float("inf")))

        buffers = {side: _RowBuffers(prepared.n_rows) for side, prepared in self._prepared.items()}
        futures: List[Future[_SourceResult]] = [self._executor.submit(_run_source_task, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            buffer = buffers[result.side]
            buffer.num_nodes[self._group_rows[result.side][result.source]] = result.num_nodes_in_paths
            rows = result.rows
            if rows.size:
                buffer.n_paths[rows] = result.n_paths
                for position, row in enumerate(rows.tolist()):
                    buffer.edges[row] = result.signed_edges[position]
                    buffer.excluded[row] = result.excluded_edges[position]
            self._cost[(result.side, result.source)] = result.elapsed_sec

        return {
            side: assemble_signed_edge_frame(
                prepared,
                n_paths_out=buffers[side].n_paths,
                num_nodes_out=buffers[side].num_nodes,
                edges_out=buffers[side].edges,
                excluded_out=buffers[side].excluded,
                include_first_edge_excluded=bool(include_first_edge_excluded),
            )
            for side, prepared in self._prepared.items()
        }

    def close(self) -> None:
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
            self._executor = None
        shutil.rmtree(self._scratch, ignore_errors=True)

    def __enter__(self) -> "Null1ParallelEvaluator":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
