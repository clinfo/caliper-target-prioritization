from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import shutil
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Iterator, Sequence, TypeVar

import pandas as pd
import yaml

from .io import ensure_dir, read_parquet_any

_T = TypeVar("_T")
_R = TypeVar("_R")


def infer_output_path(output_dir: str | Path, *, mode: str, cutoff: int, gap: int) -> Path:
    return Path(output_dir) / mode / f"cutoff={int(cutoff)}" / f"signature_reversal_paths_gap_{int(gap)}.parquet"


def write_temporary_infer_config(
    *,
    mode: str,
    cutoff: int,
    gap: int,
    input_parquet: str | Path,
    kg_tsv: str | Path,
    output_dir: str | Path,
    permutation_specs: str | Path | None = None,
    path_cache_dir: str | Path | None = None,
    path_strategy: str = "simple",
) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    cfg = {
        "mode": mode,
        "cutoff": int(cutoff),
        "gap": int(gap),
        "input_parquet": str(input_parquet),
        "kg_tsv": str(kg_tsv),
        "output_dir": str(output_dir),
        "path_strategy": str(path_strategy),
    }
    if permutation_specs is not None:
        cfg["permutation_specs"] = str(permutation_specs)
    if path_cache_dir is not None:
        cfg["path_cache_dir"] = str(path_cache_dir)
    yaml.safe_dump(cfg, tmp, sort_keys=False, allow_unicode=True)
    tmp.close()
    return Path(tmp.name)


def run_infer_subprocess(
    *,
    repo_root: str | Path,
    mode: str,
    cutoff: int,
    gap: int,
    input_parquet: str | Path,
    kg_tsv: str | Path,
    output_dir: str | Path,
    permutation_specs: str | Path | None = None,
    path_cache_dir: str | Path | None = None,
    path_strategy: str = "simple",
    log_dir: str | Path | None = None,
    log_label: str | None = None,
) -> Path:
    output_dir = ensure_dir(output_dir)
    cfg_path = write_temporary_infer_config(
        mode=mode,
        cutoff=cutoff,
        gap=gap,
        input_parquet=input_parquet,
        kg_tsv=kg_tsv,
        output_dir=output_dir,
        permutation_specs=permutation_specs,
        path_cache_dir=path_cache_dir,
        path_strategy=path_strategy,
    )
    try:
        cmd = [sys.executable, str(Path(repo_root) / "src" / "infer_signature_reversal_paths.py"), "--config", str(cfg_path)]
        if log_dir is None:
            subprocess.run(cmd, check=True)
        else:
            logs = ensure_dir(log_dir)
            label = log_label or Path(output_dir).name
            safe_label = "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in label)
            stdout_path = logs / f"{safe_label}.stdout.log"
            stderr_path = logs / f"{safe_label}.stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
                subprocess.run(cmd, check=True, stdout=stdout_f, stderr=stderr_f)
    finally:
        cfg_path.unlink(missing_ok=True)
    out = infer_output_path(output_dir, mode=mode, cutoff=cutoff, gap=gap)
    if not out.exists():
        raise FileNotFoundError(f"infer output was not generated: {out}")
    return out


def iter_pert_id_group_batches(
    input_parquet: str | Path,
    *,
    target_batches: int,
    kg_nodes: Collection[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Cut one wide infer input into contiguous ``pert_id`` group batches.

    ``signature_reversal_paths_from_graph`` walks ``groupby("pert_id", sort=False)``
    and appends its results group by group, and everything it derives outside that
    loop -- the valid gene set, the gene column positions -- comes from the columns
    rather than from the rows. Cutting the rows along contiguous runs of *whole*
    groups therefore leaves both the per-row results and the group-major output order
    untouched: batch ``i`` covers exactly the groups a single-process run would have
    reached between the end of batch ``i-1`` and the start of batch ``i+1``, so
    concatenating the parts in name order reproduces the one-file output row for row.

    ``target_batches`` is approximate. Batches close on a row budget rather than on a
    group count because the group sizes are heavily skewed, and a group is never split
    even when it alone exceeds the budget, so a few hub sources can push the realized
    batch count above the target.

    Every batch costs a few seconds of fixed overhead: the wide profile parquet carries
    one column chunk per gene, so writing it takes ~3 s and ~17 MB whatever the row
    count, and the subprocess then re-reads it and rebuilds the KG. Batching pays off
    when the traversal work per batch clears that, which is the case for a full
    observed side (tens of minutes of work) but not for a handful of rows.

    ``kg_nodes`` drops the groups whose source is absent from the KG. The infer run
    discards exactly those rows itself, so this does not change the output -- it keeps a
    batch from consisting *only* of them, which would leave a column-less part in a
    dataset whose other parts carry the full schema.
    """
    df = read_parquet_any(input_parquet)
    if "pert_id" not in df.columns:
        raise ValueError(f"Infer input must contain pert_id to be split into batches: {input_parquet}")
    known = None if kg_nodes is None else {str(node) for node in kg_nodes}
    budget = max(1, math.ceil(len(df) / max(1, int(target_batches))))
    pending: list[pd.DataFrame] = []
    pending_rows = 0
    for pert_id, sub in df.groupby("pert_id", sort=False):
        if known is not None and str(pert_id) not in known:
            continue
        pending.append(sub)
        pending_rows += len(sub)
        if pending_rows >= budget:
            yield pending[0] if len(pending) == 1 else pd.concat(pending)
            pending, pending_rows = [], 0
    if pending:
        yield pending[0] if len(pending) == 1 else pd.concat(pending)


def _map_bounded(
    tasks: Iterable[_T],
    run_one: Callable[[_T], _R],
    *,
    max_workers: int,
) -> Iterator[_R]:
    """Run ``run_one`` over ``tasks`` with at most ``max_workers`` in flight.

    ``tasks`` is pulled lazily so that only the in-flight batches are materialized;
    the decoy input batches are wide frames and holding all of them would defeat
    the batching itself. Results are yielded in completion order, which callers
    must not depend on for output naming.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending: set[Future[_R]] = set()
        for task in tasks:
            while len(pending) >= max_workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
            pending.add(executor.submit(run_one, task))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()


@dataclass(frozen=True)
class _BatchTarget:
    """Where the parts of one batched infer stream are written."""

    name: str
    mode: str
    dataset_dir: Path
    tmp_root: Path
    log_label: str


def _batch_target(
    name: str,
    *,
    mode: str,
    output_dir: str | Path,
    cutoff: int,
    gap: int,
    log_label: str | None,
) -> _BatchTarget:
    output_dir = ensure_dir(output_dir)
    dataset_dir = ensure_dir(infer_output_path(output_dir, mode=mode, cutoff=cutoff, gap=gap).with_suffix(""))
    for old_part in dataset_dir.glob("part-*.parquet"):
        old_part.unlink()
    return _BatchTarget(
        name=str(name),
        mode=str(mode),
        dataset_dir=dataset_dir,
        tmp_root=ensure_dir(output_dir / "_tmp_batches"),
        log_label=str(log_label or "infer"),
    )


_BatchTask = tuple[_BatchTarget, int, pd.DataFrame]


def _run_infer_batch(
    target: _BatchTarget,
    idx: int,
    batch: pd.DataFrame,
    *,
    repo_root: str | Path,
    cutoff: int,
    gap: int,
    kg_tsv: str | Path,
    path_cache_dir: str | Path | None,
    path_strategy: str,
    log_dir: str | Path | None,
) -> str:
    batch_input = target.tmp_root / f"input-{idx:06d}.parquet"
    batch_output = target.tmp_root / f"output-{idx:06d}"
    batch.to_parquet(batch_input, index=False)
    out = run_infer_subprocess(
        repo_root=repo_root,
        mode=target.mode,
        cutoff=cutoff,
        gap=gap,
        input_parquet=batch_input,
        kg_tsv=kg_tsv,
        output_dir=batch_output,
        path_cache_dir=path_cache_dir,
        path_strategy=path_strategy,
        log_dir=log_dir,
        log_label=f"{target.log_label}-part-{idx:06d}",
    )
    out.replace(target.dataset_dir / f"part-{idx:06d}.parquet")
    batch_input.unlink(missing_ok=True)
    shutil.rmtree(batch_output, ignore_errors=True)
    return target.name


def _drain_infer_batches(
    tasks: Iterable[_BatchTask],
    run_one: Callable[[_BatchTask], str],
    *,
    targets: Sequence[_BatchTarget],
    max_workers: int,
) -> dict[str, int]:
    workers = max(1, int(max_workers))
    completed = {target.name: 0 for target in targets}
    results = map(run_one, tasks) if workers == 1 else _map_bounded(tasks, run_one, max_workers=workers)
    for name in results:
        completed[name] += 1
    for target in targets:
        # A stream that yielded nothing still has to leave a readable parquet dataset.
        if completed[target.name] == 0:
            pd.DataFrame().to_parquet(target.dataset_dir / "part-000000.parquet", index=False)
        shutil.rmtree(target.tmp_root, ignore_errors=True)
    return completed


def run_infer_dataframe_batches(
    *,
    repo_root: str | Path,
    mode: str,
    cutoff: int,
    gap: int,
    input_batches: Iterable[pd.DataFrame],
    kg_tsv: str | Path,
    output_dir: str | Path,
    path_cache_dir: str | Path | None = None,
    path_strategy: str = "simple",
    log_dir: str | Path | None = None,
    log_label: str | None = None,
    max_workers: int = 1,
) -> Path:
    """Run infer on small in-memory batches and expose a parquet dataset output.

    Batches are independent: each one writes ``part-{idx}.parquet`` under a name
    derived from its position in ``input_batches``, and the shared path-polarity
    cache is written through a pid-suffixed temporary file. Raising ``max_workers``
    therefore changes only the completion order, not the dataset contents.
    """
    target = _batch_target(
        "infer", mode=mode, output_dir=output_dir, cutoff=cutoff, gap=gap, log_label=log_label,
    )

    def _run_one(task: _BatchTask) -> str:
        return _run_infer_batch(
            *task,
            repo_root=repo_root,
            cutoff=cutoff,
            gap=gap,
            kg_tsv=kg_tsv,
            path_cache_dir=path_cache_dir,
            path_strategy=path_strategy,
            log_dir=log_dir,
        )

    tasks = ((target, idx, batch) for idx, batch in enumerate(input_batches, start=1) if not batch.empty)
    _drain_infer_batches(tasks, _run_one, targets=[target], max_workers=max_workers)
    return target.dataset_dir


def run_infer_paired_dataframe_batches(
    *,
    repo_root: str | Path,
    cutoff: int,
    gap: int,
    input_batches: Iterable[tuple[pd.DataFrame, pd.DataFrame]],
    drug_mode: str,
    pert_mode: str,
    drug_output_dir: str | Path,
    pert_output_dir: str | Path,
    kg_tsv: str | Path,
    path_cache_dir: str | Path | None = None,
    path_strategy: str = "simple",
    log_dir: str | Path | None = None,
    log_label: str | None = None,
    max_workers: int = 1,
) -> tuple[Path, Path]:
    """Infer both sides of a paired batch stream through one bounded pool.

    The drug and pert input of a batch are built in the same call
    (``build_decoy_inputs`` returns both), so a caller that needs both sides consumes
    the pair here instead of iterating the stream once per side and throwing the other
    half away each time. The two sides also stop waiting on each other: ``max_workers``
    bounds the in-flight subprocesses across both sides together, which keeps the pool
    busy through what used to be a barrier between the drug pass and the pert pass.

    Part names still come from the batch position on both sides, so drug part *i* and
    pert part *i* stay the two halves of one batch for ``_paired_infer_part_paths``.
    """
    label = str(log_label or "infer")
    drug = _batch_target(
        "drug", mode=drug_mode, output_dir=drug_output_dir, cutoff=cutoff, gap=gap,
        log_label=f"{label}-drug",
    )
    pert = _batch_target(
        "pert", mode=pert_mode, output_dir=pert_output_dir, cutoff=cutoff, gap=gap,
        log_label=f"{label}-pert",
    )

    def _run_one(task: _BatchTask) -> str:
        return _run_infer_batch(
            *task,
            repo_root=repo_root,
            cutoff=cutoff,
            gap=gap,
            kg_tsv=kg_tsv,
            path_cache_dir=path_cache_dir,
            path_strategy=path_strategy,
            log_dir=log_dir,
        )

    def _tasks() -> Iterator[_BatchTask]:
        for idx, (drug_batch, pert_batch) in enumerate(input_batches, start=1):
            for target, batch in ((drug, drug_batch), (pert, pert_batch)):
                if not batch.empty:
                    yield target, idx, batch

    _drain_infer_batches(_tasks(), _run_one, targets=[drug, pert], max_workers=max_workers)
    return drug.dataset_dir, pert.dataset_dir
