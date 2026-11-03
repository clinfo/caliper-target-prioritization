"""Fork-friendly parallel map helper for the directed target-reach matched
STEP2+STEP4 pipeline (per-target label-free scoring).

Large read-only inputs (the directed KG, the NCS reference matrix, disease
signatures, ...) are placed in a module-level global by the caller's
``initializer`` *before* the pool is created. With the default ``fork`` start
method, worker processes inherit that state via copy-on-write memory without
any pickling, so only the (small) per-task key and per-task result dict are
ever sent through the process pool's queues.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, List, Tuple, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def resolve_n_jobs(n_jobs: int) -> int:
    n = int(n_jobs)
    if n < 1:
        raise ValueError("step4_n_jobs must be >= 1")
    cpu = os.cpu_count() or 1
    return min(n, cpu)


def get_mp_context(start_method: str):
    method = str(start_method).strip().lower()
    if method not in {"fork", "spawn", "forkserver"}:
        raise ValueError("mp_start_method must be fork, spawn, or forkserver")
    available = mp.get_all_start_methods()
    if method not in available:
        raise ValueError(
            "multiprocessing start method "
            f"{method!r} is not available on this platform; available={available}"
        )
    return mp.get_context(method)


def parallel_map_ordered(
    func: Callable[[T], R],
    tasks: List[T],
    *,
    n_jobs: int,
    start_method: str = "fork",
    initializer: Callable | None = None,
    initargs: Tuple[Any, ...] = (),
    progress: Callable | None = None,
    desc: str = "tasks",
) -> List[R]:
    """Map ``func`` over ``tasks``, preserving input order in the output.

    ``n_jobs == 1`` runs sequentially in the current process (after calling
    ``initializer(*initargs)`` once, matching the parallel path's setup) so
    results are identical regardless of ``n_jobs`` -- only wall-clock time
    changes. ``progress``, when given, wraps the iterator the same way
    ``tqdm`` does (``progress(iterable, total=..., desc=...)``).
    """
    if not tasks:
        return []
    n_jobs_resolved = resolve_n_jobs(int(n_jobs))

    def _wrap(it):
        return progress(it, total=len(tasks), desc=desc) if progress is not None else it

    if n_jobs_resolved == 1:
        if initializer is not None:
            initializer(*initargs)
        return list(_wrap(map(func, tasks)))

    ctx = get_mp_context(start_method)
    with ProcessPoolExecutor(
        max_workers=n_jobs_resolved,
        mp_context=ctx,
        initializer=initializer,
        initargs=initargs,
    ) as ex:
        return list(_wrap(ex.map(func, tasks)))
