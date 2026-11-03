from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")
R = TypeVar("R")


def resolve_n_jobs(n_jobs: int) -> int:
    n = int(n_jobs)
    if n < 1:
        raise ValueError("n_jobs must be >= 1")
    cpu = os.cpu_count() or 1
    return min(n, cpu)


def get_mp_context(start_method: str):
    method = str(start_method).strip().lower()
    if method not in {"fork", "spawn", "forkserver"}:
        raise ValueError("start_method must be fork, spawn, or forkserver")
    available = mp.get_all_start_methods()
    if method not in available:
        raise ValueError(
            f"multiprocessing start method {method!r} is not available on this platform; available={available}"
        )
    return mp.get_context(method)


def parallel_map_ordered(
    func: Callable[[T], R],
    tasks: list[T],
    *,
    n_jobs: int,
    start_method: str,
    chunksize: int = 1,
    initializer: Callable | None = None,
    initargs: tuple = (),
    progress: bool = True,
    desc: str = "tasks",
    unit: str = "task",
) -> list[R]:
    if not tasks:
        return []
    n_jobs_resolved = resolve_n_jobs(int(n_jobs))
    chunk = int(chunksize)
    if chunk < 1:
        raise ValueError("chunksize must be >= 1")
    if n_jobs_resolved == 1:
        if initializer is not None:
            initializer(*initargs)
        iterator = map(func, tasks)
        return list(
            tqdm(
                iterator,
                total=len(tasks),
                desc=desc,
                unit=unit,
                dynamic_ncols=True,
                disable=not progress,
            )
        )

    ctx = get_mp_context(start_method)
    with ProcessPoolExecutor(
        max_workers=n_jobs_resolved,
        mp_context=ctx,
        initializer=initializer,
        initargs=initargs,
    ) as ex:
        iterator = ex.map(func, tasks, chunksize=chunk)
        return list(
            tqdm(
                iterator,
                total=len(tasks),
                desc=desc,
                unit=unit,
                dynamic_ncols=True,
                disable=not progress,
            )
        )



def parallel_map_completed(
    func: Callable[[T], R],
    tasks: list[T],
    *,
    n_jobs: int,
    start_method: str,
    chunksize: int = 1,
    initializer: Callable | None = None,
    initargs: tuple = (),
    progress: bool = True,
    desc: str = "tasks",
    unit: str = "task",
    on_result: Callable[[R, int, int], None] | None = None,
    on_error: Callable[[BaseException, int, int], None] | None = None,
) -> list[R]:
    """Run tasks and update progress in completion order.

    The returned list intentionally follows completion order. Callers that need
    deterministic output ordering must sort by their stable task identifiers.
    """
    if not tasks:
        return []
    n_jobs_resolved = resolve_n_jobs(int(n_jobs))
    chunk = int(chunksize)
    if chunk < 1:
        raise ValueError("chunksize must be >= 1")
    total = len(tasks)
    results: list[R] = []

    if n_jobs_resolved == 1:
        if initializer is not None:
            initializer(*initargs)
        completed = 0
        with tqdm(
            total=total,
            desc=desc,
            unit=unit,
            dynamic_ncols=True,
            disable=not progress,
        ) as bar:
            for task in tasks:
                try:
                    result = func(task)
                except BaseException as exc:
                    if on_error is not None:
                        on_error(exc, completed, total)
                    raise
                results.append(result)
                completed += 1
                bar.update(1)
                if on_result is not None:
                    on_result(result, completed, total)
        return results

    ctx = get_mp_context(start_method)
    with ProcessPoolExecutor(
        max_workers=n_jobs_resolved,
        mp_context=ctx,
        initializer=initializer,
        initargs=initargs,
    ) as ex:
        futures = [ex.submit(func, task) for task in tasks]
        completed = 0
        with tqdm(
            total=total,
            desc=desc,
            unit=unit,
            dynamic_ncols=True,
            disable=not progress,
        ) as bar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                except BaseException as exc:
                    if on_error is not None:
                        on_error(exc, completed, total)
                    raise
                results.append(result)
                completed += 1
                bar.update(1)
                if on_result is not None:
                    on_result(result, completed, total)
    return results
