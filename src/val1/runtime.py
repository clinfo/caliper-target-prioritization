"""Path-enumeration complexity benchmark for Val1 (spec_val1_path_benchmark_v1.0).

Purpose: show that candidate-path enumeration explodes as the search distance
``k`` grows, using quantities that do not depend on the machine or the library
version. The primary metrics are therefore deterministic -- the number of simple
paths (``n_paths``), the number of reachable nodes (``N_k``), and the fraction of
targets that stay inside the path budget (``within_budget_rate``). Wall-clock
phase timings and peak RSS are recorded to satisfy P0-8 reporting, but they never
support a claim.

Enumeration is truncated by the deterministic path budget ``max_paths``; the
wall-clock ``timeout_sec`` is only a safety net so a run cannot hang, and its
firings (``censored``) are reported for confirmation rather than used as a
statistic. ``budget_exceeded`` rows keep ``n_paths = max_paths`` as a valid lower
bound and must stay in the aggregates; only ``censored`` rows are dropped.

Nothing in this module participates in ``k_star`` / ``K_valid`` selection.
"""

from __future__ import annotations

import resource
import signal
from time import perf_counter
from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from inference.signature_reversal_nodes import propagate_sign_votes_shortest_paths
from inference.signature_reversal_paths import _infer_polarity_along_path


PER_SOURCE_COLUMNS = [
    "source",
    "cutoff",
    "reach_sec",
    "enum_sec",
    "runtime_sec",
    "sign_inference_sec",
    "n_nodes",
    "n_voted",
    "n_paths",
    "budget_exceeded",
    "censored",
    "rep_idx",
]


class _EnumerationTimeout(Exception):
    """Raised when the safety-net wall-clock budget fires during enumeration."""


def _raise_timeout(signum, frame):  # noqa: ANN001 - signal handler signature
    raise _EnumerationTimeout()


def _measure_single_source_path_runtime(
    graph: nx.MultiDiGraph,
    source: str,
    cutoff: int,
    *,
    initial_polarity: int,
    timeout_sec: float,
    max_paths: int | None,
) -> Dict[str, Any]:
    """Measure neighborhood construction and path enumeration for one source.

    The enumeration loop is kept here (instead of calling
    ``_collect_path_polarities_from_source``) so that the already measured
    ``lengths`` / ``reachable`` are reused and neighborhood construction is not
    counted twice inside ``enum_sec``. Per-path polarity inference and result
    accumulation are included, matching what production does.
    """
    if source not in graph:
        return {
            "source": str(source),
            "cutoff": int(cutoff),
            "reach_sec": float("nan"),
            "enum_sec": float("nan"),
            "runtime_sec": float("nan"),
            "n_nodes": 0,
            "n_paths": float("nan"),
            "budget_exceeded": 0,
            "censored": 0,
        }

    # (1) neighborhood_construction
    t0 = perf_counter()
    lengths = nx.single_source_shortest_path_length(graph, source, int(cutoff))
    reachable = set(lengths)
    reachable.discard(source)
    t1 = perf_counter()

    # (2) path_enumeration; truncated by the path budget, timeout is the safety net
    n_paths = 0
    budget_exceeded = 0
    censored = 0
    budget = None if max_paths is None else int(max_paths)
    polarities: List[Dict[str, int]] = []  # production keeps every polarity dict
    use_timeout = timeout_sec is not None and float(timeout_sec) > 0.0
    old_handler = None
    if use_timeout:
        old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
    try:
        if reachable:
            for path in nx.all_simple_edge_paths(
                graph, source=source, target=reachable, cutoff=int(cutoff)
            ):
                polarities.append(_infer_polarity_along_path(path, graph, int(initial_polarity)))
                n_paths += 1
                if budget is not None and n_paths >= budget:
                    budget_exceeded = 1
                    break
    except _EnumerationTimeout:
        censored = 1
    finally:
        if use_timeout:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
    t2 = perf_counter()

    return {
        "source": str(source),
        "cutoff": int(cutoff),
        "reach_sec": float(t1 - t0),
        "enum_sec": float(t2 - t1),
        "runtime_sec": float(t2 - t0),
        "n_nodes": int(len(reachable)),
        # A censored source has an unknown path count; a budget-exceeded source
        # has a valid lower bound.
        "n_paths": float("nan") if censored else float(len(polarities)),
        "budget_exceeded": int(budget_exceeded),
        "censored": int(censored),
    }


def _measure_single_source_score_runtime(
    graph: nx.MultiDiGraph,
    source: str,
    cutoff: int,
    *,
    initial_polarity: int,
) -> Dict[str, Any]:
    """Measure node-level sign inference for one source.

    ``propagate_sign_votes_shortest_paths`` does its own neighborhood search, so
    this is timed as a single interval -- the same way production
    (``src/val2/legacy.py``) calls it. The pure neighborhood-construction cost is
    available separately as ``reach_sec``.
    """
    if source not in graph:
        return {"sign_inference_sec": float("nan"), "n_voted": 0}

    # (3) sign_inference
    t3 = perf_counter()
    votes = propagate_sign_votes_shortest_paths(
        graph,
        str(source),
        int(cutoff),
        start_sign=int(initial_polarity),
    )
    t4 = perf_counter()

    n_voted = 0
    for vv in votes.values():
        pos = float(vv.get(1, 0.0))
        neg = float(vv.get(-1, 0.0))
        if (pos > 0.0) or (neg > 0.0):
            n_voted += 1

    return {"sign_inference_sec": float(t4 - t3), "n_voted": int(n_voted)}


def _pct(series: pd.Series, q: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, q))


def _ratio(curr: float, prev: float | None) -> float:
    if prev is None or not np.isfinite(prev) or prev <= 0 or not np.isfinite(curr):
        return float("nan")
    return float(curr / prev)


def _median(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    return float(vals.median()) if len(vals) else float("nan")


def summarize_path_benchmark(
    per_source_df: pd.DataFrame,
    *,
    ks: Sequence[int],
    max_paths: int | None,
    timeout_sec: float,
    warmup: int,
    repeats: int,
    peak_rss_kb: Dict[int, int] | None = None,
) -> pd.DataFrame:
    """Aggregate per-source measurements into a per-``cutoff`` summary.

    Only ``censored == 1`` rows are excluded. ``budget_exceeded == 1`` rows stay
    in, because ``n_paths = max_paths`` is a valid lower bound there; dropping
    them would flatten the growth curve.
    """
    peak_rss_kb = dict(peak_rss_kb or {})
    rows: List[Dict[str, Any]] = []
    prev_paths: float | None = None
    for cutoff in ks:
        g = (
            per_source_df[per_source_df["cutoff"] == int(cutoff)]
            if not per_source_df.empty
            else per_source_df
        )
        eligible = g[g["censored"] == 0] if not g.empty else g
        npaths = eligible["n_paths"] if not eligible.empty else pd.Series(dtype=float)
        paths_median = _median(npaths)

        n_eligible = int(len(eligible))
        n_budget_exceeded = (
            int(pd.to_numeric(eligible["budget_exceeded"], errors="coerce").fillna(0).sum())
            if n_eligible
            else 0
        )
        within_budget_rate = (
            float((n_eligible - n_budget_exceeded) / n_eligible) if n_eligible else float("nan")
        )

        row: Dict[str, Any] = {
            "cutoff": int(cutoff),
            "number_of_sources": int(g["source"].nunique()) if not g.empty else 0,
            "n_paths_median": paths_median,
            "n_paths_q25": _pct(npaths, 25),
            "n_paths_q75": _pct(npaths, 75),
            "n_paths_p90": _pct(npaths, 90),
            "n_paths_p95": _pct(npaths, 95),
            "n_paths_p99": _pct(npaths, 99),
            "n_paths_max": _pct(npaths, 100),
            "n_paths_ratio_vs_prev": _ratio(paths_median, prev_paths),
            "median_N_k": _median(eligible["n_nodes"]) if n_eligible else float("nan"),
            "n_budget_exceeded": n_budget_exceeded,
            "within_budget_rate": within_budget_rate,
            "n_paths_exact_up_to_percentile": (
                float(within_budget_rate * 100.0) if np.isfinite(within_budget_rate) else float("nan")
            ),
            "n_timeout_censored": (
                int(pd.to_numeric(g["censored"], errors="coerce").fillna(0).sum())
                if not g.empty
                else 0
            ),
        }
        for phase in ("reach_sec", "enum_sec", "sign_inference_sec"):
            series = eligible[phase] if (n_eligible and phase in eligible.columns) else pd.Series(dtype=float)
            row[f"{phase}_median"] = _median(series)
            row[f"{phase}_q25"] = _pct(series, 25)
            row[f"{phase}_q75"] = _pct(series, 75)
            row[f"{phase}_p95"] = _pct(series, 95)
        row["runtime_sec_total"] = (
            float(pd.to_numeric(eligible["runtime_sec"], errors="coerce").sum())
            if n_eligible
            else float("nan")
        )
        row["peak_rss_kb"] = int(peak_rss_kb[int(cutoff)]) if int(cutoff) in peak_rss_kb else None
        row["max_paths"] = None if max_paths is None else int(max_paths)
        row["timeout_sec"] = float(timeout_sec)
        row["warmup"] = int(warmup)
        row["repeats"] = int(repeats)
        rows.append(row)
        prev_paths = paths_median

    return pd.DataFrame(rows).sort_values("cutoff").reset_index(drop=True)


def run_path_benchmark(
    *,
    G: nx.MultiDiGraph,
    sources: Sequence[str],
    ks: Sequence[int],
    initial_polarity: int,
    warmup: int,
    repeats: int,
    timeout_sec: float,
    max_paths: int | None,
    measure_node_score: bool,
    progress=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the path-enumeration benchmark over ``sources`` x ``ks``.

    Returns ``(per_source_df, summary_df)``. ``progress`` follows the Val1
    convention of a ``lambda iterable, **kwargs`` wrapper so that ``cfg.progress``
    decides whether tqdm is used.
    """
    valid_sources = [str(s) for s in sources if str(s) in G]
    warmup = int(max(0, warmup))
    repeats = int(max(1, repeats))
    cutoffs = [int(k) for k in ks]

    rows: List[Dict[str, Any]] = []
    peak_rss_kb: Dict[int, int] = {}
    for cutoff in cutoffs:
        iterable: Any = valid_sources
        if progress is not None:
            iterable = progress(
                valid_sources,
                total=len(valid_sources),
                desc=f"path benchmark k={cutoff}",
            )
        for src in iterable:
            for _ in range(warmup):
                _measure_single_source_path_runtime(
                    G,
                    src,
                    cutoff,
                    initial_polarity=int(initial_polarity),
                    timeout_sec=float(timeout_sec),
                    max_paths=max_paths,
                )
                if measure_node_score:
                    _measure_single_source_score_runtime(
                        G, src, cutoff, initial_polarity=int(initial_polarity)
                    )
            for rep_idx in range(repeats):
                row = _measure_single_source_path_runtime(
                    G,
                    src,
                    cutoff,
                    initial_polarity=int(initial_polarity),
                    timeout_sec=float(timeout_sec),
                    max_paths=max_paths,
                )
                if measure_node_score:
                    row.update(
                        _measure_single_source_score_runtime(
                            G, src, cutoff, initial_polarity=int(initial_polarity)
                        )
                    )
                else:
                    row["sign_inference_sec"] = float("nan")
                    row["n_voted"] = 0
                row["rep_idx"] = int(rep_idx)
                rows.append(row)
        # ru_maxrss is a monotone process high-water mark; because cutoffs are
        # visited in ascending order it reads as "peak RSS reached up to k".
        peak_rss_kb[int(cutoff)] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    per_source_df = pd.DataFrame(rows)
    if not per_source_df.empty:
        ordered = [c for c in PER_SOURCE_COLUMNS if c in per_source_df.columns]
        per_source_df = per_source_df[ordered + [c for c in per_source_df.columns if c not in ordered]]

    summary_df = summarize_path_benchmark(
        per_source_df,
        ks=cutoffs,
        max_paths=max_paths,
        timeout_sec=float(timeout_sec),
        warmup=warmup,
        repeats=repeats,
        peak_rss_kb=peak_rss_kb,
    )
    return per_source_df, summary_df
