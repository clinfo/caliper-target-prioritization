"""Val1 path-enumeration benchmark tests (spec_val1_path_benchmark_v1.0 §10)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference.signature_reversal_paths import _collect_path_polarities_from_source
from val1 import runtime as rt
from val1.config import _normalize_analysis


def _add(graph: nx.MultiDiGraph, u: str, v: str, polarity: int) -> None:
    graph.add_edge(u, v, polarity=int(polarity))


def _branching_kg() -> nx.MultiDiGraph:
    """Two disjoint routes A->D plus a tail, so paths outnumber reachable nodes.

    Reachable from A within k=3: B, C, D, E (4 nodes).
    Simple paths within k=3: A->B, A->C, A->B->D, A->C->D, A->B->D->E,
    A->C->D->E (6 paths).
    """
    g = nx.MultiDiGraph()
    _add(g, "A", "B", 1)
    _add(g, "A", "C", -1)
    _add(g, "B", "D", 1)
    _add(g, "C", "D", 1)
    _add(g, "D", "E", -1)
    return g


def _parallel_cycle_kg() -> nx.MultiDiGraph:
    """Parallel edges and a cycle, i.e. the cases where a path count can drift."""
    g = nx.MultiDiGraph()
    _add(g, "A", "B", 1)
    _add(g, "A", "B", -1)
    _add(g, "B", "C", 1)
    _add(g, "C", "A", -1)
    _add(g, "C", "D", 1)
    _add(g, "B", "D", -1)
    _add(g, "D", "E", 1)
    _add(g, "A", "F", -1)
    return g


def _as_multiset(polarities):
    return sorted(tuple(sorted(d.items())) for d in polarities)


def _capture_polarities(monkeypatch):
    """Record every polarity dict the benchmark loop infers, in order."""
    captured: list[dict[str, int]] = []
    real = rt._infer_polarity_along_path

    def wrapper(path, graph, initial_polarity):
        result = real(path, graph, initial_polarity)
        captured.append(result)
        return result

    monkeypatch.setattr(rt, "_infer_polarity_along_path", wrapper)
    return captured


# --- §10-1 equivalence with the production enumeration ------------------------


@pytest.mark.parametrize("cutoff", [1, 2, 3, 4])
def test_unbudgeted_enumeration_matches_production_path_set(monkeypatch, cutoff):
    graph = _parallel_cycle_kg()
    captured = _capture_polarities(monkeypatch)

    row = rt._measure_single_source_path_runtime(
        graph,
        "A",
        cutoff,
        initial_polarity=-1,
        timeout_sec=0.0,
        max_paths=None,
    )
    expected = _collect_path_polarities_from_source(
        source="A",
        graph=graph,
        initial_polarity=-1,
        cutoff=cutoff,
        path_strategy="simple",
    )

    assert row["n_paths"] == float(len(expected))
    assert row["budget_exceeded"] == 0
    assert row["censored"] == 0
    assert _as_multiset(captured) == _as_multiset(expected)


def test_unbudgeted_enumeration_matches_production_for_every_source(monkeypatch):
    graph = _parallel_cycle_kg()
    for source in sorted(graph.nodes()):
        captured = _capture_polarities(monkeypatch)
        row = rt._measure_single_source_path_runtime(
            graph,
            source,
            3,
            initial_polarity=-1,
            timeout_sec=0.0,
            max_paths=None,
        )
        expected = _collect_path_polarities_from_source(
            source=source,
            graph=graph,
            initial_polarity=-1,
            cutoff=3,
            path_strategy="simple",
        )
        assert row["n_paths"] == float(len(expected))
        assert _as_multiset(captured) == _as_multiset(expected)


# --- §10-2 path budget ------------------------------------------------------


@pytest.mark.parametrize("max_paths", [1, 2, 3])
def test_path_budget_truncates_and_flags(max_paths):
    graph = _parallel_cycle_kg()
    row = rt._measure_single_source_path_runtime(
        graph,
        "A",
        3,
        initial_polarity=-1,
        timeout_sec=0.0,
        max_paths=max_paths,
    )
    assert row["n_paths"] == float(max_paths)
    assert row["budget_exceeded"] == 1
    assert row["censored"] == 0


def test_path_budget_above_total_is_not_flagged():
    graph = _parallel_cycle_kg()
    total = len(
        _collect_path_polarities_from_source(
            source="A", graph=graph, initial_polarity=-1, cutoff=3, path_strategy="simple"
        )
    )
    row = rt._measure_single_source_path_runtime(
        graph,
        "A",
        3,
        initial_polarity=-1,
        timeout_sec=0.0,
        max_paths=total + 1,
    )
    assert row["n_paths"] == float(total)
    assert row["budget_exceeded"] == 0


# --- §10-6 n_paths is per path, not per reachable node -----------------------


def test_n_paths_counts_paths_not_reachable_nodes():
    graph = _branching_kg()
    row = rt._measure_single_source_path_runtime(
        graph,
        "A",
        3,
        initial_polarity=-1,
        timeout_sec=0.0,
        max_paths=None,
    )
    # Hand-computed on _branching_kg: 4 reachable nodes, 6 simple paths.
    assert row["n_nodes"] == 4
    assert row["n_paths"] == 6.0
    assert row["n_paths"] != float(row["n_nodes"])


def test_n_paths_counts_parallel_edges_separately():
    graph = nx.MultiDiGraph()
    _add(graph, "A", "B", 1)
    _add(graph, "A", "B", -1)
    row = rt._measure_single_source_path_runtime(
        graph,
        "A",
        1,
        initial_polarity=-1,
        timeout_sec=0.0,
        max_paths=None,
    )
    assert row["n_nodes"] == 1
    assert row["n_paths"] == 2.0


def test_missing_source_is_reported_without_measurements():
    graph = _branching_kg()
    row = rt._measure_single_source_path_runtime(
        graph,
        "ZZZ",
        3,
        initial_polarity=-1,
        timeout_sec=0.0,
        max_paths=None,
    )
    assert row["n_nodes"] == 0
    assert pd.isna(row["n_paths"])
    assert row["censored"] == 0


# --- sign inference phase ---------------------------------------------------


def test_sign_inference_reports_voted_nodes():
    graph = _branching_kg()
    out = rt._measure_single_source_score_runtime(graph, "A", 3, initial_polarity=-1)
    assert out["n_voted"] == 4
    assert out["sign_inference_sec"] >= 0.0


# --- §10-3 aggregation policy ------------------------------------------------


def _per_source_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # cutoff=2: three eligible sources (two at the budget) + one censored.
            {"source": "S1", "cutoff": 2, "reach_sec": 0.1, "enum_sec": 0.2, "runtime_sec": 0.3,
             "sign_inference_sec": 0.05, "n_nodes": 10, "n_voted": 8, "n_paths": 10.0,
             "budget_exceeded": 0, "censored": 0, "rep_idx": 0},
            {"source": "S2", "cutoff": 2, "reach_sec": 0.1, "enum_sec": 0.4, "runtime_sec": 0.5,
             "sign_inference_sec": 0.05, "n_nodes": 20, "n_voted": 15, "n_paths": 100.0,
             "budget_exceeded": 1, "censored": 0, "rep_idx": 0},
            {"source": "S3", "cutoff": 2, "reach_sec": 0.1, "enum_sec": 0.4, "runtime_sec": 0.5,
             "sign_inference_sec": 0.05, "n_nodes": 30, "n_voted": 25, "n_paths": 100.0,
             "budget_exceeded": 1, "censored": 0, "rep_idx": 0},
            {"source": "S4", "cutoff": 2, "reach_sec": 0.1, "enum_sec": 9.9, "runtime_sec": 10.0,
             "sign_inference_sec": 0.05, "n_nodes": 40, "n_voted": 35, "n_paths": float("nan"),
             "budget_exceeded": 0, "censored": 1, "rep_idx": 0},
        ]
    )


def test_budget_exceeded_rows_stay_in_and_censored_rows_drop_out():
    summary = rt.summarize_path_benchmark(
        _per_source_fixture(),
        ks=[2],
        max_paths=100,
        timeout_sec=600.0,
        warmup=0,
        repeats=1,
    )
    row = summary.iloc[0]
    # Median over the three eligible rows (10, 100, 100); dropping the budget rows
    # would give 10 and flatten the growth curve.
    assert row["n_paths_median"] == 100.0
    assert row["number_of_sources"] == 4
    assert row["n_timeout_censored"] == 1
    assert row["n_budget_exceeded"] == 2
    assert row["within_budget_rate"] == pytest.approx(1.0 / 3.0)
    assert row["n_paths_exact_up_to_percentile"] == pytest.approx(100.0 / 3.0)
    # The censored row's 10 s must not leak into the phase statistics.
    assert row["median_N_k"] == 20.0
    assert row["enum_sec_median"] == pytest.approx(0.4)
    assert row["runtime_sec_total"] == pytest.approx(1.3)


def test_summary_echoes_parameters_and_growth_ratio():
    per_source = pd.concat(
        [
            _per_source_fixture(),
            _per_source_fixture().assign(cutoff=3, n_paths=lambda d: d["n_paths"] * 10.0),
        ],
        ignore_index=True,
    )
    summary = rt.summarize_path_benchmark(
        per_source,
        ks=[2, 3],
        max_paths=100,
        timeout_sec=600.0,
        warmup=0,
        repeats=1,
        peak_rss_kb={2: 1000, 3: 2000},
    )
    assert list(summary["cutoff"]) == [2, 3]
    assert pd.isna(summary.loc[0, "n_paths_ratio_vs_prev"])
    assert summary.loc[1, "n_paths_ratio_vs_prev"] == pytest.approx(10.0)
    assert list(summary["peak_rss_kb"]) == [1000, 2000]
    assert set(summary["max_paths"]) == {100}
    assert set(summary["timeout_sec"]) == {600.0}
    assert set(summary["warmup"]) == {0}
    assert set(summary["repeats"]) == {1}


# --- run_path_benchmark end to end ------------------------------------------


def test_run_path_benchmark_shape_and_columns():
    graph = _parallel_cycle_kg()
    per_source, summary = rt.run_path_benchmark(
        G=graph,
        sources=["A", "B", "ZZZ"],
        ks=[1, 2],
        initial_polarity=-1,
        warmup=0,
        repeats=1,
        timeout_sec=0.0,
        max_paths=None,
        measure_node_score=True,
    )
    # "ZZZ" is not in the KG and is filtered out before measurement.
    assert list(per_source.columns[: len(rt.PER_SOURCE_COLUMNS)]) == rt.PER_SOURCE_COLUMNS
    assert len(per_source) == 4
    assert set(per_source["source"]) == {"A", "B"}
    assert list(summary["cutoff"]) == [1, 2]
    assert (summary["n_timeout_censored"] == 0).all()
    assert (summary["within_budget_rate"] == 1.0).all()
    assert (summary["peak_rss_kb"] > 0).all()


def test_run_path_benchmark_without_node_score_leaves_sign_inference_nan():
    graph = _branching_kg()
    per_source, summary = rt.run_path_benchmark(
        G=graph,
        sources=["A"],
        ks=[2],
        initial_polarity=-1,
        warmup=0,
        repeats=1,
        timeout_sec=0.0,
        max_paths=None,
        measure_node_score=False,
    )
    assert per_source["sign_inference_sec"].isna().all()
    assert pd.isna(summary.loc[0, "sign_inference_sec_median"])


def test_run_path_benchmark_repeats_are_indexed():
    graph = _branching_kg()
    per_source, _ = rt.run_path_benchmark(
        G=graph,
        sources=["A"],
        ks=[2],
        initial_polarity=-1,
        warmup=1,
        repeats=2,
        timeout_sec=0.0,
        max_paths=None,
        measure_node_score=False,
    )
    assert list(per_source["rep_idx"]) == [0, 1]
    assert per_source["n_paths"].nunique() == 1


# --- §10-4 config validation ------------------------------------------------


def _path_benchmark(**overrides):
    raw = {"ks_policy": "kmax", "ks": [], "path_benchmark": dict(overrides)}
    return _normalize_analysis(raw)["path_benchmark"]


def test_path_benchmark_defaults():
    pb = _normalize_analysis({"ks_policy": "kmax", "ks": []})["path_benchmark"]
    assert pb == {
        "enabled": True,
        "max_paths": 100000,
        "timeout_sec": 600.0,
        "warmup": 0,
        "repeats": 1,
        "measure_node_score": True,
        "ks": "inherit",
    }


def test_path_benchmark_rejects_warmup_two_or_more():
    with pytest.raises(ValueError):
        _path_benchmark(warmup=2)


def test_path_benchmark_rejects_non_positive_repeats():
    with pytest.raises(ValueError):
        _path_benchmark(repeats=0)


def test_path_benchmark_rejects_unknown_key():
    with pytest.raises(ValueError):
        _path_benchmark(unknown_key=1)


def test_path_benchmark_rejects_bad_max_paths_and_timeout():
    with pytest.raises(ValueError):
        _path_benchmark(max_paths=0)
    with pytest.raises(ValueError):
        _path_benchmark(timeout_sec=-1)


def test_path_benchmark_max_paths_null_means_unbounded():
    assert _path_benchmark(max_paths=None)["max_paths"] is None


def test_path_benchmark_ks_must_be_subset_of_val1_ks():
    raw = {
        "ks_policy": "config",
        "ks": [1, 2, 3, 4],
        "canonical_requires_kmax": False,
        "path_benchmark": {"ks": [1, 2, 3, 4]},
    }
    assert _normalize_analysis(raw)["path_benchmark"]["ks"] == [1, 2, 3, 4]

    raw["path_benchmark"] = {"ks": [1, 2, 9]}
    with pytest.raises(ValueError):
        _normalize_analysis(raw)


def test_path_benchmark_ks_inherit_is_allowed_under_kmax_policy():
    assert _path_benchmark(ks="inherit")["ks"] == "inherit"


# --- §10-5 the main k-selection line is untouched ---------------------------


def test_run_path_benchmark_does_not_touch_selection_outputs(tmp_path):
    run_root = tmp_path / "val1_run"
    (run_root / "metrics").mkdir(parents=True)
    (run_root / "selection").mkdir(parents=True)
    k_metrics = run_root / "metrics" / "k_metrics.tsv"
    k_selection = run_root / "selection" / "k_selection.json"
    k_metrics.write_text("k\tC\tB\tH\n1\t0.1\t0.0\t0.2\n", encoding="utf-8")
    k_selection.write_text(json.dumps({"k_star": 4, "K_valid": [3, 4, 5]}), encoding="utf-8")
    before = (k_metrics.read_bytes(), k_selection.read_bytes())

    rt.run_path_benchmark(
        G=_parallel_cycle_kg(),
        sources=["A", "B"],
        ks=[1, 2, 3],
        initial_polarity=-1,
        warmup=0,
        repeats=1,
        timeout_sec=0.0,
        max_paths=10,
        measure_node_score=True,
    )

    assert (k_metrics.read_bytes(), k_selection.read_bytes()) == before
    written = sorted(p.relative_to(run_root).as_posix() for p in run_root.rglob("*") if p.is_file())
    assert written == ["metrics/k_metrics.tsv", "selection/k_selection.json"]
