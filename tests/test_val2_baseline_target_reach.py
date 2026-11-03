from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from val2_baseline import evaluate as evaluate_mod
from val2_baseline import parallel as parallel_mod
from val2_baseline import step4_worker
from val2_baseline.availability import build_common_sets, build_common_sets_directed_target_reach
from val2_baseline.config import load_baseline_config
from val2_baseline.proximity import compute_proximity_pair_local
from val2_baseline.reach import (
    anchor_nodes,
    build_local_undirected_graph,
    build_local_undirected_graph_from_adjacency,
    build_undirected_adjacency,
    directed_reach_nodes,
)
from val2_baseline.rwr import compute_rwr_pair_local
from val2_baseline.wtcs import compute_cell_ncs_local


def _square(x: int) -> int:
    return x * x


def _write_config(path: Path, extra_external: str = "") -> Path:
    path.write_text(
        f"""
run:
  seed: 42
  date_tag: 20260702
  output_root: ../output
  output_name: auto
  registry_path: ./dataset_registry.example.yaml
data:
  kg: signor
  perturbation_mode: knockdown
  label_source: targetnexus_db
external_baseline:
{extra_external}
  k_selection_json: ../output/val1/k_selection.json
  continuous_kd_profile:
    parquet: ../preprocess/data/intermediate/BING_knockdown_signature.parquet
    id_col: ncbigene_id
    cell_col: cell_id
  val2_label_parquet: ../data/input/features_kd_targetnexus_db_signor.parquet
  rwr:
    restart_probability: 0.5
  proximity:
    n_random: 10
    min_bin_size: 2
  shared_seed: 42
  min_disease_seeds: 20
analysis:
  K_eff_cap: 10
  K_list: [10]
  tie_policy: expected
sections:
  scores: true
  evaluation: true
  make_figures: false
  make_report: false
stats:
  enabled: true
""",
        encoding="utf-8",
    )
    return path


def test_config_primary_analysis_defaults_and_aliases(tmp_path: Path) -> None:
    cfg = load_baseline_config(_write_config(tmp_path / "native.yaml"))
    assert cfg.primary_analysis == "native_shared_network"
    # Mode-aware defaults: DADA is a native-shared-network-only baseline.
    assert cfg.baselines == ["NCS", "proximity", "RWR", "DADA"]

    cfg = load_baseline_config(
        _write_config(
            tmp_path / "directed.yaml",
            extra_external="""
  primary_analysis: target_reach_matched
  baselines: [l1000-ncs, network proximity, classical_rwr, ncs]
  target_reach_matched:
    horizon: k_star
""",
        )
    )
    assert cfg.primary_analysis == "directed_target_reach_matched"
    assert cfg.baselines == ["NCS", "proximity", "RWR"]
    assert cfg.target_reach_rwr_target_only_seed_score == 1.0


def test_config_rejects_unknown_mode_and_non_kstar_horizon(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="primary_analysis"):
        load_baseline_config(_write_config(tmp_path / "bad_mode.yaml", "  primary_analysis: other\n"))

    with pytest.raises(ValueError, match="horizon"):
        load_baseline_config(
            _write_config(
                tmp_path / "bad_horizon.yaml",
                extra_external="""
  primary_analysis: directed_target_reach_matched
  target_reach_matched:
    horizon: 2
""",
            )
        )

    with pytest.raises(ValueError, match="baselines"):
        load_baseline_config(_write_config(tmp_path / "bad_baseline.yaml", "  baselines: [unknown]\n"))


def test_native_common_set_respects_selected_baselines() -> None:
    labels = pd.DataFrame(
        [
            {"disease_id": "D1", "target_gene": "T1", "cell_id": "C1", "label": 1},
            {"disease_id": "D1", "target_gene": "T2", "cell_id": "C1", "label": 0},
        ]
    )
    common = build_common_sets(
        labels_df=labels,
        H_nodes={"T1", "T2"},
        disease_info={"D1": {"ncs_query_valid": True, "s_d_h_size": 0}},
        continuous_cells={("T1", "C1"), ("T2", "C1")},
        ncs_valid_cells={("D1", "T1", "C1"), ("D1", "T2", "C1")},
        ncs_ref_counts={("D1", "C1"): (5, 5)},
        proximity_valid={},
        rwr_valid={},
        min_disease_seeds=20,
        selected_baselines=["NCS"],
    )
    assert len(common["common_candidates_df"]) == 2
    assert set(common["availability_df"]["exclusion_reason"]) == {""}

    network_common = build_common_sets(
        labels_df=labels,
        H_nodes={"T1", "T2"},
        disease_info={"D1": {"ncs_query_valid": False, "s_d_h_size": 20}},
        continuous_cells=set(),
        ncs_valid_cells=set(),
        ncs_ref_counts={},
        proximity_valid={("D1", "T1"): True, ("D1", "T2"): True},
        rwr_valid={},
        min_disease_seeds=20,
        selected_baselines=["proximity"],
    )
    assert len(network_common["common_candidates_df"]) == 2
    assert all(cells == ["C1"] for cells in network_common["common_cells"].values())


def test_reach_and_local_graph_projection() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge("A", "B", polarity=1)
    graph.add_edge("A", "B", polarity=-1)
    graph.add_edge("B", "C", polarity=1)
    graph.add_edge("C", "A", polarity=1)
    graph.add_edge("B", "B", polarity=1)
    reach = directed_reach_nodes(graph, "A", 1)
    assert reach == {"B"}
    anchors = anchor_nodes("A", reach)
    assert anchors == {"A", "B"}
    local = build_local_undirected_graph(graph, anchors)
    assert sorted(local.nodes()) == ["A", "B"]
    assert {frozenset(edge) for edge in local.edges()} == {frozenset(("A", "B"))}
    assert not list(nx.selfloop_edges(local))


def test_directed_common_set_keeps_local_fallback_pairs() -> None:
    labels = pd.DataFrame(
        [
            {"disease_id": "D1", "target_gene": "T1", "cell_id": "C1", "label": 1},
            {"disease_id": "D1", "target_gene": "T2", "cell_id": "C1", "label": 0},
        ]
    )
    common = build_common_sets_directed_target_reach(
        labels_df=labels,
        G_nodes={"T1", "T2"},
        disease_info={"D1": {"ncs_query_valid": True, "s_d_h_size": 20}},
        continuous_cells={("T1", "C1"), ("T2", "C1")},
        reach_info={
            "T1": {"n_reach_nodes": 0, "n_anchor_nodes": 1, "n_local_edges": 0},
            "T2": {"n_reach_nodes": 0, "n_anchor_nodes": 1, "n_local_edges": 0},
        },
        ncs_fallback_by_pair={("D1", "T1"): 1, ("D1", "T2"): 1},
        proximity_info={
            ("D1", "T1"): {"fallback": True, "reason": "no_seed_in_local", "n_seed_in_local": 0},
            ("D1", "T2"): {"fallback": True, "reason": "no_seed_in_local", "n_seed_in_local": 0},
        },
        rwr_info={
            ("D1", "T1"): {"fallback": True, "reason": "no_seed_in_local", "n_seed_in_local": 0},
            ("D1", "T2"): {"fallback": True, "reason": "no_seed_in_local", "n_seed_in_local": 0},
        },
        min_disease_seeds=20,
    )
    assert len(common["common_candidates_df"]) == 2
    assert set(common["availability_df"]["exclusion_reason"]) == {""}


def test_directed_common_set_excludes_disease_below_min_seeds() -> None:
    labels = pd.DataFrame(
        [
            {"disease_id": "D1", "target_gene": "T1", "cell_id": "C1", "label": 1},
            {"disease_id": "D1", "target_gene": "T2", "cell_id": "C1", "label": 0},
        ]
    )
    common = build_common_sets_directed_target_reach(
        labels_df=labels,
        G_nodes={"T1", "T2"},
        disease_info={"D1": {"ncs_query_valid": True, "s_d_h_size": 5}},
        continuous_cells={("T1", "C1"), ("T2", "C1")},
        reach_info={
            "T1": {"n_reach_nodes": 0, "n_anchor_nodes": 1, "n_local_edges": 0},
            "T2": {"n_reach_nodes": 0, "n_anchor_nodes": 1, "n_local_edges": 0},
        },
        ncs_fallback_by_pair={},
        proximity_info={
            ("D1", "T1"): {"fallback": False, "reason": "", "n_seed_in_local": 1},
            ("D1", "T2"): {"fallback": False, "reason": "", "n_seed_in_local": 1},
        },
        rwr_info={
            ("D1", "T1"): {"fallback": False, "reason": "", "n_seed_in_local": 1},
            ("D1", "T2"): {"fallback": False, "reason": "", "n_seed_in_local": 1},
        },
        selected_baselines=["proximity", "RWR"],
        min_disease_seeds=20,
    )
    assert len(common["common_candidates_df"]) == 0
    exclusion_reasons = set(common["exclusion_summary_df"]["scope_reason"])
    assert "disease:seeds_lt_20" in exclusion_reasons


def test_directed_common_set_skips_ncs_gate_when_ncs_unselected() -> None:
    labels = pd.DataFrame(
        [
            {"disease_id": "D1", "target_gene": "T1", "cell_id": "C1", "label": 1},
            {"disease_id": "D1", "target_gene": "T2", "cell_id": "C1", "label": 0},
        ]
    )
    common = build_common_sets_directed_target_reach(
        labels_df=labels,
        G_nodes={"T1", "T2"},
        disease_info={"D1": {"ncs_query_valid": False, "s_d_h_size": 20}},
        continuous_cells=set(),
        reach_info={
            "T1": {"n_reach_nodes": 0, "n_anchor_nodes": 1, "n_local_edges": 0},
            "T2": {"n_reach_nodes": 0, "n_anchor_nodes": 1, "n_local_edges": 0},
        },
        ncs_fallback_by_pair={},
        proximity_info={
            ("D1", "T1"): {"fallback": True, "reason": "no_seed_in_local", "n_seed_in_local": 0},
            ("D1", "T2"): {"fallback": True, "reason": "no_seed_in_local", "n_seed_in_local": 0},
        },
        rwr_info={},
        selected_baselines=["proximity"],
        min_disease_seeds=20,
    )
    assert len(common["common_candidates_df"]) == 2
    assert set(common["availability_df"]["exclusion_reason"]) == {""}


def test_local_ncs_empty_feature_universe_returns_invalid_finite_shape() -> None:
    ref_matrix = np.ones((2, 3), dtype=float)
    res = compute_cell_ncs_local(
        ref_matrix,
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
    )
    assert res["valid"] is False
    assert int(res["n_ref_total"]) == 2
    assert np.isnan(res["ncs"]).all()


def test_local_proximity_and_rwr_fallbacks_are_finite() -> None:
    H = nx.Graph()
    H.add_node("T")
    prox = compute_proximity_pair_local(
        H_local=H,
        target_gene="T",
        disease_seeds=set(),
        n_random=10,
        min_bin_size=2,
        shared_seed=42,
        disease_id="D1",
    )
    assert prox["valid"] is True
    assert prox["fallback"] is True
    assert prox["score"] == 0.0

    # A target-only local graph always falls back (methodology fairness fix):
    # disease seeds passed to RWR are expected to already exclude the target
    # (S_{d,-t}), so a lone seed equal to the target itself must not be
    # treated as valid self-seed evidence.
    rwr = compute_rwr_pair_local(
        H_local=H,
        target_gene="T",
        disease_seeds={"T"},
        restart_probability=0.5,
    )
    assert rwr["valid"] is True
    assert rwr["fallback"] is True
    assert rwr["score"] == 0.0
    assert rwr["reason"] == "target_only_no_seed"


def test_build_local_undirected_graph_from_adjacency_matches_full_scan() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge("A", "B", polarity=1)
    graph.add_edge("A", "B", polarity=-1)
    graph.add_edge("B", "C", polarity=1)
    graph.add_edge("C", "A", polarity=1)
    graph.add_edge("B", "B", polarity=1)
    graph.add_edge("C", "D", polarity=1)  # outside the anchor set
    anchors = {"A", "B", "C"}
    expected = build_local_undirected_graph(graph, anchors)
    adjacency = build_undirected_adjacency(graph)
    actual = build_local_undirected_graph_from_adjacency(adjacency, anchors)
    assert sorted(actual.nodes()) == sorted(expected.nodes())
    assert {frozenset(e) for e in actual.edges()} == {frozenset(e) for e in expected.edges()}


def test_apply_fallback_floor_per_disease_per_method() -> None:
    method_maps = {
        "CALIPER": {("D1", "T1"): 0.5, ("D1", "T2"): 0.0, ("D1", "T3"): 0.0},
        "NCS": {("D1", "T1"): 1.0, ("D1", "T2"): -0.2, ("D1", "T3"): -0.2},
        "proximity": {("D1", "T1"): 2.0, ("D1", "T2"): 2.0, ("D1", "T3"): 2.0},
    }
    # NCS: only T2 is a fallback pair; proximity: the whole disease is fallback.
    fallback_keys_by_method = {
        "NCS": {("D1", "T2")},
        "proximity": {("D1", "T1"), ("D1", "T2"), ("D1", "T3")},
    }
    common_candidates_df = pd.DataFrame(
        [
            {"disease_id": "D1", "target_gene": "T1"},
            {"disease_id": "D1", "target_gene": "T2"},
            {"disease_id": "D1", "target_gene": "T3"},
        ]
    )
    evaluate_mod.apply_fallback_floor(method_maps, fallback_keys_by_method, common_candidates_df)

    # NCS: valid scores are {T1: 1.0, T3: -0.2} -> floor = min(-0.2) - 1.0 = -1.2.
    assert method_maps["NCS"][("D1", "T2")] == pytest.approx(-1.2)
    assert method_maps["NCS"][("D1", "T1")] == 1.0
    assert method_maps["NCS"][("D1", "T3")] == -0.2

    # proximity: no valid score in the disease -> tie at the chance-level constant.
    assert method_maps["proximity"][("D1", "T1")] == 0.0
    assert method_maps["proximity"][("D1", "T2")] == 0.0
    assert method_maps["proximity"][("D1", "T3")] == 0.0

    # CALIPER has no fallback concept and must be untouched.
    assert method_maps["CALIPER"][("D1", "T1")] == 0.5
    assert method_maps["CALIPER"][("D1", "T2")] == 0.0


def test_parallel_map_ordered_sequential_matches_parallel() -> None:
    tasks = list(range(12))
    expected = [x * x for x in tasks]
    seq = parallel_mod.parallel_map_ordered(_square, tasks, n_jobs=1)
    assert seq == expected
    par = parallel_mod.parallel_map_ordered(_square, tasks, n_jobs=2, start_method="fork")
    assert par == expected


def test_step4_worker_excludes_target_from_seeds() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge("T", "A", polarity=1)
    graph.add_edge("A", "B", polarity=1)
    adjacency = build_undirected_adjacency(graph)
    state = step4_worker.build_worker_state(
        graph=graph,
        adjacency=adjacency,
        h=2,
        start_sign=1,
        pert_sigs={},
        dis_sigs={},
        # T's only disease seed is itself: after excluding the candidate
        # target (S_{d,-t}) this must leave no usable local seed.
        disease_info={"D1": {"seeds": ["T"], "q_up": [], "q_down": []}},
        pair_cells_by_target={"T": {"D1": []}},
        use_ncs=False,
        use_proximity=True,
        use_rwr=True,
        continuous_cells=set(),
        continuous_kd_feature_index={},
        continuous_kd_matrix=np.zeros((0, 0), dtype=np.float32),
        continuous_kd_key_to_row={},
        cell_to_rows={},
        gene_id_int=np.array([], dtype=np.int64),
        ncs_fallback_score=0.0,
        proximity_n_random=10,
        proximity_min_bin_size=2,
        proximity_fallback_score=0.0,
        shared_seed=42,
        rwr_restart_probability=0.5,
        rwr_fallback_score=0.0,
        rwr_target_only_seed_score=1.0,
    )
    step4_worker.configure_worker_state(state)
    result = step4_worker.process_target("T")
    assert result["reach_row"]["n_reach_nodes"] == 2

    assert len(result["proximity_rows"]) == 1
    prox_row = result["proximity_rows"][0]
    assert prox_row["disease_id"] == "D1" and prox_row["target_gene"] == "T"
    assert prox_row["fallback"] is True
    assert prox_row["reason"] == "no_seed_in_local"

    assert len(result["rwr_rows"]) == 1
    rwr_row = result["rwr_rows"][0]
    assert rwr_row["fallback"] is True
    assert rwr_row["reason"] == "no_seed_in_local"
