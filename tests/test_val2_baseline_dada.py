"""DADA baseline tests (docs/val2/val2_dada_implementation_spec.md §8.1-§8.3)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from val2_baseline import dada as dada_mod
from val2_baseline import rwr as rwr_mod
from val2_baseline.availability import build_common_sets
from val2_baseline.config import load_baseline_config


def _path_graph(n: int = 5) -> nx.Graph:
    return nx.path_graph([str(i) for i in range(n)])


def _context(H: nx.Graph) -> dict:
    return dada_mod.build_network_context(H)


def _rwr_vector(H: nx.Graph, seeds: list[str], r: float = 0.5) -> tuple[np.ndarray, dict]:
    node_list = sorted(H.nodes())
    node_index = {n: i for i, n in enumerate(node_list)}
    W = rwr_mod.build_transition_matrix(H, node_index)
    p, _n_iter, _l1 = rwr_mod.rwr_steady_state(W, len(node_index), [node_index[s] for s in seeds], r)
    return p, node_index


def _write_config(path: Path, extra_external: str = "") -> Path:
    path.write_text(
        f"""
run:
  seed: 42
  date_tag: 20260727
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


# ---------------------------------------------------------------- §8.1 unit


def test_centrality_is_degree_proportional_and_sums_to_one() -> None:
    H = _path_graph(5)
    ctx = _context(H)

    degrees = {n: float(d) for n, d in H.degree()}
    total = sum(degrees.values())
    assert ctx["degree"] == degrees
    for n, d in degrees.items():
        assert ctx["centrality"][n] == pytest.approx(d / total)
    assert sum(ctx["centrality"].values()) == pytest.approx(1.0)
    # lambda = 2|E| / |V| over every node of H.
    assert ctx["network_mean_degree"] == pytest.approx(
        2 * H.number_of_edges() / H.number_of_nodes()
    )

    table = dada_mod.node_centrality_table(ctx)
    assert list(table.columns) == ["gene_id", "degree_H", "centrality"]
    assert table["centrality"].sum() == pytest.approx(1.0)


def test_adjusted_score_is_log_ratio_of_raw_and_centrality() -> None:
    H = _path_graph(5)
    ctx = _context(H)
    seeds = ["0", "1"]
    p, node_index = _rwr_vector(H, seeds)
    strategy, _df = dada_mod.build_disease_strategy(
        disease_ids=["D1"],
        disease_info={"D1": {"seeds": seeds}},
        network_context=ctx,
        restart_probability=0.5,
    )
    pairs = [("D1", n) for n in sorted(H.nodes())]
    pair_df, valid_map, _selected = dada_mod.build_pair_components(
        pairs=pairs,
        disease_strategy=strategy,
        network_context=ctx,
        rwr_disease_vec={"D1": p},
        node_index=node_index,
    )

    assert all(valid_map.values())
    for r in pair_df.itertuples(index=False):
        assert r.raw_rwr_score == pytest.approx(float(p[node_index[r.target_gene]]))
        assert r.centrality_score == pytest.approx(ctx["centrality"][r.target_gene])
        assert r.adjusted_log_ratio == pytest.approx(math.log(r.raw_rwr_score / r.centrality_score))
        assert r.target_degree_H == pytest.approx(ctx["degree"][r.target_gene])
        assert r.reason == ""


def test_seed_degree_uniform_rule_picks_raw_above_lambda_and_adjusted_below() -> None:
    H = _path_graph(5)
    ctx = _context(H)
    lam = ctx["network_mean_degree"]  # 1.6 for a 5-node path
    # Interior nodes have degree 2 > lambda; endpoints have degree 1 <= lambda.
    high, _df_high = dada_mod.build_disease_strategy(
        disease_ids=["D1"],
        disease_info={"D1": {"seeds": ["1", "2", "3"]}},
        network_context=ctx,
        restart_probability=0.5,
    )
    low, _df_low = dada_mod.build_disease_strategy(
        disease_ids=["D1"],
        disease_info={"D1": {"seeds": ["0", "4"]}},
        network_context=ctx,
        restart_probability=0.5,
    )
    assert high["D1"]["mean_seed_degree"] > lam
    assert high["D1"]["selected_component"] == dada_mod.RAW_COMPONENT
    assert low["D1"]["mean_seed_degree"] < lam
    assert low["D1"]["selected_component"] == dada_mod.ADJUSTED_COMPONENT


def test_boundary_mean_seed_degree_equal_lambda_selects_adjusted() -> None:
    # Triangle: every node has degree 2 and lambda == 2, so any seed set is a
    # boundary case and must fall on the centrality-adjusted side.
    H = nx.cycle_graph(["a", "b", "c"])
    ctx = _context(H)
    assert ctx["network_mean_degree"] == pytest.approx(2.0)
    strategy, strategy_df = dada_mod.build_disease_strategy(
        disease_ids=["D1"],
        disease_info={"D1": {"seeds": ["a", "b"]}},
        network_context=ctx,
        restart_probability=0.5,
    )
    assert strategy["D1"]["mean_seed_degree"] == pytest.approx(ctx["network_mean_degree"])
    assert strategy["D1"]["selected_component"] == dada_mod.ADJUSTED_COMPONENT
    assert dada_mod.select_component(2.0, 2.0) == dada_mod.ADJUSTED_COMPONENT
    assert list(strategy_df.columns) == dada_mod.DISEASE_STRATEGY_COLUMNS
    assert float(strategy_df["restart_probability"].iloc[0]) == 0.5


def test_common_ranking_uses_average_rank_and_negative_rank_score() -> None:
    selected = {
        ("D1", "T1"): {"selected_component": dada_mod.RAW_COMPONENT, "selected_component_score": 0.9},
        ("D1", "T2"): {"selected_component": dada_mod.RAW_COMPONENT, "selected_component_score": 0.5},
        ("D1", "T3"): {"selected_component": dada_mod.RAW_COMPONENT, "selected_component_score": 0.5},
        ("D1", "T4"): {"selected_component": dada_mod.RAW_COMPONENT, "selected_component_score": 0.1},
        ("D2", "T1"): {"selected_component": dada_mod.ADJUSTED_COMPONENT, "selected_component_score": -1.0},
    }
    ranking_df, score_map = dada_mod.build_common_ranking(
        common_pairs=list(selected.keys()),
        selected_components=selected,
    )
    ranks = dict(zip(zip(ranking_df["disease_id"], ranking_df["target_gene"]), ranking_df["dada_rank"]))
    assert ranks[("D1", "T1")] == pytest.approx(1.0)
    # Tied 0.5 scores share the average of ranks 2 and 3.
    assert ranks[("D1", "T2")] == pytest.approx(2.5)
    assert ranks[("D1", "T3")] == pytest.approx(2.5)
    assert ranks[("D1", "T4")] == pytest.approx(4.0)
    # Ranks are within-disease: D2's only candidate is its top candidate.
    assert ranks[("D2", "T1")] == pytest.approx(1.0)
    for key, rank in ranks.items():
        assert score_map[key] == pytest.approx(-rank)
    assert list(ranking_df.columns) == dada_mod.COMMON_RANKING_COLUMNS
    assert score_map[("D1", "T1")] == pytest.approx(-1.0)


def test_invalid_components_report_reasons_without_fallback_scores() -> None:
    H = _path_graph(4)
    ctx = _context(H)
    seeds = ["0", "1"]
    p, node_index = _rwr_vector(H, seeds)
    strategy, _df = dada_mod.build_disease_strategy(
        disease_ids=["D1", "D2", "D3"],
        disease_info={"D1": {"seeds": seeds}, "D2": {"seeds": seeds}, "D3": {"seeds": []}},
        network_context=ctx,
        restart_probability=0.5,
    )
    assert strategy["D3"]["valid"] is False
    assert strategy["D3"]["reason"] == dada_mod.REASON_EMPTY_SEED_SET_IN_H
    assert strategy["D3"]["selected_component"] == ""

    zero_vec = np.zeros_like(p)
    pair_df, valid_map, selected = dada_mod.build_pair_components(
        pairs=[
            ("D1", "0"),          # valid
            ("D1", "OUTSIDE_H"),  # target_not_in_H
            ("D2", "0"),          # no_disease_rwr_vector (D2 has no vector)
            ("D3", "0"),          # invalid_disease_strategy
            ("D4", "0"),          # nonpositive_log_ratio_input (all-zero vector)
        ],
        disease_strategy=strategy,
        network_context=ctx,
        rwr_disease_vec={"D1": p, "D3": p, "D4": zero_vec},
        node_index=node_index,
    )
    by_key = {(r.disease_id, r.target_gene): r for r in pair_df.itertuples(index=False)}

    assert by_key[("D1", "0")].valid is True
    assert by_key[("D1", "OUTSIDE_H")].reason == dada_mod.REASON_TARGET_NOT_IN_H
    assert by_key[("D2", "0")].reason == dada_mod.REASON_NO_DISEASE_RWR_VECTOR
    assert by_key[("D3", "0")].reason == dada_mod.REASON_INVALID_DISEASE_STRATEGY
    assert by_key[("D4", "0")].reason == dada_mod.REASON_NONPOSITIVE_LOG_RATIO_INPUT

    # Invalid pairs carry no fallback score and are absent from the ranking input.
    for key, row in by_key.items():
        if not row.valid:
            assert math.isnan(float(row.selected_component_score))
            assert valid_map[key] is False
            assert key not in selected
    with pytest.raises(ValueError, match="no valid selected component"):
        dada_mod.build_common_ranking(
            common_pairs=[("D1", "OUTSIDE_H")], selected_components=selected
        )


def test_target_seed_overlap_does_not_change_seeds_or_raw_vector() -> None:
    H = _path_graph(5)
    ctx = _context(H)
    seeds = ["0", "1"]
    p, node_index = _rwr_vector(H, seeds)
    strategy, strategy_df = dada_mod.build_disease_strategy(
        disease_ids=["D1"],
        disease_info={"D1": {"seeds": seeds}},
        network_context=ctx,
        restart_probability=0.5,
    )
    pair_df, _valid, _selected = dada_mod.build_pair_components(
        pairs=[("D1", "0"), ("D1", "3")],
        disease_strategy=strategy,
        network_context=ctx,
        rwr_disease_vec={"D1": p},
        node_index=node_index,
    )
    by_target = {r.target_gene: r for r in pair_df.itertuples(index=False)}

    assert by_target["0"].seed_target_overlap is True   # candidate is a seed
    assert by_target["3"].seed_target_overlap is False
    # Seed count and raw vector are unchanged by the overlap (candidate stays in p0).
    assert int(strategy_df["n_seed_in_H"].iloc[0]) == len(seeds)
    assert strategy["D1"]["mean_seed_degree"] == pytest.approx(
        (ctx["degree"]["0"] + ctx["degree"]["1"]) / 2
    )
    for r in pair_df.itertuples(index=False):
        assert r.raw_rwr_score == pytest.approx(float(p[node_index[r.target_gene]]))
        assert r.valid is True


# --------------------------------------------- §8.2 config / availability


def test_native_default_baselines_include_dada(tmp_path: Path) -> None:
    cfg = load_baseline_config(_write_config(tmp_path / "native.yaml"))
    assert cfg.primary_analysis == "native_shared_network"
    assert cfg.baselines == ["NCS", "proximity", "RWR", "DADA"]


def test_directed_default_baselines_exclude_dada(tmp_path: Path) -> None:
    cfg = load_baseline_config(
        _write_config(
            tmp_path / "directed.yaml",
            extra_external="""
  primary_analysis: directed_target_reach_matched
  target_reach_matched:
    horizon: k_star
""",
        )
    )
    assert cfg.primary_analysis == "directed_target_reach_matched"
    assert cfg.baselines == ["NCS", "proximity", "RWR"]


def test_directed_explicit_dada_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="native_shared_network"):
        load_baseline_config(
            _write_config(
                tmp_path / "directed_dada.yaml",
                extra_external="""
  primary_analysis: directed_target_reach_matched
  baselines: [NCS, proximity, RWR, DADA]
  target_reach_matched:
    horizon: k_star
""",
            )
        )


def test_dada_alias_normalizes_to_canonical_name(tmp_path: Path) -> None:
    cfg = load_baseline_config(
        _write_config(tmp_path / "alias.yaml", extra_external="  baselines: [dada, DADA]\n")
    )
    assert cfg.baselines == ["DADA"]


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"disease_id": "D1", "target_gene": "T1", "cell_id": "C1", "label": 1},
            {"disease_id": "D1", "target_gene": "T2", "cell_id": "C1", "label": 0},
        ]
    )


def test_dada_validity_and_seed_gate_apply_only_when_selected() -> None:
    labels = _labels()
    kwargs = dict(
        labels_df=labels,
        H_nodes={"T1", "T2"},
        continuous_cells={("T1", "C1"), ("T2", "C1")},
        ncs_valid_cells=set(),
        ncs_ref_counts={},
        proximity_valid={},
        rwr_valid={},
        min_disease_seeds=20,
    )
    dada_valid = {("D1", "T1"): True, ("D1", "T2"): False}

    with_dada = build_common_sets(
        disease_info={"D1": {"ncs_query_valid": False, "s_d_h_size": 20}},
        selected_baselines=["DADA"],
        dada_valid=dada_valid,
        **kwargs,
    )
    avail = with_dada["availability_df"].set_index("target_gene")
    assert "dada_score_defined" in with_dada["availability_df"].columns
    assert bool(avail.loc["T1", "in_common_candidate"]) is True
    assert bool(avail.loc["T2", "in_common_candidate"]) is False
    assert avail.loc["T2", "exclusion_reason"] == "dada_undefined"

    # DADA activates the disease-level network seed gate.
    gated = build_common_sets(
        disease_info={"D1": {"ncs_query_valid": False, "s_d_h_size": 19}},
        selected_baselines=["DADA"],
        dada_valid={("D1", "T1"): True, ("D1", "T2"): True},
        **kwargs,
    )
    assert gated["common_diseases_df"].empty
    assert "disease:seeds_lt_20" in set(gated["exclusion_summary_df"]["scope_reason"])

    # Unselected DADA neither excludes pairs nor adds its availability column.
    without_dada = build_common_sets(
        disease_info={"D1": {"ncs_query_valid": True, "s_d_h_size": 0}},
        selected_baselines=["NCS"],
        dada_valid=dada_valid,
        ncs_valid_cells={("D1", "T1", "C1"), ("D1", "T2", "C1")},
        ncs_ref_counts={("D1", "C1"): (5, 5)},
        labels_df=labels,
        H_nodes={"T1", "T2"},
        continuous_cells={("T1", "C1"), ("T2", "C1")},
        proximity_valid={},
        rwr_valid={},
        min_disease_seeds=20,
    )
    assert len(without_dada["common_candidates_df"]) == 2
    assert "dada_score_defined" not in without_dada["availability_df"].columns


def test_explicit_three_baselines_keep_existing_common_set() -> None:
    labels = _labels()
    common = build_common_sets(
        labels_df=labels,
        H_nodes={"T1", "T2"},
        disease_info={"D1": {"ncs_query_valid": True, "s_d_h_size": 20}},
        continuous_cells={("T1", "C1"), ("T2", "C1")},
        ncs_valid_cells={("D1", "T1", "C1"), ("D1", "T2", "C1")},
        ncs_ref_counts={("D1", "C1"): (5, 5)},
        proximity_valid={("D1", "T1"): True, ("D1", "T2"): True},
        rwr_valid={("D1", "T1"): True, ("D1", "T2"): True},
        min_disease_seeds=20,
        selected_baselines=["NCS", "proximity", "RWR"],
        dada_valid={("D1", "T1"): False, ("D1", "T2"): False},
    )
    assert len(common["common_candidates_df"]) == 2
    assert set(common["availability_df"]["exclusion_reason"]) == {""}
    assert "dada_score_defined" not in common["availability_df"].columns


# --------------------------------------------------------- §8.3 integration


def test_dada_raw_component_matches_classical_rwr_score() -> None:
    """§8.3-1: the raw component is the Classical RWR score, not a recomputation."""
    H = _path_graph(6)
    ctx = _context(H)
    seeds = ["0", "1", "2"]
    p, node_index = _rwr_vector(H, seeds, r=0.3)
    # Classical RWR pair scores as the runner builds them.
    rwr_scores = {n: float(p[node_index[n]]) for n in sorted(H.nodes())}

    strategy, _df = dada_mod.build_disease_strategy(
        disease_ids=["D1"],
        disease_info={"D1": {"seeds": seeds}},
        network_context=ctx,
        restart_probability=0.3,
    )
    pair_df, _valid, _selected = dada_mod.build_pair_components(
        pairs=[("D1", n) for n in sorted(H.nodes())],
        disease_strategy=strategy,
        network_context=ctx,
        rwr_disease_vec={"D1": p},
        node_index=node_index,
    )
    for r in pair_df.itertuples(index=False):
        assert r.raw_rwr_score == pytest.approx(rwr_scores[r.target_gene], rel=0, abs=1e-12)


def test_dada_common_scores_are_finite_over_the_shared_universe() -> None:
    """§8.3-3: every common pair gets a finite DADA score on the shared universe."""
    H = _path_graph(6)
    ctx = _context(H)
    seeds = ["0", "1"]
    p, node_index = _rwr_vector(H, seeds)
    strategy, _df = dada_mod.build_disease_strategy(
        disease_ids=["D1"],
        disease_info={"D1": {"seeds": seeds}},
        network_context=ctx,
        restart_probability=0.5,
    )
    pairs = [("D1", n) for n in sorted(H.nodes())]
    _pair_df, valid_map, selected = dada_mod.build_pair_components(
        pairs=pairs,
        disease_strategy=strategy,
        network_context=ctx,
        rwr_disease_vec={"D1": p},
        node_index=node_index,
    )
    common_pairs = [key for key, ok in valid_map.items() if ok]
    ranking_df, score_map = dada_mod.build_common_ranking(
        common_pairs=common_pairs, selected_components=selected
    )
    assert len(ranking_df) == len(common_pairs)
    assert np.isfinite(ranking_df["dada_score"].to_numpy()).all()
    assert set(ranking_df["selected_component"]) == {strategy["D1"]["selected_component"]}
    assert min(score_map.values()) == pytest.approx(-float(len(common_pairs)))


def test_dada_is_a_generic_baseline_in_evaluation_and_figures() -> None:
    """§8.3-4: DADA is part of the baseline order used by evaluation and figures."""
    from val2_baseline import viz as viz_mod

    assert viz_mod.BASELINES == ["NCS", "proximity", "RWR", "DADA"]
    assert viz_mod.METHOD_ORDER == ["CALIPER", "NCS", "proximity", "RWR", "DADA"]
    assert viz_mod.METHOD_DISPLAY["DADA"] == "DADA"
    assert viz_mod.METHOD_COLOR["DADA"] == "#7B3D5E"
    assert viz_mod._method_order({"CALIPER", "DADA", "RWR"}) == ["CALIPER", "RWR", "DADA"]


def test_metadata_block_records_task_adaptation() -> None:
    block = dada_mod.metadata_block(restart_probability=0.4)
    assert block["variant"] == "centrality_adjustment_seed_degree_uniform"
    assert block["restart_probability"] == pytest.approx(0.4)
    assert block["edge_weight"] == "unit"
    assert block["seed_weighting"] == "uniform"
    assert block["candidate_seed_policy"] == "include"
    assert block["centrality"] == "r0_stationary_degree_proportional"
    assert block["adjustment"] == "natural_log_ratio"
    assert block["degree_threshold"] == "network_mean_degree"
    assert block["evaluation_score"] == "negative_descending_average_rank"
