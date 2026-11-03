from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference.signature_reversal_nodes import _score_proposed
from val2_baseline.evaluate import (
    annotate_lift_eval_flags,
    build_lift_eval_cohort,
    compute_method_summary,
    compute_paired_deltas,
    compute_ttest_table,
    expected_random_hits,
)


def test_normalize_k_list_and_multi_k_metrics() -> None:
    from val2_baseline.evaluate import (
        build_lift_eval_cohorts,
        compute_metrics_by_disease,
        lift_col,
        normalize_k_list,
    )

    assert normalize_k_list([20, 10, 10], K_eff_cap=5) == [10, 20]
    assert normalize_k_list(None, K_eff_cap=10) == [10]

    long_df = pd.DataFrame(
        [
            {"disease_id": "D1", "target_gene": "g1", "method": "CALIPER", "score": 1.0, "label": 1},
            {"disease_id": "D1", "target_gene": "g2", "method": "CALIPER", "score": 0.0, "label": 0},
            {"disease_id": "D1", "target_gene": "g1", "method": "NCS", "score": 0.5, "label": 1},
            {"disease_id": "D1", "target_gene": "g2", "method": "NCS", "score": 0.1, "label": 0},
        ]
    )
    metrics = compute_metrics_by_disease(long_df, K_list=[10, 20], tie_policy="legacy")
    assert lift_col(10) in metrics.columns
    assert lift_col(20) in metrics.columns

    cohort = build_lift_eval_cohorts(metrics, k_list=[10, 20], min_expected_hits=1.0)
    assert set(cohort["K"].astype(int)) == {10, 20}
    assert "disease_id" in cohort.columns and "in_lift_eval" in cohort.columns


def test_expected_random_hits_examples() -> None:
    assert expected_random_hits(49, 418, 10) == 49 * 10 / 418
    assert expected_random_hits(3, 423, 10) < 1.0


def test_build_lift_eval_cohort_threshold() -> None:
    metrics = pd.DataFrame(
        [
            {"disease_id": "D_high", "method": "CALIPER", "n_pos": 49, "n_candidate": 418, "Lift@10": 0.5},
            {"disease_id": "D_low", "method": "CALIPER", "n_pos": 3, "n_candidate": 423, "Lift@10": 0.0},
        ]
    )
    cohort = build_lift_eval_cohort(metrics, k_eff_cap=10, min_expected_hits=1.0)
    flags = dict(zip(cohort["disease_id"], cohort["in_lift_eval"]))
    assert flags["D_high"] is True
    assert flags["D_low"] is False


def test_compute_method_summary_dual_lift() -> None:
    metrics = pd.DataFrame(
        [
            {"disease_id": "D1", "method": "CALIPER", "AP": 0.1, "Lift@10": 1.0, "n_pos": 49, "n_candidate": 418},
            {"disease_id": "D2", "method": "CALIPER", "AP": 0.2, "Lift@10": 0.0, "n_pos": 3, "n_candidate": 423},
            {"disease_id": "D1", "method": "NCS", "AP": 0.05, "Lift@10": 0.8, "n_pos": 49, "n_candidate": 418},
            {"disease_id": "D2", "method": "NCS", "AP": 0.15, "Lift@10": 0.0, "n_pos": 3, "n_candidate": 423},
        ]
    )
    cohort = build_lift_eval_cohort(metrics, k_eff_cap=10, min_expected_hits=1.0)
    metrics = annotate_lift_eval_flags(metrics, cohort)
    summary = compute_method_summary(metrics)
    cal = summary[summary["method"] == "CALIPER"].iloc[0]
    assert int(cal["n_diseases"]) == 2
    assert int(cal["n_diseases_lift_eval"]) == 1
    assert cal["mean_Lift@10"] == 0.5
    assert cal["mean_Lift@10_eval"] == 1.0


def test_ttest_lift_eval_uses_subset() -> None:
    metrics = pd.DataFrame(
        [
            {"disease_id": "D1", "method": "CALIPER", "AP": 0.2, "Lift@10": 1.0, "n_pos": 50, "n_candidate": 100},
            {"disease_id": "D2", "method": "CALIPER", "AP": 0.1, "Lift@10": 0.0, "n_pos": 1, "n_candidate": 500},
            {"disease_id": "D1", "method": "NCS", "AP": 0.1, "Lift@10": 0.5, "n_pos": 50, "n_candidate": 100},
            {"disease_id": "D2", "method": "NCS", "AP": 0.1, "Lift@10": 0.0, "n_pos": 1, "n_candidate": 500},
        ]
    )
    cohort = build_lift_eval_cohort(metrics, k_eff_cap=10, min_expected_hits=1.0)
    metrics = annotate_lift_eval_flags(metrics, cohort)
    deltas = compute_paired_deltas(metrics, baselines=["NCS"])
    tt = compute_ttest_table(deltas, baselines=["NCS"], metrics=["Lift@10", "Lift@10_eval"], alpha=0.05)
    n_all = int(tt[tt["metric"] == "Lift@10"]["n_diseases"].iloc[0])
    n_eval = int(tt[tt["metric"] == "Lift@10_eval"]["n_diseases"].iloc[0])
    assert n_all == 2
    assert n_eval == 1
    assert "caliper_method" in deltas.columns
    # Holm is the only correction reported; the Wilcoxon/BH columns are gone.
    assert set(tt["correction"]) == {"holm"}
    assert not {"p_bh", "bh_fdr", "p_primary", "wilcoxon_p"} & set(tt.columns)


def test_multi_caliper_paired_deltas_and_stats() -> None:
    from val2_baseline.caliper import caliper_method_keys
    from val2_baseline.evaluate import (
        compute_bootstrap_summary,
        compute_metrics_by_disease,
        compute_paired_deltas,
        compute_ttest_table,
        resolve_test_caliper_methods,
    )

    caliper_methods = caliper_method_keys([3, 4])
    rows = []
    for method, score in [("CALIPER@k3", 1.0), ("CALIPER@k4", 0.8), ("NCS", 0.5)]:
        rows.extend(
            [
                {"disease_id": "D1", "target_gene": "g1", "method": method, "score": score, "label": 1},
                {"disease_id": "D1", "target_gene": "g2", "method": method, "score": score - 0.4, "label": 0},
            ]
        )
    metrics = pd.DataFrame(rows)
    metrics = compute_metrics_by_disease(metrics, K_list=[10], tie_policy="legacy")
    deltas = compute_paired_deltas(metrics, baselines=["NCS"], caliper_methods=caliper_methods, k_list=[10])
    assert set(deltas["caliper_method"].astype(str)) == {"CALIPER@k3", "CALIPER@k4"}
    assert len(deltas) == 2
    boot = compute_bootstrap_summary(deltas, baselines=["NCS"], metrics=["AP"], caliper_methods=caliper_methods)
    assert "caliper_method" in boot.columns
    assert set(boot["caliper_method"].astype(str)) == {"CALIPER@k3", "CALIPER@k4"}
    # Effect sizes cover every scored k, but only k=4 is tested.
    test_methods = resolve_test_caliper_methods(deltas, caliper_methods=caliper_methods)
    assert test_methods == ["CALIPER@k4"]
    tt = compute_ttest_table(deltas, baselines=["NCS"], metrics=["AP"], caliper_methods=test_methods)
    assert set(tt["caliper_method"].astype(str)) == {"CALIPER@k4"}


def test_config_accepts_caliper_ks(tmp_path: Path) -> None:
    from val2_baseline.config import load_baseline_config

    cfg_text = """
run:
  seed: 42
  output_root: ../output
  registry_path: ./dataset_registry.example.yaml
data:
  kg: signor
  perturbation_mode: knockdown
  label_source: targetnexus_db
external_baseline:
  k_selection_json: ../output/val1/dummy/k_selection.json
  continuous_kd_profile:
    parquet: ../preprocess/data/intermediate/BING_knockdown_signature.parquet
  val2_label_parquet: ../data/input/features_kd_targetnexus_db_signor.parquet
  rwr:
    restart_probability: 0.5
  shared_seed: 42
  caliper:
    ks: [3, 4, 5]
"""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    cfg = load_baseline_config(cfg_path)
    assert cfg.caliper_ks == [3, 4, 5]


def test_score_proposed_ratio_and_zero_support() -> None:
    votes = {"g1": {1: 2.0, -1: 0.0}}
    pert = {"g1": 1}
    dis = {"g1": -1}
    score_ratio, w_rev, n_meas, *_rest, w_sum = _score_proposed({"g1"}, votes, pert, dis)
    assert w_sum > 0
    assert score_ratio == 1.0

    score0, *_ = _score_proposed({"g2"}, {}, pert, dis)
    assert score0 == 0.0
