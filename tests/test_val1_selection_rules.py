"""Val1 label-free selection-rule sensitivity tests (P0-2 B-1..B-5)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from val1 import selection_rules as sr
from val1.config import _normalize_metrics
from val1.selection import build_k_metrics, rank_k_values, select_k


KS = [1, 2, 3, 4, 5, 6]
# Monotone increasing coverage and ambiguity, with the trade-off knee at k=4.
C_GRID = [0.00, 0.01, 0.08, 0.40, 0.69, 0.78]
B_GRID = [0.00, 0.01, 0.07, 0.26, 0.41, 0.46]


def _rule_metrics(normalize: bool = True) -> pd.DataFrame:
    return sr.build_rule_metrics(KS, C_GRID, B_GRID, normalize=normalize)


def _selection_kwargs(**overrides):
    base = dict(
        rules=list(sr.SENSITIVITY_RULES),
        normalize=True,
        zero_division=0.0,
        top_n=3,
        tie_breaker="smaller_k",
        knee_method="max_chord_distance",
        tau=0.20,
        beta=0.30,
        threshold_scale="raw",
    )
    base.update(overrides)
    return base


# --- B-1 / B-2: score rules -------------------------------------------------


def test_harmonic_and_geometric_agree_on_the_knee_grid():
    rm = _rule_metrics()
    k_h = int(rm.loc[rm["score_harmonic_mean"].idxmax(), "k"])
    k_g = int(rm.loc[rm["score_geometric_mean"].idxmax(), "k"])
    assert k_h == 4
    assert k_g == 4


def test_geometric_mean_matches_sqrt_of_components():
    rm = _rule_metrics()
    expected = np.sqrt(rm["C_tilde"].to_numpy(float) * rm["A_tilde"].to_numpy(float))
    np.testing.assert_allclose(rm["score_geometric_mean"].to_numpy(float), expected, rtol=0, atol=1e-12)


def test_recomputed_harmonic_score_matches_primary_selection_path():
    coverage_summary = pd.DataFrame({"k": KS, "cov_ratio_median": C_GRID})
    ambiguity_summary = pd.DataFrame({"k": KS, "B_amb_median": B_GRID})
    primary = build_k_metrics(
        coverage_summary=coverage_summary,
        ambiguity_summary=ambiguity_summary,
        coverage_metric="cov_ratio",
        ambiguity_metric="amb_median",
        normalize=True,
    )
    rm = _rule_metrics()
    np.testing.assert_allclose(
        rm["score_harmonic_mean"].to_numpy(float),
        primary["H"].to_numpy(float),
        rtol=0,
        atol=sr.POOLED_CONSISTENCY_TOL,
    )
    ranking = rank_k_values(primary, top_n=3, tie_breaker="smaller_k")
    assert select_k(ranking, top_n=3)["k_star"] == 4


def test_pareto_knee_selects_the_knee_and_degenerates_on_a_straight_line():
    rm = _rule_metrics()
    assert int(rm.loc[rm["knee_dist"].idxmax(), "k"]) == 4

    # The chord endpoints are never knee candidates.
    assert bool(np.isnan(rm["knee_dist"].iloc[0]))
    assert bool(np.isnan(rm["knee_dist"].iloc[-1]))

    # Perfectly linear trade-off: every interior front point lies on the chord.
    straight = sr.build_rule_metrics([1, 2, 3, 4], [0.0, 0.25, 0.50, 0.75], [0.0, 0.25, 0.50, 0.75], normalize=True)
    assert float(np.nanmax(straight["knee_dist"].to_numpy(float))) == pytest.approx(0.0, abs=1e-12)
    ranking = sr.rank_by_score(
        straight["k"].to_numpy(int),
        straight["score_pareto_knee"].to_numpy(float),
        rule=sr.RULE_PARETO_KNEE,
        top_n=3,
        tie_breaker="smaller_k",
    )
    # All-equal interior scores fall back to the smaller-k tie-breaker.
    assert sr.selection_from_ranking(ranking, rule=sr.RULE_PARETO_KNEE, top_n=3)["k_star"] == 2

    _, selections, _ = sr.select_all_rules(
        straight["k"].to_numpy(int),
        straight["C"].to_numpy(float),
        straight["B"].to_numpy(float),
        **_selection_kwargs(),
    )
    assert selections[sr.RULE_PARETO_KNEE]["degenerate_no_knee"] is True


def test_pareto_front_flags_dominated_points():
    # k=2 is dominated by k=3 on both coordinates.
    rm = sr.build_rule_metrics([1, 2, 3], [0.0, 0.2, 0.4], [0.0, 0.5, 0.5], normalize=False)
    assert list(rm["on_pareto_front"]) == [True, False, True]


def test_knee_requires_at_least_three_front_points():
    rm = sr.build_rule_metrics([1, 2], [0.0, 1.0], [0.0, 1.0], normalize=True)
    assert rm["knee_dist"].isna().all()


# --- B-3: threshold rules ---------------------------------------------------


def test_coverage_threshold_picks_smallest_feasible_k_on_a_monotone_grid():
    rm = _rule_metrics()
    for tau, expected in [(0.05, 3), (0.10, 4), (0.35, 4), (0.50, 5), (0.75, 6)]:
        _, sel = sr.select_by_coverage_threshold(rm, tau=tau, scale="raw", top_n=3, tie_breaker="smaller_k")
        assert sel["k_star"] == expected, f"tau={tau}"


def test_coverage_threshold_uses_argmin_ambiguity_on_a_non_monotone_grid():
    # k=5 clears the threshold with lower ambiguity than the smaller feasible k=4.
    rm = sr.build_rule_metrics([3, 4, 5], [0.10, 0.40, 0.45], [0.05, 0.30, 0.20], normalize=True)
    _, sel = sr.select_by_coverage_threshold(rm, tau=0.20, scale="raw", top_n=3, tie_breaker="smaller_k")
    assert sel["k_star"] == 5
    assert sel["n_feasible"] == 2


def test_ambiguity_ceiling_maximizes_coverage_under_the_cap():
    rm = _rule_metrics()
    _, sel = sr.select_by_ambiguity_ceiling(rm, beta=0.30, scale="raw", top_n=3, tie_breaker="smaller_k")
    assert sel["k_star"] == 4
    _, sel_loose = sr.select_by_ambiguity_ceiling(rm, beta=0.45, scale="raw", top_n=3, tie_breaker="smaller_k")
    assert sel_loose["k_star"] == 5


def test_threshold_sweep_and_stability_interval():
    rm = _rule_metrics()
    sweep = sr.sweep_coverage_threshold(
        rm, taus=[0.05, 0.10, 0.20, 0.30, 0.50], scale="raw", top_n=3, tie_breaker="smaller_k"
    )
    assert list(sweep["k_star"]) == [3, 4, 4, 4, 5]
    stability = sr.threshold_stability_interval(sweep, param_col="tau", k_star_primary=4)
    assert stability["n_matching"] == 3
    assert stability["interval_min"] == pytest.approx(0.10)
    assert stability["interval_max"] == pytest.approx(0.30)


def test_empty_feasible_set_yields_no_selection():
    rm = _rule_metrics()
    _, sel = sr.select_by_coverage_threshold(rm, tau=2.0, scale="raw", top_n=3, tie_breaker="smaller_k")
    assert sel["k_star"] is None
    assert sel["n_feasible"] == 0


# --- B-4: pooled recomputation and leave-one-out ----------------------------


def _td_frame() -> pd.DataFrame:
    rows = []
    targets = ["T1", "T2", "T3"]
    diseases = ["D1", "D2"]
    for ti, t in enumerate(targets):
        for k, (c, b) in enumerate(zip(C_GRID, B_GRID), start=1):
            amb = min(1.0, b + 0.01 * ti)
            for di, d in enumerate(diseases):
                rows.append(
                    {
                        "target_gene": t,
                        "disease_id": d,
                        "k": k,
                        "C_td": min(1.0, c + 0.01 * (ti + di)),
                        "B_td": amb,
                        "n_pairs_td": 2,
                    }
                )
    return pd.DataFrame(rows)


def test_pooled_recomputation_reproduces_global_metrics():
    td = _td_frame()
    prepared = sr.prepare_pooled_inputs(td, coverage_metric="cov_ratio")
    ks, c_vec, b_vec = sr.pooled_with_dropped(prepared)

    expected_c = td.groupby("k")["C_td"].median().to_numpy(float)
    expected_b = (
        td[["target_gene", "k", "B_td"]].drop_duplicates(["target_gene", "k"]).groupby("k")["B_td"].median().to_numpy(float)
    )
    np.testing.assert_allclose(c_vec, expected_c, rtol=0, atol=1e-12)
    np.testing.assert_allclose(b_vec, expected_b, rtol=0, atol=1e-12)

    k_metrics = pd.DataFrame({"k": ks, "C": expected_c, "B": expected_b})
    check = sr.check_pooled_consistency(prepared, k_metrics)
    assert check["passed"] is True
    assert check["max_abs_diff_C"] <= sr.POOLED_CONSISTENCY_TOL


def test_pooled_consistency_guard_raises_on_mismatch():
    prepared = sr.prepare_pooled_inputs(_td_frame(), coverage_metric="cov_ratio")
    ks = prepared["ks"]
    bogus = pd.DataFrame({"k": ks, "C": np.zeros(ks.size), "B": np.zeros(ks.size)})
    with pytest.raises(ValueError, match="do not match k_metrics"):
        sr.check_pooled_consistency(prepared, bogus)


def test_cov_nz_coverage_metric_is_rejected():
    with pytest.raises(NotImplementedError):
        sr.prepare_pooled_inputs(_td_frame(), coverage_metric="cov_nz")


def test_leave_one_out_reselects_per_fold():
    prepared = sr.prepare_pooled_inputs(_td_frame(), coverage_metric="cov_ratio")
    resampling = sr.run_resampling_selection(
        prepared,
        leave_one_target_out=True,
        leave_one_disease_out=True,
        n_bootstrap=2,
        seed=0,
        **_selection_kwargs(),
    )
    harmonic = resampling[resampling["rule"] == sr.RULE_HARMONIC_MEAN]
    assert set(harmonic["fold_type"]) == {"leave_one_target_out", "leave_one_disease_out", "bootstrap_targets"}
    assert int((harmonic["fold_type"] == "leave_one_target_out").sum()) == 3
    assert int((harmonic["fold_type"] == "leave_one_disease_out").sum()) == 2
    assert int((harmonic["fold_type"] == "bootstrap_targets").sum()) == 2
    assert set(harmonic["k_star"]) == {4}

    frequency, summary = sr.summarize_resampling(resampling, k_star_primary=4)
    loto = summary["folds"]
    entry = next(f for f in loto if f["fold_type"] == "leave_one_target_out" and f["rule"] == sr.RULE_HARMONIC_MEAN)
    assert entry["k_star_primary_frequency"] == pytest.approx(1.0)
    assert set(frequency["rule"]) >= {sr.RULE_HARMONIC_MEAN}


def test_bootstrap_is_seed_reproducible():
    prepared = sr.prepare_pooled_inputs(_td_frame(), coverage_metric="cov_ratio")
    kwargs = dict(leave_one_target_out=False, leave_one_disease_out=False, n_bootstrap=5, **_selection_kwargs())
    a = sr.run_resampling_selection(prepared, seed=7, **kwargs)
    b = sr.run_resampling_selection(prepared, seed=7, **kwargs)
    pd.testing.assert_frame_equal(a, b)


# --- B-5: orchestration -----------------------------------------------------


def test_run_selection_sensitivity_end_to_end():
    td = _td_frame()
    # k_metrics must be the pooled view of td_k_metrics, as the runner writes it.
    coverage_summary = pd.DataFrame(
        {"k": KS, "cov_ratio_median": td.groupby("k")["C_td"].median().to_numpy(float)}
    )
    ambiguity_summary = pd.DataFrame(
        {
            "k": KS,
            "B_amb_median": td[["target_gene", "k", "B_td"]]
            .drop_duplicates(["target_gene", "k"])
            .groupby("k")["B_td"]
            .median()
            .to_numpy(float),
        }
    )
    k_metrics = build_k_metrics(
        coverage_summary=coverage_summary,
        ambiguity_summary=ambiguity_summary,
        coverage_metric="cov_ratio",
        ambiguity_metric="amb_median",
        normalize=True,
    )
    settings = _normalize_metrics({})["selection_sensitivity"]
    result = sr.run_selection_sensitivity(
        k_metrics=k_metrics,
        td_k_metrics=td,
        primary_selection={"k_star": 4},
        settings=settings,
        coverage_metric="cov_ratio",
        normalize=True,
        zero_division=0.0,
        top_n=3,
        tie_breaker="smaller_k",
        seed=0,
    )

    agreement = result["tables"]["rule_agreement"]
    assert bool(agreement["agrees_with_primary"].all())
    assert not bool(agreement["label_dependent"].any())
    assert set(agreement["rule"]) >= {
        "harmonic_mean_primary",
        "geometric_mean",
        "pareto_knee",
        "coverage_threshold",
        "ambiguity_ceiling",
    }

    rule_selection = result["json"]["rule_selection"]
    assert rule_selection["k_star_primary"] == 4
    assert rule_selection["harmonic_consistency"]["checked"] is True
    assert rule_selection["pooled_consistency"]["passed"] is True
    assert rule_selection["label_free"] is True
    assert "resampling_k_star" in result["tables"]


def test_run_selection_sensitivity_without_td_metrics_skips_resampling():
    k_metrics = pd.DataFrame({"k": KS, "C": C_GRID, "B": B_GRID})
    settings = _normalize_metrics({})["selection_sensitivity"]
    result = sr.run_selection_sensitivity(
        k_metrics=k_metrics,
        td_k_metrics=pd.DataFrame(),
        primary_selection={},
        settings=settings,
        coverage_metric="cov_ratio",
        normalize=True,
        zero_division=0.0,
        top_n=3,
        tie_breaker="smaller_k",
    )
    assert "resampling_k_star" not in result["tables"]
    assert result["json"]["rule_selection"]["k_star_primary"] == 4
    assert result["json"]["rule_selection"]["pooled_consistency"]["checked"] is False


# --- config validation ------------------------------------------------------


def test_selection_sensitivity_defaults():
    settings = _normalize_metrics({})["selection_sensitivity"]
    assert settings["enabled"] is True
    assert settings["rules"] == list(sr.SENSITIVITY_RULES)
    assert settings["coverage_threshold"]["primary_tau"] == pytest.approx(sr.DEFAULT_PRIMARY_TAU)
    assert settings["resampling"]["leave_one_target_out"] is True


def test_primary_threshold_must_be_inside_the_sweep():
    with pytest.raises(ValueError, match="primary_tau"):
        _normalize_metrics({"selection_sensitivity": {"coverage_threshold": {"taus": [0.1, 0.2], "primary_tau": 0.15}}})
    with pytest.raises(ValueError, match="primary_beta"):
        _normalize_metrics({"selection_sensitivity": {"ambiguity_ceiling": {"betas": [0.1, 0.2], "primary_beta": 0.15}}})


def test_unknown_rule_and_knee_method_are_rejected():
    with pytest.raises(ValueError, match="Unknown metrics.selection_sensitivity.rules"):
        _normalize_metrics({"selection_sensitivity": {"rules": ["ranking_ap"]}})
    with pytest.raises(ValueError, match="knee_method"):
        _normalize_metrics({"selection_sensitivity": {"pareto": {"knee_method": "elbow"}}})


def test_primary_integration_method_is_still_locked():
    with pytest.raises(ValueError, match="metrics.integration.method"):
        _normalize_metrics({"integration": {"method": "geometric_mean"}})
