"""Tests for the per-disease degree-dependence and top-K support diagnostics."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from val2_baseline import degree_support  # noqa: E402


def _feat(rows):
    """Build the degree-diagnostics frame shape _build_degree_diagnostics_frame emits."""
    df = pd.DataFrame(rows)
    n_by_disease = df.groupby("disease_id")["target_gene"].transform("count")
    for method in ("CALIPER", "RWR"):
        df[f"{method}_rank_min"] = df.groupby("disease_id")[method].rank(method="min", ascending=False)
        df[f"{method}_rank_pct"] = df[f"{method}_rank_min"] / n_by_disease
    df["positive"] = df["label"].astype(int).eq(1)
    return df


def _monotone_disease(disease_id, n, *, caliper_sign=1, rwr_sign=1, n_positive=None):
    """One disease whose scores are (anti)monotone in target degree."""
    n_positive = n if n_positive is None else n_positive
    rows = []
    for i in range(n):
        degree = float(i + 1)
        rows.append(
            {
                "disease_id": disease_id,
                "target_gene": f"{disease_id}_G{i}",
                "degree_H": degree,
                "CALIPER": caliper_sign * degree,
                "RWR": rwr_sign * degree,
                "label": 1 if i < n_positive else 0,
            }
        )
    return rows


class TestPerDiseaseSpearman:
    def test_monotone_and_antitone_rho(self):
        feat = _feat(
            _monotone_disease("D1", 10, caliper_sign=1, rwr_sign=-1)
            + _monotone_disease("D2", 10, caliper_sign=1, rwr_sign=-1)
        )
        by_disease, summary = degree_support.compute_per_disease_degree_spearman(feat, n_boot=50)
        allset = by_disease[by_disease["candidate_set"] == "all"]
        assert set(allset["method"]) == {"CALIPER", "RWR"}
        assert np.allclose(allset.loc[allset["method"] == "CALIPER", "spearman_rho"], 1.0)
        assert np.allclose(allset.loc[allset["method"] == "RWR", "spearman_rho"], -1.0)
        row = summary[(summary["method"] == "CALIPER") & (summary["candidate_set"] == "all")].iloc[0]
        assert row["median_rho"] == pytest.approx(1.0)
        assert row["n_diseases"] == 2
        assert row["frac_rho_ge_threshold"] == pytest.approx(1.0)

    def test_pooled_rho_matches_direct_computation(self):
        feat = _feat(
            _monotone_disease("D1", 12, rwr_sign=-1) + _monotone_disease("D2", 12, rwr_sign=-1)
        )
        _by_disease, summary = degree_support.compute_per_disease_degree_spearman(feat, n_boot=50)
        for candidate_set in ("all", "positive"):
            pool = feat[feat["positive"]] if candidate_set == "positive" else feat
            for method in ("CALIPER", "RWR"):
                expected, _p = degree_support._spearman(pool["degree_H"], pool[method])
                got = summary[
                    (summary["method"] == method) & (summary["candidate_set"] == candidate_set)
                ]["pooled_rho"].iloc[0]
                assert got == pytest.approx(expected)

    def test_constant_score_disease_is_undefined_not_counted(self):
        rows = _monotone_disease("D1", 10) + _monotone_disease("D2", 10)
        feat = _feat(rows)
        feat.loc[feat["disease_id"] == "D2", "CALIPER"] = 0.5  # constant score
        by_disease, summary = degree_support.compute_per_disease_degree_spearman(feat, n_boot=50)
        d2 = by_disease[
            (by_disease["disease_id"] == "D2")
            & (by_disease["method"] == "CALIPER")
            & (by_disease["candidate_set"] == "all")
        ].iloc[0]
        assert bool(d2["included"]) is True
        assert np.isnan(d2["spearman_rho"])
        row = summary[(summary["method"] == "CALIPER") & (summary["candidate_set"] == "all")].iloc[0]
        assert row["n_diseases"] == 1
        assert row["n_diseases_undefined"] == 1
        assert row["median_rho"] == pytest.approx(1.0)

    def test_min_positives_gate_only_affects_positive_variant(self):
        feat = _feat(
            _monotone_disease("D1", 12, n_positive=10) + _monotone_disease("D2", 12, n_positive=4)
        )
        by_disease, summary = degree_support.compute_per_disease_degree_spearman(
            feat, min_positives=5, n_boot=50
        )
        gated = by_disease[
            (by_disease["disease_id"] == "D2") & (by_disease["candidate_set"] == "positive")
        ]
        assert not gated["included"].any()
        kept = by_disease[
            (by_disease["disease_id"] == "D2") & (by_disease["candidate_set"] == "all")
        ]
        assert kept["included"].all()
        assert (
            summary[(summary["method"] == "CALIPER") & (summary["candidate_set"] == "positive")][
                "n_diseases"
            ].iloc[0]
            == 1
        )
        assert (
            summary[(summary["method"] == "CALIPER") & (summary["candidate_set"] == "all")][
                "n_diseases"
            ].iloc[0]
            == 2
        )

    def test_paired_delta_and_holm_only_on_baseline_rows(self):
        feat = _feat(
            _monotone_disease("D1", 12, rwr_sign=-1)
            + _monotone_disease("D2", 12, rwr_sign=-1)
            + _monotone_disease("D3", 12, rwr_sign=-1)
        )
        _by_disease, summary = degree_support.compute_per_disease_degree_spearman(feat, n_boot=50)
        caliper = summary[(summary["method"] == "CALIPER") & (summary["candidate_set"] == "all")].iloc[0]
        assert np.isnan(caliper["delta_vs_caliper"])
        assert np.isnan(caliper["p_holm"])
        rwr = summary[(summary["method"] == "RWR") & (summary["candidate_set"] == "all")].iloc[0]
        # rho(CALIPER) = +1, rho(RWR) = -1 for every disease.
        assert rwr["delta_vs_caliper"] == pytest.approx(2.0)
        assert rwr["n_diseases_paired"] == 3
        assert np.isfinite(rwr["p_holm"])

    def test_empty_or_missing_method_returns_empty_frames(self):
        by_disease, summary = degree_support.compute_per_disease_degree_spearman(pd.DataFrame())
        assert by_disease.empty and summary.empty
        feat = _feat(_monotone_disease("D1", 6))
        by_disease, summary = degree_support.compute_per_disease_degree_spearman(
            feat, methods=("MISSING",)
        )
        assert by_disease.empty and summary.empty


class TestTopKOverlap:
    def test_disjoint_and_shared_top_k(self):
        # D1: CALIPER ranks G0..G2 top, RWR ranks G3..G5 top -> disjoint at K=3.
        # D2: identical scores -> identical top-3 -> Jaccard 1.
        rows = []
        for i, (cal, rwr) in enumerate([(6, 1), (5, 2), (4, 3), (3, 4), (2, 5), (1, 6)]):
            rows.append(
                {
                    "disease_id": "D1", "target_gene": f"G{i}", "degree_H": float(i + 1),
                    "CALIPER": float(cal), "RWR": float(rwr), "label": 0,
                }
            )
        for i in range(6):
            rows.append(
                {
                    "disease_id": "D2", "target_gene": f"H{i}", "degree_H": float(i + 1),
                    "CALIPER": float(6 - i), "RWR": float(6 - i), "label": 0,
                }
            )
        feat = _feat(rows)
        by_disease, summary = degree_support.compute_topk_overlap(feat, k_list=(3,), n_boot=50)
        d1 = by_disease[by_disease["disease_id"] == "D1"].iloc[0]
        assert (d1["n_top_a"], d1["n_top_b"], d1["n_both"]) == (3, 3, 0)
        assert d1["n_caliper_only"] == 3 and d1["n_rwr_only"] == 3
        assert d1["jaccard"] == pytest.approx(0.0)
        d2 = by_disease[by_disease["disease_id"] == "D2"].iloc[0]
        assert d2["n_both"] == 3 and d2["jaccard"] == pytest.approx(1.0)
        row = summary.iloc[0]
        assert row["K"] == 3
        assert row["pooled_n_both"] == 3
        assert row["pooled_n_caliper_only"] == 3 and row["pooled_n_rwr_only"] == 3
        assert row["median_jaccard"] == pytest.approx(0.5)

    def test_boundary_tie_inflates_selected_count(self):
        # Four targets tied at the top: rank_min <= 2 selects all four at K=2.
        rows = [
            {
                "disease_id": "D1", "target_gene": f"G{i}", "degree_H": float(i + 1),
                "CALIPER": 1.0, "RWR": float(10 - i), "label": 0,
            }
            for i in range(4)
        ]
        feat = _feat(rows)
        by_disease, summary = degree_support.compute_topk_overlap(feat, k_list=(2,), n_boot=50)
        row = by_disease.iloc[0]
        assert row["n_top_a"] == 4
        assert row["boundary_tie_size_a"] == 4
        assert row["boundary_tie_size_b"] == 0
        assert summary.iloc[0]["n_diseases_boundary_tie_a"] == 1

    def test_missing_rank_columns_returns_empty(self):
        by_disease, summary = degree_support.compute_topk_overlap(
            pd.DataFrame({"disease_id": ["D1"], "target_gene": ["G0"]}), k_list=(3,)
        )
        assert by_disease.empty and summary.empty


class TestTopKGroupProfile:
    def _feat_two_groups(self):
        rows = []
        for i, (cal, rwr) in enumerate([(6, 1), (5, 2), (4, 3), (3, 4), (2, 5), (1, 6)]):
            rows.append(
                {
                    "disease_id": "D1", "target_gene": f"G{i}", "degree_H": float(10 * (i + 1)),
                    "CALIPER": float(cal), "RWR": float(rwr), "label": 0,
                }
            )
        return _feat(rows)

    def test_groups_partition_pairs_and_profile_medians(self):
        feat = self._feat_two_groups()
        support = pd.DataFrame(
            {
                "disease_id": ["D1"] * 6,
                "target_gene": [f"G{i}" for i in range(6)],
                "n_common_cells": [1, 2, 3, 8, 9, 10],
                "mean_covAlign": [0.1] * 6,
                "mean_support": [1.0] * 6,
            }
        )
        pair_df, profile_df, tests_df = degree_support.compute_topk_group_profile(
            feat, support, k_list=(2,)
        )
        groups = degree_support.assign_topk_groups(feat, 2)
        assert list(groups) == [
            degree_support.GROUP_A_ONLY,
            degree_support.GROUP_A_ONLY,
            degree_support.GROUP_NEITHER,
            degree_support.GROUP_NEITHER,
            degree_support.GROUP_B_ONLY,
            degree_support.GROUP_B_ONLY,
        ]
        # pair_df keeps only the three top-K groups; 'neither' lives in profile_df.
        assert set(pair_df["group"]) == {degree_support.GROUP_A_ONLY, degree_support.GROUP_B_ONLY}
        assert len(pair_df) == 4
        a_only = profile_df[profile_df["group"] == degree_support.GROUP_A_ONLY].iloc[0]
        b_only = profile_df[profile_df["group"] == degree_support.GROUP_B_ONLY].iloc[0]
        assert a_only["degree_H_median"] == pytest.approx(15.0)
        assert b_only["degree_H_median"] == pytest.approx(55.0)
        assert a_only["n_common_cells_median"] == pytest.approx(1.5)
        assert b_only["n_common_cells_median"] == pytest.approx(9.5)
        assert degree_support.GROUP_NEITHER in set(profile_df["group"])
        assert set(tests_df["K"]) == {2}
        assert int(tests_df["n_tests_in_family"].iloc[0]) == len(tests_df)

    def test_missing_support_rows_yield_nan_not_error(self):
        feat = self._feat_two_groups()
        support = pd.DataFrame(
            {
                "disease_id": ["D1"],
                "target_gene": ["G0"],
                "n_common_cells": [4],
                "mean_covAlign": [0.2],
                "mean_support": [3.0],
            }
        )
        pair_df, profile_df, _tests = degree_support.compute_topk_group_profile(
            feat, support, k_list=(2,)
        )
        assert pair_df["n_common_cells"].isna().sum() == 3
        b_only = profile_df[profile_df["group"] == degree_support.GROUP_B_ONLY].iloc[0]
        assert b_only["n_common_cells_n"] == 0
        assert np.isnan(b_only["n_common_cells_median"])

    def test_empty_support_still_profiles_degree(self):
        feat = self._feat_two_groups()
        pair_df, profile_df, tests_df = degree_support.compute_topk_group_profile(
            feat, pd.DataFrame(), k_list=(2,)
        )
        assert not pair_df.empty
        assert "degree_H_median" in profile_df.columns
        assert set(tests_df["metric"]) == {"degree_H"}


class TestLoadSupportFeatures:
    def test_reads_k_star_rows_and_merges_availability(self, tmp_path):
        (tmp_path / "scores").mkdir()
        (tmp_path / "common_sets").mkdir()
        (tmp_path / "metadata.json").write_text('{"k_star": 4}')
        pd.DataFrame(
            {
                "disease_id": ["D1", "D1"],
                "target_gene": ["G0", "G0"],
                "n_common_cells": [9, 1],
                "mean_n_meas": [22.0, 3.0],
                "mean_covAlign": [0.03, 0.01],
                "mean_support": [2.0, 1.0],
                "k": [4, 3],
            }
        ).to_csv(tmp_path / "scores" / "caliper_target_level.tsv", sep="\t", index=False)
        pd.DataFrame(
            {
                "disease_id": ["D1"],
                "target_gene": ["G0"],
                "ncs_n_common_cells": [7],
                "s_d_h_size": [820],
            }
        ).to_csv(tmp_path / "common_sets" / "availability_by_pair.tsv", sep="\t", index=False)

        support = degree_support.load_support_features(tmp_path)
        assert len(support) == 1
        row = support.iloc[0]
        assert row["n_common_cells"] == 9  # k_star row, not the k=3 row
        assert row["ncs_n_common_cells"] == 7
        assert row["s_d_h_size"] == 820

    def test_missing_files_return_empty(self, tmp_path):
        support = degree_support.load_support_features(tmp_path)
        assert support.empty


class TestPairedEstimators:
    def test_paired_median_ci_p_matches_median_and_bounds_ci(self):
        rng = np.random.default_rng(0)
        deltas = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        median, lo, hi, p = degree_support.paired_median_ci_p(deltas, rng, n_boot=200)
        assert median == pytest.approx(0.3)
        assert lo <= median <= hi
        assert 0.0 < p <= 1.0

    def test_paired_median_ci_p_all_zero_gives_nan_p(self):
        rng = np.random.default_rng(0)
        median, lo, hi, p = degree_support.paired_median_ci_p(np.zeros(5), rng, n_boot=50)
        assert median == pytest.approx(0.0)
        assert lo == pytest.approx(0.0) and hi == pytest.approx(0.0)
        assert np.isnan(p)

    def test_empty_input_is_all_nan(self):
        rng = np.random.default_rng(0)
        assert all(np.isnan(v) for v in degree_support.paired_median_ci_p(np.array([]), rng, n_boot=10))
        assert all(np.isnan(v) for v in degree_support.median_ci_bootstrap(np.array([]), rng, n_boot=10))
