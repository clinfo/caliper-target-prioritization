"""Verify that n_jobs > 1 produces numerically identical outputs to n_jobs = 1.

Uses a small synthetic KG with multiple targets, diseases, and cells so that
all three scoring methods (proposed, kg_nosign, noKG_sizematch) and the
precompute+size-match parallel paths are exercised.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from val2 import legacy


# ---------------------------------------------------------------------------
# Fixtures: a 7-node KG with 3 targets and 2 diseases × 2 cell lines
# ---------------------------------------------------------------------------

def _build_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    # Target T1 neighbourhood
    G.add_edge("T1", "g1", polarity=1)
    G.add_edge("T1", "g2", polarity=-1)
    G.add_edge("g1", "g3", polarity=-1)
    G.add_edge("g2", "g3", polarity=1)
    # Target T2 neighbourhood
    G.add_edge("T2", "g3", polarity=1)
    G.add_edge("T2", "g4", polarity=-1)
    G.add_edge("g3", "g5", polarity=1)
    # Target T3 (no KG edges -> nodes_k empty)
    G.add_node("T3")
    return G


def _build_inputs() -> Tuple[
    nx.MultiDiGraph,
    Dict,
    Dict,
    pd.DataFrame,
]:
    G = _build_graph()
    pert_sigs = {
        ("T1", "C1"): {"g1": 1, "g2": -1, "g3": 1, "g4": -1, "g5": 1},
        ("T1", "C2"): {"g1": -1, "g2": 1, "g3": -1, "g4": 1, "g5": -1},
        ("T2", "C1"): {"g1": -1, "g2": 1, "g3": -1, "g4": 1, "g5": 1},
        ("T2", "C2"): {"g1": 1, "g2": -1, "g3": 1, "g4": -1, "g5": -1},
        ("T3", "C1"): {"g1": 1, "g2": 1, "g3": -1, "g4": 0, "g5": 0},
    }
    dis_sigs = {
        "D1": {"g1": -1, "g2": 1, "g3": -1, "g4": 1, "g5": 1},
        "D2": {"g1": 1, "g2": -1, "g3": 1, "g4": -1, "g5": -1},
        "D3": {},  # empty disease -> score_pair returns (0,0,0,...)
    }
    labels = pd.DataFrame([
        {"disease_id": "D1", "target_gene": "T1", "cell_id": "C1", "label": 1},
        {"disease_id": "D1", "target_gene": "T1", "cell_id": "C2", "label": 1},
        {"disease_id": "D1", "target_gene": "T2", "cell_id": "C1", "label": 0},
        {"disease_id": "D1", "target_gene": "T3", "cell_id": "C1", "label": 0},
        {"disease_id": "D2", "target_gene": "T1", "cell_id": "C1", "label": 0},
        {"disease_id": "D2", "target_gene": "T2", "cell_id": "C2", "label": 1},
        {"disease_id": "D3", "target_gene": "T1", "cell_id": "C1", "label": 1},
        {"disease_id": "D3", "target_gene": "T2", "cell_id": "C1", "label": 0},
    ])
    return G, pert_sigs, dis_sigs, labels


def _sort_df(df: pd.DataFrame, key_cols: list) -> pd.DataFrame:
    """Sort by key_cols (all present) and reset index."""
    present = [c for c in key_cols if c in df.columns]
    return df.sort_values(present).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

METHODS_FULL = ["proposed", "kg_nosign", "kg_sign_noalign", "noKG_sizematch"]
METHODS_NO_SM = ["proposed", "kg_nosign", "kg_sign_noalign"]
KS = [1, 2, 3]
K_LIST = [5]
SM_REPS = 20  # enough reps, small enough for fast tests


@pytest.mark.parametrize("methods,n_jobs", [
    (METHODS_FULL, 2),
    (METHODS_FULL, 4),
    (METHODS_NO_SM, 2),
])
def test_parallel_scores_match_sequential(methods, n_jobs):
    """n_jobs > 1 scores must be numerically identical to n_jobs = 1."""
    G, pert_sigs, dis_sigs, labels = _build_inputs()
    common_kwargs = dict(
        pert_sigs=pert_sigs, dis_sigs=dis_sigs, labels=labels,
        ks=KS, methods=methods, start_sign=-1,
        sizematch_reps=SM_REPS, sizematch_seed=42,
        K_list=K_LIST, tie_policy="expected",
    )

    scores1, metrics1, _ = legacy.run_val2(G, **common_kwargs, n_jobs=1)
    scoresN, metricsN, _ = legacy.run_val2(G, **common_kwargs, n_jobs=n_jobs)

    # Align both DataFrames by the same sort key
    key_scores = ["disease_id", "target_gene", "cell_id", "k", "method"]
    key_metrics = ["disease_id", "k", "method"]

    s1 = _sort_df(scores1, key_scores)
    sN = _sort_df(scoresN, key_scores)
    m1 = _sort_df(metrics1, key_metrics)
    mN = _sort_df(metricsN, key_metrics)

    assert list(s1.columns) == list(sN.columns), "scores columns differ"
    assert list(m1.columns) == list(mN.columns), "metrics columns differ"
    assert len(s1) == len(sN), f"scores row count differs: {len(s1)} vs {len(sN)}"
    assert len(m1) == len(mN), f"metrics row count differs: {len(m1)} vs {len(mN)}"

    for col in s1.select_dtypes(include="number").columns:
        v1 = s1[col].to_numpy(dtype=float)
        vN = sN[col].to_numpy(dtype=float)
        ok = np.allclose(v1, vN, equal_nan=True)
        assert ok, f"scores column '{col}' differs between n_jobs=1 and n_jobs={n_jobs}"

    for col in m1.select_dtypes(include="number").columns:
        v1 = m1[col].to_numpy(dtype=float)
        vN = mN[col].to_numpy(dtype=float)
        ok = np.allclose(v1, vN, equal_nan=True)
        assert ok, f"metrics column '{col}' differs between n_jobs=1 and n_jobs={n_jobs}"


def test_parallel_pred_ambiguity_matches_sequential():
    """pred_ambiguity_df must also be identical across n_jobs."""
    G, pert_sigs, dis_sigs, labels = _build_inputs()
    common_kwargs = dict(
        pert_sigs=pert_sigs, dis_sigs=dis_sigs, labels=labels,
        ks=KS, methods=METHODS_NO_SM, start_sign=-1,
        K_list=K_LIST, tie_policy="expected",
    )
    _, _, amb1 = legacy.run_val2(G, **common_kwargs, n_jobs=1)
    _, _, ambN = legacy.run_val2(G, **common_kwargs, n_jobs=2)

    key = ["target_gene", "k", "start_sign"]
    a1 = _sort_df(amb1, key)
    aN = _sort_df(ambN, key)
    assert len(a1) == len(aN)
    for col in a1.select_dtypes(include="number").columns:
        assert np.allclose(
            a1[col].to_numpy(dtype=float),
            aN[col].to_numpy(dtype=float),
            equal_nan=True,
        ), f"pred_ambiguity column '{col}' differs"


def test_n_jobs_1_is_unchanged():
    """Sanity check: n_jobs=1 still returns the expected number of rows."""
    G, pert_sigs, dis_sigs, labels = _build_inputs()
    scores, metrics, _ = legacy.run_val2(
        G, pert_sigs=pert_sigs, dis_sigs=dis_sigs, labels=labels,
        ks=KS, methods=["proposed", "kg_nosign"], start_sign=-1,
        K_list=K_LIST, tie_policy="expected", n_jobs=1,
    )
    assert len(scores) > 0
    assert len(metrics) > 0
    assert set(metrics["method"].unique()) == {"proposed", "kg_nosign"}
    assert set(metrics["k"].astype(int).unique()) == set(KS)
