"""Tests for the NCS (L1000 NCS) ablation reference line in src/val2/.

Covers both the standalone target-level scorer (``val2.ncs``) and its
integration into ``val2.legacy.run_val2`` as a special method that does not
perturb the deterministic methods' evaluation space.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from val2 import legacy
from val2.ncs import compute_ncs_target_scores


# Feature space F = 6 genes (NCBIGENE:1..6). Disease D1: up = {1,2}, down = {5,6}.
FEATURE_GENES = [f"NCBIGENE:{i}" for i in range(1, 7)]


def _write_continuous_kd(tmp_path: Path) -> Path:
    """Continuous KD reference profiles for one cell line (C1).

    Mirror-image profiles guarantee both positive and negative WTCS values in
    the reference population so the per-cell NCS normalization is defined.
    """
    rows = [
        # (ncbigene_id, cell_id, values over FEATURE_GENES)
        ("T1", "C1", [10.0, 9.0, 0.0, 0.0, -9.0, -10.0]),   # up-high/down-low -> WTCS +
        ("T2", "C1", [-10.0, -9.0, 0.0, 0.0, 9.0, 10.0]),   # up-low/down-high -> WTCS -
        ("T3", "C1", [8.0, 7.0, 1.0, 1.0, -7.0, -8.0]),     # WTCS +
        ("T4", "C1", [-8.0, -7.0, 1.0, 1.0, 7.0, 8.0]),     # WTCS -
    ]
    records = []
    for gene_id, cell_id, vals in rows:
        rec = {"ncbigene_id": gene_id, "cell_id": cell_id}
        rec.update({g: v for g, v in zip(FEATURE_GENES, vals)})
        records.append(rec)
    df = pd.DataFrame.from_records(records, columns=["ncbigene_id", "cell_id", *FEATURE_GENES])
    path = tmp_path / "continuous_kd.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


def _disease_signatures() -> dict:
    return {
        # D1: valid query (both up and down present in F).
        "D1": {"NCBIGENE:1": 1, "NCBIGENE:2": 1, "NCBIGENE:5": -1, "NCBIGENE:6": -1},
        # D2: invalid query (no down gene) -> must produce no NCS rows.
        "D2": {"NCBIGENE:1": 1, "NCBIGENE:2": 1},
    }


def _labels() -> pd.DataFrame:
    # T_missing has no continuous KD profile -> undefined NCS pair.
    return pd.DataFrame(
        [
            {"disease_id": "D1", "target_gene": "T1", "cell_id": "C1", "label": 1},
            {"disease_id": "D1", "target_gene": "T2", "cell_id": "C1", "label": 0},
            {"disease_id": "D1", "target_gene": "T_missing", "cell_id": "C1", "label": 0},
            {"disease_id": "D2", "target_gene": "T1", "cell_id": "C1", "label": 1},
        ]
    )


def test_ncs_target_scores_floor_and_query_skip(tmp_path: Path) -> None:
    parquet = _write_continuous_kd(tmp_path)
    out = compute_ncs_target_scores(
        labels_df=_labels(),
        disease_signatures=_disease_signatures(),
        continuous_kd_parquet=parquet,
        disease_signature_source="val2",
        undefined_policy="floor",
    )

    # D2 has an invalid query -> no rows at all.
    assert set(out["disease_id"].unique()) == {"D1"}

    d1 = out.set_index("target_gene")
    # Valid targets present in the KD get a finite, non-floor score.
    assert set(d1.index) == {"T1", "T2", "T_missing"}
    for t in ("T1", "T2"):
        assert bool(d1.loc[t, "is_floor"]) is False
        assert np.isfinite(d1.loc[t, "score"])
        assert int(d1.loc[t, "n_valid_cells"]) == 1

    # Missing profile -> floored to (min valid score in disease) - 1, ranks last.
    assert bool(d1.loc["T_missing", "is_floor"]) is True
    assert int(d1.loc["T_missing", "n_valid_cells"]) == 0
    valid_min = min(float(d1.loc["T1", "score"]), float(d1.loc["T2", "score"]))
    assert d1.loc["T_missing", "score"] == pytest.approx(valid_min - 1.0)


def test_ncs_target_scores_drop_policy(tmp_path: Path) -> None:
    parquet = _write_continuous_kd(tmp_path)
    out = compute_ncs_target_scores(
        labels_df=_labels(),
        disease_signatures=_disease_signatures(),
        continuous_kd_parquet=parquet,
        disease_signature_source="val2",
        undefined_policy="drop",
    )
    # Under 'drop', the undefined pair is omitted entirely.
    assert set(zip(out["disease_id"], out["target_gene"])) == {("D1", "T1"), ("D1", "T2")}
    assert not out["is_floor"].any()


def _toy_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_edge("T1", "g1", polarity=1)
    g.add_edge("g1", "g2", polarity=-1)
    g.add_edge("T2", "g2", polarity=1)
    g.add_edge("g2", "g3", polarity=1)
    return g


def _toy_run_inputs():
    graph = _toy_graph()
    pert_sigs = {
        ("T1", "C1"): {"g1": 1, "g2": -1, "g3": 1},
        ("T2", "C1"): {"g1": -1, "g2": 1, "g3": -1},
    }
    dis_sigs = {"D1": {"g1": -1, "g2": 1, "g3": -1}}
    labels = pd.DataFrame(
        [
            {"disease_id": "D1", "target_gene": "T1", "cell_id": "C1", "label": 1},
            {"disease_id": "D1", "target_gene": "T2", "cell_id": "C1", "label": 0},
        ]
    )
    return graph, pert_sigs, dis_sigs, labels


def test_run_val2_ncs_flat_across_k_and_no_perturbation() -> None:
    graph, pert_sigs, dis_sigs, labels = _toy_run_inputs()
    ncs_scores = pd.DataFrame(
        [
            {"disease_id": "D1", "target_gene": "T1", "score": 2.0},
            {"disease_id": "D1", "target_gene": "T2", "score": 1.0},
        ]
    )
    ks = [1, 2]

    scores_with, metrics_with, _ = legacy.run_val2(
        graph, pert_sigs=pert_sigs, dis_sigs=dis_sigs, labels=labels,
        ks=ks, methods=["proposed", "ncs"], start_sign=-1,
        K_list=[10], tie_policy="expected", ncs_scores=ncs_scores,
    )

    # NCS metrics exist for every k and are identical across k (flat line).
    ncs_metrics = metrics_with[metrics_with["method"] == "ncs"]
    assert set(ncs_metrics["k"].astype(int)) == set(ks)
    ap_by_k = ncs_metrics.groupby("k")["AP"].first()
    assert ap_by_k.nunique() == 1

    # NCS scores are injected for both label pairs at every k.
    ncs_scores_rows = scores_with[scores_with["method"] == "ncs"]
    assert len(ncs_scores_rows) == len(ks) * 2

    # Proposed's evaluated (disease, target, cell, k) set is unchanged whether or
    # not NCS is included (NCS never enters filter_by_min_n_meas_all_methods).
    scores_only, _, _ = legacy.run_val2(
        graph, pert_sigs=pert_sigs, dis_sigs=dis_sigs, labels=labels,
        ks=ks, methods=["proposed"], start_sign=-1,
        K_list=[10], tie_policy="expected",
    )
    key = ["disease_id", "target_gene", "cell_id", "k"]
    prop_with = scores_with[scores_with["method"] == "proposed"][key]
    prop_only = scores_only[scores_only["method"] == "proposed"][key]
    assert set(map(tuple, prop_with.to_numpy())) == set(map(tuple, prop_only.to_numpy()))


def test_run_val2_ncs_requires_scores() -> None:
    graph, pert_sigs, dis_sigs, labels = _toy_run_inputs()
    with pytest.raises(ValueError, match="ncs_scores"):
        legacy.run_val2(
            graph, pert_sigs=pert_sigs, dis_sigs=dis_sigs, labels=labels,
            ks=[1], methods=["proposed", "ncs"], start_sign=-1,
            K_list=[10], tie_policy="expected", ncs_scores=None,
        )
