from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from val2.label_confidence import (
    assign_confidence_score_groups,
    confidence_positive_pairs,
    evaluate_label_confidence_by_disease,
    map_target_disease_confidence,
    summarize_label_confidence,
    validate_all_evidence_labels,
)


def _confidence_inputs() -> tuple[pd.DataFrame, ...]:
    target_disease = pd.DataFrame(
        [
            {
                "target_node_id": "n:target:E1",
                "disease_node_id": "n:disease:D1",
                "tier": 1,
                "direction": "beneficial_inhibition",
            },
            {
                "target_node_id": "n:target:E2",
                "disease_node_id": "n:disease:D2",
                "tier": 2,
                "direction": "beneficial_inhibition",
            },
        ]
    )
    triplets = pd.DataFrame(
        [
            {
                "triplet_id": "T1",
                "target_node_id": "n:target:E1",
                "disease_node_id": "n:disease:D1",
            },
            {
                "triplet_id": "T2",
                "target_node_id": "n:target:E2",
                "disease_node_id": "n:disease:D2",
            },
        ]
    )
    triplet_labels = pd.DataFrame(
        [{"triplet_id": "T1", "phase": 4.0}, {"triplet_id": "T2", "phase": 2.0}]
    )
    nodes = pd.DataFrame(
        [
            {"node_id": "n:target:E1", "canonical_curie": "E1"},
            {"node_id": "n:target:E2", "canonical_curie": "E2"},
        ]
    )
    node_xrefs = pd.DataFrame(
        [
            {"node_id": "n:disease:D1", "xref_id": "DOID:1"},
            {"node_id": "n:disease:D2", "xref_id": "DOID:2"},
        ]
    )
    crosswalk = pd.DataFrame(
        [
            {"target_id": "E1", "gene_id": "NCBIGENE:1"},
            {"target_id": "E2", "gene_id": "NCBIGENE:2"},
        ]
    )
    return target_disease, triplets, triplet_labels, nodes, node_xrefs, crosswalk


def test_map_target_disease_confidence_and_strata():
    confidence = map_target_disease_confidence(*_confidence_inputs())

    assert set(confidence["target_gene"]) == {"NCBIGENE:1", "NCBIGENE:2"}
    assert confidence_positive_pairs(confidence, "approved") == {
        ("DOID:1", "NCBIGENE:1")
    }
    assert confidence_positive_pairs(confidence, "phase_ge_2") == {
        ("DOID:1", "NCBIGENE:1"),
        ("DOID:2", "NCBIGENE:2"),
    }


def test_summary_relabels_fixed_scores_and_validates_all_evidence():
    confidence = map_target_disease_confidence(*_confidence_inputs())
    scores = pd.DataFrame(
        [
            {
                "disease_id": "DOID:1",
                "target_gene": "NCBIGENE:1",
                "method": "proposed",
                "score": 0.9,
                "pair_label": 1,
            },
            {
                "disease_id": "DOID:1",
                "target_gene": "NCBIGENE:2",
                "method": "proposed",
                "score": 0.1,
                "pair_label": 0,
            },
            {
                "disease_id": "DOID:2",
                "target_gene": "NCBIGENE:1",
                "method": "proposed",
                "score": 0.2,
                "pair_label": 0,
            },
            {
                "disease_id": "DOID:2",
                "target_gene": "NCBIGENE:2",
                "method": "proposed",
                "score": 0.8,
                "pair_label": 1,
            },
        ]
    )

    audit = validate_all_evidence_labels(scores, confidence)
    summary = summarize_label_confidence(scores, confidence)

    assert audit == {"n_pairs": 4, "n_positive_pairs": 2, "n_mismatches": 0}
    all_evidence = summary[summary["stratum"].eq("all_evidence")].iloc[0]
    approved = summary[summary["stratum"].eq("approved")].iloc[0]
    assert all_evidence["mAP"] == pytest.approx(1.0)
    assert int(all_evidence["n_diseases"]) == 2
    assert approved["mAP"] == pytest.approx(1.0)
    assert int(approved["n_diseases"]) == 1


def test_normalized_ap_uses_disease_prevalence_and_common_cohort():
    confidence = map_target_disease_confidence(*_confidence_inputs())
    scores = pd.DataFrame(
        [
            {
                "disease_id": "DOID:1",
                "target_gene": "NCBIGENE:1",
                "method": "proposed",
                "score": 0.5,
                "pair_label": 1,
            },
            {
                "disease_id": "DOID:1",
                "target_gene": "NCBIGENE:2",
                "method": "proposed",
                "score": 0.9,
                "pair_label": 0,
            },
            {
                "disease_id": "DOID:1",
                "target_gene": "NCBIGENE:3",
                "method": "proposed",
                "score": 0.1,
                "pair_label": 0,
            },
            {
                "disease_id": "DOID:2",
                "target_gene": "NCBIGENE:2",
                "method": "proposed",
                "score": 0.8,
                "pair_label": 1,
            },
        ]
    )

    disease_level = evaluate_label_confidence_by_disease(
        scores,
        confidence,
        evaluation_diseases={"DOID:1"},
    )
    summary = summarize_label_confidence(
        scores,
        confidence,
        evaluation_diseases={"DOID:1"},
    )

    all_evidence = disease_level[
        disease_level["stratum"].eq("all_evidence")
    ].iloc[0]
    all_summary = summary[summary["stratum"].eq("all_evidence")].iloc[0]
    assert all_evidence["AP"] == pytest.approx(0.5)
    assert all_evidence["chance"] == pytest.approx(1 / 3)
    assert all_evidence["normalized_AP"] == pytest.approx(0.25)
    assert all_summary["normalized_mAP"] == pytest.approx(0.25)
    assert int(all_summary["n_diseases"]) == 1
    assert set(disease_level["disease_id"]) == {"DOID:1"}


def test_confidence_score_groups_are_mutually_exclusive():
    confidence = pd.DataFrame(
        [
            {
                "disease_id": "DOID:1",
                "target_gene": f"NCBIGENE:{index}",
                "tier": tier,
                "phase": phase,
                "direction": "beneficial_inhibition",
            }
            for index, tier, phase in (
                (1, 3, 1),
                (2, 2, 2),
                (3, 2, 3),
                (4, 1, 4),
            )
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "disease_id": "DOID:1",
                "target_gene": f"NCBIGENE:{index}",
                "method": "proposed",
                "score": score,
                "pair_label": 1,
            }
            for index, score in enumerate((0.1, 0.2, 0.3, 0.4), start=1)
        ]
    )

    grouped = assign_confidence_score_groups(scores, confidence)

    assert len(grouped) == 4
    assert grouped["confidence_group"].is_unique
    assert grouped["confidence_group"].tolist() == [
        "evidence_below_phase_2",
        "phase_2_only",
        "phase_3_plus_not_approved",
        "approved",
    ]
    assert grouped["within_disease_score_percentile"].tolist() == pytest.approx(
        [0.25, 0.5, 0.75, 1.0]
    )
