from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from val2.support_robustness import (
    aggregate_target_scores,
    compute_support_robustness,
    load_cell_scores,
    prepare_support_score_distributions,
    summarize_target_scores,
)


def _cell_scores() -> pd.DataFrame:
    rows = []
    scores = {
        ("D1", "T1"): (1, [0.9, 0.8, 0.7]),
        ("D1", "T2"): (0, [0.2, 0.3, 0.4]),
        ("D1", "T3"): (0, [0.1, 0.2, 0.1]),
        ("D2", "T1"): (0, [0.2, 0.1, 0.3]),
        ("D2", "T2"): (1, [0.8, 0.9, 0.7]),
        ("D2", "T3"): (0, [0.3, 0.2, 0.4]),
    }
    cells = ["A", "B", "C"]
    for (disease_id, target_gene), (label, values) in scores.items():
        for index, (cell_id, score) in enumerate(zip(cells, values)):
            rows.append(
                {
                    "disease_id": disease_id,
                    "target_gene": target_gene,
                    "cell_id": cell_id,
                    "k": 4,
                    "method": "proposed",
                    "score": score,
                    "n_meas": 25 + 25 * index,
                    "pair_label": label,
                }
            )
    return pd.DataFrame(rows)


def test_load_cell_scores_selects_requested_method_and_k(tmp_path):
    scores = _cell_scores()
    extra = scores.copy()
    extra["k"] = 3
    extra["method"] = "kg_nosign"
    path = tmp_path / "scores.csv"
    pd.concat([scores, extra], ignore_index=True).to_csv(path, index=False)

    loaded = load_cell_scores(path, k=4, method="proposed", chunksize=5)

    assert len(loaded) == len(scores)
    assert set(loaded["k"]) == {4}
    assert set(loaded["method"]) == {"proposed"}


def test_aggregate_target_scores_applies_gene_and_cell_filters():
    scores = _cell_scores()

    target = aggregate_target_scores(
        scores,
        aggregation="mean",
        min_n_meas=50,
        min_cells=2,
    )

    assert len(target) == 6
    d1_t1 = target[
        target["disease_id"].eq("D1") & target["target_gene"].eq("T1")
    ].iloc[0]
    assert d1_t1["score"] == pytest.approx(0.75)
    assert int(d1_t1["n_evaluable_cells"]) == 2


def test_summary_and_full_robustness_are_tie_aware_and_complete():
    scores = _cell_scores()
    target = aggregate_target_scores(scores)

    summary = summarize_target_scores(target)
    robustness = compute_support_robustness(
        scores,
        gene_thresholds=(0, 50),
        cell_thresholds=(1, 2),
    )

    assert summary["mAP"] == pytest.approx(1.0)
    assert summary["n_diseases"] == 2
    assert set(robustness["analysis"]) == {
        "minimum_measured_genes",
        "minimum_evaluable_cells",
        "cell_aggregation",
        "leave_one_cell_line_out",
    }
    assert len(
        robustness[robustness["analysis"].eq("leave_one_cell_line_out")]
    ) == 3
    primary = robustness[
        robustness["analysis"].eq("cell_aggregation")
        & robustness["setting"].eq("mean")
    ].iloc[0]
    assert primary["mAP"] == pytest.approx(1.0)
    assert primary["pair_retention"] == pytest.approx(1.0)
    assert np.isclose(primary["delta_mAP"], 0.0)


def test_support_score_distributions_use_non_overlapping_bins():
    scores = _cell_scores()
    scores.loc[scores["target_gene"].eq("T1"), "n_meas"] = 25
    scores.loc[scores["target_gene"].eq("T2"), "n_meas"] = 75
    scores.loc[scores["target_gene"].eq("T3"), "n_meas"] = 225

    grouped = prepare_support_score_distributions(scores, positive_only=False)

    assert len(grouped) == 6
    assert grouped["measured_genes_group_label"].notna().all()
    assert grouped["evaluable_cells_group_label"].notna().all()
    assert set(
        grouped.loc[
            grouped["target_gene"].eq("T1"),
            "measured_genes_group_label",
        ]
    ) == {"<50"}
    assert set(
        grouped.loc[
            grouped["target_gene"].eq("T2"),
            "measured_genes_group_label",
        ]
    ) == {"50–99"}
    assert set(
        grouped.loc[
            grouped["target_gene"].eq("T3"),
            "measured_genes_group_label",
        ]
    ) == {"≥200"}
    assert set(grouped["evaluable_cells_group_label"]) == {"3–4"}
