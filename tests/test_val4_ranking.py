from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from val2.legacy import _ranking_metrics_for_scores  # noqa: E402
from val4.ranking import evaluate_disease_metrics  # noqa: E402


def test_evaluate_disease_metrics_uses_val2_expected_tie_policy() -> None:
    pair_scores = pd.DataFrame(
        [
            {"disease_id": "D1", "target_gene": "T1", "cell_id": "C1", "pair_label": 1, "score": 0.5},
            {"disease_id": "D1", "target_gene": "T2", "cell_id": "C1", "pair_label": 0, "score": 0.5},
            {"disease_id": "D1", "target_gene": "T3", "cell_id": "C1", "pair_label": 1, "score": 0.5},
            {"disease_id": "D1", "target_gene": "T4", "cell_id": "C1", "pair_label": 0, "score": 0.1},
        ]
    )

    observed = evaluate_disease_metrics(pair_scores).iloc[0]
    expected = _ranking_metrics_for_scores([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.1], [10], tie_policy="expected")

    assert observed["tie_policy"] == "expected"
    assert math.isclose(float(observed["AP"]), float(expected["AP"]))
    assert math.isclose(float(observed["P@10"]), float(expected["P@10"]))
    assert math.isclose(float(observed["Lift@10"]), float(expected["Lift@10"]))
    assert int(observed["n_tie_groups"]) == int(expected["n_tie_groups"])
