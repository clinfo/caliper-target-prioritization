from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .legacy import _ranking_metrics_for_scores


DEFAULT_GENE_THRESHOLDS = (0, 50, 100, 200)
DEFAULT_CELL_THRESHOLDS = (1, 3, 5, 8)
MEASURED_GENE_SCORE_GROUPS = (
    ("lt_50", "<50"),
    ("50_99", "50–99"),
    ("100_199", "100–199"),
    ("ge_200", "≥200"),
)
EVALUABLE_CELL_SCORE_GROUPS = (
    ("1_2", "1–2"),
    ("3_4", "3–4"),
    ("5_7", "5–7"),
    ("ge_8", "≥8"),
)
SCORE_COLUMNS = (
    "disease_id",
    "target_gene",
    "cell_id",
    "k",
    "method",
    "score",
    "n_meas",
    "pair_label",
)


def load_cell_scores(
    path: str | Path,
    *,
    k: int,
    method: str = "proposed",
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load one method and distance from the potentially large Val2 score CSV."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=SCORE_COLUMNS)

    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=list(SCORE_COLUMNS), chunksize=int(chunksize)):
        k_values = pd.to_numeric(chunk["k"], errors="coerce")
        keep = k_values.eq(int(k)) & chunk["method"].astype(str).eq(str(method))
        if keep.any():
            selected.append(chunk.loc[keep, list(SCORE_COLUMNS)].copy())
    if not selected:
        return pd.DataFrame(columns=SCORE_COLUMNS)

    out = pd.concat(selected, ignore_index=True)
    for column in ("k", "score", "n_meas", "pair_label"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(
        subset=["disease_id", "target_gene", "cell_id", "score", "pair_label"]
    )
    out["pair_label"] = out["pair_label"].astype(int)
    return out


def aggregate_target_scores(
    cell_scores: pd.DataFrame,
    *,
    aggregation: str = "mean",
    min_n_meas: int = 0,
    min_cells: int = 1,
    omitted_cell: str | None = None,
) -> pd.DataFrame:
    """Aggregate cell-level CALIPER scores after an evidence-support filter."""
    required = {
        "disease_id",
        "target_gene",
        "cell_id",
        "score",
        "n_meas",
        "pair_label",
    }
    if cell_scores.empty or not required.issubset(cell_scores.columns):
        return pd.DataFrame(
            columns=[
                "disease_id",
                "target_gene",
                "label",
                "score",
                "mean_n_meas",
                "n_evaluable_cells",
            ]
        )
    if aggregation not in {"mean", "median"}:
        raise ValueError("aggregation must be 'mean' or 'median'")

    work = cell_scores.copy()
    if omitted_cell is not None:
        work = work[work["cell_id"].astype(str).ne(str(omitted_cell))]
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work["n_meas"] = pd.to_numeric(work["n_meas"], errors="coerce")
    work = work[
        np.isfinite(work["score"])
        & work["n_meas"].fillna(0).ge(int(min_n_meas))
    ]
    if work.empty:
        return pd.DataFrame(
            columns=[
                "disease_id",
                "target_gene",
                "label",
                "score",
                "mean_n_meas",
                "n_evaluable_cells",
            ]
        )

    grouped = work.groupby(["disease_id", "target_gene"], as_index=False)
    if aggregation == "mean":
        aggregated = grouped.agg(
            label=("pair_label", "max"),
            score=("score", "mean"),
            mean_n_meas=("n_meas", "mean"),
            n_evaluable_cells=("cell_id", "nunique"),
        )
    else:
        aggregated = grouped.agg(
            label=("pair_label", "max"),
            score=("score", "median"),
            mean_n_meas=("n_meas", "mean"),
            n_evaluable_cells=("cell_id", "nunique"),
        )
    aggregated = aggregated[
        aggregated["n_evaluable_cells"].ge(int(min_cells))
    ].reset_index(drop=True)
    return aggregated


def prepare_support_score_distributions(
    cell_scores: pd.DataFrame,
    *,
    positive_only: bool = True,
) -> pd.DataFrame:
    """Group target-pair CALIPER scores into non-overlapping support bins."""
    target_scores = aggregate_target_scores(cell_scores, aggregation="mean")
    if target_scores.empty:
        return target_scores
    if positive_only:
        target_scores = target_scores[
            pd.to_numeric(target_scores["label"], errors="coerce").fillna(0).gt(0)
        ].copy()
    if target_scores.empty:
        return target_scores

    gene_keys = [name for name, _label in MEASURED_GENE_SCORE_GROUPS]
    gene_labels = dict(MEASURED_GENE_SCORE_GROUPS)
    target_scores["measured_genes_group"] = pd.cut(
        pd.to_numeric(target_scores["mean_n_meas"], errors="coerce"),
        bins=[-np.inf, 50, 100, 200, np.inf],
        labels=gene_keys,
        right=False,
    ).astype("object")
    target_scores["measured_genes_group_label"] = target_scores[
        "measured_genes_group"
    ].map(gene_labels)
    target_scores["measured_genes_group_order"] = target_scores[
        "measured_genes_group"
    ].map({name: index for index, name in enumerate(gene_keys)})

    cell_keys = [name for name, _label in EVALUABLE_CELL_SCORE_GROUPS]
    cell_labels = dict(EVALUABLE_CELL_SCORE_GROUPS)
    target_scores["evaluable_cells_group"] = pd.cut(
        pd.to_numeric(target_scores["n_evaluable_cells"], errors="coerce"),
        bins=[1, 3, 5, 8, np.inf],
        labels=cell_keys,
        right=False,
        include_lowest=True,
    ).astype("object")
    target_scores["evaluable_cells_group_label"] = target_scores[
        "evaluable_cells_group"
    ].map(cell_labels)
    target_scores["evaluable_cells_group_order"] = target_scores[
        "evaluable_cells_group"
    ].map({name: index for index, name in enumerate(cell_keys)})
    return target_scores.sort_values(
        ["disease_id", "target_gene"], kind="mergesort"
    ).reset_index(drop=True)


def summarize_target_scores(target_scores: pd.DataFrame) -> dict[str, float | int]:
    """Summarize disease-level AP using Val2's tie-aware primary metric."""
    if target_scores.empty:
        return {
            "mAP": np.nan,
            "AP_SEM": np.nan,
            "n_diseases": 0,
            "n_pairs": 0,
            "n_positive_pairs": 0,
        }

    ap_values: list[float] = []
    for _disease_id, group in target_scores.groupby("disease_id", sort=False):
        labels = pd.to_numeric(group["label"], errors="coerce").fillna(0).astype(int)
        scores = pd.to_numeric(group["score"], errors="coerce")
        finite = np.isfinite(scores)
        labels = labels[finite]
        scores = scores[finite]
        if labels.empty or int(labels.sum()) == 0:
            continue
        metrics = _ranking_metrics_for_scores(
            labels.tolist(),
            scores.astype(float).tolist(),
            K_list=[10],
            tie_policy="expected",
        )
        ap = float(metrics["AP"])
        if np.isfinite(ap):
            ap_values.append(ap)

    n_diseases = len(ap_values)
    ap_array = np.asarray(ap_values, dtype=float)
    sem = (
        float(np.std(ap_array, ddof=1) / np.sqrt(n_diseases))
        if n_diseases >= 2
        else np.nan
    )
    return {
        "mAP": float(np.mean(ap_array)) if n_diseases else np.nan,
        "AP_SEM": sem,
        "n_diseases": int(n_diseases),
        "n_pairs": int(len(target_scores)),
        "n_positive_pairs": int(
            pd.to_numeric(target_scores["label"], errors="coerce").fillna(0).gt(0).sum()
        ),
    }


def _summary_row(
    cell_scores: pd.DataFrame,
    *,
    analysis: str,
    setting: str,
    aggregation: str = "mean",
    min_n_meas: int = 0,
    min_cells: int = 1,
    omitted_cell: str | None = None,
    baseline_pairs: int,
) -> dict[str, object]:
    target_scores = aggregate_target_scores(
        cell_scores,
        aggregation=aggregation,
        min_n_meas=min_n_meas,
        min_cells=min_cells,
        omitted_cell=omitted_cell,
    )
    summary = summarize_target_scores(target_scores)
    n_pairs = int(summary["n_pairs"])
    return {
        "analysis": analysis,
        "setting": setting,
        "aggregation": aggregation,
        "min_n_meas": int(min_n_meas),
        "min_cells": int(min_cells),
        "omitted_cell": omitted_cell or "",
        **summary,
        "pair_retention": (
            float(n_pairs / baseline_pairs) if int(baseline_pairs) > 0 else np.nan
        ),
    }


def compute_support_robustness(
    cell_scores: pd.DataFrame,
    *,
    gene_thresholds: Iterable[int] = DEFAULT_GENE_THRESHOLDS,
    cell_thresholds: Iterable[int] = DEFAULT_CELL_THRESHOLDS,
) -> pd.DataFrame:
    """Compute the planned gene/cell/aggregation/cell-line sensitivity panel."""
    if cell_scores.empty:
        return pd.DataFrame()

    baseline_targets = aggregate_target_scores(cell_scores)
    baseline_pairs = int(len(baseline_targets))
    rows: list[dict[str, object]] = []

    for threshold in gene_thresholds:
        rows.append(
            _summary_row(
                cell_scores,
                analysis="minimum_measured_genes",
                setting=str(int(threshold)),
                min_n_meas=int(threshold),
                baseline_pairs=baseline_pairs,
            )
        )
    for threshold in cell_thresholds:
        rows.append(
            _summary_row(
                cell_scores,
                analysis="minimum_evaluable_cells",
                setting=str(int(threshold)),
                min_cells=int(threshold),
                baseline_pairs=baseline_pairs,
            )
        )
    for aggregation in ("mean", "median"):
        rows.append(
            _summary_row(
                cell_scores,
                analysis="cell_aggregation",
                setting=aggregation,
                aggregation=aggregation,
                baseline_pairs=baseline_pairs,
            )
        )
    for cell_id in sorted(cell_scores["cell_id"].dropna().astype(str).unique()):
        rows.append(
            _summary_row(
                cell_scores,
                analysis="leave_one_cell_line_out",
                setting=cell_id,
                omitted_cell=cell_id,
                baseline_pairs=baseline_pairs,
            )
        )

    result = pd.DataFrame(rows)
    primary = result[
        result["analysis"].eq("cell_aggregation")
        & result["setting"].eq("mean")
    ]
    primary_map = float(primary["mAP"].iloc[0]) if not primary.empty else np.nan
    result["delta_mAP"] = pd.to_numeric(result["mAP"], errors="coerce") - primary_map
    return result


def build_support_robustness_table(
    run_root: str | Path,
    *,
    k: int = 4,
    method: str = "proposed",
) -> pd.DataFrame:
    run_root = Path(run_root)
    scores_path = run_root / "ablation" / "val2_scores_k1-10.csv"
    cell_scores = load_cell_scores(scores_path, k=int(k), method=method)
    return compute_support_robustness(cell_scores)
