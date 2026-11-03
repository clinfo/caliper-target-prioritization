from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from val2.legacy import _ranking_metrics_for_scores, normalize_tie_policy
except Exception:  # pragma: no cover
    def normalize_tie_policy(tie_policy: str = "expected") -> str:
        policy = str(tie_policy or "expected").strip().lower()
        aliases = {
            "default": "expected",
            "tie_aware": "expected",
            "tie-aware": "expected",
            "average": "expected",
            "average_ties": "expected",
            "expected_ties": "expected",
        }
        policy = aliases.get(policy, policy)
        if policy not in {"legacy", "expected"}:
            raise ValueError(f"Unknown tie_policy={tie_policy!r}. valid=['expected', 'legacy']")
        return policy

    def precision_at_k(y_true, y_score, k: int) -> float:
        idx = np.argsort(-np.array(y_score, dtype=float))[: int(k)]
        yt = np.array(y_true, dtype=int)[idx]
        return float(yt.mean()) if len(yt) > 0 else float("nan")

    def average_precision(y_true, y_score) -> float:
        y_true = np.array(y_true, dtype=int)
        y_score = np.array(y_score, dtype=float)
        order = np.argsort(-y_score)
        y_true = y_true[order]
        n_pos = int(y_true.sum())
        if n_pos == 0:
            return float("nan")
        cum_tp = 0
        ap = 0.0
        for i, yt in enumerate(y_true, start=1):
            if yt == 1:
                cum_tp += 1
                ap += cum_tp / i
        return float(ap / n_pos)

    def _ranking_metrics_for_scores(y_true, y_score, K_list, tie_policy: str = "expected") -> dict[str, object]:
        policy = normalize_tie_policy(tie_policy)
        y_true_arr = np.asarray(y_true, dtype=int)
        y_score_arr = np.asarray(y_score, dtype=float)
        finite = np.isfinite(y_score_arr)
        y_true_arr = y_true_arr[finite]
        y_score_arr = y_score_arr[finite]
        chance = float(np.mean(y_true_arr)) if len(y_true_arr) > 0 else float("nan")
        out: dict[str, object] = {
            "chance": chance,
            "n_pos": int(y_true_arr.sum()),
            "n_neg": int(len(y_true_arr) - int(y_true_arr.sum())),
            "tie_policy": policy,
        }
        if policy == "legacy":
            out["AP"] = average_precision(y_true_arr.tolist(), y_score_arr.tolist())
            out["n_score_groups"] = np.nan
            out["n_tie_groups"] = np.nan
            out["largest_tie_group"] = np.nan
            for K in K_list:
                p_at_k = precision_at_k(y_true_arr.tolist(), y_score_arr.tolist(), int(K))
                out[f"P@{int(K)}"] = p_at_k
                out[f"Lift@{int(K)}"] = (p_at_k / chance) if chance > 0 and not np.isnan(p_at_k) else np.nan
                out[f"top{int(K)}_boundary_tie_size"] = np.nan
            return out

        order = np.argsort(-y_score_arr, kind="mergesort")
        y_true_sorted = y_true_arr[order]
        y_score_sorted = y_score_arr[order]
        n_pos = int(y_true_sorted.sum())
        if n_pos == 0:
            out["AP"] = np.nan
        else:
            ap = 0.0
            positives_before = 0
            rows_before = 0
            start = 0
            while start < len(y_true_sorted):
                end = start + 1
                while end < len(y_true_sorted) and y_score_sorted[end] == y_score_sorted[start]:
                    end += 1
                block_size = int(end - start)
                block_pos = int(y_true_sorted[start:end].sum())
                if block_pos > 0:
                    if block_size == 1:
                        ap += (positives_before + 1) / (rows_before + 1)
                    else:
                        for j in range(1, block_size + 1):
                            expected_prev = ((j - 1) * (block_pos - 1) / (block_size - 1)) if block_pos > 1 else 0.0
                            ap += (block_pos / block_size) * (positives_before + 1.0 + expected_prev) / (rows_before + j)
                positives_before += block_pos
                rows_before += block_size
                start = end
            out["AP"] = float(ap / n_pos)

        groups = []
        start = 0
        while start < len(y_score_sorted):
            end = start + 1
            while end < len(y_score_sorted) and y_score_sorted[end] == y_score_sorted[start]:
                end += 1
            groups.append((start, end))
            start = end
        out["n_score_groups"] = int(len(groups))
        tie_sizes = [end - start for start, end in groups if end - start > 1]
        out["n_tie_groups"] = int(len(tie_sizes))
        out["largest_tie_group"] = int(max(tie_sizes)) if tie_sizes else 0

        for K in K_list:
            k_eff = min(int(K), len(y_true_sorted))
            expected_pos = 0.0
            rows_seen = 0
            boundary_tie_size = 0
            for start, end in groups:
                if rows_seen >= k_eff:
                    break
                block_size = int(end - start)
                block_pos = int(y_true_sorted[start:end].sum())
                remaining = k_eff - rows_seen
                if block_size <= remaining:
                    expected_pos += block_pos
                    rows_seen += block_size
                else:
                    expected_pos += remaining * (block_pos / block_size)
                    boundary_tie_size = block_size
                    rows_seen = k_eff
            p_at_k = float(expected_pos / k_eff) if k_eff > 0 else np.nan
            out[f"P@{int(K)}"] = p_at_k
            out[f"Lift@{int(K)}"] = (p_at_k / chance) if chance > 0 and not np.isnan(p_at_k) else np.nan
            out[f"top{int(K)}_boundary_tie_size"] = int(boundary_tie_size)
        return out


def _empty_metric_frame(k: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "disease_id",
            "AP",
            f"P@{int(k)}",
            f"Lift@{int(k)}",
            "chance",
            "n_pos",
            "n_neg",
            "n_targets",
            "tie_policy",
            "n_score_groups",
            "n_tie_groups",
            "largest_tie_group",
            f"top{int(k)}_boundary_tie_size",
        ]
    )


def aggregate_target_scores(pair_scores: pd.DataFrame) -> pd.DataFrame:
    if pair_scores is None or len(pair_scores) == 0:
        return pd.DataFrame(columns=["disease_id", "target_gene", "label", "score"])
    return (
        pair_scores.groupby(["disease_id", "target_gene"], as_index=False)
        .agg(label=("pair_label", "max"), score=("score", "mean"))
    )


def evaluate_disease_metrics(
    pair_scores: pd.DataFrame,
    *,
    score_col: str = "score",
    label_col: str = "pair_label",
    k: int = 10,
    tie_policy: str = "expected",
) -> pd.DataFrame:
    tie_policy = normalize_tie_policy(tie_policy)
    if pair_scores is None or len(pair_scores) == 0:
        return _empty_metric_frame(k)
    df = pair_scores.dropna(subset=[score_col]).copy()
    if len(df) == 0:
        return _empty_metric_frame(k)
    agg = (
        df.groupby(["disease_id", "target_gene"], as_index=False)
        .agg(label=(label_col, "max"), score=(score_col, "mean"))
    )
    rows: list[dict[str, object]] = []
    for disease_id, g in agg.groupby("disease_id", sort=False):
        y_true = g["label"].astype(int).tolist()
        y_score = g["score"].astype(float).tolist()
        n_pos = int(sum(y_true))
        n_targets = int(len(y_true))
        n_neg = n_targets - n_pos
        if n_pos == 0:
            continue
        metric_values = _ranking_metrics_for_scores(y_true, y_score, [int(k)], tie_policy=tie_policy)
        rows.append(
            {
                "disease_id": disease_id,
                **metric_values,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_targets": n_targets,
            }
        )
    return pd.DataFrame(rows) if rows else _empty_metric_frame(k)
