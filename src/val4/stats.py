from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_median_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or int(n_boot) <= 0:
        val = float(np.median(arr))
        return val, val
    rng = np.random.default_rng(int(seed))
    meds = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        meds[i] = np.median(sample)
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def empirical_p_greater_equal(obs: float, null_values: np.ndarray) -> float:
    null = np.asarray(null_values, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or null.size == 0:
        return float("nan")
    return float((1 + np.sum(null >= float(obs))) / (1 + null.size))


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float)
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return out
    ranks = np.arange(1, m + 1, dtype=float)
    q = valid.to_numpy(dtype=float) * m / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    out.loc[valid.index] = np.minimum(q, 1.0)
    return out
