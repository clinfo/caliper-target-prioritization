from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


@dataclass(frozen=True)
class KeyPayload:
    key: tuple[str, str]
    sign_vec: np.ndarray
    row_gene_idx: np.ndarray
    fam_idx: np.ndarray
    align_pos: np.ndarray
    align_neg: np.ndarray
    obs_align: np.ndarray
    obs_sum: np.ndarray
    obs_count: np.ndarray
    row_obs_index: np.ndarray


def build_observation_rows(
    *,
    perturbation_keys: list[tuple[str, str]],
    perturbation_signatures: dict[tuple[str, str], dict[str, int]],
    shortest_path_results: dict[
        tuple[str, str],
        tuple[
            dict[str, int],
            dict[tuple[str, str], dict[int, float]],
            dict[tuple[str, str], int],
        ],
    ],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key in perturbation_keys:
        pert_sig = perturbation_signatures[key]
        dist, family_sign_counts, family_total_counts = shortest_path_results[key]
        target_gene, cell_id = key
        for (gene_id, family), total in family_total_counts.items():
            obs_sign = int(pert_sig.get(gene_id, 0))
            if obs_sign == 0:
                continue
            sign_counts = family_sign_counts[(gene_id, family)]
            n_pos = float(sign_counts.get(1, 0.0))
            n_neg = float(sign_counts.get(-1, 0.0))
            n_total = float(total)
            if n_total <= 0:
                continue
            alignment_prob = (n_pos / n_total) if obs_sign > 0 else (n_neg / n_total)
            rows.append(
                {
                    "target_gene": str(target_gene),
                    "cell_id": str(cell_id),
                    "gene_id": str(gene_id),
                    "family": str(family),
                    "observed_sign": int(obs_sign),
                    "dist": int(dist[str(gene_id)]),
                    "n_paths_pos": float(n_pos),
                    "n_paths_neg": float(n_neg),
                    "n_paths_total": float(n_total),
                    "alignment_prob": float(alignment_prob),
                }
            )
    if len(rows) == 0:
        return pd.DataFrame(
            columns=[
                "target_gene", "cell_id", "gene_id", "family", "observed_sign",
                "dist", "n_paths_pos", "n_paths_neg", "n_paths_total", "alignment_prob",
            ]
        )
    return pd.DataFrame(rows)


def prepare_key_payloads(
    observation_rows: pd.DataFrame,
    perturbation_signatures: dict[tuple[str, str], dict[str, int]],
    family_order: list[str],
) -> list[KeyPayload]:
    family_to_idx = {family: i for i, family in enumerate(family_order)}
    payloads: list[KeyPayload] = []
    if len(observation_rows) == 0:
        return payloads

    for (target_gene, cell_id), sub_df in observation_rows.groupby(
        ["target_gene", "cell_id"], sort=False
    ):
        key = (str(target_gene), str(cell_id))
        sig = perturbation_signatures[key]
        nonzero_items = [(str(g), int(v)) for g, v in sig.items() if int(v) != 0]
        if len(nonzero_items) == 0:
            continue
        gene_order = [g for g, _ in nonzero_items]
        gene_to_idx = {g: i for i, g in enumerate(gene_order)}
        sign_vec = np.asarray([v for _, v in nonzero_items], dtype=np.int8)

        row_gene_idx = np.asarray(
            [gene_to_idx[str(g)] for g in sub_df["gene_id"].tolist()], dtype=np.int32
        )
        fam_idx = np.asarray(
            [family_to_idx[str(f)] for f in sub_df["family"].tolist()], dtype=np.int16
        )
        align_pos = (
            sub_df["n_paths_pos"].to_numpy(dtype=np.float64)
            / sub_df["n_paths_total"].to_numpy(dtype=np.float64)
        )
        align_neg = (
            sub_df["n_paths_neg"].to_numpy(dtype=np.float64)
            / sub_df["n_paths_total"].to_numpy(dtype=np.float64)
        )
        obs_sign = sub_df["observed_sign"].to_numpy(dtype=np.int8)
        obs_align = np.where(obs_sign > 0, align_pos, align_neg)
        obs_sum = np.bincount(
            fam_idx, weights=obs_align, minlength=len(family_order)
        ).astype(np.float64)
        obs_count = np.bincount(fam_idx, minlength=len(family_order)).astype(np.int64)
        row_obs_index = sub_df.index.to_numpy()
        payloads.append(
            KeyPayload(
                key=key,
                sign_vec=sign_vec,
                row_gene_idx=row_gene_idx,
                fam_idx=fam_idx,
                align_pos=align_pos,
                align_neg=align_neg,
                obs_align=obs_align,
                obs_sum=obs_sum,
                obs_count=obs_count,
                row_obs_index=row_obs_index,
            )
        )
    return payloads


_PAIRED_OUTPUT_KEYS = (
    "family_summary_df",
    "family_summary_cluster_df",
    "family_null_df",
    "family_null_cluster_df",
    "family_boot_df",
    "family_boot_cluster_df",
    "key_delta_df",
    "row_delta_df",
)


def _empty_paired_outputs(family_order: list[str]) -> dict[str, pd.DataFrame]:
    n_fam = len(family_order)
    nan_col = [np.nan] * n_fam
    zero_col = [0] * n_fam
    family_summary_df = pd.DataFrame(
        {
            "family": family_order,
            "delta_f": nan_col,
            "null_mean_f": nan_col,
            "null_sd_f": nan_col,
            "p_empirical": nan_col,
            "z_f": nan_col,
            "ci_low": nan_col,
            "ci_high": nan_col,
            "n_evaluable_rows": zero_col,
            "n_perturbations": zero_col,
        }
    )
    family_summary_cluster_df = pd.DataFrame(
        {
            "family": family_order,
            "delta_f_cluster": nan_col,
            "null_mean_f_cluster": nan_col,
            "null_sd_f_cluster": nan_col,
            "p_empirical_cluster": nan_col,
            "z_f_cluster": nan_col,
            "ci_low_cluster": nan_col,
            "ci_high_cluster": nan_col,
            "n_evaluable_rows": zero_col,
            "n_perturbations": zero_col,
        }
    )
    family_null_df = pd.DataFrame(columns=["family", "null_rep", "null_delta_f"])
    family_null_cluster_df = pd.DataFrame(
        columns=["family", "null_rep", "null_delta_f_cluster"]
    )
    family_boot_df = pd.DataFrame(columns=["family", "bootstrap_rep", "delta_f_boot"])
    family_boot_cluster_df = pd.DataFrame(
        columns=["family", "bootstrap_rep", "delta_f_cluster_boot"]
    )
    key_delta_df = pd.DataFrame(
        columns=[
            "target_gene", "cell_id", "family",
            "n_paths_used", "obs_a_key", "null_a_key_mean", "delta_key",
        ]
    )
    row_delta_df = pd.DataFrame(
        columns=[
            "target_gene", "cell_id", "gene_id", "family", "observed_sign",
            "dist", "alignment_prob", "null_alignment_prob_mean", "delta_row",
        ]
    )
    return {
        "family_summary_df": family_summary_df,
        "family_summary_cluster_df": family_summary_cluster_df,
        "family_null_df": family_null_df,
        "family_null_cluster_df": family_null_cluster_df,
        "family_boot_df": family_boot_df,
        "family_boot_cluster_df": family_boot_cluster_df,
        "key_delta_df": key_delta_df,
        "row_delta_df": row_delta_df,
    }


def _safe_z(delta: float, null_mean: float, null_sd: float) -> float:
    if not np.isfinite(null_sd) or null_sd == 0.0:
        if delta > null_mean:
            return float("inf")
        if delta < null_mean:
            return float("-inf")
        return 0.0
    return float((delta - null_mean) / null_sd)


def compute_null_and_bootstrap(
    *,
    payloads: list[KeyPayload],
    family_order: list[str],
    n_null: int,
    bootstrap_reps: int,
    seed: int,
    progress: bool = True,
    observation_rows: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    n_fam = len(family_order)
    n_keys = len(payloads)

    if n_keys == 0:
        return _empty_paired_outputs(family_order)

    obs_sum = np.vstack([p.obs_sum for p in payloads]).astype(np.float64)
    obs_count = np.vstack([p.obs_count for p in payloads]).astype(np.int64)

    null_sum = np.zeros((n_null, n_keys, n_fam), dtype=np.float32)
    null_row_mean_parts: list[np.ndarray] = []

    rng_null = np.random.default_rng(seed)
    payload_iter = tqdm(
        payloads, total=len(payloads), desc="h1 null by key", disable=not progress
    )
    for key_idx, payload in enumerate(payload_iter):
        n_rows_p = payload.fam_idx.size
        null_align_acc = np.zeros(n_rows_p, dtype=np.float64)
        if n_rows_p == 0:
            null_row_mean_parts.append(null_align_acc)
            continue
        for rep in range(n_null):
            perm_sign = rng_null.permutation(payload.sign_vec)
            row_sign = perm_sign[payload.row_gene_idx]
            align = np.where(row_sign > 0, payload.align_pos, payload.align_neg)
            null_align_acc += align
            null_sum[rep, key_idx, :] = np.bincount(
                payload.fam_idx, weights=align, minlength=n_fam
            ).astype(np.float32)
        null_row_mean_parts.append(null_align_acc / float(n_null))

    null_row_mean_all = (
        np.concatenate(null_row_mean_parts)
        if null_row_mean_parts
        else np.zeros(0, dtype=np.float64)
    )
    obs_align_all = (
        np.concatenate([p.obs_align for p in payloads])
        if payloads
        else np.zeros(0, dtype=np.float64)
    )
    row_obs_index_all = (
        np.concatenate([p.row_obs_index for p in payloads])
        if payloads
        else np.zeros(0, dtype=np.int64)
    )

    valid_key = obs_count > 0
    obs_count_f = obs_count.astype(np.float64)
    obs_key_mean = np.divide(
        obs_sum, obs_count_f,
        out=np.full_like(obs_sum, np.nan, dtype=np.float64),
        where=valid_key,
    )
    null_sum_total_per_key = null_sum.sum(axis=0).astype(np.float64)  # (n_keys, n_fam)
    null_key_mean_avg = np.divide(
        null_sum_total_per_key, (float(n_null) * obs_count_f),
        out=np.full_like(obs_sum, np.nan, dtype=np.float64),
        where=valid_key,
    )
    delta_key = obs_key_mean - null_key_mean_avg

    T_i_f = np.zeros_like(obs_sum)
    cursor = 0
    for key_idx, payload in enumerate(payloads):
        n_rows_p = payload.fam_idx.size
        if n_rows_p == 0:
            continue
        null_rm = null_row_mean_all[cursor:cursor + n_rows_p]
        T_i_f[key_idx, :] = np.bincount(
            payload.fam_idx, weights=null_rm, minlength=n_fam
        )
        cursor += n_rows_p

    obs_total = obs_sum.sum(axis=0)
    T_total = T_i_f.sum(axis=0)
    cnt_total = obs_count_f.sum(axis=0)
    valid_fam_row = cnt_total > 0
    delta_f_row = np.divide(
        obs_total - T_total, cnt_total,
        out=np.full(n_fam, np.nan, dtype=np.float64),
        where=valid_fam_row,
    )
    delta_f_cluster = np.full(n_fam, np.nan, dtype=np.float64)
    for f in range(n_fam):
        col = delta_key[:, f]
        col_valid = col[~np.isnan(col)]
        if col_valid.size > 0:
            delta_f_cluster[f] = float(col_valid.mean())

    null_sum_per_rep_total = null_sum.sum(axis=1).astype(np.float64)  # (n_null, n_fam)
    null_delta_row = np.full((n_null, n_fam), np.nan, dtype=np.float64)
    for f in range(n_fam):
        if cnt_total[f] > 0:
            null_delta_row[:, f] = (
                null_sum_per_rep_total[:, f] - T_total[f]
            ) / cnt_total[f]

    null_delta_cluster = np.full((n_null, n_fam), np.nan, dtype=np.float64)
    for f in range(n_fam):
        valid_i = obs_count[:, f] > 0
        if not valid_i.any():
            continue
        cnt_col = obs_count_f[valid_i, f]
        nkm = null_sum[:, valid_i, f].astype(np.float64) / cnt_col[None, :]
        avg = null_key_mean_avg[valid_i, f][None, :]
        null_delta_cluster[:, f] = (nkm - avg).mean(axis=1)

    p_emp_row = np.full(n_fam, np.nan, dtype=np.float64)
    z_row = np.full(n_fam, np.nan, dtype=np.float64)
    null_mean_row = np.full(n_fam, np.nan, dtype=np.float64)
    null_sd_row = np.full(n_fam, np.nan, dtype=np.float64)
    p_emp_cluster = np.full(n_fam, np.nan, dtype=np.float64)
    z_cluster = np.full(n_fam, np.nan, dtype=np.float64)
    null_mean_cluster = np.full(n_fam, np.nan, dtype=np.float64)
    null_sd_cluster = np.full(n_fam, np.nan, dtype=np.float64)
    for f in range(n_fam):
        if not np.isnan(delta_f_row[f]):
            ndr = null_delta_row[:, f]
            ndr_valid = ndr[~np.isnan(ndr)]
            if ndr_valid.size > 0:
                null_mean_row[f] = float(ndr_valid.mean())
                null_sd_row[f] = float(ndr_valid.std(ddof=0))
                p_emp_row[f] = float(
                    (1 + np.sum(ndr_valid >= delta_f_row[f])) / (ndr_valid.size + 1)
                )
                z_row[f] = _safe_z(
                    float(delta_f_row[f]), null_mean_row[f], null_sd_row[f]
                )
        if not np.isnan(delta_f_cluster[f]):
            ndc = null_delta_cluster[:, f]
            ndc_valid = ndc[~np.isnan(ndc)]
            if ndc_valid.size > 0:
                null_mean_cluster[f] = float(ndc_valid.mean())
                null_sd_cluster[f] = float(ndc_valid.std(ddof=0))
                p_emp_cluster[f] = float(
                    (1 + np.sum(ndc_valid >= delta_f_cluster[f])) / (ndc_valid.size + 1)
                )
                z_cluster[f] = _safe_z(
                    float(delta_f_cluster[f]),
                    null_mean_cluster[f],
                    null_sd_cluster[f],
                )

    rng_boot = np.random.default_rng(seed + 1)
    delta_boot_row = np.full((bootstrap_reps, n_fam), np.nan, dtype=np.float64)
    delta_boot_cluster = np.full((bootstrap_reps, n_fam), np.nan, dtype=np.float64)
    for b in tqdm(
        range(bootstrap_reps), desc="h1 bootstrap", disable=not progress
    ):
        sampled = rng_boot.integers(0, n_keys, size=n_keys)
        S_b = obs_sum[sampled].sum(axis=0)
        T_b = T_i_f[sampled].sum(axis=0)
        N_b = obs_count_f[sampled].sum(axis=0)
        nonzero = N_b > 0
        delta_boot_row[b, nonzero] = (S_b[nonzero] - T_b[nonzero]) / N_b[nonzero]
        for f in range(n_fam):
            col = delta_key[sampled, f]
            col_valid = col[~np.isnan(col)]
            if col_valid.size > 0:
                delta_boot_cluster[b, f] = float(col_valid.mean())

    ci_low_row = np.nanpercentile(delta_boot_row, 2.5, axis=0)
    ci_high_row = np.nanpercentile(delta_boot_row, 97.5, axis=0)
    ci_low_cluster = np.nanpercentile(delta_boot_cluster, 2.5, axis=0)
    ci_high_cluster = np.nanpercentile(delta_boot_cluster, 97.5, axis=0)

    n_evaluable_rows = obs_count.sum(axis=0).astype(int)
    n_perturbations = (obs_count > 0).sum(axis=0).astype(int)

    family_summary_df = pd.DataFrame(
        {
            "family": family_order,
            "delta_f": delta_f_row,
            "null_mean_f": null_mean_row,
            "null_sd_f": null_sd_row,
            "p_empirical": p_emp_row,
            "z_f": z_row,
            "ci_low": ci_low_row,
            "ci_high": ci_high_row,
            "n_evaluable_rows": n_evaluable_rows,
            "n_perturbations": n_perturbations,
        }
    )
    family_summary_cluster_df = pd.DataFrame(
        {
            "family": family_order,
            "delta_f_cluster": delta_f_cluster,
            "null_mean_f_cluster": null_mean_cluster,
            "null_sd_f_cluster": null_sd_cluster,
            "p_empirical_cluster": p_emp_cluster,
            "z_f_cluster": z_cluster,
            "ci_low_cluster": ci_low_cluster,
            "ci_high_cluster": ci_high_cluster,
            "n_evaluable_rows": n_evaluable_rows,
            "n_perturbations": n_perturbations,
        }
    )

    family_null_records: list[dict[str, object]] = []
    family_null_cluster_records: list[dict[str, object]] = []
    for r in range(n_null):
        for f_idx, fam in enumerate(family_order):
            family_null_records.append(
                {
                    "family": fam,
                    "null_rep": int(r),
                    "null_delta_f": float(null_delta_row[r, f_idx]),
                }
            )
            family_null_cluster_records.append(
                {
                    "family": fam,
                    "null_rep": int(r),
                    "null_delta_f_cluster": float(null_delta_cluster[r, f_idx]),
                }
            )
    family_null_df = pd.DataFrame(family_null_records)
    family_null_cluster_df = pd.DataFrame(family_null_cluster_records)

    family_boot_records: list[dict[str, object]] = []
    family_boot_cluster_records: list[dict[str, object]] = []
    for b in range(bootstrap_reps):
        for f_idx, fam in enumerate(family_order):
            family_boot_records.append(
                {
                    "family": fam,
                    "bootstrap_rep": int(b),
                    "delta_f_boot": float(delta_boot_row[b, f_idx]),
                }
            )
            family_boot_cluster_records.append(
                {
                    "family": fam,
                    "bootstrap_rep": int(b),
                    "delta_f_cluster_boot": float(delta_boot_cluster[b, f_idx]),
                }
            )
    family_boot_df = pd.DataFrame(family_boot_records)
    family_boot_cluster_df = pd.DataFrame(family_boot_cluster_records)

    key_delta_records: list[dict[str, object]] = []
    for i, p in enumerate(payloads):
        t, c = p.key
        for f_idx, fam in enumerate(family_order):
            if obs_count[i, f_idx] > 0:
                key_delta_records.append(
                    {
                        "target_gene": t,
                        "cell_id": c,
                        "family": fam,
                        "n_paths_used": int(obs_count[i, f_idx]),
                        "obs_a_key": float(obs_key_mean[i, f_idx]),
                        "null_a_key_mean": float(null_key_mean_avg[i, f_idx]),
                        "delta_key": float(delta_key[i, f_idx]),
                    }
                )
    key_delta_df = pd.DataFrame(
        key_delta_records,
        columns=[
            "target_gene", "cell_id", "family",
            "n_paths_used", "obs_a_key", "null_a_key_mean", "delta_key",
        ],
    )

    if observation_rows is not None and row_obs_index_all.size > 0:
        meta_cols = [
            "target_gene", "cell_id", "gene_id", "family", "observed_sign", "dist"
        ]
        sub = observation_rows.loc[row_obs_index_all, meta_cols].reset_index(drop=True)
        sub["alignment_prob"] = obs_align_all.astype(np.float64)
        sub["null_alignment_prob_mean"] = null_row_mean_all.astype(np.float64)
        sub["delta_row"] = (obs_align_all - null_row_mean_all).astype(np.float64)
        row_delta_df = sub
    else:
        row_delta_df = pd.DataFrame(
            columns=[
                "target_gene", "cell_id", "gene_id", "family", "observed_sign",
                "dist", "alignment_prob", "null_alignment_prob_mean", "delta_row",
            ]
        )

    return {
        "family_summary_df": family_summary_df,
        "family_summary_cluster_df": family_summary_cluster_df,
        "family_null_df": family_null_df,
        "family_null_cluster_df": family_null_cluster_df,
        "family_boot_df": family_boot_df,
        "family_boot_cluster_df": family_boot_cluster_df,
        "key_delta_df": key_delta_df,
        "row_delta_df": row_delta_df,
    }
