from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .cache import SharedCache
from .io import DataBundle, load_legacy_val2

def run_ablation_section(
    *,
    data_bundle: DataBundle,
    output_dir: Path,
    ks: Sequence[int],
    methods: Sequence[str],
    start_sign: int,
    sizematch_reps: int,
    seed: int,
    k_list: Sequence[int],
    save_pred_sign_ambiguity: bool,
    save_proposed_w: bool,
    save_scores: bool = True,
    save_metrics: bool = True,
    tie_policy: str = "expected",
    ncs_cfg: Optional[dict] = None,
    n_jobs: int = 1,
) -> Dict[str, str]:
    legacy_val2 = load_legacy_val2()

    methods = [str(m) for m in methods]

    # NCS (L1000-NCS reference line) is scored via val2_baseline's algorithm on
    # val2's own evaluation space, then injected into run_val2 as a target-level
    # score table. It is computed here because it needs the DataBundle + the
    # ablation.ncs config block.
    ncs_scores = None
    if "ncs" in methods:
        if not ncs_cfg:
            raise ValueError(
                "ablation.methods contains 'ncs' but no ablation.ncs config was provided."
            )
        from .ncs import compute_ncs_target_scores

        ckd = ncs_cfg["continuous_kd_profile"]
        ds = ncs_cfg["disease_signature"]
        ncs_scores = compute_ncs_target_scores(
            labels_df=data_bundle.labels_df,
            disease_signatures=data_bundle.disease_signatures,
            continuous_kd_parquet=ckd["parquet"],
            id_col=ckd.get("id_col", "ncbigene_id"),
            cell_col=ckd.get("cell_col", "cell_id"),
            disease_signature_source=ds.get("source", "full_creeds"),
            creeds_parquet=ds.get("parquet"),
            creeds_id_col=ds.get("id_col", "doid"),
            undefined_policy=ncs_cfg.get("undefined_policy", "floor"),
        )

    scores_df, metrics_df, pred_ambiguity_df = legacy_val2.run_val2(
        data_bundle.graph,
        pert_sigs=data_bundle.perturbation_signatures,
        dis_sigs=data_bundle.disease_signatures,
        labels=data_bundle.labels_df,
        ks=[int(k) for k in ks],
        methods=methods,
        start_sign=int(start_sign),
        sizematch_reps=int(sizematch_reps),
        sizematch_seed=int(seed),
        K_list=[int(k) for k in k_list],
        tie_policy=str(tie_policy),
        ncs_scores=ncs_scores,
        n_jobs=int(n_jobs),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    k_tag = f"k{min(ks)}-{max(ks)}"
    outputs = {}

    if ncs_scores is not None and bool(save_scores):
        ncs_path = output_dir / f"val2_ncs_target_level_{k_tag}.csv"
        ncs_scores.to_csv(ncs_path, index=False)
        outputs["ncs_target_level"] = ncs_path.name

    if bool(save_scores):
        scores_path = output_dir / f"val2_scores_{k_tag}.csv"
        scores_df.to_csv(scores_path, index=False)
        outputs["scores"] = scores_path.name

    if bool(save_metrics):
        metrics_path = output_dir / f"val2_metrics_{k_tag}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        outputs["metrics"] = metrics_path.name

    if bool(save_pred_sign_ambiguity):
        ambiguity_path = output_dir / f"val2_pred_sign_ambiguity_{k_tag}.csv"
        pred_ambiguity_df.to_csv(ambiguity_path, index=False)
        outputs["pred_sign_ambiguity"] = ambiguity_path.name

    if bool(save_proposed_w):
        w_path = output_dir / f"val2_proposed_w_values_{k_tag}.csv"
        legacy_val2.export_proposed_w_values(
            G=data_bundle.graph,
            pert_sigs=data_bundle.perturbation_signatures,
            dis_sigs=data_bundle.disease_signatures,
            labels_df=data_bundle.labels_df,
            ks=[int(k) for k in ks],
            out_path=w_path,
            start_sign=int(start_sign),
        )
        outputs["proposed_w_values"] = w_path.name

    return outputs


def _build_score_cache(data_bundle: DataBundle):
    legacy_val2 = load_legacy_val2()
    return legacy_val2._build_score_cache(
        data_bundle.graph,
        data_bundle.perturbation_signatures,
        data_bundle.disease_signatures,
    )


def _precompute_proposed_base(
    *,
    data_bundle: DataBundle,
    ks: Sequence[int],
    start_sign: int,
    tie_policy: str = "expected",
) -> tuple[Dict[int, Tuple[float, float]], Dict[int, pd.DataFrame], dict]:
    legacy_val2 = load_legacy_val2()
    base_cache = _build_score_cache(data_bundle)

    targets = sorted(data_bundle.labels_df["target_gene"].dropna().astype(str).unique().tolist())
    nodes_k_cache = base_cache.setdefault("nodes_k", {})
    for target_gene in targets:
        if target_gene in data_bundle.graph:
            for k in sorted(dict.fromkeys(int(x) for x in ks)):
                nodes_k_cache[(target_gene, int(k))] = legacy_val2.khop_nodes(data_bundle.graph, target_gene, int(k))

    base_metrics: Dict[int, Tuple[float, float]] = {}
    base_by_disease: Dict[int, pd.DataFrame] = {}
    for k in tqdm(sorted(dict.fromkeys(int(x) for x in ks)), desc="Base (unpermuted)", total=len(set(ks))):
        pair_scores = legacy_val2._compute_scores_for_pairs_single_k(
            G=data_bundle.graph,
            pert_sigs=data_bundle.perturbation_signatures,
            dis_sigs=data_bundle.disease_signatures,
            labels_df=data_bundle.labels_df,
            k=int(k),
            method="proposed",
            start_sign=int(start_sign),
            cache=base_cache,
        )
        mean_ap, mean_lift10 = legacy_val2._compute_mean_ap_and_lift10_from_pair_scores(
            data_bundle.labels_df, pair_scores, tie_policy=str(tie_policy)
        )
        base_metrics[int(k)] = (float(mean_ap), float(mean_lift10))
        base_by_disease[int(k)] = legacy_val2._compute_per_disease_ap_lift10_from_pair_scores(
            data_bundle.labels_df, pair_scores, tie_policy=str(tie_policy)
        )
    return base_metrics, base_by_disease, base_cache


def run_permutation_section(
    *,
    data_bundle: DataBundle,
    output_dir: Path,
    ks: Sequence[int],
    start_sign: int,
    n_perm: int,
    seed: int,
    shared_cache: SharedCache,
    save_by_disease: bool = True,
    tie_policy: str = "expected",
) -> Dict[str, str]:
    legacy_val2 = load_legacy_val2()
    if int(n_perm) <= 0:
        return {}

    if not shared_cache.proposed_base_metrics:
        base_metrics, base_by_disease, proposed_score_cache = _precompute_proposed_base(
            data_bundle=data_bundle,
            ks=ks,
            start_sign=start_sign,
            tie_policy=str(tie_policy),
        )
        shared_cache.proposed_base_metrics = base_metrics
        shared_cache.proposed_base_by_disease = base_by_disease
        shared_cache.proposed_score_cache = proposed_score_cache

    ks_sorted = sorted(dict.fromkeys(int(k) for k in ks))
    ks_tag = ",".join(str(k) for k in ks_sorted)
    summary_rows = []
    by_disease_rows = []
    base_cache = shared_cache.proposed_score_cache

    for perm_type in ["disease_sign_shuffle", "pert_sign_shuffle", "kg_sign_shuffle"]:
        for perm_index in tqdm(range(int(n_perm)), desc=perm_type, total=int(n_perm)):
            rng = np.random.default_rng(
                legacy_val2.stable_uint32_seed(int(seed), perm_type, int(perm_index))
            )

            if perm_type == "disease_sign_shuffle":
                disease_signatures = {
                    disease_id: legacy_val2.permute_signs_preserving_counts(signature, rng)
                    for disease_id, signature in sorted(data_bundle.disease_signatures.items())
                }
                graph = data_bundle.graph
                perturbation_signatures = data_bundle.perturbation_signatures
                score_cache = base_cache
            elif perm_type == "pert_sign_shuffle":
                perturbation_signatures = {
                    key: legacy_val2.permute_signs_preserving_counts(signature, rng)
                    for key, signature in sorted(data_bundle.perturbation_signatures.items())
                }
                graph = data_bundle.graph
                disease_signatures = data_bundle.disease_signatures
                score_cache = base_cache
            else:
                graph = legacy_val2.shuffle_kg_polarity_across_edges(data_bundle.graph, rng)
                perturbation_signatures = data_bundle.perturbation_signatures
                disease_signatures = data_bundle.disease_signatures
                score_cache = legacy_val2._build_score_cache(
                    graph, perturbation_signatures, disease_signatures
                )
                score_cache["nodes_k"] = dict(base_cache.get("nodes_k", {}))

            for k in ks_sorted:
                pair_scores = legacy_val2._compute_scores_for_pairs_single_k(
                    G=graph,
                    pert_sigs=perturbation_signatures,
                    dis_sigs=disease_signatures,
                    labels_df=data_bundle.labels_df,
                    k=int(k),
                    method="proposed",
                    start_sign=int(start_sign),
                    cache=score_cache,
                )
                mean_ap, mean_lift10 = legacy_val2._compute_mean_ap_and_lift10_from_pair_scores(
                    data_bundle.labels_df, pair_scores, tie_policy=str(tie_policy)
                )
                base_ap, base_lift10 = shared_cache.proposed_base_metrics[int(k)]
                summary_rows.append(
                    {
                        "perm_type": perm_type,
                        "perm_idx": int(perm_index),
                        "k_star": int(k),
                        "ks": ks_tag,
                        "mAP": float(mean_ap),
                        "mean_Lift@10": float(mean_lift10),
                        "mAP_base": float(base_ap),
                        "mean_Lift@10_base": float(base_lift10),
                    }
                )
                per_disease = legacy_val2._compute_per_disease_ap_lift10_from_pair_scores(
                    data_bundle.labels_df, pair_scores, tie_policy=str(tie_policy)
                )
                base_per_disease = shared_cache.proposed_base_by_disease[int(k)].rename(
                    columns={"AP": "AP_base", "Lift@10": "Lift@10_base"}
                )
                merged = per_disease.merge(base_per_disease, on="disease_id", how="inner")
                for _, row in merged.iterrows():
                    by_disease_rows.append(
                        {
                            "perm_type": perm_type,
                            "perm_idx": int(perm_index),
                            "k_star": int(k),
                            "ks": ks_tag,
                            "disease_id": str(row["disease_id"]),
                            "AP": float(row["AP"]),
                            "Lift@10": float(row["Lift@10"]),
                            "AP_base": float(row["AP_base"]),
                            "Lift@10_base": float(row["Lift@10_base"]),
                        }
                    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "val2_2_permutation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    outputs = {"permutation_summary": summary_path.name}

    if bool(save_by_disease):
        by_disease_df = pd.DataFrame(by_disease_rows)
        by_disease_path = output_dir / "val2_2_permutation_summary_by_disease.csv"
        by_disease_df.to_csv(by_disease_path, index=False)
        outputs["permutation_by_disease"] = by_disease_path.name

    return outputs
