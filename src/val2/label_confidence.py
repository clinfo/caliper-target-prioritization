from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .legacy import _ranking_metrics_for_scores


SCORE_COLUMNS = (
    "disease_id",
    "target_gene",
    "k",
    "method",
    "score",
    "pair_label",
)
CONFIDENCE_STRATA = (
    ("all_evidence", "All evidence"),
    ("phase_ge_2", "Phase ≥2"),
    ("phase_ge_3", "Phase ≥3"),
    ("approved", "Approved"),
)
CONFIDENCE_SCORE_GROUPS = (
    ("evidence_below_phase_2", "Evidence below Phase 2"),
    ("phase_2_only", "Phase 2 only"),
    ("phase_3_plus_not_approved", "Phase 3+ not approved"),
    ("approved", "Approved"),
)
INHIBITORY_DIRECTION = "beneficial_inhibition"
UNKNOWN_DIRECTION = "unknown"
_CURIE_RE = re.compile(r"^(?P<prefix>[A-Za-z0-9_]+):(?P<num>\d+)$")


def _read_parquet(path: str | Path) -> pd.DataFrame:
    """Read TargetNexus parquet files with their release-compatible engine."""
    return pd.read_parquet(Path(path), engine="fastparquet")


def _normalize_doid(value: object) -> str | None:
    if value is None:
        return None
    match = _CURIE_RE.match(str(value).strip())
    if match is None:
        return None
    prefix = re.sub(r"[\s_]+", "", match.group("prefix").upper())
    if prefix != "DOID":
        return None
    return f"DOID:{match.group('num')}"


def _aggregate_direction(values: pd.Series) -> str:
    observed = {
        str(value).strip()
        for value in values.dropna().astype(str)
        if str(value).strip() and str(value).strip() != UNKNOWN_DIRECTION
    }
    if len(observed) == 1:
        return next(iter(observed))
    return UNKNOWN_DIRECTION


def map_target_disease_confidence(
    target_disease_labels: pd.DataFrame,
    triplets: pd.DataFrame,
    triplet_labels: pd.DataFrame,
    nodes: pd.DataFrame,
    node_xrefs: pd.DataFrame,
    gene_crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """Map release-matched TargetNexus confidence fields to CALIPER pair IDs."""
    td_required = {"target_node_id", "disease_node_id", "tier", "direction"}
    triplet_required = {"triplet_id", "target_node_id", "disease_node_id"}
    triplet_label_required = {"triplet_id", "phase"}
    node_required = {"node_id", "canonical_curie"}
    xref_required = {"node_id", "xref_id"}
    crosswalk_required = {"target_id", "gene_id"}
    required_sets = (
        (target_disease_labels, td_required, "target_disease_labels"),
        (triplets, triplet_required, "triplets"),
        (triplet_labels, triplet_label_required, "triplet_labels"),
        (nodes, node_required, "nodes"),
        (node_xrefs, xref_required, "node_xrefs"),
        (gene_crosswalk, crosswalk_required, "gene_crosswalk"),
    )
    for frame, required, name in required_sets:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    phase = triplets[list(triplet_required)].merge(
        triplet_labels[["triplet_id", "phase"]],
        on="triplet_id",
        how="inner",
        validate="one_to_one",
    )
    phase["phase"] = pd.to_numeric(phase["phase"], errors="coerce")
    phase = (
        phase.groupby(["target_node_id", "disease_node_id"], as_index=False)
        .agg(phase=("phase", "max"))
    )

    target_ids = (
        nodes[["node_id", "canonical_curie"]]
        .drop_duplicates("node_id")
        .rename(columns={"node_id": "target_node_id", "canonical_curie": "target_id"})
    )
    crosswalk = gene_crosswalk[["target_id", "gene_id"]].copy()
    crosswalk["target_id"] = crosswalk["target_id"].astype(str).str.strip()
    crosswalk["gene_id"] = crosswalk["gene_id"].astype(str).str.strip()
    crosswalk = crosswalk[
        crosswalk["gene_id"].str.fullmatch(r"NCBIGENE:\d+", na=False)
    ].drop_duplicates()
    target_map = target_ids.merge(
        crosswalk,
        on="target_id",
        how="inner",
        validate="many_to_many",
    )[["target_node_id", "gene_id"]].drop_duplicates()

    disease_map = node_xrefs[["node_id", "xref_id"]].copy()
    disease_map["disease_id"] = disease_map["xref_id"].map(_normalize_doid)
    disease_map = (
        disease_map.dropna(subset=["disease_id"])
        .rename(columns={"node_id": "disease_node_id"})[
            ["disease_node_id", "disease_id"]
        ]
        .drop_duplicates()
    )

    mapped = (
        target_disease_labels[
            ["target_node_id", "disease_node_id", "tier", "direction"]
        ]
        .merge(
            phase,
            on=["target_node_id", "disease_node_id"],
            how="left",
            validate="one_to_one",
        )
        .merge(target_map, on="target_node_id", how="inner", validate="many_to_many")
        .merge(
            disease_map,
            on="disease_node_id",
            how="inner",
            validate="many_to_many",
        )
    )
    mapped["tier"] = pd.to_numeric(mapped["tier"], errors="coerce")
    mapped["phase"] = pd.to_numeric(mapped["phase"], errors="coerce")
    mapped = mapped.dropna(
        subset=["gene_id", "disease_id", "tier", "phase", "direction"]
    )
    collapsed = (
        mapped.groupby(["gene_id", "disease_id"], as_index=False)
        .agg(
            tier=("tier", "min"),
            phase=("phase", "max"),
            direction=("direction", _aggregate_direction),
        )
        .rename(columns={"gene_id": "target_gene"})
        .sort_values(["disease_id", "target_gene"], kind="mergesort")
        .reset_index(drop=True)
    )
    collapsed["tier"] = collapsed["tier"].astype(int)
    return collapsed


def load_targetnexus_confidence(
    targetnexus_root: str | Path,
    *,
    release: str = "25.12",
    build_profile: str = "cc0_core",
) -> pd.DataFrame:
    """Load and map TargetNexus confidence metadata for one fixed release."""
    targetnexus_root = Path(targetnexus_root)
    processed = (
        targetnexus_root
        / "data"
        / "processed"
        / "targetnexus_db"
        / build_profile
        / release
        / "parquet"
    )
    return map_target_disease_confidence(
        _read_parquet(
            processed
            / "tnx_triplet"
            / "target_disease_label"
            / "part-00000.snappy.parquet"
        ),
        _read_parquet(
            processed / "tnx_triplet" / "triplet" / "part-00000.snappy.parquet"
        ),
        _read_parquet(
            processed
            / "tnx_triplet"
            / "triplet_label"
            / "part-00000.snappy.parquet"
        ),
        _read_parquet(processed / "tnx_core" / "node" / "part-00000.snappy.parquet"),
        _read_parquet(
            processed / "tnx_core" / "node_xref" / "part-00000.snappy.parquet"
        ),
        pd.read_csv(
            targetnexus_root / "data" / "mappings" / "target_gene_crosswalk.tsv",
            sep="\t",
            dtype=str,
        ),
    )


def load_target_scores(
    path: str | Path,
    *,
    k: int = 4,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Aggregate an existing Val2 cell-level score file to target level."""
    partial: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        Path(path),
        usecols=list(SCORE_COLUMNS),
        chunksize=int(chunksize),
    ):
        chunk["k"] = pd.to_numeric(chunk["k"], errors="coerce")
        chunk = chunk[chunk["k"].eq(int(k))].copy()
        if chunk.empty:
            continue
        chunk["score"] = pd.to_numeric(chunk["score"], errors="coerce")
        chunk["pair_label"] = pd.to_numeric(
            chunk["pair_label"], errors="coerce"
        ).fillna(0)
        chunk = chunk.dropna(
            subset=["disease_id", "target_gene", "method", "score"]
        )
        if chunk.empty:
            continue
        partial.append(
            chunk.groupby(
                ["disease_id", "target_gene", "method"], as_index=False
            ).agg(
                score_sum=("score", "sum"),
                score_count=("score", "count"),
                pair_label=("pair_label", "max"),
            )
        )
    if not partial:
        return pd.DataFrame(
            columns=["disease_id", "target_gene", "method", "score", "pair_label"]
        )
    combined = (
        pd.concat(partial, ignore_index=True)
        .groupby(["disease_id", "target_gene", "method"], as_index=False)
        .agg(
            score_sum=("score_sum", "sum"),
            score_count=("score_count", "sum"),
            pair_label=("pair_label", "max"),
        )
    )
    combined["score"] = combined["score_sum"] / combined["score_count"]
    return combined[
        ["disease_id", "target_gene", "method", "score", "pair_label"]
    ].sort_values(
        ["method", "disease_id", "target_gene"], kind="mergesort"
    ).reset_index(drop=True)


def confidence_positive_pairs(
    confidence: pd.DataFrame,
    stratum: str,
    *,
    direction: str = INHIBITORY_DIRECTION,
) -> set[tuple[str, str]]:
    """Return positive (disease, target) pairs for a cumulative confidence stratum."""
    selected = confidence[confidence["direction"].astype(str).eq(str(direction))].copy()
    phase = pd.to_numeric(selected["phase"], errors="coerce")
    tier = pd.to_numeric(selected["tier"], errors="coerce")
    if stratum == "all_evidence":
        keep = phase.notna()
    elif stratum == "phase_ge_2":
        keep = phase.ge(2)
    elif stratum == "phase_ge_3":
        keep = phase.ge(3)
    elif stratum == "approved":
        keep = tier.eq(1)
    else:
        raise ValueError(f"Unknown confidence stratum: {stratum}")
    selected = selected[keep]
    return set(
        map(
            tuple,
            selected[["disease_id", "target_gene"]]
            .astype(str)
            .itertuples(index=False, name=None),
        )
    )


def validate_all_evidence_labels(
    target_scores: pd.DataFrame,
    confidence: pd.DataFrame,
) -> dict[str, int]:
    """Require release-derived all-evidence labels to match the saved Val2 labels."""
    original = (
        target_scores.groupby(["disease_id", "target_gene"], as_index=False)
        .agg(pair_label=("pair_label", "max"))
    )
    positive = confidence_positive_pairs(confidence, "all_evidence")
    original["derived_label"] = [
        int((str(row.disease_id), str(row.target_gene)) in positive)
        for row in original.itertuples(index=False)
    ]
    mismatch = original[
        original["pair_label"].astype(int).ne(original["derived_label"].astype(int))
    ]
    if not mismatch.empty:
        preview = mismatch.head(5).to_dict(orient="records")
        raise ValueError(
            "Release-derived all-evidence labels do not match saved Val2 labels: "
            f"n_mismatch={len(mismatch)}, examples={preview}"
        )
    return {
        "n_pairs": int(len(original)),
        "n_positive_pairs": int(original["pair_label"].astype(int).sum()),
        "n_mismatches": 0,
    }


def assign_confidence_score_groups(
    target_scores: pd.DataFrame,
    confidence: pd.DataFrame,
    *,
    evaluation_diseases: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Assign evidence-positive pairs to mutually exclusive confidence groups."""
    if target_scores.empty:
        return pd.DataFrame()

    work = target_scores.copy()
    if evaluation_diseases is not None:
        disease_filter = {str(disease_id) for disease_id in evaluation_diseases}
        work = work[work["disease_id"].astype(str).isin(disease_filter)].copy()
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work = work[np.isfinite(work["score"])].copy()
    if work.empty:
        return pd.DataFrame()

    work["within_disease_score_percentile"] = work.groupby(
        ["method", "disease_id"],
        sort=False,
    )["score"].rank(method="average", pct=True)
    positive = {
        stratum: confidence_positive_pairs(confidence, stratum)
        for stratum, _label in CONFIDENCE_STRATA
    }

    def _group_for_pair(pair: tuple[str, str]) -> str | None:
        if pair in positive["approved"]:
            return "approved"
        if pair in positive["phase_ge_3"]:
            return "phase_3_plus_not_approved"
        if pair in positive["phase_ge_2"]:
            return "phase_2_only"
        if pair in positive["all_evidence"]:
            return "evidence_below_phase_2"
        return None

    work["confidence_group"] = [
        _group_for_pair((str(row.disease_id), str(row.target_gene)))
        for row in work.itertuples(index=False)
    ]
    work = work.dropna(subset=["confidence_group"]).copy()
    group_order = {
        name: index for index, (name, _label) in enumerate(CONFIDENCE_SCORE_GROUPS)
    }
    group_label = dict(CONFIDENCE_SCORE_GROUPS)
    work["confidence_group_order"] = work["confidence_group"].map(group_order)
    work["confidence_group_label"] = work["confidence_group"].map(group_label)
    return work.sort_values(
        ["confidence_group_order", "method", "disease_id", "target_gene"],
        kind="mergesort",
    ).reset_index(drop=True)


def evaluate_label_confidence_by_disease(
    target_scores: pd.DataFrame,
    confidence: pd.DataFrame,
    *,
    evaluation_diseases: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute disease-level raw and prevalence-normalized AP by stratum."""
    if target_scores.empty:
        return pd.DataFrame()

    disease_filter = (
        None
        if evaluation_diseases is None
        else {str(disease_id) for disease_id in evaluation_diseases}
    )
    rows: list[dict[str, object]] = []
    for stratum, stratum_label in CONFIDENCE_STRATA:
        positive = confidence_positive_pairs(confidence, stratum)
        for method, method_scores in target_scores.groupby("method", sort=False):
            work = method_scores.copy()
            if disease_filter is not None:
                work = work[
                    work["disease_id"].astype(str).isin(disease_filter)
                ].copy()
            work["label"] = [
                int((str(row.disease_id), str(row.target_gene)) in positive)
                for row in work.itertuples(index=False)
            ]
            for disease_id, disease in work.groupby("disease_id", sort=False):
                labels = disease["label"].astype(int)
                n_positive = int(labels.sum())
                if n_positive == 0:
                    continue
                metrics = _ranking_metrics_for_scores(
                    labels.tolist(),
                    pd.to_numeric(disease["score"], errors="coerce")
                    .astype(float)
                    .tolist(),
                    K_list=[10],
                    tie_policy="expected",
                )
                ap = float(metrics["AP"])
                chance = float(metrics["chance"])
                normalized_ap = (
                    float((ap - chance) / (1.0 - chance))
                    if np.isfinite(ap) and np.isfinite(chance) and chance < 1.0
                    else np.nan
                )
                rows.append(
                    {
                        "stratum": stratum,
                        "stratum_label": stratum_label,
                        "method": str(method),
                        "disease_id": str(disease_id),
                        "AP": ap,
                        "chance": chance,
                        "normalized_AP": normalized_ap,
                        "n_positive_pairs": n_positive,
                        "n_ranked_pairs": int(len(disease)),
                        "tie_policy": "expected",
                        "k": 4,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    order = {name: index for index, (name, _label) in enumerate(CONFIDENCE_STRATA)}
    result["_order"] = result["stratum"].map(order)
    return result.sort_values(
        ["_order", "method", "disease_id"], kind="mergesort"
    ).drop(columns="_order").reset_index(drop=True)


def summarize_label_confidence(
    target_scores: pd.DataFrame,
    confidence: pd.DataFrame,
    *,
    evaluation_diseases: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Macro-average disease-level raw and normalized AP by confidence stratum."""
    disease_level = evaluate_label_confidence_by_disease(
        target_scores,
        confidence,
        evaluation_diseases=evaluation_diseases,
    )
    if disease_level.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (stratum, stratum_label, method), group in disease_level.groupby(
        ["stratum", "stratum_label", "method"],
        sort=False,
    ):
        ap_array = pd.to_numeric(group["AP"], errors="coerce").to_numpy(dtype=float)
        normalized_array = pd.to_numeric(
            group["normalized_AP"], errors="coerce"
        ).to_numpy(dtype=float)
        ap_array = ap_array[np.isfinite(ap_array)]
        normalized_array = normalized_array[np.isfinite(normalized_array)]
        n_diseases = int(normalized_array.size)
        rows.append(
            {
                "stratum": str(stratum),
                "stratum_label": str(stratum_label),
                "method": str(method),
                "mAP": float(np.mean(ap_array)) if ap_array.size else np.nan,
                "mAP_SEM": (
                    float(np.std(ap_array, ddof=1) / np.sqrt(ap_array.size))
                    if ap_array.size >= 2
                    else np.nan
                ),
                "normalized_mAP": (
                    float(np.mean(normalized_array)) if n_diseases else np.nan
                ),
                "normalized_mAP_SEM": (
                    float(
                        np.std(normalized_array, ddof=1) / np.sqrt(n_diseases)
                    )
                    if n_diseases >= 2
                    else np.nan
                ),
                "mean_chance": float(
                    pd.to_numeric(group["chance"], errors="coerce").mean()
                ),
                "n_diseases": n_diseases,
                "n_positive_pairs": int(group["n_positive_pairs"].sum()),
                "n_ranked_pairs": int(group["n_ranked_pairs"].sum()),
                "tie_policy": "expected",
                "k": 4,
            }
        )

    result = pd.DataFrame(rows)
    order = {name: index for index, (name, _label) in enumerate(CONFIDENCE_STRATA)}
    result["_order"] = result["stratum"].map(order)
    return result.sort_values(
        ["_order", "method"], kind="mergesort"
    ).drop(columns="_order").reset_index(drop=True)
