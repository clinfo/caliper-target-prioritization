from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from .io import ensure_dir, write_json, write_parquet
from .prepare import build_pert_input


DECOY_NULL_TYPE = "observed_target_pool_decoy_control"
LEGACY_DECOY_NULL_TYPE = "evaluable_decoy_anchor_resampling"
TARGET_DECOY_NULL_TYPE = "topology_profile_matched_target_decoy_control"

# The observed-target-pool control and its pre-v2.6 alias are the legacy decoy nulls.
# They stay readable for reproducing existing runs but are never silently converted
# into the topology- and profile-matched control.
LEGACY_DECOY_NULL_TYPES = frozenset({DECOY_NULL_TYPE, LEGACY_DECOY_NULL_TYPE})
DECOY_NULL_TYPES = frozenset(LEGACY_DECOY_NULL_TYPES | {TARGET_DECOY_NULL_TYPE})

# Fixed model definition of the topology- and profile-matched decoy target control.
TARGET_DECOY_CANDIDATE_POOL_TYPE = "l1000_kd_profile_x_signor_nodes"
TARGET_DECOY_CANDIDATE_POOL_TYPE_LABELED = "l1000_kd_profile_x_signor_nodes_x_labeled_targets"
# Primary universe: exactly the profile-supported observed-target population used by
# the legacy observed-target-pool control, while retaining the new matching rule.
TARGET_DECOY_CANDIDATE_POOL_TYPE_PROFILE_SUPPORTED_OBSERVED = (
    "l1000_kd_profile_x_signor_nodes_x_profile_supported_observed_targets"
)
TARGET_DECOY_CANDIDATE_POOL_TYPES = (
    TARGET_DECOY_CANDIDATE_POOL_TYPE,
    TARGET_DECOY_CANDIDATE_POOL_TYPE_LABELED,
    TARGET_DECOY_CANDIDATE_POOL_TYPE_PROFILE_SUPPORTED_OBSERVED,
)
TARGET_DECOY_POSITIVE_EXCLUSION_SCOPE = "same_drug_disease"
TARGET_DECOY_N_DECOYS_PER_TRIPLET = 10
TARGET_DECOY_MATCHING_VARIABLES: tuple[str, ...] = (
    "directed_out_degree",
    "reachable_unique_nodes_k4",
    "unique_kd_cell_lines",
)
TARGET_DECOY_DIAGNOSTIC_VARIABLES: tuple[str, ...] = ("total_degree_diagnostic",)
TARGET_DECOY_RANK_COLUMNS: dict[str, str] = {
    "directed_out_degree": "rank_out_degree",
    "reachable_unique_nodes_k4": "rank_reachable_k4",
    "unique_kd_cell_lines": "rank_kd_cell_lines",
}
TARGET_DECOY_RANK_TRANSFORM = "midrank_percentile"
TARGET_DECOY_DISTANCE = "equal_weight_euclidean"
TARGET_DECOY_GRAPH_TYPE = "networkx.MultiDiGraph"
TARGET_DECOY_EDGE_COUNT_RULE = "parallel_signed_edges_counted"
# Primary matching acceptance: |SMD| < 0.1 on the three matching variables, judged on
# the primary path-evaluable triplets before inference with triplet weighting.
TARGET_DECOY_SMD_ACCEPTANCE_THRESHOLD = 0.1
TARGET_DECOY_ACCEPTANCE_ANALYSIS_SCOPE = "primary_path_evaluable"
TARGET_DECOY_ACCEPTANCE_STAGE = "pre_inference_selected"
TARGET_DECOY_ACCEPTANCE_WEIGHTING = "triplet_weighted"

# How the decoys are drawn out of the candidate universe. Both modes share the
# universe and the exclusion rules; only the selection rule differs.
TARGET_DECOY_SELECTION_MATCHED = "matched"
TARGET_DECOY_SELECTION_RANDOM = "random"
TARGET_DECOY_SELECTIONS = (TARGET_DECOY_SELECTION_MATCHED, TARGET_DECOY_SELECTION_RANDOM)
TARGET_DECOY_SELECTION_NOT_APPLICABLE = "not_applicable"


def canonical_decoy_null_type(null_type: Any) -> str:
    """Map the pre-v2.6 alias onto the observed-target-pool control name.

    The topology- and profile-matched control is never rewritten: legacy and new
    decoy nulls stay distinguishable in every output row.
    """
    raw = str(null_type)
    return DECOY_NULL_TYPE if raw == LEGACY_DECOY_NULL_TYPE else raw


def null_representative_kind(null_type: Any) -> str:
    """Per-triplet null representative used for paired deltas.

    Every null model uses the median over the triplet's null draws / evaluable decoys.
    The function and the ``baseline_representative`` column are kept so that output
    tables stay self-describing.
    """
    return "median"



def _unique_observed_triplets(df: pd.DataFrame, *, direction: str | None = None) -> pd.DataFrame:
    required = {"triplet_id", "drug_id", "target_id", "disease_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"observed_triplets missing columns: {sorted(missing)}")
    cols = ["triplet_id", "drug_id", "target_id", "disease_id"]
    if "direction" in df.columns:
        cols.append("direction")
    out = df[cols].drop_duplicates("triplet_id").copy()
    if "direction" not in out.columns:
        out["direction"] = str(direction or "")
    out["direction"] = out["direction"].astype(str)
    return out


def build_decoy_candidate_table(
    observed_triplets: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    kg_nodes: set[str],
    direction: str,
    exclude_known_positive_targets: bool = True,
    max_decoy_anchors_per_triplet: int | None = None,
    rng: np.random.Generator | None = None,
    anchor_universe_triplets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build observed-universe decoy anchors before path-evaluability filtering.

    Null 3 controls are target-like anchors drawn from the same profile-supported
    observed triplet universe.  For a given drug-disease pair, true positive
    targets for that same pair are excluded from the decoy pool.
    """
    if "pert_id" not in target_df.columns:
        raise ValueError("target_df must contain pert_id to build decoy anchors.")
    obs = _unique_observed_triplets(observed_triplets, direction=direction)
    anchor_source = _unique_observed_triplets(
        obs if anchor_universe_triplets is None else anchor_universe_triplets,
        direction=direction,
    )
    kg = set(map(str, kg_nodes))
    target_ids_with_profile = set(target_df["pert_id"].dropna().astype(str))
    anchors = sorted(set(anchor_source["target_id"].dropna().astype(str)) & target_ids_with_profile & kg)
    rng = rng or np.random.default_rng()
    positives_by_drug_disease = (
        anchor_source.groupby(["drug_id", "disease_id"], dropna=False)["target_id"]
        .apply(lambda s: set(s.astype(str))).to_dict()
        if exclude_known_positive_targets else {}
    )
    target_profile_pairs: set[tuple[str, str]] | None = None
    if "disease_id" in target_df.columns:
        target_profile_pairs = set(
            zip(target_df["pert_id"].astype(str), target_df["disease_id"].astype(str))
        )

    rows: list[dict[str, Any]] = []
    for rec in obs.to_dict(orient="records"):
        original_triplet_id = str(rec["triplet_id"])
        drug_id = str(rec["drug_id"])
        original_target_id = str(rec["target_id"])
        disease_id = str(rec["disease_id"])
        excluded = {original_target_id}
        if exclude_known_positive_targets:
            excluded |= set(positives_by_drug_disease.get((drug_id, disease_id), set()))
        candidates = [a for a in anchors if a not in excluded]
        if target_profile_pairs is not None:
            candidates = [a for a in candidates if (str(a), disease_id) in target_profile_pairs]
        if max_decoy_anchors_per_triplet is not None and int(max_decoy_anchors_per_triplet) >= 0:
            max_n = int(max_decoy_anchors_per_triplet)
            if len(candidates) > max_n:
                idx = rng.choice(len(candidates), size=max_n, replace=False)
                candidates = [candidates[int(i)] for i in sorted(idx)]
        for anchor in candidates:
            pseudo = f"{original_triplet_id}__decoy__{anchor}"
            rows.append({
                "original_triplet_id": original_triplet_id,
                "pseudo_triplet_id": pseudo,
                "drug_id": drug_id,
                "original_target_id": original_target_id,
                "decoy_anchor_id": str(anchor),
                "disease_id": disease_id,
                "direction": str(rec.get("direction", direction)),
                "candidate_pool_type": "observed_target_pool",
                "exclude_positive_scope": "same_drug_disease",
            })
    return pd.DataFrame(rows, columns=[
        "original_triplet_id", "pseudo_triplet_id", "drug_id", "original_target_id",
        "decoy_anchor_id", "disease_id", "direction", "candidate_pool_type",
        "exclude_positive_scope",
    ])


def build_decoy_inputs(
    decoy_candidates: pd.DataFrame,
    drug_df: pd.DataFrame,
    target_df: pd.DataFrame,
    disease_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build drug/pert infer inputs for all decoy pseudo-triplets."""
    if decoy_candidates.empty:
        empty = pd.DataFrame()
        return empty, empty, {"n_decoy_candidates": 0, "n_drug_rows": 0, "n_pert_rows": 0}
    required = {"original_triplet_id", "pseudo_triplet_id", "drug_id", "original_target_id", "decoy_anchor_id", "disease_id", "direction"}
    missing = required - set(decoy_candidates.columns)
    if missing:
        raise ValueError(f"decoy_candidates missing columns: {sorted(missing)}")

    drug_parts: list[pd.DataFrame] = []
    for cand in decoy_candidates.to_dict(orient="records"):
        sub = drug_df.loc[drug_df["triplet_id"].astype(str).eq(str(cand["original_triplet_id"]))].copy()
        if sub.empty:
            continue
        sub["triplet_id"] = str(cand["pseudo_triplet_id"])
        sub["target_id"] = str(cand["decoy_anchor_id"])
        sub["pert_id"] = str(cand["decoy_anchor_id"])
        sub["drug_id"] = str(cand["drug_id"])
        sub["do_id"] = str(cand["disease_id"])
        sub["original_triplet_id"] = str(cand["original_triplet_id"])
        sub["original_target_id"] = str(cand["original_target_id"])
        sub["decoy_anchor_id"] = str(cand["decoy_anchor_id"])
        drug_parts.append(sub)
    drug_input = pd.concat(drug_parts, ignore_index=True) if drug_parts else pd.DataFrame(columns=drug_df.columns)
    if not drug_input.empty:
        drug_input["row_id"] = np.arange(len(drug_input), dtype=np.int64)

    pseudo_triplets = decoy_candidates.rename(columns={
        "pseudo_triplet_id": "triplet_id",
        "decoy_anchor_id": "target_id",
    })[["triplet_id", "drug_id", "target_id", "disease_id", "direction", "original_triplet_id", "original_target_id"]].copy()
    pert_input, pert_summary = build_pert_input(pseudo_triplets, target_df, disease_df)
    meta = decoy_candidates[["pseudo_triplet_id", "original_triplet_id", "original_target_id", "decoy_anchor_id"]].rename(columns={"pseudo_triplet_id": "triplet_id"})
    if not pert_input.empty:
        pert_input = pert_input.merge(meta, on="triplet_id", how="left")
        pert_input["row_id"] = np.arange(len(pert_input), dtype=np.int64)

    summary = {
        "n_decoy_candidates": int(len(decoy_candidates)),
        "n_drug_rows": int(len(drug_input)),
        "n_pert_rows": int(len(pert_input)),
        "pert_summary": pert_summary,
    }
    return drug_input, pert_input, summary


def summarize_decoy_candidates(
    decoy_candidates: pd.DataFrame,
    *,
    null_type: str = DECOY_NULL_TYPE,
) -> pd.DataFrame:
    """Per-triplet candidate counts.

    The same ``decoy_candidate_diagnostics.tsv`` filename is written by both decoy
    controls, where the count means a variable observed-pool anchor count for the
    legacy control and the fixed 1:N match count for the matched control, so the
    responsible ``null_type`` is recorded in the table itself.
    """
    columns = ["null_type", "triplet_id", "drug_id", "target_id", "disease_id", "direction", "n_decoy_candidates"]
    if decoy_candidates.empty:
        return pd.DataFrame(columns=columns)
    out = decoy_candidates.groupby(
        ["original_triplet_id", "drug_id", "original_target_id", "disease_id", "direction"],
        dropna=False,
    ).agg(n_decoy_candidates=("decoy_anchor_id", "nunique")).reset_index()
    out = out.rename(columns={"original_triplet_id": "triplet_id", "original_target_id": "target_id"})
    out["null_type"] = str(null_type)
    return out[columns]


def save_decoy_candidate_inputs(
    *,
    out_dir: str | Path,
    decoy_candidates: pd.DataFrame,
    drug_input: pd.DataFrame,
    pert_input: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, str]:
    out = ensure_dir(out_dir)
    candidate_path = out / "decoy_candidate_table.tsv"
    drug_path = out / "drug.parquet"
    pert_path = out / "pert.parquet"
    summary_path = out / "decoy_input_summary.json"
    decoy_candidates.to_csv(candidate_path, sep="\t", index=False)
    write_parquet(drug_input, drug_path)
    write_parquet(pert_input, pert_path)
    write_json(summary, summary_path)
    return {
        "candidate_table": str(candidate_path),
        "candidate_drug_input": str(drug_path),
        "candidate_pert_input": str(pert_path),
        "summary": str(summary_path),
    }


def save_decoy_candidate_table(
    *,
    out_dir: str | Path,
    decoy_candidates: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, str]:
    out = ensure_dir(out_dir)
    candidate_path = out / "decoy_candidate_table.tsv"
    summary_path = out / "decoy_input_summary.json"
    decoy_candidates.to_csv(candidate_path, sep="\t", index=False)
    write_json(summary, summary_path)
    return {
        "candidate_table": str(candidate_path),
        "summary": str(summary_path),
    }


def iter_decoy_paired_input_batches(
    *,
    decoy_candidates: pd.DataFrame,
    drug_df: pd.DataFrame,
    target_df: pd.DataFrame,
    disease_df: pd.DataFrame | None,
    batch_size_candidates: int = 250,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield the drug and pert infer input of each candidate batch together.

    ``build_decoy_inputs`` builds both sides in one call, so iterating per side
    rebuilds -- and discards -- the other side once per batch. Callers that infer both
    sides consume the pair and pay that construction once.
    """
    n = len(decoy_candidates)
    if n <= 0:
        return
    batch_size = max(1, int(batch_size_candidates))
    for start in range(0, n, batch_size):
        sub = decoy_candidates.iloc[start:start + batch_size].copy()
        drug_input, pert_input, _ = build_decoy_inputs(sub, drug_df, target_df, disease_df)
        yield drug_input, pert_input


def iter_decoy_input_batches(
    *,
    decoy_candidates: pd.DataFrame,
    drug_df: pd.DataFrame,
    target_df: pd.DataFrame,
    disease_df: pd.DataFrame | None,
    side: str,
    batch_size_candidates: int = 250,
) -> Iterator[pd.DataFrame]:
    if side not in {"drug", "pert"}:
        raise ValueError("side must be 'drug' or 'pert'.")
    for drug_input, pert_input in iter_decoy_paired_input_batches(
        decoy_candidates=decoy_candidates,
        drug_df=drug_df,
        target_df=target_df,
        disease_df=disease_df,
        batch_size_candidates=batch_size_candidates,
    ):
        yield drug_input if side == "drug" else pert_input


def decoy_candidate_manifest(paths: dict[str, str], *, null_type: str = DECOY_NULL_TYPE) -> dict[str, Any]:
    return {"null_type": str(null_type), **paths}


# ---------------------------------------------------------------------------
# topology- and profile-matched decoy target control
# ---------------------------------------------------------------------------

def load_directed_kg_graph(kg_tsv: str | Path) -> nx.MultiDiGraph:
    """Load the resolved directed KG used by observed path inference."""
    kg = pd.read_csv(kg_tsv, sep="\t")
    required = {"source", "target", "polarity"}
    missing = required - set(kg.columns)
    if missing:
        raise ValueError(f"Resolved KG file missing columns: {sorted(missing)}")
    kg = kg.copy()
    kg["source"] = kg["source"].astype(str)
    kg["target"] = kg["target"].astype(str)
    kg["polarity"] = pd.to_numeric(kg["polarity"], errors="raise").astype(int)
    return nx.from_pandas_edgelist(
        kg,
        source="source",
        target="target",
        edge_attr=["polarity"],
        create_using=nx.MultiDiGraph(),
    )


def build_target_decoy_candidate_universe(
    target_df: pd.DataFrame,
    *,
    kg_nodes: set[str] | Sequence[str],
    labeled_target_ids: set[str] | Sequence[str] | None = None,
) -> list[str]:
    """Candidate universe = L1000 KD targets ∩ resolved KG nodes [∩ allowed targets].

    Without ``labeled_target_ids`` the universe is deliberately not restricted to
    a label-derived target set, and nodes with zero out-degree or zero reach are kept.
    When ``labeled_target_ids`` is given, the universe is further restricted to that
    target set. The caller supplies either the full label target set for an
    exploratory run or the profile-supported observed target set for the primary
    legacy-population-matched run.
    """
    if "pert_id" not in target_df.columns:
        raise ValueError("target_df must contain pert_id to build the decoy candidate universe.")
    kd_targets = set(target_df["pert_id"].dropna().astype(str))
    kg = {str(n) for n in kg_nodes}
    universe = kd_targets & kg
    if labeled_target_ids is not None:
        universe &= {str(t) for t in labeled_target_ids}
    return sorted(universe)


def midrank_percentile(values: pd.Series) -> pd.Series:
    """Candidate-universe mid-rank percentile: (average_rank - 0.5) / n."""
    x = pd.to_numeric(pd.Series(values), errors="coerce").astype(float)
    n = int(len(x))
    if n == 0:
        return pd.Series(dtype=float)
    midrank = x.rank(method="average", ascending=True)
    return (midrank - 0.5) / float(n)


def build_target_decoy_matching_features(
    candidate_universe: Sequence[str],
    *,
    graph: nx.MultiDiGraph,
    target_df: pd.DataFrame,
    reach_cutoff: int,
    candidate_pool_type: str = TARGET_DECOY_CANDIDATE_POOL_TYPE,
) -> pd.DataFrame:
    """Matching-feature table over the candidate universe.

    Ranks are fixed here, before any triplet-specific candidate exclusion.
    ``candidate_pool_type`` only labels which universe ``candidate_universe`` was
    built from; it does not change how features are computed.
    """
    candidates = [str(c) for c in candidate_universe]
    if not candidates:
        raise ValueError("Target-Decoy candidate universe is empty.")
    if "cell_id" not in target_df.columns:
        raise ValueError("target_df must contain cell_id to count unique KD cell lines.")
    cutoff = int(reach_cutoff)
    if cutoff < 1:
        raise ValueError("reach_cutoff must be a positive integer.")

    kd_cells = (
        target_df[["pert_id", "cell_id"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .groupby("pert_id")["cell_id"]
        .nunique()
        .to_dict()
    )
    deg_out = dict(graph.out_degree())
    deg_in = dict(graph.in_degree())

    rows: list[dict[str, Any]] = []
    for c in candidates:
        n_out = int(deg_out.get(c, 0))
        n_in = int(deg_in.get(c, 0))
        if c in graph:
            reachable = len(nx.single_source_shortest_path_length(graph, c, cutoff=cutoff)) - 1
        else:
            reachable = 0
        n_cells = int(kd_cells.get(c, 0))
        if n_cells < 1:
            raise ValueError(
                f"Candidate {c} has no KD profile cell line; the candidate universe is inconsistent."
            )
        rows.append({
            "target_id": c,
            "directed_out_degree": n_out,
            "reachable_unique_nodes_k4": int(max(0, reachable)),
            "unique_kd_cell_lines": n_cells,
            "total_degree_diagnostic": n_in + n_out,
        })
    out = pd.DataFrame(rows)
    for var, rank_col in TARGET_DECOY_RANK_COLUMNS.items():
        out[rank_col] = midrank_percentile(out[var]).to_numpy(dtype=float)
    out["reach_cutoff"] = cutoff
    out["candidate_pool_type"] = str(candidate_pool_type)
    out["graph_type"] = TARGET_DECOY_GRAPH_TYPE
    out["edge_count_rule"] = TARGET_DECOY_EDGE_COUNT_RULE
    return out


def build_positive_targets_by_drug_disease(label_triplets: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    """Same drug–disease positive targets from the full label input.

    ``direction`` is never part of the exclusion key, and the label input must not
    be pre-filtered by path evaluability.
    """
    required = {"drug_id", "target_id", "disease_id"}
    missing = required - set(label_triplets.columns)
    if missing:
        raise ValueError(f"positive-label input missing columns: {sorted(missing)}")
    sub = label_triplets[["drug_id", "target_id", "disease_id"]].dropna().astype(str)
    out: dict[tuple[str, str], set[str]] = {}
    for drug_id, disease_id, target_id in sub[["drug_id", "disease_id", "target_id"]].itertuples(index=False, name=None):
        out.setdefault((drug_id, disease_id), set()).add(target_id)
    return out


def target_decoy_tie_break_key(seed: Any, triplet_id: str, candidate_target_id: str) -> str:
    """Platform-stable tie-break key; Python's builtin ``hash`` is never used."""
    payload = "\0".join([str(seed), str(triplet_id), str(candidate_target_id)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


TARGET_DECOY_ASSIGNMENT_COLUMNS: list[str] = (
    [
        "original_triplet_id",
        "pseudo_triplet_id",
        "drug_id",
        "original_target_id",
        "decoy_target_id",
        "decoy_anchor_id",
        "disease_id",
        "direction",
        "match_rank",
        "matching_distance",
        "candidate_pool_type",
        "positive_exclusion_scope",
        "selection_mode",
        "n_eligible_candidates",
    ]
    + [f"observed_{v}" for v in TARGET_DECOY_MATCHING_VARIABLES + TARGET_DECOY_DIAGNOSTIC_VARIABLES]
    + [f"decoy_{v}" for v in TARGET_DECOY_MATCHING_VARIABLES + TARGET_DECOY_DIAGNOSTIC_VARIABLES]
    + [f"observed_{TARGET_DECOY_RANK_COLUMNS[v]}" for v in TARGET_DECOY_MATCHING_VARIABLES]
    + [f"decoy_{TARGET_DECOY_RANK_COLUMNS[v]}" for v in TARGET_DECOY_MATCHING_VARIABLES]
)


def build_target_decoy_assignments(
    observed_triplets: pd.DataFrame,
    *,
    feature_table: pd.DataFrame,
    positive_targets_by_drug_disease: dict[tuple[str, str], set[str]],
    direction: str,
    n_decoys_per_triplet: int = TARGET_DECOY_N_DECOYS_PER_TRIPLET,
    seed: int = 0,
    exclude_known_positive_targets: bool = True,
    decoy_selection: str = TARGET_DECOY_SELECTION_MATCHED,
) -> pd.DataFrame:
    """Target-Decoy assignment over the shared candidate universe.

    ``matched`` selects the 1:N percentile-rank nearest neighbours; ``random`` draws
    uniformly from the same eligible pool.  The candidate universe, the exclusion
    rules and the recorded raw / rank features are identical in both modes, so the
    two selections differ only in which candidates are kept.

    Selection uses only the pre-inference matching features: path evaluability is
    never consulted here, and candidates are never replaced afterwards.  The uniform
    draw is realized as the ascending order of the SHA-256 tie-break key rather than
    an ``np.random`` stream, so it depends on ``seed`` and ``triplet_id`` alone and is
    independent of worker count and processing order.

    A triplet whose eligible pool is smaller than ``n_decoys_per_triplet`` is dropped
    from the output entirely rather than raising or reusing candidates: it simply has
    no rows in the returned table, so it falls out of every downstream comparison the
    same way a triplet with zero evaluable decoys does.
    """
    obs = _unique_observed_triplets(observed_triplets, direction=direction)
    n_decoys = int(n_decoys_per_triplet)
    if n_decoys < 1:
        raise ValueError("n_decoys_per_triplet must be >= 1.")
    if feature_table.empty:
        raise ValueError("Target-Decoy matching-feature table is empty.")
    selection_mode = str(decoy_selection)
    if selection_mode not in TARGET_DECOY_SELECTIONS:
        raise ValueError(f"decoy_selection must be one of {sorted(TARGET_DECOY_SELECTIONS)}.")
    pseudo_infix = "matched_decoy" if selection_mode == TARGET_DECOY_SELECTION_MATCHED else "random_decoy"

    features = feature_table.copy()
    features["target_id"] = features["target_id"].astype(str)
    pool_type = (
        str(features["candidate_pool_type"].iloc[0])
        if "candidate_pool_type" in features.columns and not features.empty
        else TARGET_DECOY_CANDIDATE_POOL_TYPE
    )
    candidate_ids = np.array(features["target_id"].to_numpy(), dtype=object)
    rank_cols = [TARGET_DECOY_RANK_COLUMNS[v] for v in TARGET_DECOY_MATCHING_VARIABLES]
    rank_matrix = features[rank_cols].to_numpy(dtype=float)
    raw_cols = list(TARGET_DECOY_MATCHING_VARIABLES) + list(TARGET_DECOY_DIAGNOSTIC_VARIABLES)
    raw_lookup = {
        col: dict(zip(features["target_id"], features[col].to_numpy()))
        for col in raw_cols
    }
    rank_lookup = {
        col: dict(zip(features["target_id"], features[col].to_numpy(dtype=float)))
        for col in rank_cols
    }

    rows: list[dict[str, Any]] = []
    for rec in obs.to_dict(orient="records"):
        triplet_id = str(rec["triplet_id"])
        drug_id = str(rec["drug_id"])
        observed_target_id = str(rec["target_id"])
        disease_id = str(rec["disease_id"])
        if observed_target_id not in rank_lookup[rank_cols[0]]:
            raise ValueError(
                f"Observed target {observed_target_id} of triplet {triplet_id} is not in the "
                "Target-Decoy candidate universe."
            )
        excluded = {observed_target_id}
        if exclude_known_positive_targets:
            excluded |= set(positive_targets_by_drug_disease.get((drug_id, disease_id), set()))
        eligible_mask = np.array([cid not in excluded for cid in candidate_ids], dtype=bool)
        n_eligible = int(eligible_mask.sum())
        if n_eligible < n_decoys:
            # Candidates are never reused within a triplet, so a pool this small cannot
            # fill the request; drop the triplet instead of raising or reusing candidates.
            continue
        observed_ranks = np.array([rank_lookup[c][observed_target_id] for c in rank_cols], dtype=float)
        diff = rank_matrix[eligible_mask] - observed_ranks
        distances = np.sqrt(np.sum(diff * diff, axis=1))
        eligible_ids = candidate_ids[eligible_mask]

        if selection_mode == TARGET_DECOY_SELECTION_MATCHED:
            # Only candidates at or below the N-th smallest distance can be selected, so
            # the stable tie-break hash is computed for that small subset instead of the
            # whole eligible pool.
            threshold = float(np.partition(distances, n_decoys - 1)[n_decoys - 1])
            near_idx = np.flatnonzero(distances <= threshold)
            ordered = sorted(
                (
                    (
                        float(distances[i]),
                        target_decoy_tie_break_key(seed, triplet_id, str(eligible_ids[i])),
                        str(eligible_ids[i]),
                    )
                    for i in near_idx
                )
            )
        else:
            # A uniform draw without within-triplet replacement: the tie-break key is a
            # seed-dependent pseudo-random permutation of the eligible pool, so taking its
            # first N entries is the same as sampling N distinct candidates uniformly.
            # The rank distance is still recorded, but it does not drive the selection.
            ordered = sorted(
                (
                    (
                        target_decoy_tie_break_key(seed, triplet_id, str(eligible_ids[i])),
                        float(distances[i]),
                        str(eligible_ids[i]),
                    )
                    for i in range(len(eligible_ids))
                )
            )
            ordered = [(distance, tie_key, cid) for tie_key, distance, cid in ordered]
        for match_rank, (distance, _tie_key, decoy_target_id) in enumerate(ordered[:n_decoys], start=1):
            row: dict[str, Any] = {
                "original_triplet_id": triplet_id,
                "pseudo_triplet_id": f"{triplet_id}__{pseudo_infix}__{decoy_target_id}",
                "drug_id": drug_id,
                "original_target_id": observed_target_id,
                "decoy_target_id": decoy_target_id,
                "decoy_anchor_id": decoy_target_id,
                "disease_id": disease_id,
                "direction": str(rec.get("direction", direction)),
                "match_rank": int(match_rank),
                "matching_distance": float(distance),
                "candidate_pool_type": pool_type,
                "positive_exclusion_scope": TARGET_DECOY_POSITIVE_EXCLUSION_SCOPE,
                "selection_mode": selection_mode,
                "n_eligible_candidates": n_eligible,
            }
            for col in raw_cols:
                row[f"observed_{col}"] = raw_lookup[col][observed_target_id]
                row[f"decoy_{col}"] = raw_lookup[col][decoy_target_id]
            for col in rank_cols:
                row[f"observed_{col}"] = float(rank_lookup[col][observed_target_id])
                row[f"decoy_{col}"] = float(rank_lookup[col][decoy_target_id])
            rows.append(row)
    return pd.DataFrame(rows, columns=TARGET_DECOY_ASSIGNMENT_COLUMNS)


def summarize_target_decoy_assignments(
    assignments: pd.DataFrame,
    *,
    n_input_triplets: int | None = None,
) -> dict[str, Any]:
    """Summarize the assignment table.

    ``n_input_triplets`` is the number of profile-supported observed triplets offered
    to :func:`build_target_decoy_assignments`; when given, the gap between it and the
    number of triplets that actually received decoys is reported as
    ``n_triplets_excluded_insufficient_eligible`` (triplets dropped because their
    eligible pool was smaller than ``n_decoys_per_triplet``).
    """
    if assignments.empty:
        out: dict[str, Any] = {
            "null_type": TARGET_DECOY_NULL_TYPE,
            "n_matching_assignments": 0,
            "n_original_triplets": 0,
        }
        if n_input_triplets is not None:
            out["n_triplets_excluded_insufficient_eligible"] = int(n_input_triplets)
        return out
    per_triplet = assignments.groupby("original_triplet_id", dropna=False)["decoy_target_id"].nunique()
    selection_modes = sorted(
        set(assignments["selection_mode"].astype(str))
        if "selection_mode" in assignments.columns else {TARGET_DECOY_SELECTION_MATCHED}
    )
    pool_types = sorted(
        set(assignments["candidate_pool_type"].astype(str))
        if "candidate_pool_type" in assignments.columns else {TARGET_DECOY_CANDIDATE_POOL_TYPE}
    )
    n_original_triplets = int(assignments["original_triplet_id"].astype(str).nunique())
    out = {
        "null_type": TARGET_DECOY_NULL_TYPE,
        "candidate_pool_type": pool_types[0] if len(pool_types) == 1 else pool_types,
        "positive_exclusion_scope": TARGET_DECOY_POSITIVE_EXCLUSION_SCOPE,
        # For a random draw the distance summary below is a diagnostic of how far the
        # drawn decoys happened to land, not a property of the selection rule.
        "selection_mode": selection_modes[0] if len(selection_modes) == 1 else selection_modes,
        "n_matching_assignments": int(len(assignments)),
        "n_original_triplets": n_original_triplets,
        "n_unique_decoy_targets": int(assignments["decoy_target_id"].astype(str).nunique()),
        "n_decoys_per_triplet_min": int(per_triplet.min()),
        "n_decoys_per_triplet_max": int(per_triplet.max()),
        "matching_distance_mean": float(pd.to_numeric(assignments["matching_distance"], errors="coerce").mean()),
        "matching_distance_max": float(pd.to_numeric(assignments["matching_distance"], errors="coerce").max()),
        "within_triplet_replacement": False,
        "across_triplet_reuse": True,
        "path_evaluability_prefilter": False,
        "post_inference_backfill": False,
        "materialized_prepared_inputs": False,
    }
    if n_input_triplets is not None:
        out["n_triplets_excluded_insufficient_eligible"] = int(n_input_triplets) - n_original_triplets
    return out


def _weighted_mean_and_population_var(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    w_sum = float(np.sum(weights))
    if w_sum <= 0:
        return float("nan"), float("nan")
    mean = float(np.sum(weights * values) / w_sum)
    var = float(np.sum(weights * (values - mean) ** 2) / w_sum)
    return mean, var


TARGET_DECOY_BALANCE_COLUMNS = [
    "analysis_scope",
    "stage",
    "weighting",
    "variable",
    "n_observed_rows",
    "n_decoy_rows",
    "observed_mean",
    "decoy_mean",
    "observed_sd",
    "decoy_sd",
    "smd",
    "abs_smd",
    "is_matching_variable",
]


def build_target_decoy_matching_balance(
    assignments: pd.DataFrame,
    *,
    analysis_scope: str,
    stage: str,
    triplet_ids: set[str] | None = None,
    surviving_col: str | None = None,
) -> pd.DataFrame:
    """Standardized mean differences between observed and matched decoy targets.

    ``triplet_weighted`` gives every observed row weight 1 and splits weight 1 over
    the triplet's decoy rows.  ``target_weighted`` additionally divides by the number
    of triplets sharing the observed target, so each observed-target stratum carries
    total weight 1 in both groups.
    """
    if assignments.empty:
        return pd.DataFrame(columns=TARGET_DECOY_BALANCE_COLUMNS)
    df = assignments.copy()
    df["original_triplet_id"] = df["original_triplet_id"].astype(str)
    df["original_target_id"] = df["original_target_id"].astype(str)
    if triplet_ids is not None:
        df = df.loc[df["original_triplet_id"].isin({str(t) for t in triplet_ids})].copy()
    if surviving_col is not None:
        if surviving_col not in df.columns:
            raise ValueError(f"assignments missing surviving flag column: {surviving_col}")
        df = df.loc[df[surviving_col].fillna(False).astype(bool)].copy()
    if df.empty:
        return pd.DataFrame(columns=TARGET_DECOY_BALANCE_COLUMNS)

    n_decoys_by_triplet = df.groupby("original_triplet_id", dropna=False).size().to_dict()
    observed_rows = df.drop_duplicates("original_triplet_id").copy()
    n_triplets_by_target = observed_rows.groupby("original_target_id", dropna=False).size().to_dict()

    variables = list(TARGET_DECOY_MATCHING_VARIABLES) + list(TARGET_DECOY_DIAGNOSTIC_VARIABLES)
    rows: list[dict[str, Any]] = []
    for weighting in ["triplet_weighted", "target_weighted"]:
        if weighting == "triplet_weighted":
            obs_w = np.ones(len(observed_rows), dtype=float)
            decoy_w = np.array(
                [1.0 / float(n_decoys_by_triplet[t]) for t in df["original_triplet_id"]],
                dtype=float,
            )
        else:
            obs_w = np.array(
                [1.0 / float(n_triplets_by_target[t]) for t in observed_rows["original_target_id"]],
                dtype=float,
            )
            decoy_w = np.array(
                [
                    1.0 / (float(n_decoys_by_triplet[t]) * float(n_triplets_by_target[g]))
                    for t, g in zip(df["original_triplet_id"], df["original_target_id"])
                ],
                dtype=float,
            )
        for var in variables:
            obs_values = pd.to_numeric(observed_rows[f"observed_{var}"], errors="coerce").to_numpy(dtype=float)
            decoy_values = pd.to_numeric(df[f"decoy_{var}"], errors="coerce").to_numpy(dtype=float)
            obs_mean, obs_var = _weighted_mean_and_population_var(obs_values, obs_w)
            decoy_mean, decoy_var = _weighted_mean_and_population_var(decoy_values, decoy_w)
            pooled = (obs_var + decoy_var) / 2.0
            smd = float((obs_mean - decoy_mean) / np.sqrt(pooled)) if pooled > 0 else float("nan")
            rows.append({
                "analysis_scope": str(analysis_scope),
                "stage": str(stage),
                "weighting": weighting,
                "variable": var,
                "n_observed_rows": int(len(observed_rows)),
                "n_decoy_rows": int(len(df)),
                "observed_mean": obs_mean,
                "decoy_mean": decoy_mean,
                "observed_sd": float(np.sqrt(obs_var)),
                "decoy_sd": float(np.sqrt(decoy_var)),
                "smd": smd,
                "abs_smd": float(abs(smd)),
                "is_matching_variable": var in TARGET_DECOY_MATCHING_VARIABLES,
            })
    return pd.DataFrame(rows, columns=TARGET_DECOY_BALANCE_COLUMNS)


TARGET_DECOY_ACCEPTANCE_COLUMNS = [
    "null_type",
    "evaluable_set_type",
    "analysis_scope",
    "stage",
    "weighting",
    "variable",
    "n_observed_rows",
    "n_decoy_rows",
    "observed_mean",
    "decoy_mean",
    "smd",
    "abs_smd",
    "abs_smd_threshold",
    "is_within_threshold",
    "n_matching_variables",
    "max_abs_smd",
    "all_matching_variables_within_threshold",
]


def _as_bool_series(values: pd.Series) -> pd.Series:
    """Balance tables are also read back from TSV, where booleans arrive as strings."""
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def build_target_decoy_matching_acceptance(
    balance: pd.DataFrame,
    *,
    threshold: float = TARGET_DECOY_SMD_ACCEPTANCE_THRESHOLD,
) -> pd.DataFrame:
    """Primary matching acceptance rows: |SMD| < threshold on the 3 matching variables.

    Acceptance is judged only on ``primary_path_evaluable`` × ``pre_inference_selected``
    × ``triplet_weighted``.  Target-weighted rows, post-evaluability attrition rows,
    the all-profile-supported scope and the total-degree diagnostic stay in the balance
    table but never gate acceptance, so they are excluded here instead of being mixed
    into the same maximum.
    """
    required = {"analysis_scope", "stage", "weighting", "variable", "is_matching_variable", "smd", "abs_smd"}
    if balance.empty or not required <= set(balance.columns):
        return pd.DataFrame(columns=TARGET_DECOY_ACCEPTANCE_COLUMNS)
    df = balance.copy()
    selected = (
        df["analysis_scope"].astype(str).eq(TARGET_DECOY_ACCEPTANCE_ANALYSIS_SCOPE)
        & df["stage"].astype(str).eq(TARGET_DECOY_ACCEPTANCE_STAGE)
        & df["weighting"].astype(str).eq(TARGET_DECOY_ACCEPTANCE_WEIGHTING)
        & _as_bool_series(df["is_matching_variable"])
    )
    df = df.loc[selected].copy()
    if df.empty:
        return pd.DataFrame(columns=TARGET_DECOY_ACCEPTANCE_COLUMNS)
    if "evaluable_set_type" not in df.columns:
        df["evaluable_set_type"] = pd.NA
    df["null_type"] = TARGET_DECOY_NULL_TYPE
    df["smd"] = pd.to_numeric(df["smd"], errors="coerce")
    df["abs_smd"] = pd.to_numeric(df["abs_smd"], errors="coerce")
    df["abs_smd_threshold"] = float(threshold)
    # A missing SMD is not within the threshold: ``lt`` already yields False for NaN.
    df["is_within_threshold"] = df["abs_smd"].lt(float(threshold))
    grouped = df.groupby("evaluable_set_type", dropna=False)
    df["n_matching_variables"] = grouped["variable"].transform("nunique").astype(int)
    df["max_abs_smd"] = grouped["abs_smd"].transform("max")
    df["all_matching_variables_within_threshold"] = grouped["is_within_threshold"].transform("all") & df[
        "n_matching_variables"
    ].eq(len(TARGET_DECOY_MATCHING_VARIABLES))
    for col in TARGET_DECOY_ACCEPTANCE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    out = df[TARGET_DECOY_ACCEPTANCE_COLUMNS].copy()
    return out.sort_values(["evaluable_set_type", "variable"], kind="stable").reset_index(drop=True)


def target_decoy_acceptance_metadata(acceptance: pd.DataFrame) -> dict[str, Any]:
    """Compact acceptance record for run metadata and the evaluation summary."""
    base: dict[str, Any] = {
        "null_type": TARGET_DECOY_NULL_TYPE,
        "analysis_scope": TARGET_DECOY_ACCEPTANCE_ANALYSIS_SCOPE,
        "stage": TARGET_DECOY_ACCEPTANCE_STAGE,
        "weighting": TARGET_DECOY_ACCEPTANCE_WEIGHTING,
        "matching_variables": list(TARGET_DECOY_MATCHING_VARIABLES),
        "abs_smd_threshold": float(TARGET_DECOY_SMD_ACCEPTANCE_THRESHOLD),
    }
    if acceptance is None or acceptance.empty:
        return {**base, "evaluated": False, "by_evaluable_set_type": {}}
    df = acceptance.copy()
    df["abs_smd"] = pd.to_numeric(df["abs_smd"], errors="coerce")
    df["is_within_threshold"] = _as_bool_series(df["is_within_threshold"])
    by_set: dict[str, Any] = {}
    for eval_type, sub in df.groupby("evaluable_set_type", dropna=False):
        by_set[str(eval_type)] = {
            "max_abs_smd": float(sub["abs_smd"].max()),
            "n_matching_variables": int(sub["variable"].nunique()),
            "all_matching_variables_within_threshold": bool(
                sub["is_within_threshold"].all()
                and sub["variable"].nunique() == len(TARGET_DECOY_MATCHING_VARIABLES)
            ),
            "abs_smd_by_variable": {
                str(v): float(x) for v, x in zip(sub["variable"].astype(str), sub["abs_smd"])
            },
        }
    return {
        **base,
        "evaluated": True,
        "all_matching_variables_within_threshold": bool(
            all(rec["all_matching_variables_within_threshold"] for rec in by_set.values())
        ),
        "by_evaluable_set_type": by_set,
    }


TARGET_DECOY_EVALUABILITY_COLUMNS = [
    "original_triplet_id",
    "pseudo_triplet_id",
    "drug_id",
    "original_target_id",
    "decoy_target_id",
    "disease_id",
    "direction",
    "match_rank",
    "matching_distance",
    "evaluable_set_type",
    "is_decoy_evaluable",
    "n_raw_cell_pairs",
    "n_path_existing_cell_pairs",
    "n_drug_paths",
    "n_kd_paths",
    "joint_jaccard",
    "has_joint_score",
]


def build_target_decoy_evaluability(
    assignments: pd.DataFrame,
    candidate_joint_df: pd.DataFrame,
    *,
    evaluable_set_type: str,
) -> pd.DataFrame:
    """Per-assignment post-inference evaluability, keeping non-evaluable candidates.

    Non-evaluable candidates keep a missing score: they are never zero-filled and
    never dropped from the assignment set.
    """
    if assignments.empty:
        return pd.DataFrame(columns=TARGET_DECOY_EVALUABILITY_COLUMNS)
    keep = [
        c for c in [
            "original_triplet_id", "pseudo_triplet_id", "drug_id", "original_target_id",
            "decoy_target_id", "disease_id", "direction", "match_rank", "matching_distance",
        ] if c in assignments.columns
    ]
    out = assignments[keep].copy()
    out["evaluable_set_type"] = str(evaluable_set_type)
    metric_cols = ["n_raw_cell_pairs", "n_path_existing_cell_pairs", "n_drug_paths", "n_kd_paths", "joint_jaccard"]
    if candidate_joint_df is None or candidate_joint_df.empty:
        for col in metric_cols:
            out[col] = np.nan
    else:
        joint = candidate_joint_df.loc[
            candidate_joint_df.get("evaluable_set_type", pd.Series("", index=candidate_joint_df.index))
            .astype(str)
            .eq(str(evaluable_set_type))
        ].copy()
        joint["pseudo_triplet_id"] = joint["triplet_id"].astype(str)
        agg_cols = ["pseudo_triplet_id"] + [c for c in metric_cols if c in joint.columns]
        joint = joint[agg_cols].drop_duplicates("pseudo_triplet_id")
        out["pseudo_triplet_id"] = out["pseudo_triplet_id"].astype(str)
        out = out.merge(joint, on="pseudo_triplet_id", how="left")
        for col in metric_cols:
            if col not in out.columns:
                out[col] = np.nan
    out["has_joint_score"] = pd.to_numeric(out["joint_jaccard"], errors="coerce").notna()
    out["is_decoy_evaluable"] = (
        pd.to_numeric(out["n_path_existing_cell_pairs"], errors="coerce").fillna(0).gt(0)
        & out["has_joint_score"]
    )
    return out[TARGET_DECOY_EVALUABILITY_COLUMNS]


TARGET_DECOY_TRIPLET_EVALUABILITY_COLUMNS = [
    "original_triplet_id",
    "drug_id",
    "original_target_id",
    "disease_id",
    "evaluable_set_type",
    "n_selected_decoys",
    "n_evaluable_decoys",
    "has_evaluable_decoy",
]


def build_target_decoy_triplet_evaluability(evaluability: pd.DataFrame) -> pd.DataFrame:
    """Per observed triplet: selected decoys, evaluable decoys and |E_i| >= 1."""
    if evaluability.empty:
        return pd.DataFrame(columns=TARGET_DECOY_TRIPLET_EVALUABILITY_COLUMNS)
    df = evaluability.copy()
    df["is_decoy_evaluable"] = df["is_decoy_evaluable"].fillna(False).astype(bool)
    group_cols = [
        c for c in ["original_triplet_id", "drug_id", "original_target_id", "disease_id", "evaluable_set_type"]
        if c in df.columns
    ]
    out = df.groupby(group_cols, dropna=False).agg(
        n_selected_decoys=("pseudo_triplet_id", "nunique"),
        n_evaluable_decoys=("is_decoy_evaluable", "sum"),
    ).reset_index()
    out["n_evaluable_decoys"] = out["n_evaluable_decoys"].astype(int)
    out["has_evaluable_decoy"] = out["n_evaluable_decoys"].ge(1)
    return out[[c for c in TARGET_DECOY_TRIPLET_EVALUABILITY_COLUMNS if c in out.columns]]


TARGET_DECOY_EVALUABILITY_SUMMARY_COLUMNS = [
    "null_type",
    "evaluable_set_type",
    "n_selected_decoys_total",
    "n_evaluable_decoys_total",
    "candidate_level_evaluable_rate",
    "candidate_level_evaluable_rate_denominator",
    "n_profile_supported_triplets",
    "n_observed_path_evaluable_triplets",
    "n_triplets_with_evaluable_decoy",
    "triplet_level_coverage",
    "n_comparison_triplets",
    "n_evaluable_decoys_histogram",
    "n_unique_drugs",
    "n_unique_observed_targets",
    "n_unique_diseases",
    "n_unique_decoy_targets",
]


def summarize_target_decoy_evaluability(
    evaluability: pd.DataFrame,
    *,
    evaluable_set_type: str,
    observed_evaluable_triplet_ids: set[str],
    comparison_triplet_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Candidate-level and triplet-level evaluability report.

    The candidate-level rate denominator is all selected candidates, and the column
    name states that denominator explicitly.
    """
    if evaluability.empty:
        return pd.DataFrame(columns=TARGET_DECOY_EVALUABILITY_SUMMARY_COLUMNS)
    df = evaluability.loc[
        evaluability["evaluable_set_type"].astype(str).eq(str(evaluable_set_type))
    ].copy()
    if df.empty:
        return pd.DataFrame(columns=TARGET_DECOY_EVALUABILITY_SUMMARY_COLUMNS)
    df["original_triplet_id"] = df["original_triplet_id"].astype(str)
    df["is_decoy_evaluable"] = df["is_decoy_evaluable"].fillna(False).astype(bool)

    observed_ids = {str(t) for t in observed_evaluable_triplet_ids}
    n_evaluable_by_triplet = df.groupby("original_triplet_id")["is_decoy_evaluable"].sum().astype(int)
    n_selected = int(len(df))
    n_evaluable = int(df["is_decoy_evaluable"].sum())
    with_evaluable = set(n_evaluable_by_triplet.loc[n_evaluable_by_triplet.ge(1)].index.astype(str))
    hist = n_evaluable_by_triplet.value_counts().sort_index()
    evaluable_rows = df.loc[df["is_decoy_evaluable"]]
    row = {
        "null_type": TARGET_DECOY_NULL_TYPE,
        "evaluable_set_type": str(evaluable_set_type),
        "n_selected_decoys_total": n_selected,
        "n_evaluable_decoys_total": n_evaluable,
        "candidate_level_evaluable_rate": float(n_evaluable / n_selected) if n_selected else float("nan"),
        "candidate_level_evaluable_rate_denominator": "n_selected_decoys_total",
        "n_profile_supported_triplets": int(df["original_triplet_id"].nunique()),
        "n_observed_path_evaluable_triplets": int(len(observed_ids)),
        "n_triplets_with_evaluable_decoy": int(len(with_evaluable & observed_ids)) if observed_ids else int(len(with_evaluable)),
        "triplet_level_coverage": (
            float(len(with_evaluable & observed_ids) / len(observed_ids)) if observed_ids else float("nan")
        ),
        "n_comparison_triplets": int(len(comparison_triplet_ids)) if comparison_triplet_ids is not None else pd.NA,
        "n_evaluable_decoys_histogram": json_dumps_int_histogram(hist),
        "n_unique_drugs": int(df["drug_id"].astype(str).nunique()) if "drug_id" in df.columns else 0,
        "n_unique_observed_targets": int(df["original_target_id"].astype(str).nunique()),
        "n_unique_diseases": int(df["disease_id"].astype(str).nunique()) if "disease_id" in df.columns else 0,
        "n_unique_decoy_targets": int(evaluable_rows["decoy_target_id"].astype(str).nunique()),
    }
    return pd.DataFrame([row], columns=TARGET_DECOY_EVALUABILITY_SUMMARY_COLUMNS)


def json_dumps_int_histogram(counts: pd.Series) -> str:
    return json.dumps({str(int(k)): int(v) for k, v in counts.items()}, sort_keys=True)


TARGET_DECOY_DELTA_STRATUM_COLUMNS = [
    "null_type",
    "evaluable_set_type",
    "statistic",
    "n_evaluable_decoys",
    "n_triplets",
    "delta_mean",
    "delta_sd",
]


def summarize_target_decoy_delta_by_n_evaluable(
    triplet_delta: pd.DataFrame,
    *,
    evaluable_set_type: str,
) -> pd.DataFrame:
    """Diagnostic stratification of the triplet-wise delta by |E_i|."""
    required = {"n_decoy_anchors", "delta", "statistic"}
    if triplet_delta.empty or not required <= set(triplet_delta.columns):
        return pd.DataFrame(columns=TARGET_DECOY_DELTA_STRATUM_COLUMNS)
    df = triplet_delta.copy()
    df["n_decoy_anchors"] = pd.to_numeric(df["n_decoy_anchors"], errors="coerce")
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
    df = df.dropna(subset=["n_decoy_anchors", "delta"])
    if df.empty:
        return pd.DataFrame(columns=TARGET_DECOY_DELTA_STRATUM_COLUMNS)
    rows: list[dict[str, Any]] = []
    for (stat, k), sub in df.groupby(["statistic", "n_decoy_anchors"], dropna=False):
        rows.append({
            "null_type": TARGET_DECOY_NULL_TYPE,
            "evaluable_set_type": str(evaluable_set_type),
            "statistic": str(stat),
            "n_evaluable_decoys": int(k),
            "n_triplets": int(len(sub)),
            "delta_mean": float(sub["delta"].mean()),
            "delta_sd": float(sub["delta"].std(ddof=1)) if len(sub) > 1 else float("nan"),
        })
    return pd.DataFrame(rows, columns=TARGET_DECOY_DELTA_STRATUM_COLUMNS)


def target_decoy_model_metadata(
    *,
    n_candidate_universe: int,
    reach_cutoff: int,
    n_decoys_per_triplet: int,
    tie_break_seed: int,
    is_primary_spec: bool,
    decoy_selection: str = TARGET_DECOY_SELECTION_MATCHED,
    candidate_pool_type: str = TARGET_DECOY_CANDIDATE_POOL_TYPE,
) -> dict[str, Any]:
    """Model definition actually used by the run.

    ``matching_variables`` / ``rank_transform`` / ``distance`` describe the matching
    rule, so a random draw records them as ``not_applicable`` instead of implying a
    matching it never performed.
    """
    matched = str(decoy_selection) == TARGET_DECOY_SELECTION_MATCHED
    return {
        "null_type": TARGET_DECOY_NULL_TYPE,
        "decoy_selection": str(decoy_selection),
        "candidate_universe_type": str(candidate_pool_type),
        "n_candidate_universe": int(n_candidate_universe),
        "matching_variables": (
            list(TARGET_DECOY_MATCHING_VARIABLES) if matched else TARGET_DECOY_SELECTION_NOT_APPLICABLE
        ),
        "rank_transform": TARGET_DECOY_RANK_TRANSFORM if matched else TARGET_DECOY_SELECTION_NOT_APPLICABLE,
        "distance": TARGET_DECOY_DISTANCE if matched else TARGET_DECOY_SELECTION_NOT_APPLICABLE,
        "reach_cutoff": int(reach_cutoff),
        "n_decoys_per_triplet": int(n_decoys_per_triplet),
        "within_triplet_replacement": False,
        "across_triplet_reuse": True,
        "positive_exclusion_scope": TARGET_DECOY_POSITIVE_EXCLUSION_SCOPE,
        "path_evaluability_prefilter": False,
        "post_inference_backfill": False,
        "non_evaluable_decoy_score": "missing",
        "tie_break_seed": int(tie_break_seed),
        "graph_type": TARGET_DECOY_GRAPH_TYPE,
        "edge_count_rule": TARGET_DECOY_EDGE_COUNT_RULE,
        "is_primary_spec": bool(is_primary_spec),
    }
