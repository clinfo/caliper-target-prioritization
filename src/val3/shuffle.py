from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .io import ensure_dir, write_json, write_parquet


@dataclass(frozen=True)
class TargetKhopPermutationContext:
    genes: list[str]
    groups: list[str]
    local_idx_by_triplet: dict[str, np.ndarray]
    diagnostics: pd.DataFrame


def _genes_from_cols(cols: Sequence[str], suffix: str) -> set[str]:
    return {str(c[: -len(suffix)]) for c in cols if str(c).endswith(suffix)}


def common_shuffle_genes(drug_df: pd.DataFrame, pert_df: pd.DataFrame, kg_nodes: set[str] | None = None) -> list[str]:
    drug_gene = _genes_from_cols(drug_df.columns, "_gene")
    pert_gene = _genes_from_cols(pert_df.columns, "_gene")
    drug_dis = _genes_from_cols(drug_df.columns, "_disease")
    pert_dis = _genes_from_cols(pert_df.columns, "_disease")
    genes = drug_gene & pert_gene & drug_dis & pert_dis
    if kg_nodes is not None:
        genes &= set(map(str, kg_nodes))
    return sorted(genes)


def _permutation_groups(
    drug_df: pd.DataFrame,
    pert_df: pd.DataFrame,
    *,
    triplet_col: str,
    shuffle_per_triplet: bool,
) -> list[str]:
    if shuffle_per_triplet:
        drug_triplets = set(drug_df[triplet_col].astype(str))
        pert_triplets = set(pert_df[triplet_col].astype(str))
        return sorted(drug_triplets.intersection(pert_triplets))
    return ["__global__"]


def make_global_paired_gene_label_permutation_specs(
    drug_df: pd.DataFrame,
    pert_df: pd.DataFrame,
    *,
    rng: np.random.Generator,
    kg_nodes: set[str] | None = None,
    triplet_col: str = "triplet_id",
    shuffle_per_triplet: bool = True,
) -> dict[str, Any]:
    genes = common_shuffle_genes(drug_df, pert_df, kg_nodes=kg_nodes)
    if not genes:
        raise ValueError("No common drug/pert/disease/KG genes found for global paired permutation.")

    permutation_specs: dict[str, Any] = {"gene_order": genes, "per_triplet": {}}
    for group in _permutation_groups(
        drug_df,
        pert_df,
        triplet_col=triplet_col,
        shuffle_per_triplet=shuffle_per_triplet,
    ):
        perm = rng.permutation(len(genes))
        permutation_specs["per_triplet"][str(group)] = perm.astype(int).tolist()
    return permutation_specs


def _row_indices_by_group(df: pd.DataFrame, *, groups: list[str], triplet_col: str, shuffle_per_triplet: bool) -> dict[str, np.ndarray]:
    if not shuffle_per_triplet:
        return {"__global__": np.arange(len(df), dtype=np.int64)}
    triplets = df[triplet_col].astype(str).to_numpy()
    return {group: np.flatnonzero(triplets == str(group)) for group in groups}


def _apply_column_permutations_from_specs(
    df: pd.DataFrame,
    *,
    cols: list[str],
    row_indices: dict[str, np.ndarray],
    permutation_specs: dict[str, Any],
) -> None:
    if not cols:
        return
    block = df.loc[:, cols].to_numpy(copy=True)
    for group, perm_list in permutation_specs["per_triplet"].items():
        rows = row_indices.get(str(group))
        if rows is None or rows.size == 0:
            continue
        perm = np.asarray(perm_list, dtype=np.intp)
        block[rows, :] = block[np.ix_(rows, perm)]
    df.loc[:, cols] = block


def apply_global_paired_gene_label_permutation(
    drug_df: pd.DataFrame,
    pert_df: pd.DataFrame,
    *,
    rng: np.random.Generator,
    kg_nodes: set[str] | None = None,
    triplet_col: str = "triplet_id",
    shuffle_per_triplet: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply the same gene-label permutation to drug/pert gene and disease blocks.

    Columns stay named by the original KG gene. Values are copied from permuted genes,
    implementing s_null(g_j) = s(tau(g_j)) while preserving drug-target-disease profile
    tuples under the same tau.
    """
    out_drug = drug_df.copy()
    out_pert = pert_df.copy()
    permutation_specs = make_global_paired_gene_label_permutation_specs(
        drug_df,
        pert_df,
        rng=rng,
        kg_nodes=kg_nodes,
        triplet_col=triplet_col,
        shuffle_per_triplet=shuffle_per_triplet,
    )

    genes = list(permutation_specs["gene_order"])
    gene_cols = [f"{g}_gene" for g in genes]
    disease_cols = [f"{g}_disease" for g in genes]
    groups = list(permutation_specs["per_triplet"])
    drug_rows = _row_indices_by_group(out_drug, groups=groups, triplet_col=triplet_col, shuffle_per_triplet=shuffle_per_triplet)
    pert_rows = _row_indices_by_group(out_pert, groups=groups, triplet_col=triplet_col, shuffle_per_triplet=shuffle_per_triplet)

    for cols in (gene_cols, disease_cols):
        _apply_column_permutations_from_specs(out_drug, cols=cols, row_indices=drug_rows, permutation_specs=permutation_specs)
        _apply_column_permutations_from_specs(out_pert, cols=cols, row_indices=pert_rows, permutation_specs=permutation_specs)

    return out_drug, out_pert, permutation_specs


def save_baseline_inputs(
    *,
    drug_df: pd.DataFrame,
    pert_df: pd.DataFrame,
    out_dir: str | Path,
    iteration: int,
    permutation_specs: dict[str, Any],
) -> dict[str, str]:
    iter_dir = ensure_dir(Path(out_dir) / f"iter={iteration}")
    drug_path = iter_dir / "drug.parquet"
    pert_path = iter_dir / "pert.parquet"
    spec_path = iter_dir / "permutation_specs.json"
    write_parquet(drug_df, drug_path)
    write_parquet(pert_df, pert_path)
    write_json({"iteration": int(iteration), "permutation_specs": permutation_specs}, spec_path)
    return {"baseline_drug_input": str(drug_path), "baseline_pert_input": str(pert_path), "permutation_specs": str(spec_path)}


def save_baseline_permutation_specs(
    *,
    out_dir: str | Path,
    iteration: int,
    permutation_specs: dict[str, Any],
) -> dict[str, str]:
    iter_dir = ensure_dir(Path(out_dir) / f"iter={iteration}")
    spec_path = iter_dir / "permutation_specs.json"
    write_json({"iteration": int(iteration), "permutation_specs": permutation_specs}, spec_path)
    return {"permutation_specs": str(spec_path)}


def _load_directed_kg_edges(kg_tsv: str | Path) -> pd.DataFrame:
    kg = pd.read_csv(kg_tsv, sep="\t")
    required = {"source", "target"}
    missing = required - set(kg.columns)
    if missing:
        raise ValueError(f"KG file missing columns for local permutation: {sorted(missing)}")
    out = kg[["source", "target"]].copy()
    out["source"] = out["source"].astype(str)
    out["target"] = out["target"].astype(str)
    return out


def reachable_nodes_from_targets(
    kg_tsv: str | Path,
    *,
    targets: Sequence[str],
    cutoff: int,
) -> dict[str, set[str]]:
    """Return directed target-centered k-hop reachable nodes for Null 1."""
    kg = _load_directed_kg_edges(kg_tsv)
    adj: dict[str, set[str]] = {}
    for u, v in kg[["source", "target"]].itertuples(index=False, name=None):
        adj.setdefault(str(u), set()).add(str(v))
    out: dict[str, set[str]] = {}
    for target in sorted({str(t) for t in targets}):
        seen = {target}
        frontier = {target}
        for _ in range(int(cutoff)):
            nxt: set[str] = set()
            for node in frontier:
                nxt.update(adj.get(node, set()))
            nxt -= seen
            if not nxt:
                break
            seen.update(nxt)
            frontier = nxt
        seen.discard(target)
        out[target] = seen
    return out


def _nonzero_count_for_triplet(df: pd.DataFrame, *, triplet_id: str, cols: list[str]) -> int:
    if not cols or "triplet_id" not in df.columns:
        return 0
    sub = df.loc[df["triplet_id"].astype(str).eq(str(triplet_id)), cols]
    if sub.empty:
        return 0
    arr = sub.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    return int((arr != 0).any(axis=0).sum())


def build_target_khop_restricted_paired_gene_label_permutation_context(
    drug_df: pd.DataFrame,
    pert_df: pd.DataFrame,
    *,
    kg_tsv: str | Path,
    cutoff: int,
    kg_nodes: set[str] | None = None,
    triplet_col: str = "triplet_id",
    target_col: str = "target_id",
) -> TargetKhopPermutationContext:
    """Build iteration-invariant Null 1 local permutation context."""
    genes = common_shuffle_genes(drug_df, pert_df, kg_nodes=kg_nodes)
    if not genes:
        raise ValueError("No common drug/pert/disease/KG genes found for target k-hop local permutation.")
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    drug_triplets = set(drug_df[triplet_col].astype(str)) if triplet_col in drug_df.columns else set()
    pert_triplets = set(pert_df[triplet_col].astype(str)) if triplet_col in pert_df.columns else set()
    groups = sorted(drug_triplets & pert_triplets)
    if not groups:
        raise ValueError("No common triplet_id values found for target k-hop local permutation.")

    frames = []
    if {triplet_col, target_col} <= set(drug_df.columns):
        frames.append(drug_df[[triplet_col, target_col]].drop_duplicates())
    if {triplet_col, target_col} <= set(pert_df.columns):
        frames.append(pert_df[[triplet_col, target_col]].drop_duplicates())
    target_map_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=[triplet_col]) if frames else pd.DataFrame(columns=[triplet_col, target_col])
    target_map_df[triplet_col] = target_map_df[triplet_col].astype(str)
    target_map_df[target_col] = target_map_df[target_col].astype(str)
    target_by_triplet = target_map_df.set_index(triplet_col)[target_col].to_dict()
    reachable = reachable_nodes_from_targets(kg_tsv, targets=[target_by_triplet.get(g, "") for g in groups], cutoff=int(cutoff))

    local_idx_by_triplet: dict[str, np.ndarray] = {}
    diag_rows: list[dict[str, Any]] = []
    for triplet_id in groups:
        target_id = str(target_by_triplet.get(triplet_id, ""))
        local_genes = sorted((reachable.get(target_id, set()) & set(genes)))
        local_idx = np.asarray([gene_to_idx[g] for g in local_genes], dtype=np.intp)
        local_idx_by_triplet[str(triplet_id)] = local_idx

        gene_cols = [f"{g}_gene" for g in local_genes]
        disease_cols = [f"{g}_disease" for g in local_genes]
        meta = {}
        for c in ["drug_id", "disease_id", "direction"]:
            if c in drug_df.columns:
                vals = drug_df.loc[drug_df[triplet_col].astype(str).eq(str(triplet_id)), c]
                if not vals.empty:
                    meta[c] = vals.iloc[0]
        diag_rows.append({
            "triplet_id": str(triplet_id),
            "drug_id": str(meta.get("drug_id", "")),
            "target_id": target_id,
            "disease_id": str(meta.get("disease_id", "")),
            "direction": str(meta.get("direction", "")),
            "n_local_bg_genes": int(len(local_genes)),
            "n_local_bg_nonzero_drug": _nonzero_count_for_triplet(drug_df, triplet_id=str(triplet_id), cols=gene_cols),
            "n_local_bg_nonzero_pert": _nonzero_count_for_triplet(pert_df, triplet_id=str(triplet_id), cols=gene_cols),
            "n_local_bg_nonzero_disease": max(
                _nonzero_count_for_triplet(drug_df, triplet_id=str(triplet_id), cols=disease_cols),
                _nonzero_count_for_triplet(pert_df, triplet_id=str(triplet_id), cols=disease_cols),
            ),
            "n_unique_permutable_genes": int(local_idx.size),
        })
    return TargetKhopPermutationContext(
        genes=genes,
        groups=groups,
        local_idx_by_triplet=local_idx_by_triplet,
        diagnostics=pd.DataFrame(diag_rows),
    )


def make_target_khop_restricted_paired_gene_label_permutation_specs_from_context(
    context: TargetKhopPermutationContext,
    *,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], pd.DataFrame]:
    permutation_specs: dict[str, Any] = {"gene_order": context.genes, "per_triplet": {}}
    identity = np.arange(len(context.genes), dtype=np.intp)
    for triplet_id in context.groups:
        local_idx = context.local_idx_by_triplet[str(triplet_id)]
        perm = identity.copy()
        if local_idx.size >= 2:
            perm[local_idx] = rng.permutation(local_idx)
        permutation_specs["per_triplet"][str(triplet_id)] = perm.astype(int).tolist()
    return permutation_specs, context.diagnostics.copy()


def make_target_khop_restricted_paired_gene_label_permutation_specs(
    drug_df: pd.DataFrame,
    pert_df: pd.DataFrame,
    *,
    kg_tsv: str | Path,
    cutoff: int,
    rng: np.random.Generator,
    kg_nodes: set[str] | None = None,
    triplet_col: str = "triplet_id",
    target_col: str = "target_id",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build Null 1 local paired gene-label permutation specs."""
    context = build_target_khop_restricted_paired_gene_label_permutation_context(
        drug_df,
        pert_df,
        kg_tsv=kg_tsv,
        cutoff=cutoff,
        kg_nodes=kg_nodes,
        triplet_col=triplet_col,
        target_col=target_col,
    )
    return make_target_khop_restricted_paired_gene_label_permutation_specs_from_context(context, rng=rng)


def save_null1_permutation_specs(
    *,
    out_dir: str | Path,
    iteration: int,
    permutation_specs: dict[str, Any],
) -> dict[str, str]:
    return save_baseline_permutation_specs(
        out_dir=out_dir,
        iteration=iteration,
        permutation_specs=permutation_specs,
    )
