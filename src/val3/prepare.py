from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .io import read_parquet_any, read_table, write_json, write_parquet, ensure_dir
from .registry import Val3RegistrySpec

_DRUG_ID_RE = re.compile(r"^PUBCHEM\.COMPOUND:(\d+)$")
_PUBCHEM_RE = re.compile(r"^PUBCHEM:(\d+)$")
_CID_RE = re.compile(r"^(\d+)$")


def normalize_pubchem_compound_id(value: Any) -> str | None:
    if value is None:
        return None
    v = str(value).strip()
    if not v or v.lower() == "nan":
        return None
    for rx in (_DRUG_ID_RE, _PUBCHEM_RE, _CID_RE):
        m = rx.fullmatch(v)
        if m:
            return f"PUBCHEM.COMPOUND:{m.group(1)}"
    return v


def load_drug_id_mapper(path: str | Path | None, *, source_col: str | None, target_col: str | None) -> dict[str, str]:
    if path is None:
        return {}
    df = read_table(path, sep="\t")
    if not source_col or not target_col:
        raise ValueError("Drug ID mapping requires source_col and target_col.")
    missing = {source_col, target_col} - set(df.columns)
    if missing:
        raise ValueError(f"Drug ID mapping missing columns: {sorted(missing)}")
    return {
        str(src): normalize_pubchem_compound_id(dst)
        for src, dst in zip(df[source_col], df[target_col])
        if pd.notna(src) and pd.notna(dst)
    }


def _normalize_ncbigene_id(value: Any) -> str:
    v = str(value).strip()
    if not v or v.lower() == "nan":
        return v
    if v.startswith("NCBIGENE:"):
        return v
    if v.isdigit():
        return f"NCBIGENE:{v}"
    return v


def _ternarize_sign_frame(df: pd.DataFrame, columns: Iterable[str], *, threshold: float = 1.0) -> pd.DataFrame:
    out = df.copy()
    thr = float(threshold)
    if thr < 0:
        raise ValueError("sign threshold must be >= 0.")
    for c in columns:
        x = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        out[c] = np.where(x >= thr, 1, np.where(x <= -thr, -1, 0)).astype(np.int8)
    return out


def _binarize_disease_sign_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        x = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        out[c] = np.where(x > 0, 1, np.where(x < 0, -1, 0)).astype(np.int8)
    return out


def _non_meta_cols(df: pd.DataFrame, meta_cols: set[str]) -> list[str]:
    return [c for c in df.columns if c not in meta_cols]


def _rename_profile_gene_cols(
    df: pd.DataFrame,
    *,
    suffix: str,
    meta_cols: set[str],
    threshold: float = 1.0,
    normalize_gene_ids: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    rename: dict[Any, str] = {}
    for c in _non_meta_cols(out, meta_cols):
        c_str = str(c)
        if c_str.endswith(suffix):
            base = c_str[: -len(suffix)]
            rename[c] = f"{_normalize_ncbigene_id(base) if normalize_gene_ids else base}{suffix}"
        elif c_str.endswith("_gene") or c_str.endswith("_disease"):
            # Do not accidentally convert a disease column to a gene column.
            continue
        else:
            base = _normalize_ncbigene_id(c_str) if normalize_gene_ids else c_str
            rename[c] = f"{base}{suffix}"
    out = out.rename(columns=rename)
    sign_cols = [c for c in out.columns if c.endswith(suffix)]
    if suffix == "_disease":
        return _binarize_disease_sign_frame(out, sign_cols)
    return _ternarize_sign_frame(out, sign_cols, threshold=threshold)


def load_triplets(path: str | Path, *, direction: str) -> pd.DataFrame:
    df = read_table(path, sep="\t")
    required = {"triplet_id", "drug_id", "target_id", "disease_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Triplet table missing columns: {sorted(missing)}")
    out = df.copy()
    out["drug_id"] = out["drug_id"].map(normalize_pubchem_compound_id)
    out["target_id"] = out["target_id"].map(_normalize_ncbigene_id)
    out["disease_id"] = out["disease_id"].astype(str)
    out["direction"] = direction
    return out


def load_drug_profiles(spec: Val3RegistrySpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = read_parquet_any(spec.drug_profile_parquet)
    required = {spec.drug_id_col, spec.drug_cell_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Drug profile missing columns: {sorted(missing)}")
    mapper = load_drug_id_mapper(
        spec.drug_id_mapping_tsv,
        source_col=spec.drug_id_mapping_source_col,
        target_col=spec.drug_id_mapping_target_col,
    )
    out = df.copy()
    if mapper:
        out["drug_id"] = out[spec.drug_id_col].astype(str).map(mapper)
    else:
        out["drug_id"] = out[spec.drug_id_col].map(normalize_pubchem_compound_id)
    out = out[out["drug_id"].notna()].copy()
    if spec.drug_cell_col != "cell_id":
        out = out.rename(columns={spec.drug_cell_col: "cell_id"})
    meta = {spec.drug_id_col, spec.drug_cell_col, "drug_id", "cell_id"}
    out = _rename_profile_gene_cols(out, suffix="_gene", meta_cols=meta)
    gene_cols = sorted(c for c in out.columns if c.endswith("_gene"))
    out = out[["drug_id", "cell_id"] + gene_cols].reset_index(drop=True)
    summary = {"n_rows": int(len(out)), "n_drugs": int(out["drug_id"].nunique()), "n_cells": int(out["cell_id"].nunique()), "n_gene_cols": int(len(gene_cols))}
    return out, summary


def load_disease_profiles(spec: Val3RegistrySpec) -> pd.DataFrame | None:
    if spec.disease_profile_parquet is None:
        return None
    if not spec.disease_profile_parquet.exists():
        return None
    if not spec.disease_col:
        raise ValueError("Disease registry entry must define disease_col when disease parquet is used.")
    df = read_parquet_any(spec.disease_profile_parquet)
    if spec.disease_col not in df.columns:
        raise ValueError(f"Disease profile missing disease_col: {spec.disease_col}")
    out = df.copy()
    if spec.disease_col != "disease_id":
        out = out.rename(columns={spec.disease_col: "disease_id"})
    out["disease_id"] = out["disease_id"].astype(str)
    out = _rename_profile_gene_cols(out, suffix="_disease", meta_cols={"disease_id"})
    disease_cols = sorted(c for c in out.columns if c.endswith("_disease"))
    return out[["disease_id"] + disease_cols].drop_duplicates("disease_id").reset_index(drop=True)


def load_target_profiles(spec: Val3RegistrySpec) -> pd.DataFrame:
    df = read_parquet_any(spec.target_profile_parquet)

    if spec.target_format == "caliper_feature_matrix":
        required = {spec.target_pert_col, spec.target_disease_col, spec.target_cell_col, spec.target_label_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Target profile missing columns: {sorted(missing)}")
        rename = {
            spec.target_pert_col: "pert_id",
            spec.target_disease_col: "disease_id",
            spec.target_cell_col: "cell_id",
            spec.target_label_col: "label",
        }
        out = df.rename(columns=rename).copy()
        out["pert_id"] = out["pert_id"].map(_normalize_ncbigene_id)
        out["disease_id"] = out["disease_id"].astype(str)
        out["cell_id"] = out["cell_id"].astype(str)
        # Normalize sign columns but preserve existing suffixes.
        gene_cols = [c for c in out.columns if str(c).endswith(spec.target_gene_suffix)]
        if spec.target_gene_suffix != "_gene":
            out = out.rename(columns={c: str(c)[: -len(spec.target_gene_suffix)] + "_gene" for c in gene_cols})
        disease_cols = [c for c in out.columns if str(c).endswith(spec.target_disease_suffix)]
        if spec.target_disease_suffix != "_disease":
            out = out.rename(columns={c: str(c)[: -len(spec.target_disease_suffix)] + "_disease" for c in disease_cols})
        gene_cols = [c for c in out.columns if str(c).endswith("_gene")]
        disease_cols = [c for c in out.columns if str(c).endswith("_disease")]
        out = _ternarize_sign_frame(out, gene_cols, threshold=spec.target_sign_threshold)
        out = _binarize_disease_sign_frame(out, disease_cols)
        meta_cols = ["pert_id", "disease_id", "cell_id", "label"]
        return out[meta_cols + sorted(c for c in out.columns if str(c).endswith("_gene") or str(c).endswith("_disease"))].reset_index(drop=True)

    if spec.target_format == "l1000_bing_signature":
        required = {spec.target_pert_col, spec.target_cell_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"L1000 target profile missing columns: {sorted(missing)}")
        out = df.rename(columns={spec.target_pert_col: "pert_id", spec.target_cell_col: "cell_id"}).copy()
        out = out[out["pert_id"].notna() & out["cell_id"].notna()].copy()
        out["pert_id"] = out["pert_id"].map(_normalize_ncbigene_id)
        out["cell_id"] = out["cell_id"].astype(str)
        if out.duplicated(["pert_id", "cell_id"]).any():
            n_dup = int(out.duplicated(["pert_id", "cell_id"]).sum())
            raise ValueError(
                "L1000 target profile must be unique by (pert_id, cell_id). "
                f"Found {n_dup} duplicate rows; re-run preprocess notebook 02."
            )
        out = _rename_profile_gene_cols(
            out,
            suffix="_gene",
            meta_cols={"pert_id", "cell_id"},
            threshold=spec.target_sign_threshold,
            normalize_gene_ids=True,
        )
        gene_cols = sorted(c for c in out.columns if str(c).endswith("_gene"))
        return out[["pert_id", "cell_id"] + gene_cols].reset_index(drop=True)

    raise ValueError(f"Unsupported target profile format: {spec.target_format}")


def _merge_disease_for_drug(merged: pd.DataFrame, disease_df: pd.DataFrame | None) -> pd.DataFrame:
    if disease_df is None:
        raise ValueError("A separate disease profile is required to prepare drug-side infer input.")
    return merged.merge(disease_df, on="disease_id", how="inner")


def derive_disease_profiles_from_target(target_df: pd.DataFrame) -> pd.DataFrame | None:
    disease_cols = sorted(c for c in target_df.columns if c.endswith("_disease"))
    if not disease_cols or "disease_id" not in target_df.columns:
        return None
    return target_df[["disease_id"] + disease_cols].drop_duplicates("disease_id").reset_index(drop=True)


def build_drug_input(triplets: pd.DataFrame, drug_df: pd.DataFrame, disease_df: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = triplets.merge(drug_df, on="drug_id", how="inner")
    after_drug = len(merged)
    merged = _merge_disease_for_drug(merged, disease_df)
    merged = merged.copy()
    merged.insert(0, "row_id", np.arange(len(merged), dtype=np.int64))
    merged["input_side"] = "drug"
    # Critical revised-val3 requirement: target-centered skeleton.
    merged["pert_id"] = merged["target_id"]
    merged["do_id"] = merged["disease_id"]
    merged["label"] = 1
    meta_cols = ["row_id", "triplet_id", "drug_id", "target_id", "direction", "input_side", "pert_id", "cell_id", "do_id", "label"]
    gene_cols = sorted(c for c in merged.columns if c.endswith("_gene"))
    disease_cols = sorted(c for c in merged.columns if c.endswith("_disease"))
    out = merged[meta_cols + gene_cols + disease_cols].reset_index(drop=True)
    summary = summarize_prepared_dataset(out)
    summary.update({"join_rows_after_drug": int(after_drug), "join_rows_after_disease": int(len(out))})
    return out, summary


def build_pert_input(triplets: pd.DataFrame, target_df: pd.DataFrame, disease_df: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "disease_id" in target_df.columns:
        merged = triplets.merge(
            target_df,
            left_on=["target_id", "disease_id"],
            right_on=["pert_id", "disease_id"],
            how="inner",
        )
        join_rows_after_target = len(merged)
        if disease_df is not None:
            # Use the common disease source for both drug and pert sides.
            drop_cols = [c for c in merged.columns if str(c).endswith("_disease")]
            merged = merged.drop(columns=drop_cols).merge(disease_df, on="disease_id", how="inner")
    else:
        if disease_df is None:
            raise ValueError("L1000 target profile requires a separate disease profile.")
        merged = triplets.merge(
            target_df,
            left_on="target_id",
            right_on="pert_id",
            how="inner",
        )
        join_rows_after_target = len(merged)
        merged = merged.merge(disease_df, on="disease_id", how="inner")
        merged["label"] = 1
    merged = merged.copy()
    merged.insert(0, "row_id", np.arange(len(merged), dtype=np.int64))
    merged["input_side"] = "perturbation"
    merged["pert_id"] = merged["target_id"]
    merged["do_id"] = merged["disease_id"]
    if "label" not in merged.columns:
        merged["label"] = 1
    meta_cols = ["row_id", "triplet_id", "drug_id", "target_id", "direction", "input_side", "pert_id", "cell_id", "do_id", "label"]
    gene_cols = sorted(c for c in merged.columns if str(c).endswith("_gene"))
    disease_cols = sorted(c for c in merged.columns if str(c).endswith("_disease"))
    out = merged[meta_cols + gene_cols + disease_cols].reset_index(drop=True)
    summary = summarize_prepared_dataset(out)
    summary.update({"join_rows_after_target": int(join_rows_after_target), "join_rows_after_disease": int(len(out))})
    return out, summary


def summarize_prepared_dataset(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_rows": int(len(df)),
        "n_triplets": int(df["triplet_id"].nunique()) if "triplet_id" in df else 0,
        "n_drugs": int(df["drug_id"].nunique()) if "drug_id" in df else 0,
        "n_targets": int(df["target_id"].nunique()) if "target_id" in df else 0,
        "n_diseases": int(df["do_id"].nunique()) if "do_id" in df else 0,
        "n_cells": int(df["cell_id"].nunique()) if "cell_id" in df else 0,
        "n_gene_cols": int(sum(c.endswith("_gene") for c in df.columns)),
        "n_disease_cols": int(sum(c.endswith("_disease") for c in df.columns)),
    }


def build_profile_supported_observed_triplets(
    triplets: pd.DataFrame,
    drug_df: pd.DataFrame,
    target_df: pd.DataFrame,
    disease_df: pd.DataFrame | None,
    *,
    kg_nodes: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Restrict observed inference universe before path exploration.

    The restriction uses only profile/label/KG-node support.  It deliberately does
    not condition on whether a path exists; path-based evaluability is decided
    after inference by the existing evaluable-set logic.
    """
    out = triplets.copy()
    if "triplet_id" not in out.columns:
        raise ValueError("Triplet table must contain triplet_id.")
    out["triplet_id"] = out["triplet_id"].astype(str)
    out["drug_id"] = out["drug_id"].astype(str)
    out["target_id"] = out["target_id"].astype(str)
    out["disease_id"] = out["disease_id"].astype(str)

    drug_ids = set(drug_df["drug_id"].dropna().astype(str)) if "drug_id" in drug_df.columns else set()
    mask_drug = out["drug_id"].isin(drug_ids)

    if "disease_id" in target_df.columns:
        target_pairs = set(zip(target_df["pert_id"].astype(str), target_df["disease_id"].astype(str)))
        mask_target = pd.Series(
            [(t, d) in target_pairs for t, d in zip(out["target_id"], out["disease_id"])],
            index=out.index,
        )
    else:
        target_ids = set(target_df["pert_id"].dropna().astype(str))
        mask_target = out["target_id"].isin(target_ids)

    if disease_df is not None and "disease_id" in disease_df.columns:
        disease_ids = set(disease_df["disease_id"].dropna().astype(str))
        mask_disease = out["disease_id"].isin(disease_ids)
    elif "disease_id" in target_df.columns:
        disease_ids = set(target_df["disease_id"].dropna().astype(str))
        mask_disease = out["disease_id"].isin(disease_ids)
    else:
        mask_disease = pd.Series(True, index=out.index)

    if "label" in out.columns:
        mask_label = out["label"].notna()
    else:
        mask_label = pd.Series(True, index=out.index)

    if kg_nodes is not None:
        kg = set(map(str, kg_nodes))
        mask_kg = out["target_id"].isin(kg)
    else:
        mask_kg = pd.Series(True, index=out.index)

    keep = mask_drug & mask_target & mask_disease & mask_label & mask_kg
    supported = out.loc[keep].drop_duplicates("triplet_id").reset_index(drop=True)
    summary = {
        "definition": "drug_profile_and_target_profile_and_disease_profile_and_triplet_label_and_target_kg_node",
        "n_triplets_raw": int(len(out.drop_duplicates("triplet_id"))),
        "n_profile_supported_observed_triplets": int(len(supported)),
        "n_excluded_no_drug_profile": int((~mask_drug).sum()),
        "n_excluded_no_target_profile": int((~mask_target).sum()),
        "n_excluded_no_disease_profile": int((~mask_disease).sum()),
        "n_excluded_no_triplet_label": int((~mask_label).sum()),
        "n_excluded_target_not_kg_node": int((~mask_kg).sum()),
        "path_existence_filter_applied": False,
    }
    return supported, summary


def prepare_observed_inputs(
    *,
    spec: Val3RegistrySpec,
    direction: str,
    out_dir: str | Path,
    kg_nodes: set[str] | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    triplets = load_triplets(spec.triplet_tsv, direction=direction)
    drug_df, drug_profile_summary = load_drug_profiles(spec)
    target_df = load_target_profiles(spec)
    disease_df = load_disease_profiles(spec)
    if disease_df is None:
        disease_df = derive_disease_profiles_from_target(target_df)

    profile_supported_triplets, universe_summary = build_profile_supported_observed_triplets(
        triplets,
        drug_df,
        target_df,
        disease_df,
        kg_nodes=kg_nodes,
    )
    if profile_supported_triplets.empty:
        raise ValueError("No profile-supported observed triplets remain before path exploration.")

    drug_input, drug_summary = build_drug_input(profile_supported_triplets, drug_df, disease_df)
    pert_input, pert_summary = build_pert_input(profile_supported_triplets, target_df, disease_df)

    drug_path = Path(out_dir) / "drug.parquet"
    pert_path = Path(out_dir) / "pert.parquet"
    universe_path = Path(out_dir) / "profile_supported_observed_triplets.tsv"
    write_parquet(drug_input, drug_path)
    write_parquet(pert_input, pert_path)
    profile_supported_triplets.to_csv(universe_path, sep="	", index=False)

    common_triplets = sorted(set(drug_input["triplet_id"]).intersection(set(pert_input["triplet_id"])))
    summary = {
        "direction": direction,
        "observed_inference_universe": universe_summary,
        "profile_supported_observed_triplets": str(universe_path),
        "drug_profile_summary": drug_profile_summary,
        "drug_summary": drug_summary,
        "pert_summary": pert_summary,
        "n_common_triplets": int(len(common_triplets)),
        "drug_path": str(drug_path),
        "pert_path": str(pert_path),
    }
    write_json(summary, Path(out_dir) / "input_summary.json")
    return {
        "drug_path": drug_path,
        "pert_path": pert_path,
        "drug_df": drug_input,
        "pert_df": pert_input,
        "triplets": profile_supported_triplets,
        "profile_supported_observed_triplets": profile_supported_triplets,
        "profile_supported_observed_triplets_path": universe_path,
        "all_triplets": triplets,
        "drug_profile_df": drug_df,
        "target_df": target_df,
        "disease_df": disease_df,
        "summary": summary,
    }
