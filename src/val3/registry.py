from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import read_yaml_or_json, resolve_path


@dataclass(frozen=True)
class Val3RegistrySpec:
    kg_path: Path
    kg_sep: str
    kg_source_col: str
    kg_target_col: str
    kg_sign_col: str
    triplet_tsv: Path
    drug_profile_parquet: Path
    drug_id_col: str
    drug_cell_col: str
    drug_id_mapping_tsv: Path | None
    drug_id_mapping_source_col: str | None
    drug_id_mapping_target_col: str | None
    target_profile_key: str
    target_profile_parquet: Path
    target_format: str
    target_pert_col: str
    target_disease_col: str
    target_cell_col: str
    target_label_col: str
    target_gene_suffix: str
    target_disease_suffix: str
    target_sign_threshold: float
    disease_profile_parquet: Path | None
    disease_col: str | None


def load_dataset_registry(path: str | Path) -> dict[str, Any]:
    return read_yaml_or_json(path)


def _lookup(mapping: dict[str, Any], *keys: str) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError("Missing registry key: " + ".".join(keys))
        cur = cur[key]
    return cur


def _maybe_path(base: Path, value: Any) -> Path | None:
    if value in (None, "", "null"):
        return None
    return resolve_path(base, value)


def _target_profile_key_for_direction(
    *,
    requested_key: str,
    target_profiles: dict[str, Any],
    direction: str,
) -> str:
    if requested_key in target_profiles:
        return requested_key

    suffix_by_direction = {
        "inhibitory": "knockdown",
        "activatory": "overexpression",
    }
    suffix = suffix_by_direction.get(str(direction))
    if suffix is None:
        raise ValueError(f"Unsupported direction for target_profile auto-selection: {direction}")

    candidate = f"{requested_key}_{suffix}"
    if candidate in target_profiles:
        return candidate

    raise KeyError(
        "Missing registry key for data.target_profile auto-selection: "
        f"profiles.target.{candidate}. "
        f"Either define it, or set data.target_profile to one of {sorted(target_profiles)}."
    )


def resolve_val3_registry_spec(
    data_cfg: dict[str, Any],
    registry: dict[str, Any],
    registry_base_dir: str | Path,
    *,
    direction: str,
) -> Val3RegistrySpec:
    base = Path(registry_base_dir)
    kg_key = str(data_cfg.get("kg"))
    triplet_source = str(data_cfg.get("triplet_source"))
    drug_key = str(data_cfg.get("drug_profile"))
    target_key = str(data_cfg.get("target_profile"))
    disease_key = str(data_cfg.get("disease_profile"))

    # val3 は datasets / triplets / profiles を必須とする。
    if "datasets" in registry and "triplets" in registry and "profiles" in registry:
        kg = _lookup(registry, "datasets", kg_key)
        trips = _lookup(registry, "triplets", triplet_source)
        triplet_path_key = f"{direction}_tsv"
        if triplet_path_key not in trips:
            raise KeyError(f"Missing registry triplet path: triplets.{triplet_source}.{triplet_path_key}")
        drug = _lookup(registry, "profiles", "drug", drug_key)
        target_profiles = _lookup(registry, "profiles", "target")
        if not isinstance(target_profiles, dict):
            raise ValueError("profiles.target must be a mapping.")
        target_key = _target_profile_key_for_direction(
            requested_key=target_key,
            target_profiles=target_profiles,
            direction=direction,
        )
        target = _lookup(registry, "profiles", "target", target_key)
        disease = _lookup(registry, "profiles", "disease", disease_key) if disease_key and disease_key != "None" else {}
        target_format = str(target.get("format", "caliper_feature_matrix"))
        allowed_target_formats = {"caliper_feature_matrix", "l1000_bing_signature"}
        if target_format not in allowed_target_formats:
            raise ValueError(
                "profiles.target.{key}.format must be one of {allowed}; got {got}".format(
                    key=target_key,
                    allowed=sorted(allowed_target_formats),
                    got=target_format,
                )
            )

        return Val3RegistrySpec(
            kg_path=resolve_path(base, kg["ppi_path"]),
            kg_sep=str(kg.get("sep", "\t")),
            kg_source_col=str(kg.get("source_col", "source")),
            kg_target_col=str(kg.get("target_col", "target")),
            kg_sign_col=str(kg.get("sign_col", "polarity")),
            triplet_tsv=resolve_path(base, trips[triplet_path_key]),
            drug_profile_parquet=resolve_path(base, drug["parquet"]),
            drug_id_col=str(drug.get("id_col", "pert_iname")),
            drug_cell_col=str(drug.get("cell_col", "cell_id")),
            drug_id_mapping_tsv=_maybe_path(base, drug.get("id_mapping_tsv")),
            drug_id_mapping_source_col=drug.get("id_mapping_source_col"),
            drug_id_mapping_target_col=drug.get("id_mapping_target_col"),
            target_profile_key=target_key,
            target_profile_parquet=resolve_path(base, target["parquet"]),
            target_format=target_format,
            target_pert_col=str(target.get("pert_col", "pert_id")),
            target_disease_col=str(target.get("disease_col", "do_id")),
            target_cell_col=str(target.get("cell_col", "cell_id")),
            target_label_col=str(target.get("label_col", "label")),
            target_gene_suffix=str(target.get("gene_suffix", "_gene")),
            target_disease_suffix=str(target.get("disease_suffix", "_disease")),
            target_sign_threshold=float(target.get("sign_threshold", 1.0)),
            disease_profile_parquet=_maybe_path(base, disease.get("parquet")),
            disease_col=disease.get("disease_col"),
        )

    raise ValueError(
        "dataset_registry.yaml must contain val3 blocks: datasets, triplets, profiles."
    )
