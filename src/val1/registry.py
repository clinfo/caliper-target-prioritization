from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from common.registry import (
    load_registry_mapping as _load_mapping,
    resolve_parquet_path as _resolve_parquet_path,
    resolve_registry_path as _resolve_path,
)


@dataclass(frozen=True)
class DatasetSpec:
    kg: str
    perturbation_mode: str
    label_source: str
    ppi_path: Path
    parquet_path: Path
    sep: str
    source_col: str
    target_col: str
    sign_col: str
    pert_suffix: str
    disease_suffix: str



def load_dataset_registry(path: str | Path) -> Dict[str, Any]:
    return _load_mapping(Path(path).resolve())



def resolve_dataset_spec(
    data_cfg: Dict[str, str], registry: Dict[str, Any], base_dir: Path,
) -> DatasetSpec:
    kg = str(data_cfg.get("kg", "")).strip()
    perturbation_mode = str(data_cfg.get("perturbation_mode", "")).strip()
    label_source = str(data_cfg.get("label_source", "")).strip()

    datasets = registry.get("datasets")
    labels = registry.get("labels")
    if not isinstance(datasets, dict) or not isinstance(labels, dict):
        raise ValueError("Registry must contain 'datasets' and 'labels' mappings.")

    kg_entry = datasets.get(kg)
    if not isinstance(kg_entry, dict):
        raise ValueError(f"Unknown kg in registry: {kg!r}")

    label_source_entry = labels.get(label_source)
    if not isinstance(label_source_entry, dict):
        raise ValueError(f"Unknown label_source in registry: {label_source!r}")

    label_entry = label_source_entry.get(perturbation_mode)
    if not isinstance(label_entry, dict):
        raise ValueError(
            f"Unknown perturbation_mode/label_source in registry: "
            f"label_source={label_source!r}, perturbation_mode={perturbation_mode!r}"
        )

    return DatasetSpec(
        kg=kg,
        perturbation_mode=perturbation_mode,
        label_source=label_source,
        ppi_path=_resolve_path(base_dir, kg_entry["ppi_path"]),
        parquet_path=_resolve_path(
            base_dir,
            _resolve_parquet_path(label_entry, kg, label_source, perturbation_mode),
        ),
        sep=str(kg_entry.get("sep", "\t")),
        source_col=str(kg_entry.get("source_col", "source")),
        target_col=str(kg_entry.get("target_col", "target")),
        sign_col=str(kg_entry.get("sign_col", "polarity")),
        pert_suffix=str(label_entry.get("pert_suffix", "_gene")),
        disease_suffix=str(label_entry.get("disease_suffix", "_disease")),
    )
