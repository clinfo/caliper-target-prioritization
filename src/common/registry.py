from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .io import read_yaml_or_json


def load_registry(path: str | Path) -> dict[str, Any]:
    return read_yaml_or_json(path)


def get_registry_entry(registry: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = registry.get(key)
    if not isinstance(value, Mapping):
        raise KeyError(f"Registry entry not found or not a mapping: {key!r}")
    return value


def load_registry_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")
    return read_yaml_or_json(path)


def resolve_registry_path(base_dir: str | Path, raw_path: str | Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = (Path(base_dir) / path).resolve()
    return path


def resolve_parquet_path(
    label_entry: Mapping[str, Any],
    kg: str,
    label_source: str,
    perturbation_mode: str,
) -> str:
    parquet = label_entry.get("parquet")
    if isinstance(parquet, Mapping):
        if kg not in parquet:
            raise ValueError(
                f"No parquet path for kg={kg!r} in labels.{label_source}.{perturbation_mode}.parquet. "
                f"Available: {list(parquet.keys())}"
            )
        return str(parquet[kg])
    if "parquet_path" in label_entry:
        return str(label_entry["parquet_path"])
    raise ValueError(
        f"labels.{label_source}.{perturbation_mode} must have 'parquet' (dict by kg) or 'parquet_path' (str)."
    )
