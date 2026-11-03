from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .io import read_yaml_or_json


def load_mapping(path: str | Path) -> dict[str, Any]:
    return read_yaml_or_json(path)


def require_mapping(obj: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = obj.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping at key {key!r}")
    return value
