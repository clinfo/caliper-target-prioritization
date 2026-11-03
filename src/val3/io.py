from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml_or_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        if p.suffix.lower() == ".json":
            data = json.load(f)
        else:
            data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level: {p}")
    return data


def write_json(obj: Mapping[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(x: Any) -> Any:
    try:
        import numpy as np
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    if isinstance(x, Path):
        return str(x)
    return str(x)


def read_table(path: str | Path, sep: str = "\t") -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)


def read_parquet_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    try:
        return pd.read_parquet(p, engine="pyarrow")
    except Exception:
        return pd.read_parquet(p)


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def resolve_path(base_dir: str | Path, path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    p = Path(str(path_like)).expanduser()
    if p.is_absolute():
        return p
    return (Path(base_dir) / p).resolve()


def signed_edge_set_to_json_str(edges: Any) -> str:
    if edges is None or (isinstance(edges, float) and pd.isna(edges)):
        return "[]"
    s = edges if isinstance(edges, set) else set(edges or [])
    triples = sorted((str(u), str(v), int(sig)) for (u, v, sig) in s)
    return json.dumps(triples, ensure_ascii=False, separators=(",", ":"))
