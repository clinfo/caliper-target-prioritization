from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


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


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def read_yaml_or_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix in {".yml", ".yaml"}:
        import yaml
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix in {".tsv", ".txt"}:
        df.to_csv(path, sep="\t", index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported table extension: {suffix}")
