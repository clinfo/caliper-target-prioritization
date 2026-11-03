from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .io import write_json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_metadata(metadata: Mapping[str, Any], path: str | Path) -> None:
    write_json(dict(metadata), path)


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    write_json(dict(manifest), path)


def json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def relative_path(run_root: str | Path, path: str | Path) -> str:
    root = Path(run_root)
    p = Path(path)
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def write_json_safe(data: Any, path: str | Path) -> Path:
    path = Path(path)
    write_json(json_safe(data), path)
    return path
