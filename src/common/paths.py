from __future__ import annotations

from pathlib import Path


def resolve_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute() or base_dir is None:
        return p
    return Path(base_dir) / p


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
