from __future__ import annotations

from val4.config import (  # noqa: F401
    Val4Config,
    load_val4_config,
    resolve_start_sign,
    resolve_val4_output_dir,
)
from val4.config import load_val4_h2_config as _base_load_val4_h2_config

_UNUSED_H2_ANALYSIS_KEYS = {
    "metrics",
    "save_null_pair_scores",
    "min_family_removed_path_votes",
    "min_evaluable_diseases",
    "gap",
    "path_strategy",
}
_UNUSED_H2_LOFO_KEYS = {
    "strict_edge_removal",
    "common_evaluable",
}
_UNUSED_H2_LOFO_NULL_KEYS = {
    "enabled",
    "mode",
}


def _drop_key(mapping: dict, key: str) -> None:
    if isinstance(mapping, dict):
        mapping.pop(key, None)


def load_val4_h2_config(config_path: str):
    """Load H2 config and remove H2 runtime keys that are not used by implementation."""
    cfg = _base_load_val4_h2_config(config_path)

    for key in _UNUSED_H2_ANALYSIS_KEYS:
        _drop_key(cfg.analysis, key)
        if isinstance(cfg.raw.get("analysis"), dict):
            cfg.raw["analysis"].pop(key, None)

    if isinstance(cfg.lofo, dict):
        for key in _UNUSED_H2_LOFO_KEYS:
            cfg.lofo.pop(key, None)
        for null_key in ("null", None):
            null_cfg = cfg.lofo.get(null_key)
            if isinstance(null_cfg, dict):
                for key in _UNUSED_H2_LOFO_NULL_KEYS:
                    null_cfg.pop(key, None)
                if not null_cfg:
                    cfg.lofo.pop(null_key, None)

    if isinstance(cfg.raw.get("lofo"), dict):
        for key in _UNUSED_H2_LOFO_KEYS:
            cfg.raw["lofo"].pop(key, None)
        for null_key in ("null", None):
            null_cfg = cfg.raw["lofo"].get(null_key)
            if isinstance(null_cfg, dict):
                for key in _UNUSED_H2_LOFO_NULL_KEYS:
                    null_cfg.pop(key, None)
                if not null_cfg:
                    cfg.raw["lofo"].pop(null_key, None)

    if isinstance(cfg.raw.get("sections"), dict):
        cfg.raw.pop("sections", None)
    cfg.sections.clear()

    return cfg
