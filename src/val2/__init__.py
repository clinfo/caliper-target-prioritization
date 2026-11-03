"""Unified Val2 runner package.

The legacy Val2 scoring, ranking, and permutation implementation is
kept intact in :mod:`val2.legacy`. This package adds Val1-style configuration,
section execution, output organization, figure generation, and report writing.
"""

from .config import load_unified_config, resolve_output_dir, resolve_start_sign
from .registry import load_dataset_registry, resolve_dataset_spec
from .io import load_data_bundle, load_legacy_val2
from .sections import (
    run_ablation_section,
    run_permutation_section,
)
from .metadata import initialize_run_context, finalize_run_metadata
from .viz import make_all_figures
from .report import write_val2_report
from .runner import main

__all__ = [
    "load_unified_config",
    "resolve_output_dir",
    "resolve_start_sign",
    "load_dataset_registry",
    "resolve_dataset_spec",
    "load_data_bundle",
    "load_legacy_val2",
    "run_ablation_section",
    "run_permutation_section",
    "initialize_run_context",
    "finalize_run_metadata",
    "make_all_figures",
    "write_val2_report",
    "main",
]
