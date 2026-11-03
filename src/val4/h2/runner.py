from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_val4_h2_config, resolve_start_sign, resolve_val4_output_dir
from val4.family import (
    aggregate_rare_families_as_other,
    build_edge_family_inventory,
    build_edge_family_memberships,
    build_family_eligibility_table,
    build_mechanism_to_family_map,
    get_primary_target_families,
)
from val4.io import load_caliper_profiles_for_val4, load_mechanism_multidigraph_from_tsv
from val4.metadata import (
    append_outputs_to_manifest,
    finalize_run_metadata,
    initialize_run_context,
    mark_section_end,
    mark_section_start,
    write_config_snapshot,
)
from val4.registry import load_dataset_registry, resolve_dataset_spec
from .lofo import run_lofo_analysis
from .progress import H2ProgressRecorder, write_metadata_snapshot
from .report import write_val4_h2_report
from .scoring import score_val2_proposed_pairs
from .sign_shuffle import run_sign_shuffle_analysis
from .viz import make_all_figures

VALID_STEPS = {"all", "compute", "figures", "report"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_steps(raw_steps: list[str] | None) -> set[str]:
    if not raw_steps:
        return {"compute", "figures", "report"}
    steps: set[str] = set()
    for raw in raw_steps:
        for token in str(raw).split(","):
            step = token.strip().lower()
            if step:
                steps.add(step)
    unknown = steps - VALID_STEPS
    if unknown:
        raise ValueError(f"Invalid --steps value(s): {sorted(unknown)}. Valid steps: {sorted(VALID_STEPS)}")
    if "all" in steps:
        return {"compute", "figures", "report"}
    return steps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Val4/H2 family contribution analysis.")
    parser.add_argument("--config", default=None, help="Path to config/val4/h2.yaml.")
    parser.add_argument("--run-root", default=None, help="Existing output/val4/h2/<run_name> directory.")
    parser.add_argument("--steps", action="append", default=None, help="all,compute,figures,report")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args(argv)


def _section_enabled(section: dict, *, default: bool) -> bool:
    return bool(section.get("enabled", default))


def _mark_section_start(run_context: dict, output_dir: Path, section: str) -> None:
    mark_section_start(run_context, section)
    write_metadata_snapshot(run_context, output_dir)


def _mark_section_end(run_context: dict, output_dir: Path, section: str) -> None:
    mark_section_end(run_context, section)
    write_metadata_snapshot(run_context, output_dir)


def _run_compute(config_path: Path, *, progress: bool = True) -> Path:
    cfg = load_val4_h2_config(config_path)
    registry = load_dataset_registry(cfg.registry_path)
    dataset_spec = resolve_dataset_spec(cfg.data, registry, cfg.registry_path.parent)
    output_dir = resolve_val4_output_dir(cfg, task="h2")
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    resolved_start_sign = resolve_start_sign(dataset_spec.perturbation_mode)
    run_context = initialize_run_context(cfg=cfg, dataset_spec=dataset_spec, output_dir=output_dir, task="h2")
    progress_recorder = H2ProgressRecorder(run_context, output_dir, enabled=True)
    write_metadata_snapshot(run_context, output_dir)

    config_outputs: dict[str, str] = write_config_snapshot(cfg=cfg, output_dir=output_dir)
    metrics_outputs: dict[str, str] = {}
    print(f"[CONFIG] {cfg.config_path}")
    print(f"[OUTPUT] {output_dir}")
    print(f"[ANALYSIS] multi_family_edges={cfg.analysis['multi_family_edges']}")
    print("[ANALYSIS] scoring_core=src/inference/signature_reversal_nodes.py")
    print("[ANALYSIS] scoring_method=val2_proposed")
    print(f"[ANALYSIS] h2_n_jobs={cfg.analysis['h2_n_jobs']}")
    print(f"[ANALYSIS] h2_parallel_start_method={cfg.analysis['h2_parallel_start_method']}")

    _mark_section_start(run_context, output_dir, "load_inputs")
    G = load_mechanism_multidigraph_from_tsv(
        str(dataset_spec.ppi_path),
        src_col=dataset_spec.source_col,
        dst_col=dataset_spec.target_col,
        sign_col=dataset_spec.sign_col,
        mechanism_col=dataset_spec.mechanism_col,
        sep=dataset_spec.sep,
    )
    kg_nodes = set(map(str, G.nodes()))
    labels_df, pert_sigs, dis_sigs, common_genes = load_caliper_profiles_for_val4(
        str(dataset_spec.parquet_path),
        pert_suffix=dataset_spec.pert_suffix,
        disease_suffix=dataset_spec.disease_suffix,
        kg_nodes=kg_nodes,
    )
    _mark_section_end(run_context, output_dir, "load_inputs")
    print(f"[KG] nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}")
    print(f"[TABLE] pairs={len(labels_df):,} pert_sigs={len(pert_sigs):,} diseases={len(dis_sigs):,}")

    _mark_section_start(run_context, output_dir, "edge_family_inventory")
    mechanism_to_family = build_mechanism_to_family_map(cfg.family_definition)
    edge_inventory_df = build_edge_family_inventory(G, mechanism_to_family)
    eligibility_df = build_family_eligibility_table(
        edge_inventory_df,
        min_physical_edges=int(cfg.family_selection["min_physical_edges"]),
    )
    edge_inventory_df, eligibility_df = aggregate_rare_families_as_other(edge_inventory_df, eligibility_df)
    edge_inventory_df.to_csv(metrics_dir / "val4_h2_edge_family_inventory.csv", index=False)
    metrics_outputs["edge_family_inventory"] = "metrics/val4_h2_edge_family_inventory.csv"
    eligibility_df.to_csv(metrics_dir / "val4_family_eligibility.csv", index=False)
    metrics_outputs["family_eligibility"] = "metrics/val4_family_eligibility.csv"
    primary_families = get_primary_target_families(eligibility_df)
    if not primary_families:
        raise ValueError("No primary target families passed family_selection.min_physical_edges.")
    edge_id_to_families = build_edge_family_memberships(
        edge_inventory_df,
        str(cfg.analysis["multi_family_edges"]),
        allow_split_equal=False,
    )
    _mark_section_end(run_context, output_dir, "edge_family_inventory")
    print(f"[FAMILY] primary={primary_families}")

    lofo_enabled = _section_enabled(cfg.lofo, default=True)
    sign_shuffle_enabled = _section_enabled(cfg.sign_shuffle, default=True)
    full_pair_scores = None
    if lofo_enabled or sign_shuffle_enabled:
        _mark_section_start(run_context, output_dir, "shared_val2_proposed_full_scores")
        full_pair_scores = score_val2_proposed_pairs(
            G=G,
            labels_df=labels_df,
            pert_sigs=pert_sigs,
            dis_sigs=dis_sigs,
            k_star=int(cfg.analysis["k_star"]),
            start_sign=int(resolved_start_sign),
        )
        if bool(cfg.analysis.get("save_pair_scores", False)):
            full_pair_scores.to_parquet(metrics_dir / "val4_h2_full_pair_scores.parquet", index=False)
            metrics_outputs["full_pair_scores"] = "metrics/val4_h2_full_pair_scores.parquet"
        _mark_section_end(run_context, output_dir, "shared_val2_proposed_full_scores")

    if lofo_enabled:
        _mark_section_start(run_context, output_dir, "lofo")
        lofo_disease, lofo_null, lofo_summary = run_lofo_analysis(
            G=G,
            labels_df=labels_df,
            pert_sigs=pert_sigs,
            dis_sigs=dis_sigs,
            k_star=int(cfg.analysis["k_star"]),
            start_sign=int(resolved_start_sign),
            edge_id_to_families=edge_id_to_families,
            family_order=primary_families,
            n_null=int(cfg.analysis["n_null"]),
            bootstrap_reps=int(cfg.analysis["bootstrap_reps"]),
            seed=int(cfg.seed),
            full_pair_scores=full_pair_scores,
            progress=progress,
            n_jobs=int(cfg.analysis["h2_n_jobs"]),
            parallel_start_method=str(cfg.analysis["h2_parallel_start_method"]),
            parallel_chunksize=int(cfg.analysis["h2_parallel_chunksize"]),
            progress_recorder=progress_recorder,
        )
        lofo_summary.to_csv(metrics_dir / "val4_h2_lofo_family_summary.csv", index=False)
        lofo_disease.to_csv(metrics_dir / "val4_h2_lofo_disease_metrics.csv", index=False)
        lofo_null.to_csv(metrics_dir / "val4_h2_lofo_null_disease_metrics.csv", index=False)
        metrics_outputs.update(
            {
                "lofo_family_summary": "metrics/val4_h2_lofo_family_summary.csv",
                "lofo_disease_metrics": "metrics/val4_h2_lofo_disease_metrics.csv",
                "lofo_null_disease_metrics": "metrics/val4_h2_lofo_null_disease_metrics.csv",
            }
        )
        _mark_section_end(run_context, output_dir, "lofo")

    if sign_shuffle_enabled:
        _mark_section_start(run_context, output_dir, "sign_shuffle")
        sign_disease, sign_summary = run_sign_shuffle_analysis(
            G=G,
            labels_df=labels_df,
            pert_sigs=pert_sigs,
            dis_sigs=dis_sigs,
            k_star=int(cfg.analysis["k_star"]),
            start_sign=int(resolved_start_sign),
            edge_id_to_families=edge_id_to_families,
            family_order=primary_families,
            n_null=int(cfg.analysis["n_null"]),
            bootstrap_reps=int(cfg.analysis["bootstrap_reps"]),
            seed=int(cfg.seed),
            full_pair_scores=full_pair_scores,
            progress=progress,
            n_jobs=int(cfg.analysis["h2_n_jobs"]),
            parallel_start_method=str(cfg.analysis["h2_parallel_start_method"]),
            parallel_chunksize=int(cfg.analysis["h2_parallel_chunksize"]),
            progress_recorder=progress_recorder,
        )
        sign_summary.to_csv(metrics_dir / "val4_h2_sign_shuffle_family_summary.csv", index=False)
        sign_disease.to_csv(metrics_dir / "val4_h2_sign_shuffle_disease_metrics.csv", index=False)
        metrics_outputs.update(
            {
                "sign_shuffle_family_summary": "metrics/val4_h2_sign_shuffle_family_summary.csv",
                "sign_shuffle_disease_metrics": "metrics/val4_h2_sign_shuffle_disease_metrics.csv",
            }
        )
        _mark_section_end(run_context, output_dir, "sign_shuffle")

    section_outputs = {
        "config": config_outputs,
        "metrics": metrics_outputs,
    }

    metadata_path, manifest_path, summary_path = finalize_run_metadata(
        run_context=run_context,
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        n_pairs=len(labels_df),
        n_perturbations=len(pert_sigs),
        n_common_genes=len(common_genes),
        n_families=len(primary_families),
        n_disease_signatures=len(dis_sigs),
        output_dir=output_dir,
        resolved_start_sign=resolved_start_sign,
        section_outputs=section_outputs,
        summary={
            "steps": ["compute"],
            "task": "h2",
            "scoring_core": "src/inference/signature_reversal_nodes.py",
            "scoring_method": "val2_proposed",
            "parallel_unit": "condition",
            "h2_n_jobs": int(cfg.analysis["h2_n_jobs"]),
            "h2_parallel_start_method": str(cfg.analysis["h2_parallel_start_method"]),
        },
    )
    print(f"[WRITE] {metadata_path.name}")
    print(f"[WRITE] {manifest_path.name}")
    print(f"[WRITE] {summary_path.name}")
    return output_dir


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    steps = _parse_steps(args.steps)
    run_root = Path(args.run_root).resolve() if args.run_root else None
    if "compute" in steps:
        config_path = Path(args.config).resolve() if args.config else _repo_root() / "config" / "val4" / "h2.yaml"
        run_root = _run_compute(config_path, progress=not bool(args.no_progress))
    elif run_root is None:
        raise ValueError("--run-root is required when running figures/report without compute.")
    assert run_root is not None
    extra_outputs: dict[str, str] = {}
    if "figures" in steps:
        figure_outputs = make_all_figures(run_root=run_root)
        for key, path in figure_outputs.items():
            p = Path(path)
            extra_outputs[f"figure_{key}"] = str(p.relative_to(run_root)) if p.is_absolute() else str(p)
        if figure_outputs:
            print(f"[WRITE] figures={len(figure_outputs)}")
    if "report" in steps:
        report_outputs = write_val4_h2_report(run_root=run_root)
        for key, path in report_outputs.items():
            p = Path(path)
            extra_outputs[key] = str(p.relative_to(run_root)) if p.is_absolute() else str(p)
            print(f"[WRITE] {key}: {path}")
    figure_outputs = {k: v for k, v in extra_outputs.items() if k.startswith("figure_")}
    report_outputs = {k: v for k, v in extra_outputs.items() if not k.startswith("figure_")}
    append_outputs_to_manifest(run_root, figure_outputs, section_name="figures")
    append_outputs_to_manifest(run_root, report_outputs, section_name="report")
    print("[DONE]")


if __name__ == "__main__":
    main()
