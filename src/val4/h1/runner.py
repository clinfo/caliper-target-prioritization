from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .config import load_val4_h1_config, resolve_output_dir, resolve_start_sign
from .core import (
    aggregate_rare_families_as_other,
    build_edge_family_inventory,
    build_family_eligibility_table,
    build_h1_family_lookups,
    build_mechanism_to_family_map,
    get_primary_target_families,
    shortest_path_family_sign_counts,
)
from .io import load_knockdown_parquet_as_dicts, load_mechanism_multidigraph_from_tsv
from .metadata import append_outputs_to_manifest, finalize_run_metadata, initialize_run_context, write_config_snapshot
from .metrics import build_observation_rows, compute_null_and_bootstrap, prepare_key_payloads
from .registry import load_dataset_registry, resolve_dataset_spec
from .report import write_val4_h1_report
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
    parser = argparse.ArgumentParser(
        description="Run Val4/H1 family-wise label-free observability and report generation."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to Val4/H1 config YAML. When omitted, defaults to <repo>/config/val4/h1.yaml.",
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help="Existing or target output/val4/h1/<run_name> directory. Required for figures/report-only runs.",
    )
    parser.add_argument(
        "--steps",
        action="append",
        default=None,
        help="Comma-separated Val4/H1 steps to run: all,compute,figures,report.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args(argv)


def _build_edge_family_lookups(
    edge_inventory_df: pd.DataFrame,
    *,
    multi_family_edges: str,
):
    return build_h1_family_lookups(edge_inventory_df, multi_family_edges)

def _run_compute(config_path: Path, *, progress: bool = True) -> Path:
    cfg = load_val4_h1_config(config_path)
    registry = load_dataset_registry(cfg.registry_path)
    dataset_spec = resolve_dataset_spec(cfg.data, registry, cfg.registry_path.parent)
    output_dir = resolve_output_dir(cfg)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    resolved_start_sign = resolve_start_sign(dataset_spec.perturbation_mode)

    run_context = initialize_run_context(cfg=cfg, dataset_spec=dataset_spec, output_dir=output_dir)
    config_outputs = write_config_snapshot(cfg=cfg, output_dir=output_dir)

    print(f"[CONFIG] {cfg.config_path}")
    print(
        f"[DATA] kg={dataset_spec.kg} "
        f"perturbation_mode={dataset_spec.perturbation_mode} "
        f"label_source={dataset_spec.label_source}"
    )
    print(f"[OUTPUT] {output_dir}")
    print(f"[ANALYSIS] multi_family_edges={cfg.analysis['multi_family_edges']}")

    G = load_mechanism_multidigraph_from_tsv(
        str(dataset_spec.ppi_path),
        src_col=dataset_spec.source_col,
        dst_col=dataset_spec.target_col,
        sign_col=dataset_spec.sign_col,
        mechanism_col=dataset_spec.mechanism_col,
        sep=dataset_spec.sep,
    )
    kg_nodes = set(map(str, G.nodes()))
    print(f"[KG] nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}")

    labels_df, pert_sigs, _dis_sigs, common_genes = load_knockdown_parquet_as_dicts(
        str(dataset_spec.parquet_path),
        pert_suffix=dataset_spec.pert_suffix,
        disease_suffix=dataset_spec.disease_suffix,
        kg_nodes=kg_nodes,
    )
    print(
        f"[TABLE] pairs={len(labels_df):,} "
        f"pert_sigs={len(pert_sigs):,} "
        f"common_genes={len(common_genes):,}"
    )

    perturbation_keys = sorted(
        [
            (str(t), str(c))
            for (t, c) in pert_sigs.keys()
            if str(t) in kg_nodes
        ]
    )
    if len(perturbation_keys) == 0:
        raise ValueError("No perturbation keys left after KG filtering.")

    mechanism_to_family = build_mechanism_to_family_map(cfg.family_definition)
    edge_inventory_df = build_edge_family_inventory(G, mechanism_to_family)
    family_eligibility_df = build_family_eligibility_table(
        edge_inventory_df,
        min_physical_edges=int(cfg.family_selection["min_physical_edges"]),
    )
    edge_inventory_df, family_eligibility_df = aggregate_rare_families_as_other(
        edge_inventory_df,
        family_eligibility_df,
    )
    edge_inventory_df.to_csv(metrics_dir / "val4_h1_edge_family_inventory.csv", index=False)
    family_eligibility_df.to_csv(metrics_dir / "val4_family_eligibility.csv", index=False)
    primary_families = get_primary_target_families(family_eligibility_df)
    if not primary_families:
        raise ValueError("No primary target families passed family_selection.min_physical_edges.")

    mfe = str(cfg.analysis["multi_family_edges"])
    (
        edge_id_to_family,
        edge_id_is_multi_family,
        edge_id_split_families,
        edge_id_include_families,
    ) = _build_edge_family_lookups(
        edge_inventory_df,
        multi_family_edges=mfe,
    )

    shortest_path_results = {}
    key_iter = tqdm(
        perturbation_keys,
        total=len(perturbation_keys),
        desc="h1 shortest paths",
        disable=not progress,
    )
    for key in key_iter:
        target_gene, _cell_id = key
        shortest_path_results[key] = shortest_path_family_sign_counts(
            G,
            source=target_gene,
            k_star=int(cfg.analysis["k_star"]),
            start_sign=int(resolved_start_sign),
            edge_id_to_family=edge_id_to_family,
            edge_id_is_multi_family=edge_id_is_multi_family,
            edge_id_split_families=edge_id_split_families,
            edge_id_include_families=edge_id_include_families,
        )

    observation_rows = build_observation_rows(
        perturbation_keys=perturbation_keys,
        perturbation_signatures=pert_sigs,
        shortest_path_results=shortest_path_results,
    )
    if bool(cfg.analysis["save_observation_rows"]):
        observation_rows.to_parquet(metrics_dir / "val4_h1_observation_rows.parquet", index=False)

    family_order = primary_families
    observation_rows_for_summary = observation_rows[observation_rows["family"].isin(family_order)].copy()
    payloads = prepare_key_payloads(
        observation_rows=observation_rows_for_summary,
        perturbation_signatures=pert_sigs,
        family_order=family_order,
    )
    paired_outputs = compute_null_and_bootstrap(
        payloads=payloads,
        family_order=family_order,
        n_null=int(cfg.analysis["n_null"]),
        bootstrap_reps=int(cfg.analysis["bootstrap_reps"]),
        seed=int(cfg.seed),
        progress=progress,
        observation_rows=observation_rows_for_summary,
    )
    family_summary_df = paired_outputs["family_summary_df"]
    family_summary_cluster_df = paired_outputs["family_summary_cluster_df"]
    family_null_df = paired_outputs["family_null_df"]
    family_null_cluster_df = paired_outputs["family_null_cluster_df"]
    family_boot_df = paired_outputs["family_boot_df"]
    family_boot_cluster_df = paired_outputs["family_boot_cluster_df"]
    key_delta_df = paired_outputs["key_delta_df"]
    row_delta_df = paired_outputs["row_delta_df"]

    family_summary_df.to_csv(metrics_dir / "val4_h1_family_summary.csv", index=False)
    family_summary_cluster_df.to_csv(metrics_dir / "val4_h1_family_summary_cluster.csv", index=False)
    family_null_df.to_csv(metrics_dir / "val4_h1_family_null.csv", index=False)
    family_null_cluster_df.to_csv(metrics_dir / "val4_h1_family_null_cluster.csv", index=False)
    family_boot_df.to_csv(metrics_dir / "val4_h1_family_bootstrap.csv", index=False)
    family_boot_cluster_df.to_csv(metrics_dir / "val4_h1_family_bootstrap_cluster.csv", index=False)
    key_delta_df.to_csv(metrics_dir / "val4_h1_paired_key_deltas.csv", index=False)
    row_delta_df.to_parquet(metrics_dir / "val4_h1_paired_row_deltas.parquet", index=False)

    metrics_outputs = {
        "edge_family_inventory": "metrics/val4_h1_edge_family_inventory.csv",
        "family_eligibility": "metrics/val4_family_eligibility.csv",
        "family_summary": "metrics/val4_h1_family_summary.csv",
        "family_summary_cluster": "metrics/val4_h1_family_summary_cluster.csv",
        "family_null": "metrics/val4_h1_family_null.csv",
        "family_null_cluster": "metrics/val4_h1_family_null_cluster.csv",
        "family_bootstrap": "metrics/val4_h1_family_bootstrap.csv",
        "family_bootstrap_cluster": "metrics/val4_h1_family_bootstrap_cluster.csv",
        "paired_key_deltas": "metrics/val4_h1_paired_key_deltas.csv",
        "paired_row_deltas": "metrics/val4_h1_paired_row_deltas.parquet",
    }
    if bool(cfg.analysis["save_observation_rows"]):
        metrics_outputs["observation_rows"] = "metrics/val4_h1_observation_rows.parquet"

    section_outputs = {
        "config": config_outputs,
        "metrics": metrics_outputs,
    }

    metadata_path, manifest_path, summary_path = finalize_run_metadata(
        run_context=run_context,
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        n_pairs=len(labels_df),
        n_perturbations=len(perturbation_keys),
        n_common_genes=len(common_genes),
        n_families=len(primary_families),
        output_dir=output_dir,
        resolved_start_sign=resolved_start_sign,
        section_outputs=section_outputs,
        summary={"steps": ["compute"], "task": "h1", "method_version": "v3"},
    )
    print(f"[WRITE] {metadata_path.name}")
    print(f"[WRITE] {manifest_path.name}")
    print(f"[WRITE] {summary_path.name}")
    return output_dir



def main(argv: list[str] | None = None) -> None:
    cli_args = parse_args(argv)
    steps = _parse_steps(cli_args.steps)

    run_root: Path | None = Path(cli_args.run_root).resolve() if cli_args.run_root else None

    if "compute" in steps:
        config_path = Path(cli_args.config).resolve() if cli_args.config else _repo_root() / "config" / "val4" / "h1.yaml"
        run_root = _run_compute(config_path, progress=not bool(cli_args.no_progress))
    elif run_root is None:
        raise ValueError("--run-root is required when running figures/report without compute.")

    assert run_root is not None

    extra_outputs: dict[str, str] = {}
    if "figures" in steps:
        figure_outputs = make_all_figures(run_root=run_root)
        extra_outputs.update({f"figure_{key}": path for key, path in figure_outputs.items()})
        if figure_outputs:
            print(f"[WRITE] figures={len(figure_outputs)}")

    if "report" in steps:
        report_outputs = write_val4_h1_report(run_root=run_root)
        for logical_name, path in report_outputs.items():
            path_obj = Path(path)
            relative = str(path_obj.relative_to(run_root)) if path_obj.is_absolute() else str(path_obj)
            extra_outputs[logical_name] = relative
            print(f"[WRITE] {logical_name}: {path}")

    figure_outputs = {k: v for k, v in extra_outputs.items() if k.startswith("figure_")}
    report_outputs = {k: v for k, v in extra_outputs.items() if not k.startswith("figure_")}
    append_outputs_to_manifest(run_root, figure_outputs, section_name="figures")
    append_outputs_to_manifest(run_root, report_outputs, section_name="report")

    print("[DONE]")


if __name__ == "__main__":
    main()
