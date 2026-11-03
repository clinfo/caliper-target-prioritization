from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .config import (
    Val1Config,
    load_run_metrics_settings,
    load_run_seed,
    load_val1_config,
    resolve_output_dir,
    resolve_start_sign,
    tqdm_kwargs,
)
from .core import (
    propagate_sign_votes_shortest_paths,
    shortest_path_lengths_up_to_k,
    summarize_target_hop_depths,
)
from .io import (
    ensure_dir,
    load_knockdown_parquet_as_dicts,
    load_signed_multidigraph_from_tsv,
    write_json,
    write_table,
)
from .metadata import (
    finalize_run_metadata,
    initialize_run_metadata,
    record_path_benchmark_metadata,
    refresh_posthoc_run_metadata,
)
from .metrics import (
    compute_pair_coverage,
    compute_target_ambiguity,
    summarize_ambiguity,
    summarize_coverage,
    summarize_coverage_by_td,
)
from .registry import load_dataset_registry, resolve_dataset_spec
from .runtime import run_path_benchmark
from .selection import (
    build_k_metrics,
    build_td_k_metrics,
    rank_k_values,
    rank_td_k_values,
    select_k,
    select_td_k,
)
from .selection_rules import run_selection_sensitivity
from .viz import make_all_figures
from .report import write_val1_report


VALID_STEPS = {"all", "kmax", "metrics", "selection", "td", "pathbench", "rules", "figures", "report"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Val1 k-range analysis and report generation.")
    p.add_argument(
        "--config",
        default=None,
        help=(
            "Path to Val1 config YAML. When omitted (except rules/figures/report-only with "
            "--run-root), defaults to <repo>/config/val1.yaml. Step 'pathbench' also requires "
            "--config because it needs the KG and the target list."
        ),
    )
    p.add_argument("--run-root", default=None, help="Existing or target output/val1/<run_name> directory.")
    p.add_argument(
        "--steps",
        action="append",
        default=None,
        help=(
            "Comma-separated Val1 steps to run: all,kmax,metrics,selection,td,pathbench,rules,"
            "figures,report. 'pathbench' needs --config (KG + targets); it cannot run from "
            "--run-root alone."
        ),
    )
    return p.parse_args(argv)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _progress_iter(cfg: Val1Config, iterable: Iterable, **kwargs):
    progress = cfg.progress
    if not progress.get("enabled", True) or not progress.get("use_tqdm", True):
        return iterable
    from tqdm import tqdm
    merged = tqdm_kwargs(cfg)
    merged.update(kwargs)
    return tqdm(iterable, **merged)


def _parse_steps(raw_steps: list[str] | None) -> set[str]:
    if not raw_steps:
        return {"all"}
    steps: set[str] = set()
    for chunk in raw_steps:
        for item in str(chunk).split(","):
            item = item.strip().lower()
            if item:
                steps.add(item)
    unknown = steps - VALID_STEPS
    if unknown:
        raise ValueError(f"Unknown Val1 step(s): {sorted(unknown)}. valid={sorted(VALID_STEPS)}")
    if "all" in steps and len(steps) > 1:
        raise ValueError("Do not combine step 'all' with other steps.")
    return steps or {"all"}


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Val1 output is missing: {path}")
    return pd.read_csv(path, sep="\t")


def _run_selection_sensitivity_step(
    run_root: str | Path,
    *,
    metrics_settings: Dict[str, Any],
    seed: int,
    progress=None,
) -> Dict[str, Any]:
    """Post-hoc label-free selection-rule sensitivity on a completed run."""
    run_root = Path(run_root)
    out_dir = ensure_dir(run_root / "selection_sensitivity")
    k_metrics = _read_tsv(run_root / "metrics" / "k_metrics.tsv")
    td_path = run_root / "metrics" / "td_k_metrics.tsv"
    td_k_metrics = pd.read_csv(td_path, sep="\t") if td_path.exists() else pd.DataFrame()
    sel_path = run_root / "selection" / "k_selection.json"
    primary_selection = _read_json(sel_path) if sel_path.exists() else {}

    result = run_selection_sensitivity(
        k_metrics=k_metrics,
        td_k_metrics=td_k_metrics,
        primary_selection=primary_selection,
        settings=metrics_settings["selection_sensitivity"],
        coverage_metric=metrics_settings["coverage"]["primary"],
        normalize=bool(metrics_settings["integration"]["normalize"]),
        zero_division=float(metrics_settings["integration"].get("zero_division", 0.0)),
        top_n=int(metrics_settings["selection"]["top_n"]),
        tie_breaker=str(metrics_settings["selection"]["tie_breaker"]),
        seed=int(seed),
        progress=progress,
    )

    written: Dict[str, str] = {}
    for name, table in result["tables"].items():
        path = out_dir / f"{name}.tsv"
        write_table(table, path)
        written[name] = _relative(run_root, path)
    for name, payload in result["json"].items():
        path = out_dir / f"{name}.json"
        write_json(payload, path)
        written[name] = _relative(run_root, path)
    return written


def _run_output_steps(
    run_root: str | Path,
    steps: set[str],
    *,
    metrics_settings: Dict[str, Any] | None = None,
    seed: int | None = None,
    progress=None,
) -> Dict[str, Any]:
    run_root = Path(run_root)
    out: Dict[str, Any] = {"run_root": str(run_root), "steps": sorted(steps)}
    section_outputs: Dict[str, Any] = {}
    if "rules" in steps:
        settings = metrics_settings if metrics_settings is not None else load_run_metrics_settings(run_root)
        out["selection_sensitivity"] = _run_selection_sensitivity_step(
            run_root,
            metrics_settings=settings,
            seed=load_run_seed(run_root) if seed is None else int(seed),
            progress=progress,
        )
        section_outputs["selection_sensitivity"] = out["selection_sensitivity"]
    if "figures" in steps:
        out["figures"] = make_all_figures(run_root)
        section_outputs["figures"] = {
            key: _relative(run_root, path)
            for key, path in out["figures"].items()
        }
    if "report" in steps:
        out["report"] = write_val1_report(run_root=run_root)
        section_outputs["report"] = {
            "val1_report": _relative(run_root, out["report"]["report_md"])
        }
    refresh_posthoc_run_metadata(run_root, section_outputs=section_outputs)
    if "report" in steps:
        out["report"] = write_val1_report(run_root=run_root)
        refresh_posthoc_run_metadata(run_root, section_outputs=section_outputs)
    return out


def _resolve_ks(cfg: Val1Config, k_max: int) -> list[int]:
    policy = cfg.analysis.get("ks_policy", "kmax")
    if policy == "kmax":
        return list(range(1, int(k_max) + 1))
    ks = [int(k) for k in cfg.analysis.get("ks", [])]
    if not ks:
        raise ValueError("analysis.ks_policy='config' requires non-empty analysis.ks.")
    return sorted(set(k for k in ks if k >= 1))


def _load_inputs(cfg: Val1Config, spec) -> Dict[str, Any]:
    G = load_signed_multidigraph_from_tsv(
        str(spec.ppi_path),
        src_col=spec.source_col,
        dst_col=spec.target_col,
        sign_col=spec.sign_col,
        sep=spec.sep,
    )
    labels_df, pert_sigs, disease_sigs, common_genes = load_knockdown_parquet_as_dicts(
        str(spec.parquet_path),
        pert_suffix=spec.pert_suffix,
        disease_suffix=spec.disease_suffix,
        kg_nodes=set(map(str, G.nodes())),
    )
    pairs = labels_df[["disease_id", "target_gene", "cell_id"]].drop_duplicates().copy()
    pairs = pairs.sort_values(["target_gene", "disease_id", "cell_id"]).reset_index(drop=True)
    targets = sorted(pairs["target_gene"].astype(str).unique().tolist())

    disease_gene_sets = {d: set(sig.keys()) for d, sig in disease_sigs.items()}
    perturbation_gene_sets = {k: set(sig.keys()) for k, sig in pert_sigs.items()}

    return {
        "G": G,
        "labels_df": labels_df,
        "pairs": pairs,
        "targets": targets,
        "disease_gene_sets": disease_gene_sets,
        "perturbation_gene_sets": perturbation_gene_sets,
        "common_genes": common_genes,
    }


def _make_dirs(run_root: Path) -> Dict[str, Path]:
    dirs = {
        "config": run_root / "config",
        "kmax": run_root / "kmax",
        "metrics": run_root / "metrics",
        "selection": run_root / "selection",
        "selection_sensitivity": run_root / "selection_sensitivity",
        "runtime": run_root / "runtime",
        "report": run_root / "report",
        "figures": run_root / "report" / "figures",
    }
    for p in dirs.values():
        ensure_dir(p)
    return dirs


def _relative(run_root: Path, path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(run_root))
    except Exception:
        return str(p)


def _run(
    cfg: Val1Config,
    *,
    run_root_override: str | Path | None = None,
    steps: set[str] | None = None,
) -> Dict[str, Any]:
    requested = steps or {"all"}
    is_all = "all" in requested
    run_kmax_step = is_all or "kmax" in requested
    run_metrics_step = (is_all and cfg.sections.get("compute_metrics", True)) or "metrics" in requested or "td" in requested
    run_selection_step = (is_all and cfg.sections.get("select_k", True)) or "selection" in requested
    run_td_step = (is_all and cfg.sections.get("compute_td_individual_stats", True)) or "td" in requested
    run_pathbench_step = (
        is_all
        and cfg.sections.get("run_path_benchmark", False)
        and bool(cfg.analysis["path_benchmark"].get("enabled", True))
    ) or "pathbench" in requested
    run_rules_step = (
        is_all
        and cfg.sections.get("run_selection_sensitivity", True)
        and bool(cfg.metrics["selection_sensitivity"].get("enabled", True))
    ) or "rules" in requested
    run_figures_step = (is_all and cfg.sections.get("make_figures", True)) or "figures" in requested
    run_report_step = (is_all and cfg.sections.get("make_report", True)) or "report" in requested

    if requested <= {"rules", "figures", "report"}:
        run_root = Path(run_root_override) if run_root_override is not None else resolve_output_dir(cfg)
        return _run_output_steps(
            run_root,
            requested,
            metrics_settings=cfg.metrics,
            seed=cfg.seed,
            progress=lambda iterable, **kw: _progress_iter(cfg, iterable, **kw),
        )

    registry = load_dataset_registry(cfg.registry_path)
    spec = resolve_dataset_spec(cfg.data, registry, cfg.registry_path.parent)
    run_root = Path(run_root_override) if run_root_override is not None else resolve_output_dir(cfg)
    dirs = _make_dirs(run_root)
    meta = initialize_run_metadata(cfg, run_root) if (run_kmax_step or run_metrics_step) else None
    section_outputs: Dict[str, Any] = {}

    inputs = None
    G = None
    pairs = pd.DataFrame()
    targets: list[str] = []
    kg_nodes: set[str] = set()
    ks: list[int] = []

    # 'pathbench' needs the KG and the target list too, so it shares the input
    # load and the ks resolution; the metrics computation below stays off.
    if run_kmax_step or run_metrics_step or run_pathbench_step:
        inputs = _load_inputs(cfg, spec)
        G = inputs["G"]
        pairs = inputs["pairs"]
        targets = inputs["targets"]
        kg_nodes = set(map(str, G.nodes()))

        target_depths, kmax_summary = summarize_target_hop_depths(G, targets)
        k_max = int(kmax_summary["K_max"])
        ks = _resolve_ks(cfg, k_max)
        if not ks:
            raise ValueError("No candidate k values resolved. Check K_max and analysis.ks settings.")

        if meta is not None:
            # Update metadata with resolved k values before downstream writes.
            meta["params"]["ks"] = list(ks)
            write_json({k: v for k, v in meta.items() if not k.startswith("_")}, run_root / "metadata.json")

        if run_kmax_step or run_metrics_step:
            write_table(target_depths, dirs["kmax"] / "target_hop_depths.tsv")
            kmax_summary["ks"] = ks
            write_json(kmax_summary, dirs["kmax"] / "kmax_summary.json")
            section_outputs["kmax"] = {
                "target_hop_depths": _relative(run_root, dirs["kmax"] / "target_hop_depths.tsv"),
                "kmax_summary": _relative(run_root, dirs["kmax"] / "kmax_summary.json"),
            }

    @functools.lru_cache(maxsize=None)
    def get_dist_k(source: str, k: int):
        return shortest_path_lengths_up_to_k(G, str(source), int(k))

    @functools.lru_cache(maxsize=None)
    def get_nodes_k(source: str, k: int):
        dist = get_dist_k(str(source), int(k))
        nodes = set(dist.keys())
        nodes.discard(str(source))
        return nodes

    @functools.lru_cache(maxsize=None)
    def get_votes(source: str, k: int, start_sign: int):
        dist = get_dist_k(str(source), int(k))
        return propagate_sign_votes_shortest_paths(
            G,
            str(source),
            int(k),
            start_sign=int(start_sign),
            dist=dist,
        )

    coverage_summary = pd.DataFrame()
    ambiguity_summary = pd.DataFrame()
    pair_coverage = pd.DataFrame()
    target_ambiguity = pd.DataFrame()
    td_coverage_summary = pd.DataFrame()

    if run_metrics_step:
        if inputs is None or G is None:
            raise RuntimeError("Val1 metrics require loaded inputs.")
        pair_coverage = compute_pair_coverage(
            pairs=pairs,
            ks=ks,
            disease_gene_sets=inputs["disease_gene_sets"],
            perturbation_gene_sets=inputs["perturbation_gene_sets"],
            kg_nodes=kg_nodes,
            get_nodes_k=get_nodes_k,
            progress=lambda iterable, **kw: _progress_iter(cfg, iterable, **kw),
        )
        coverage_summary = summarize_coverage(pair_coverage)
        td_coverage_summary = summarize_coverage_by_td(
            pair_coverage,
            progress=lambda iterable, **kw: _progress_iter(cfg, iterable, **kw),
        )

        target_ambiguity = compute_target_ambiguity(
            targets=targets,
            ks=ks,
            start_sign=resolve_start_sign(cfg.data["perturbation_mode"]),
            get_nodes_k=get_nodes_k,
            get_votes=get_votes,
            progress=lambda iterable, **kw: _progress_iter(cfg, iterable, **kw),
        )
        ambiguity_summary = summarize_ambiguity(target_ambiguity)

        write_table(coverage_summary, dirs["metrics"] / "coverage_summary.tsv")
        write_table(ambiguity_summary, dirs["metrics"] / "ambiguity_summary.tsv")
        section_outputs["metrics_input"] = {
            "coverage_summary": _relative(run_root, dirs["metrics"] / "coverage_summary.tsv"),
            "ambiguity_summary": _relative(run_root, dirs["metrics"] / "ambiguity_summary.tsv"),
        }
    elif is_all and not cfg.sections.get("compute_metrics", True):
        raise ValueError("sections.compute_metrics=false is not supported unless existing metrics are loaded explicitly.")
    elif run_selection_step:
        coverage_summary = _read_tsv(dirs["metrics"] / "coverage_summary.tsv")
        ambiguity_summary = _read_tsv(dirs["metrics"] / "ambiguity_summary.tsv")

    selection_summary: Dict[str, Any] = {}
    if run_selection_step or run_td_step:
        cov_metric = cfg.metrics["coverage"]["primary"]
        amb_metric = cfg.metrics["ambiguity"]["primary"]
        normalize = bool(cfg.metrics["integration"]["normalize"])
        zero_division = float(cfg.metrics["integration"].get("zero_division", 0.0))
        top_n = int(cfg.metrics["selection"]["top_n"])
        tie_breaker = str(cfg.metrics["selection"]["tie_breaker"])

    if run_selection_step:
        k_metrics = build_k_metrics(
            coverage_summary=coverage_summary,
            ambiguity_summary=ambiguity_summary,
            coverage_metric=cov_metric,
            ambiguity_metric=amb_metric,
            normalize=normalize,
            zero_division=zero_division,
        )
        k_ranking = rank_k_values(k_metrics, top_n=top_n, tie_breaker=tie_breaker)
        selection_summary = select_k(k_ranking, top_n=top_n)
        k_valid = k_ranking.loc[k_ranking["in_K_valid"], ["k", "rank", "H", "selected"]].copy()

        write_table(k_metrics, dirs["metrics"] / "k_metrics.tsv")
        write_table(k_ranking, dirs["selection"] / "k_ranking.tsv")
        write_json(selection_summary, dirs["selection"] / "k_selection.json")
        write_table(k_valid, dirs["selection"] / "k_valid.tsv")

        section_outputs["global_selection"] = {
            "k_metrics": _relative(run_root, dirs["metrics"] / "k_metrics.tsv"),
            "k_ranking": _relative(run_root, dirs["selection"] / "k_ranking.tsv"),
            "k_selection": _relative(run_root, dirs["selection"] / "k_selection.json"),
            "k_valid": _relative(run_root, dirs["selection"] / "k_valid.tsv"),
        }

    if run_td_step:
        if not run_metrics_step:
            raise ValueError("Step 'td' requires metrics recomputation in the current compact-output mode.")
        td_k_metrics = build_td_k_metrics(
            td_coverage_summary=td_coverage_summary,
            target_ambiguity=target_ambiguity,
            coverage_metric=cov_metric,
            ambiguity_metric=amb_metric,
            normalize=normalize,
            zero_division=zero_division,
            progress=lambda iterable, **kw: _progress_iter(cfg, iterable, **kw),
        )
        td_k_ranking = rank_td_k_values(
            td_k_metrics,
            top_n=top_n,
            tie_breaker=tie_breaker,
            progress=lambda iterable, **kw: _progress_iter(cfg, iterable, **kw),
        )
        td_k_selection, td_k_valid = select_td_k(
            td_k_ranking,
            top_n=top_n,
            all_td_pairs=td_k_metrics[["target_gene", "disease_id"]],
            progress=lambda iterable, **kw: _progress_iter(cfg, iterable, **kw),
        )

        td_metric_cols = [
            c
            for c in [
                "target_gene",
                "disease_id",
                "k",
                "C_td",
                "B_td",
                "A_td",
                "C_td_tilde",
                "A_td_tilde",
                "H_td",
                "n_pairs_td",
                "td_valid_for_ranking",
            ]
            if c in td_k_metrics.columns
        ]
        write_table(td_k_metrics[td_metric_cols], dirs["metrics"] / "td_k_metrics.tsv")
        write_table(td_k_selection, dirs["selection"] / "td_k_selection.tsv")
        write_table(td_k_valid, dirs["selection"] / "td_k_valid.tsv")
        section_outputs["td_individual_stats"] = {
            "td_k_metrics": _relative(run_root, dirs["metrics"] / "td_k_metrics.tsv"),
            "td_k_selection": _relative(run_root, dirs["selection"] / "td_k_selection.tsv"),
            "td_k_valid": _relative(run_root, dirs["selection"] / "td_k_valid.tsv"),
        }

    if run_pathbench_step:
        if G is None:
            raise RuntimeError("Step 'pathbench' requires loaded inputs (--config and the KG).")
        pb = cfg.analysis["path_benchmark"]
        ks_pb = ks if pb["ks"] == "inherit" else [int(x) for x in pb["ks"]]
        per_source, pb_summary = run_path_benchmark(
            G=G,
            sources=targets,
            ks=ks_pb,
            initial_polarity=resolve_start_sign(cfg.data["perturbation_mode"]),
            warmup=pb["warmup"],
            repeats=pb["repeats"],
            timeout_sec=pb["timeout_sec"],
            max_paths=pb["max_paths"],
            measure_node_score=pb["measure_node_score"],
            progress=lambda iterable, **kw: _progress_iter(cfg, iterable, **kw),
        )
        write_table(per_source, dirs["runtime"] / "per_source.tsv")
        write_table(pb_summary, dirs["runtime"] / "per_source_summary.tsv")
        section_outputs["path_benchmark"] = {
            "per_source": _relative(run_root, dirs["runtime"] / "per_source.tsv"),
            "per_source_summary": _relative(run_root, dirs["runtime"] / "per_source_summary.tsv"),
        }
        if meta is None:
            # pathbench-only run: patch the existing metadata.json instead of
            # re-initializing it, so an existing run's metadata is preserved
            # while the execution environment of this benchmark is recorded.
            record_path_benchmark_metadata(
                run_root,
                path_benchmark=pb,
                ks=ks_pb,
                section_outputs=section_outputs["path_benchmark"],
            )

    if run_rules_step:
        section_outputs["selection_sensitivity"] = _run_selection_sensitivity_step(
            run_root,
            metrics_settings=cfg.metrics,
            seed=cfg.seed,
            progress=lambda iterable, **kw: _progress_iter(cfg, iterable, **kw),
        )

    figure_outputs: Dict[str, str] = {}
    if run_figures_step:
        figure_outputs = make_all_figures(run_root)
        section_outputs["figures"] = {k: _relative(run_root, v) for k, v in figure_outputs.items()}

    # Two-pass finalization keeps the final manifest and report consistent while
    # still allowing the report to read metadata.json.
    summary: Dict[str, Any] = {
        "run_root": str(run_root),
        "make_report": bool(cfg.sections.get("make_report", True)),
        "K_max": int(k_max) if "k_max" in locals() else None,
        "ks": ks,
        "selection": selection_summary,
        "canonical_definition": bool(cfg.analysis.get("ks_policy") == "kmax"),
    }
    if meta is not None:
        finalize_run_metadata(meta, run_root, section_outputs=section_outputs, summary=summary)

    if run_report_step:
        report_out = write_val1_report(run_root=run_root)
        section_outputs["report"] = {"val1_report": _relative(run_root, report_out["report_md"])}
        if meta is not None:
            finalize_run_metadata(meta, run_root, section_outputs=section_outputs, summary=summary)
        write_val1_report(run_root=run_root)

    if meta is None:
        return {"run_root": str(run_root), "steps": sorted(requested)}
    return _read_json(run_root / "run_summary.json")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> Dict[str, Any]:
    args = parse_args(argv)
    if args.config is None:
        steps = _parse_steps(args.steps)
        if args.run_root is not None and steps <= {"rules", "figures", "report"}:
            return _run_output_steps(args.run_root, steps)
        args.config = str(_repo_root() / "config" / "val1.yaml")
    cfg = load_val1_config(args.config)
    steps = _parse_steps(args.steps or [str(cfg.steps.get("default", "all"))])
    return _run(cfg, run_root_override=args.run_root, steps=steps)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
