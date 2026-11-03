from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict

from .cache import SharedCache
from .config import ResolvedConfig, load_unified_config, resolve_output_dir, resolve_start_sign
from .io import load_data_bundle
from .metadata import finalize_run_metadata, initialize_run_context, write_json
from .registry import load_dataset_registry, resolve_dataset_spec
from .report import write_val2_report
from .sections import run_ablation_section, run_permutation_section
from .stats import make_all_stats
from .viz import make_all_figures


VALID_STEPS = {"all", "ablation", "permutation", "stats", "figures", "report"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Val2 target ranking evaluation and report generation."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Val2 config YAML/JSON. Required for scoring steps. Defaults to <repo>/config/val2.yaml when omitted outside report/figures-only mode.",
    )
    parser.add_argument("--run-root", default=None, help="Existing or target output/val2/<run_name> directory.")
    parser.add_argument(
        "--steps",
        action="append",
        default=None,
        help="Comma-separated Val2 steps: all,ablation,permutation,figures,report.",
    )
    parser.add_argument(
        "--targetnexus-root",
        default=None,
        help=(
            "Release-matched TargetNexus-DB root used only to regenerate the "
            "Open Targets evidence-stratification figure."
        ),
    )
    return parser.parse_args(argv)


def _parse_steps(raw_steps: list[str] | None, default: str = "all") -> set[str]:
    chunks = raw_steps if raw_steps is not None else [default]
    steps: set[str] = set()
    for chunk in chunks:
        for item in str(chunk).split(","):
            item = item.strip().lower()
            if item:
                steps.add(item)
    unknown = steps - VALID_STEPS
    if unknown:
        raise ValueError(f"Unknown Val2 step(s): {sorted(unknown)}. valid={sorted(VALID_STEPS)}")
    if "all" in steps and len(steps) > 1:
        raise ValueError("Do not combine step 'all' with other Val2 steps.")
    return steps or {"all"}


def _relative(run_root: Path, path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(run_root))
    except Exception:
        return str(p)


def _prefix_outputs(section_dir: Path, outputs: Dict[str, str], run_root: Path) -> Dict[str, str]:
    return {key: _relative(run_root, section_dir / value) for key, value in outputs.items()}


def _make_dirs(run_root: Path) -> Dict[str, Path]:
    dirs = {
        "config": run_root / "config",
        "ablation": run_root / "ablation",
        "permutation": run_root / "permutation",
        "report": run_root / "report",
        "figures": run_root / "report" / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_config_snapshot(cfg: ResolvedConfig, dirs: Dict[str, Path], run_root: Path) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    config_copy = dirs["config"] / "val2.yaml"
    try:
        shutil.copy2(cfg.config_path, config_copy)
    except Exception:
        config_copy.write_text("# Original config could not be copied.\n", encoding="utf-8")
    outputs["config"] = _relative(run_root, config_copy)

    resolved = {
        "config_path": str(cfg.config_path),
        "registry_path": str(cfg.registry_path),
        "seed": cfg.seed,
        "output_root": str(cfg.output_root),
        "output_name": cfg.output_name,
        "date_tag": cfg.date_tag,
        "data": cfg.data,
        "sections": cfg.sections,
        "analysis": cfg.analysis,
        "ablation": cfg.ablation,
        "permutation": cfg.permutation,
        "stats": cfg.stats,
        "steps": cfg.steps,
        "report": cfg.report,
    }
    resolved_path = dirs["config"] / "resolved_config.json"
    write_json(resolved, resolved_path)
    outputs["resolved_config"] = _relative(run_root, resolved_path)
    return outputs


def _load_output_manifest(run_root: Path) -> Dict[str, Any]:
    path = run_root / "output_manifest.json"
    if not path.exists():
        return {"sections": {}, "all_outputs": []}
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        return {"sections": {}, "all_outputs": []}
    manifest.setdefault("sections", {})
    manifest.setdefault("all_outputs", [])
    return manifest


def _manifest_entries(run_root: Path, section_name: str, outputs: Dict[str, str]) -> list[Dict[str, Any]]:
    entries = []
    for logical_name, relative_path in outputs.items():
        path = run_root / str(relative_path)
        entries.append(
            {
                "section": section_name,
                "logical_name": logical_name,
                "relative_path": str(relative_path),
                "exists": path.exists(),
                "size_bytes": int(path.stat().st_size) if path.exists() and path.is_file() else None,
            }
        )
    return entries


def _update_manifest_section(run_root: Path, section_name: str, outputs: Dict[str, str]) -> None:
    manifest = _load_output_manifest(run_root)
    entries = _manifest_entries(run_root, section_name, outputs)
    manifest["sections"][section_name] = entries
    manifest["all_outputs"] = [
        row for row in manifest.get("all_outputs", []) if row.get("section") != section_name
    ] + entries
    write_json(manifest, run_root / "output_manifest.json")


def _run_output_steps(
    run_root: str | Path,
    steps: set[str],
    *,
    targetnexus_root: str | Path | None = None,
) -> Dict[str, Any]:
    run_root = Path(run_root)
    out: Dict[str, Any] = {"run_root": str(run_root), "steps": sorted(steps)}
    # Significance tables are read by figures/report, so (re)generate them first
    # whenever any output step is requested.
    if steps & {"stats", "figures", "report"}:
        stats_outputs = make_all_stats(run_root)
        if stats_outputs:
            out["stats"] = stats_outputs
            _update_manifest_section(run_root, "stats", stats_outputs)
    if "figures" in steps:
        out["figures"] = make_all_figures(
            run_root,
            targetnexus_root=targetnexus_root,
        )
        _update_manifest_section(run_root, "figures", out["figures"])
    if "report" in steps:
        out["report"] = write_val2_report(run_root=run_root)
        _update_manifest_section(run_root, "report", {"val2_report": _relative(run_root, out["report"]["report_md"])})
        out["report"] = write_val2_report(run_root=run_root)
        _update_manifest_section(
            run_root,
            "report",
            {"val2_report": _relative(run_root, out["report"]["report_md"])},
        )
    return out


def _run(
    cfg: ResolvedConfig,
    *,
    run_root_override: str | Path | None = None,
    steps: set[str] | None = None,
    targetnexus_root: str | Path | None = None,
) -> Dict[str, Any]:
    requested = steps or {"all"}
    is_all = "all" in requested
    run_ablation = (is_all and cfg.sections.get("ablation", True)) or "ablation" in requested
    run_permutation = (is_all and cfg.sections.get("permutation", True)) or "permutation" in requested
    run_stats = (is_all and bool(cfg.stats.get("enabled", True))) or "stats" in requested
    run_figures = (is_all and cfg.sections.get("make_figures", True)) or "figures" in requested
    run_report = (is_all and cfg.sections.get("make_report", True)) or "report" in requested

    if requested <= {"stats", "figures", "report"}:
        run_root = Path(run_root_override) if run_root_override is not None else resolve_output_dir(cfg)
        return _run_output_steps(
            run_root,
            requested,
            targetnexus_root=targetnexus_root,
        )

    registry = load_dataset_registry(cfg.registry_path)
    dataset_spec = resolve_dataset_spec(cfg.data, registry, cfg.registry_path.parent)
    run_root = Path(run_root_override) if run_root_override is not None else resolve_output_dir(cfg)
    dirs = _make_dirs(run_root)
    resolved_start_sign = resolve_start_sign(dataset_spec.perturbation_mode)

    section_outputs: Dict[str, Dict[str, str]] = {}
    section_outputs["config"] = _write_config_snapshot(cfg, dirs, run_root)

    print(f"[CONFIG] {cfg.config_path}")
    print(
        "[DATA] "
        f"kg={dataset_spec.kg} perturbation_mode={dataset_spec.perturbation_mode} "
        f"label_source={dataset_spec.label_source}"
    )
    print(f"[OUTPUT] {run_root}")

    run_context = None
    data_bundle = None
    shared_cache = SharedCache()

    if run_ablation or run_permutation:
        run_context = initialize_run_context(cfg=cfg, dataset_spec=dataset_spec, output_dir=run_root)
        data_bundle = load_data_bundle(dataset_spec)
        print(f"[KG] nodes={data_bundle.graph.number_of_nodes():,}, edges={data_bundle.graph.number_of_edges():,}")
        print(
            "[TABLE] "
            f"pairs={len(data_bundle.labels_df):,} "
            f"pert_sigs={len(data_bundle.perturbation_signatures):,} "
            f"dis_sigs={len(data_bundle.disease_signatures):,} "
            f"common_genes={len(data_bundle.common_genes):,}"
        )

    if run_ablation:
        assert data_bundle is not None
        print("[RUN] ablation")
        outputs = run_ablation_section(
            data_bundle=data_bundle,
            output_dir=dirs["ablation"],
            ks=cfg.analysis["ks"],
            methods=cfg.ablation["methods"],
            start_sign=resolved_start_sign,
            sizematch_reps=cfg.ablation["sizematch_reps"],
            seed=cfg.seed,
            k_list=cfg.analysis["K_list"],
            tie_policy=cfg.analysis["tie_policy"],
            save_pred_sign_ambiguity=cfg.ablation["save_pred_sign_ambiguity"],
            save_proposed_w=cfg.ablation["save_proposed_w"],
            save_scores=cfg.ablation["save_scores"],
            save_metrics=cfg.ablation["save_metrics"],
            ncs_cfg=cfg.ablation.get("ncs"),
            n_jobs=cfg.analysis.get("n_jobs", 1),
        )
        section_outputs["ablation"] = _prefix_outputs(dirs["ablation"], outputs, run_root)

    if run_permutation:
        assert data_bundle is not None
        print("[RUN] permutation")
        outputs = run_permutation_section(
            data_bundle=data_bundle,
            output_dir=dirs["permutation"],
            ks=cfg.analysis["ks"],
            start_sign=resolved_start_sign,
            n_perm=cfg.permutation["n_perm"],
            seed=cfg.seed,
            shared_cache=shared_cache,
            save_by_disease=cfg.permutation["save_by_disease"],
            tie_policy=cfg.analysis["tie_policy"],
        )
        section_outputs["permutation"] = _prefix_outputs(dirs["permutation"], outputs, run_root)

    if run_stats:
        print("[RUN] stats")
        stats_outputs = make_all_stats(run_root)
        if stats_outputs:
            section_outputs["stats"] = stats_outputs

    figure_outputs: Dict[str, str] = {}
    if run_figures:
        print("[RUN] figures")
        figure_outputs = make_all_figures(
            run_root,
            targetnexus_root=targetnexus_root,
        )
        section_outputs["figures"] = figure_outputs

    summary = {
        "steps": sorted(requested),
        "ks": list(cfg.analysis["ks"]),
        "k_valid": list(cfg.analysis.get("k_valid", cfg.analysis["ks"])),
        "K_list": list(cfg.analysis["K_list"]),
        "tie_policy": str(cfg.analysis["tie_policy"]),
        "make_report": bool(run_report),
    }
    if run_context is not None and data_bundle is not None:
        metadata_path, manifest_path, summary_path = finalize_run_metadata(
            run_context=run_context,
            data_bundle=data_bundle,
            output_dir=run_root,
            resolved_start_sign=resolved_start_sign,
            section_outputs=section_outputs,
            summary=summary,
        )
        print(f"[WRITE] {metadata_path}")
        print(f"[WRITE] {manifest_path}")
        print(f"[WRITE] {summary_path}")

    if run_report:
        print("[RUN] report")
        report_out = write_val2_report(run_root=run_root, max_rows=int(cfg.report.get("max_rows", 20)))
        section_outputs["report"] = {"val2_report": _relative(run_root, report_out["report_md"])}
        if run_context is not None and data_bundle is not None:
            metadata_path, manifest_path, summary_path = finalize_run_metadata(
                run_context=run_context,
                data_bundle=data_bundle,
                output_dir=run_root,
                resolved_start_sign=resolved_start_sign,
                section_outputs=section_outputs,
                summary=summary,
            )
            print(f"[WRITE] {metadata_path}")
            print(f"[WRITE] {manifest_path}")
            print(f"[WRITE] {summary_path}")
            # Re-write after manifest includes the report entry.
            report_out = write_val2_report(run_root=run_root, max_rows=int(cfg.report.get("max_rows", 20)))

    print("[DONE]")
    if run_root.joinpath("run_summary.json").exists():
        with (run_root / "run_summary.json").open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"run_root": str(run_root), "steps": sorted(requested), "figures": figure_outputs}


def main(argv: list[str] | None = None) -> Dict[str, Any]:
    args = parse_args(argv)
    if args.config is None:
        preliminary_steps = _parse_steps(args.steps, default="all")
        if args.run_root is not None and preliminary_steps <= {"figures", "report"}:
            return _run_output_steps(
                args.run_root,
                preliminary_steps,
                targetnexus_root=args.targetnexus_root,
            )
        args.config = str(_repo_root() / "config" / "val2.yaml")
    cfg = load_unified_config(args.config)
    steps = _parse_steps(args.steps, default=str(cfg.steps.get("default", "all")))
    return _run(
        cfg,
        run_root_override=args.run_root,
        steps=steps,
        targetnexus_root=args.targetnexus_root,
    )


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
