from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np
import pandas as pd

from .summary import (
    PRIMARY_K,
    describe_numeric,
    load_ablation_metrics,
    load_ablation_summary,
    load_k_valid_method_summary,
    proposed_ablation_deltas,
    resolve_k_valid,
    summarize_fixed_k_internal_ablation,
)


METHOD_LABEL = {
    "proposed": "CALIPER",
    "kg_sign_noalign": "w/o KD alignment",
    "kg_nosign": "w/o edge polarity",
    "noKG_sizematch": "Size-matched no-KG",
    "ncs": "L1000 NCS",
}
PRESENTATION_EXCLUDED_METHODS = frozenset({"noKG"})

PERM_LABEL = {
    "disease_sign_shuffle": "disease-sign shuffle",
    "pert_sign_shuffle": "KD-sign shuffle",
    "kg_sign_shuffle": "KG-polarity shuffle",
}


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _fmt(x: Any) -> str:
    if x is None:
        return "NA"
    try:
        if pd.isna(x):
            return "NA"
    except Exception:
        pass
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.6g}"
    return str(x)


def _md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "（該当データなし）\n"
    d = df.copy()
    if max_rows is not None:
        d = d.head(max_rows)
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _existing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[col for col in columns if col in df.columns]].copy()


def _bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items) + "\n"


def _append_figure(
    lines: list[str],
    *,
    report_dir: Path,
    figures_dir: Path,
    filename: str,
    title: str,
    notes: Iterable[str],
) -> None:
    fig_path = figures_dir / filename
    if not fig_path.exists():
        return
    lines.append(f"### {title}")
    lines.append(_bullets(notes))
    rel = fig_path.relative_to(report_dir)
    lines.append(f"![{filename}]({rel.as_posix()})")
    lines.append("")


def _manifest_df(run_root: Path) -> pd.DataFrame:
    manifest = _read_json(run_root / "output_manifest.json", {"all_outputs": []})
    rows = manifest.get("all_outputs", []) if isinstance(manifest, dict) else []
    if not rows:
        return pd.DataFrame()
    refreshed = []
    for row in rows:
        item = dict(row)
        relative_path = item.get("relative_path")
        if relative_path:
            path = run_root / str(relative_path)
            item["exists"] = path.exists()
            item["size_bytes"] = int(path.stat().st_size) if path.exists() and path.is_file() else None
        refreshed.append(item)
    df = pd.DataFrame(refreshed)
    cols = [c for c in ["section", "logical_name", "relative_path", "exists", "size_bytes"] if c in df.columns]
    return df[cols]


def _core_ablation_tables(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    overview = load_ablation_summary(run_root)
    if overview.empty:
        return overview, pd.DataFrame()
    overview = overview.copy()
    if "method" in overview.columns:
        overview = overview[
            ~overview["method"].astype(str).isin(PRESENTATION_EXCLUDED_METHODS)
        ].copy()
        overview["method_label"] = overview["method"].map(METHOD_LABEL).fillna(overview["method"])
    keep = [
        c
        for c in [
            "method",
            "method_label",
            "k",
            "n_diseases",
            "mAP",
            "mAP_SEM",
            "mean_Lift@10",
            "mean_Lift@10_SEM",
            "mean_Lift@20",
            "mean_Lift@20_SEM",
        ]
        if c in overview.columns
    ]
    overview = overview[keep].reset_index(drop=True)
    deltas = proposed_ablation_deltas(overview)
    if not deltas.empty and "comparison" in deltas.columns:
        for method, label in METHOD_LABEL.items():
            deltas["comparison"] = deltas["comparison"].str.replace(
                f"proposed - {method}",
                f"CALIPER - {label}",
                regex=False,
            )
    return overview, deltas


def _relabel_significance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "baseline" in out.columns:
        out = out[
            ~out["baseline"].astype(str).isin(PRESENTATION_EXCLUDED_METHODS)
        ].copy()
        out.insert(
            out.columns.get_loc("baseline") + 1,
            "baseline_label",
            out["baseline"].map(METHOD_LABEL).fillna(out["baseline"]),
        )
    return out


def _append_ablation_significance_section(
    lines: list[str],
    run_root: Path,
    *,
    report_dir: Path,
    figures_dir: Path,
) -> None:
    stat_dir = run_root / "stat_tests"
    pairwise = _read_csv(stat_dir / "val2_pairwise_significance.csv")
    best = _read_csv(stat_dir / "val2_best_baseline_significance.csv")
    if pairwise.empty and best.empty:
        return

    lines.append(f"### Disease-level paired significance at k={PRIMARY_K} (bootstrap CI + paired t-test)")
    lines.append(_bullets([
        "見方: 同一 disease の AP / Lift@10 をペアにし、`CALIPER - baseline` の疾患単位 Δ を評価する。",
        "見方: effect size は median Δ と win-rate（Δ>0 の疾患割合）。CI は疾患リサンプリングの paired bootstrap 95% CI。",
        f"検定: 主要距離 k={PRIMARY_K} のみを対象とする両側 paired t 検定（疾患単位 Δ の平均が 0 か否か）。他の k は曲線として示すだけで検定しない。",
        "補正: Holm 法のみ。`significant` は `p_holm < alpha`。補正 family は metric 内の baseline 群（best baseline 比較は metric ごとに 1 検定）。",
    ]))
    sig_cols = [
        "metric", "k", "baseline", "baseline_label", "n_diseases", "n_nonzero",
        "median_delta", "mean_delta", "win_rate", "ci_low", "ci_high",
        "t_stat", "p_value", "p_holm", "significant",
    ]
    if not pairwise.empty:
        lines.append("#### 個別 baseline との比較")
        lines.append(_md_table(_existing_columns(_relabel_significance(pairwise), sig_cols), max_rows=None))
    if not best.empty:
        best_cols = [c for c in sig_cols if c != "baseline" and c != "baseline_label"]
        lines.append("#### disease ごとの best baseline との保守的比較")
        lines.append(_md_table(_existing_columns(best, best_cols), max_rows=None))

    ki = PRIMARY_K
    for metric in ["AP", "Lift@10"]:
        suffix = "ap" if metric == "AP" else "lift10"
        _append_figure(
            lines,
            report_dir=report_dir,
            figures_dir=figures_dir,
            filename=f"val2_effect_size_forest_{suffix}_k{ki}.png",
            title=f"Effect size forest: Δ{metric} (k={ki})",
            notes=[
                f"見方: k={ki} の baseline ごとの median Δ{metric} と paired bootstrap 95% CI。0 縦線より右は CALIPER 優位。",
                "`*`: Holm 補正後 p<0.05（paired t 検定）。",
            ],
        )
        _append_figure(
            lines,
            report_dir=report_dir,
            figures_dir=figures_dir,
            filename=f"val2_win_rate_{suffix}_k{ki}.png",
            title=f"Win-rate across diseases: Δ{metric} (k={ki})",
            notes=[f"見方: k={ki} で Δ{metric}>0 となる疾患割合。0.5 を基準線として表示。"],
        )
        _append_figure(
            lines,
            report_dir=report_dir,
            figures_dir=figures_dir,
            filename=f"val2_delta_dist_{suffix}_k{ki}.png",
            title=f"Per-disease Δ{metric} distribution (k={ki})",
            notes=[f"見方: k={ki} における疾患単位 Δ{metric} の box plot（ヒゲ + 各疾患のジッター点）。"],
        )

    _append_figure(
        lines,
        report_dir=report_dir,
        figures_dir=figures_dir,
        filename="val2_effect_size_forest_bestbaseline.png",
        title="Effect size forest vs best baseline",
        notes=[
            f"見方: k={ki} における best baseline との median ΔAP / ΔLift@10 と paired bootstrap 95% CI。0 横線より上は CALIPER 優位。",
            "`*`: Holm 補正後 p<0.05（paired t 検定）。",
        ],
    )

    for metric in ["AP", "Lift@10"]:
        suffix = "ap" if metric == "AP" else "lift10"
        _append_figure(
            lines,
            report_dir=report_dir,
            figures_dir=figures_dir,
            filename=f"val2_win_rate_{suffix}_bestbaseline.png",
            title=f"Win-rate vs best baseline: Δ{metric}",
            notes=[f"見方: k={ki} で best baseline に対し Δ{metric}>0 となる疾患割合。0.5 を基準線として表示。"],
        )


def _summarize_permutation(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _read_csv(run_root / "permutation" / "val2_2_permutation_summary.csv")
    if df.empty:
        return df, pd.DataFrame()
    if "perm_type" in df.columns:
        df = df.copy()
        df["perm_label"] = df["perm_type"].map(PERM_LABEL).fillna(df["perm_type"])
    for metric, base, delta in [
        ("mAP", "mAP_base", "delta_mAP"),
        ("mean_Lift@10", "mean_Lift@10_base", "delta_mean_Lift@10"),
    ]:
        if metric in df.columns and base in df.columns and delta not in df.columns:
            df[delta] = pd.to_numeric(df[metric], errors="coerce") - pd.to_numeric(df[base], errors="coerce")
    group_cols = [c for c in ["perm_type", "perm_label", "k_star"] if c in df.columns]
    value_cols = [c for c in ["mAP", "mean_Lift@10", "mAP_base", "mean_Lift@10_base", "delta_mAP", "delta_mean_Lift@10"] if c in df.columns]
    summary = df.groupby(group_cols, as_index=False)[value_cols].median(numeric_only=True) if group_cols and value_cols else pd.DataFrame()

    by_dis = _read_csv(run_root / "permutation" / "val2_2_permutation_summary_by_disease.csv")
    if not by_dis.empty:
        if "perm_type" in by_dis.columns:
            by_dis["perm_label"] = by_dis["perm_type"].map(PERM_LABEL).fillna(by_dis["perm_type"])
        if {"AP", "AP_base"}.issubset(by_dis.columns) and "delta_AP" not in by_dis.columns:
            by_dis["delta_AP"] = pd.to_numeric(by_dis["AP"], errors="coerce") - pd.to_numeric(by_dis["AP_base"], errors="coerce")
        if {"Lift@10", "Lift@10_base"}.issubset(by_dis.columns) and "delta_Lift@10" not in by_dis.columns:
            by_dis["delta_Lift@10"] = pd.to_numeric(by_dis["Lift@10"], errors="coerce") - pd.to_numeric(by_dis["Lift@10_base"], errors="coerce")
        dcols = [c for c in ["perm_type", "perm_label", "k_star"] if c in by_dis.columns]
        vcols = [c for c in ["delta_AP", "delta_Lift@10"] if c in by_dis.columns]
        by_dis_summary = by_dis.groupby(dcols, as_index=False)[vcols].median(numeric_only=True) if dcols and vcols else pd.DataFrame()
    else:
        by_dis_summary = pd.DataFrame()
    return summary, by_dis_summary


def _caliper_vs_sizematched_nokg_standardized_delta_table_path(
    run_root: Path,
    metric: str,
    *,
    selection: str = "top10",
) -> Path:
    suffix = "ap" if metric == "AP" else "lift10"
    return (
        run_root
        / "report"
        / "tables"
        / f"k4_caliper_vs_size_matched_no_kg_standardized_delta_{suffix}_{selection}.csv"
    )


def _caliper_vs_sizematched_nokg_standardized_delta_table(
    run_root: Path,
    *,
    metric: str,
    k: int = 4,
    n: int = 10,
    ascending: bool = False,
) -> pd.DataFrame:
    selection = "bottom10" if ascending else "top10"
    table_path = _caliper_vs_sizematched_nokg_standardized_delta_table_path(
        run_root,
        metric,
        selection=selection,
    )
    if table_path.exists():
        return _read_csv(table_path)

    metrics = load_ablation_metrics(run_root)
    sem_col = f"{metric}_SEM"
    required = {"disease_id", "k", "method", metric, sem_col}
    if metrics.empty or not required.issubset(metrics.columns):
        return pd.DataFrame()
    df = metrics.copy()
    df["method"] = df["method"].astype(str)
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df[sem_col] = pd.to_numeric(df[sem_col], errors="coerce")
    df = df[
        (df["k"] == int(k))
        & df["method"].isin(["proposed", "noKG_sizematch"])
    ].dropna(subset=["disease_id", metric])
    if df.empty or set(df["method"]) != {"proposed", "noKG_sizematch"}:
        return pd.DataFrame()

    caliper = df[df["method"] == "proposed"].copy()
    baseline = df[df["method"] == "noKG_sizematch"].copy()
    value_cols = ["disease_id", metric]
    optional_cols = [col for col in ["chance", "n_pos", "n_neg"] if col in caliper.columns]
    merged = caliper[value_cols + optional_cols].merge(
        baseline[value_cols + [sem_col]],
        on="disease_id",
        how="inner",
        suffixes=("_caliper", "_size_matched_noKG"),
    )
    if merged.empty:
        return pd.DataFrame()
    baseline_sem_col = f"{sem_col}_size_matched_noKG" if f"{sem_col}_size_matched_noKG" in merged.columns else sem_col
    output_baseline_sem_col = f"{sem_col}_size_matched_noKG"
    if output_baseline_sem_col not in merged.columns:
        merged[output_baseline_sem_col] = merged[baseline_sem_col]
    merged[f"delta_{metric}"] = merged[f"{metric}_caliper"] - merged[f"{metric}_size_matched_noKG"]
    merged[f"standardized_delta_{metric}"] = merged[f"delta_{metric}"] / merged[baseline_sem_col].replace(0, np.nan)
    out = (
        merged.dropna(subset=[f"standardized_delta_{metric}"])
        .sort_values(f"standardized_delta_{metric}", ascending=ascending)
        .head(int(n))
        .copy()
    )
    out.insert(0, "standardized_rank", np.arange(1, len(out) + 1))
    out.insert(1, "k", int(k))
    out.insert(2, "comparison", "CALIPER - Size-matched no-KG")
    keep = [
        c
        for c in [
            "standardized_rank",
            "k",
            "comparison",
            "disease_id",
            f"{metric}_caliper",
            f"{metric}_size_matched_noKG",
            f"delta_{metric}",
            output_baseline_sem_col,
            f"standardized_delta_{metric}",
            "chance",
            "n_pos",
            "n_neg",
        ]
        if c in out.columns
    ]
    out = out[keep].reset_index(drop=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(table_path, index=False)
    out.to_csv(table_path.with_suffix(".tsv"), sep="\t", index=False)
    return out


def _ap_positive_gene_scatter_summary_by_method(run_root: Path, k_valid: list[int]) -> pd.DataFrame:
    metrics = load_ablation_metrics(run_root)
    required = {"method", "k", "AP", "n_pos"}
    if metrics.empty or not required.issubset(metrics.columns):
        return pd.DataFrame()
    df = metrics.copy()
    df["method"] = df["method"].astype(str)
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df["AP"] = pd.to_numeric(df["AP"], errors="coerce")
    df["n_pos"] = pd.to_numeric(df["n_pos"], errors="coerce")
    if k_valid:
        df = df[df["k"].isin([int(k) for k in k_valid])]
    df = df.dropna(subset=["k", "AP", "n_pos"])
    if df.empty:
        return pd.DataFrame()

    rows = []
    for (method, k), grp in df.groupby(["method", "k"], sort=True):
        ap_values = pd.to_numeric(grp["AP"], errors="coerce")
        n_pos_values = pd.to_numeric(grp["n_pos"], errors="coerce")
        finite = grp[np.isfinite(ap_values) & np.isfinite(n_pos_values)]
        corr = finite["AP"].corr(finite["n_pos"], method="spearman") if len(finite) > 1 else np.nan
        rows.append(
            {
                "method": str(method),
                "method_label": METHOD_LABEL.get(str(method), str(method)),
                "k": int(k),
                "count": int(len(finite)),
                "AP_mean": float(finite["AP"].mean()) if len(finite) else np.nan,
                "AP_median": float(finite["AP"].median()) if len(finite) else np.nan,
                "n_pos_median": float(finite["n_pos"].median()) if len(finite) else np.nan,
                "n_pos_min": float(finite["n_pos"].min()) if len(finite) else np.nan,
                "n_pos_max": float(finite["n_pos"].max()) if len(finite) else np.nan,
                "spearman_AP_vs_n_pos": float(corr) if pd.notna(corr) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[
        ~out["method"].astype(str).isin(PRESENTATION_EXCLUDED_METHODS)
    ].copy()
    if out.empty:
        return out
    method_order = [method for method in METHOD_LABEL if method in set(out["method"])]
    method_order.extend(sorted(set(out["method"]) - set(method_order)))
    out["method"] = pd.Categorical(out["method"], categories=method_order, ordered=True)
    return out.sort_values(["method", "k"]).reset_index(drop=True)


def write_val2_report(run_root: str | Path, *, max_rows: int = 20) -> dict[str, str]:
    run_root = Path(run_root)
    report_dir = run_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = report_dir / "figures"
    report_path = report_dir / "val2_report.md"

    metadata = _read_json(run_root / "metadata.json", {})
    summary = _read_json(run_root / "run_summary.json", {})
    manifest = _manifest_df(run_root)
    k_valid = resolve_k_valid(run_root)

    lines: list[str] = []
    lines.append("# Visualization for `run_val2_unified.py`")
    lines.append("")
    lines.append("This report reproduces the visualization contents for unified Val2 outputs.")
    lines.append("")
    lines.append("Main sections:")
    lines.append("1. Ablation")
    lines.append("2. Permutation")
    lines.append("")
    lines.append(f"Results directory: `{run_root}`")
    lines.append("")

    lines.append("## Run metadata")
    lines.append(_bullets([
        "見方: この表は解析対象データ、実行時刻、評価した k、乱数 seed を確認するための表。",
        "見方: `ks` はこの run で評価した distance k の一覧。",
    ]))
    params = metadata.get("params", {}) if isinstance(metadata, dict) else {}
    data = metadata.get("data", {}) if isinstance(metadata, dict) else {}
    meta_rows = []
    for key in ["run_id", "started_at", "finished_at", "duration_sec", "duration_min"]:
        meta_rows.append({"key": key, "value": metadata.get(key) or summary.get(key)})
    for key in ["kg", "perturbation_mode", "label_source"]:
        meta_rows.append({"key": key, "value": data.get(key)})
    for key in ["ks", "K_list", "tie_policy", "resolved_start_sign"]:
        meta_rows.append({"key": key, "value": params.get(key)})
    meta_rows.append({"key": "k_valid", "value": params.get("k_valid") or summary.get("k_valid") or k_valid})
    meta_rows.append({"key": "seed", "value": metadata.get("seed")})
    lines.append(_md_table(pd.DataFrame(meta_rows)))

    lines.append("## Method / score definition summary")
    lines.append(_bullets([
        "Val2 は target ranking performance evaluation である。",
        "Ablation は `AP`, `Lift@10`, `Lift@20` を疾患単位で読み、method × k ごとに mean±SEM を集約する。",
        "`tie_policy=expected` では、同一 score の target を同順位ブロックとして扱い、ブロック内順序を一様ランダムとした期待 AP / 期待 Lift@K を用いる。",
        "`mAP` は疾患ごとの `AP` の平均で、高いほど positive target が上位に来やすい。",
        "`mean_Lift@10` は疾患ごとの `Lift@10` の平均で、高いほど上位10件に positive target が濃縮している。",
        "`method_label` は命名 crosswalk の Figure 表記に対応する。",
        "`L1000 NCS` は k 非依存のため、k を横軸に取る figure ではエラーバーなしの点線として表示する。",
        f"k={PRIMARY_K} 固定の internal ablation figure には external baseline の `L1000 NCS` を含めない。",
        "Permutation は `disease-sign shuffle`, `KD-sign shuffle`, `KG-polarity shuffle` を base と比較する。",
    ]))

    lines.append("## 1. Ablation")
    lines.append("### Metrics by method and k")
    lines.append(_bullets([
        "見方: notebook の `metrics_df` から、method × k ごとに疾患単位 metric の mean±SEM を集約した表。",
        "見方: `method_label` は命名 crosswalk の Figure 表記に対応する。",
        "見方: 横軸に使う k は notebook と同じ distance k。",
    ]))
    ablation_overview, ablation_delta = _core_ablation_tables(run_root)
    lines.append(_md_table(ablation_overview, max_rows=max_rows))
    _append_figure(
        lines,
        report_dir=report_dir,
        figures_dir=figures_dir,
        filename="mean_ap_by_distance_k.png",
        title="Mean AP by distance k",
        notes=[
            "見方: notebook の `Fig_val2_meanAP_vs_k` に対応する、method ごとの Mean AP by distance k。",
            "横軸: distance k。",
            "縦軸: Mean AP。点と線は method × k ごとの mean。",
        ],
    )
    _append_figure(
        lines,
        report_dir=report_dir,
        figures_dir=figures_dir,
        filename="mean_lift10_by_distance_k.png",
        title="Mean Lift@10 by distance k",
        notes=[
            "見方: notebook の `Fig_val2_meanLift10_vs_k` に対応する、method ごとの Mean Lift@10 by distance k。",
            "横軸: distance k。",
            "縦軸: Mean Lift@10。点と線は method × k ごとの mean。",
        ],
    )
    if k_valid:
        valid_overview = ablation_overview[
            pd.to_numeric(ablation_overview["k"], errors="coerce").isin(k_valid)
        ].copy()
        lines.append("### Candidate k AP / Lift@10 results")
        lines.append(_bullets([
            "見方: config の `analysis.k_valid`（未指定の既存 run では `analysis.ks`）に含まれる candidate k だけを抜き出した結果。",
            "見方: notebook と同じく、疾患単位 `AP` / `Lift@10` を method × k ごとに mean±SEM で表示する。",
            f"見方: この run の candidate k は `{k_valid}`。",
        ]))
        valid_cols = [
            c
            for c in [
                "method",
                "method_label",
                "k",
                "n_diseases",
                "mAP",
                "mAP_SEM",
                "mean_Lift@10",
                "mean_Lift@10_SEM",
            ]
            if c in valid_overview.columns
        ]
        lines.append(_md_table(valid_overview[valid_cols], max_rows=None))

        k_valid_methods = load_k_valid_method_summary(run_root)
        if not k_valid_methods.empty and "method" in k_valid_methods.columns:
            k_valid_methods = k_valid_methods[
                ~k_valid_methods["method"].astype(str).isin(
                    PRESENTATION_EXCLUDED_METHODS
                )
            ].copy()
            k_valid_methods.insert(
                1,
                "method_label",
                k_valid_methods["method"].map(METHOD_LABEL).fillna(k_valid_methods["method"]),
            )
        lines.append("### Candidate k mean by method")
        lines.append(_bullets([
            "見方: candidate k に含まれる各 distance k の疾患単位結果をまとめ、method ごとの平均値と SEM を比較する表。",
            "見方: 次の figure はこの表を論文用に整形したもの。",
        ]))
        lines.append(_md_table(k_valid_methods, max_rows=None))
        _append_figure(
            lines,
            report_dir=report_dir,
            figures_dir=figures_dir,
            filename="candidate_k_method_comparison_ap.png",
            title="Candidate k Mean AP Comparison",
            notes=[
                "見方: config で指定した candidate k に対する AP の method 間比較。",
                "横軸: candidate k。各 candidate k 内で method ごとの点を横にずらして表示する。",
                "縦軸: 各 k における疾患単位 AP の平均値。点は平均、エラーバーは SEM。ただし k 非依存の L1000 NCS はエラーバーなしの点線。",
            ],
        )
        _append_figure(
            lines,
            report_dir=report_dir,
            figures_dir=figures_dir,
            filename="candidate_k_method_comparison_lift10.png",
            title="Candidate k Mean Lift@10 Comparison",
            notes=[
                "見方: config で指定した candidate k に対する Lift@10 の method 間比較。",
                "横軸: candidate k。各 candidate k 内で method ごとの点を横にずらして表示する。",
                "縦軸: 各 k における疾患単位 Lift@10 の平均値。点は平均、エラーバーは SEM。ただし k 非依存の L1000 NCS はエラーバーなしの点線。",
            ],
        )
    fixed_k_overview = summarize_fixed_k_internal_ablation(ablation_overview, PRIMARY_K)
    if not fixed_k_overview.empty:
        fixed_k_method_order = "、".join(
            fixed_k_overview["method"].map(METHOD_LABEL).fillna(fixed_k_overview["method"]).tolist()
        )
        lines.append(f"### Fixed k={PRIMARY_K} internal ablation results")
        fixed_k_notes = [
            f"見方: 主解析は k={PRIMARY_K} 固定。{fixed_k_method_order} を同一疾患集合で比較する。",
        ]
        fixed_k_notes.append("見方: `L1000 NCS` は external baseline なのでこの比較には含めない。")
        fixed_k_notes.append("見方: 次の figure はこの表を論文用に整形したもの。")
        lines.append(_bullets(fixed_k_notes))
        fixed_k_cols = [
            c
            for c in [
                "method",
                "method_label",
                "k",
                "n_diseases",
                "mAP",
                "mAP_SEM",
                "mean_Lift@10",
                "mean_Lift@10_SEM",
            ]
            if c in fixed_k_overview.columns
        ]
        lines.append(_md_table(fixed_k_overview[fixed_k_cols], max_rows=None))
        _append_figure(
            lines,
            report_dir=report_dir,
            figures_dir=figures_dir,
            filename=f"k{PRIMARY_K}_internal_ablation_ap.png",
            title=f"Fixed k={PRIMARY_K} Internal Ablation Mean AP",
            notes=[
                f"見方: k={PRIMARY_K} 固定での CALIPER と internal ablation の Mean AP 比較。",
                "縦軸: method。Mean AP の高い順に上から並べる。",
                "横軸: 疾患単位 AP の平均値。点は平均、横向きエラーバーは SEM。",
                "CALIPER はアクセントカラーの菱形、それ以外はグレーの円で示す。",
            ],
        )
        _append_figure(
            lines,
            report_dir=report_dir,
            figures_dir=figures_dir,
            filename=f"k{PRIMARY_K}_internal_ablation_lift10.png",
            title=f"Fixed k={PRIMARY_K} Internal Ablation Mean Lift@10",
            notes=[
                f"見方: k={PRIMARY_K} 固定での CALIPER と internal ablation の Mean Lift@10 比較。",
                f"横軸: method。{fixed_k_method_order} の順。",
                "縦軸: 疾患単位 Lift@10 の平均値。点は平均、エラーバーは SEM。",
            ],
        )
    support_robustness = _read_tsv(
        report_dir / "tables" / f"support_robustness_k{PRIMARY_K}.tsv"
    )
    if not support_robustness.empty:
        lines.append(f"### Fixed k={PRIMARY_K} evidence-support robustness")
        lines.append(
            _bullets(
                [
                    f"見方: k={PRIMARY_K} の CALIPER score について、minimum measured genes、minimum evaluable cell lines、mean/median cell-line aggregation、leave-one-cell-line-out を比較する。",
                    "縦軸の mAP は疾患単位 AP の平均。c1-c3 のエラーバーは疾患間 SEM。",
                    "leave-one-cell-line-out は full mean aggregation との差 ΔmAP を表示する。",
                    "pair_retention は無フィルタの mean aggregation に対して残る target-disease pair の割合。",
                ]
            )
        )
        support_columns = [
            "analysis",
            "setting",
            "aggregation",
            "min_n_meas",
            "min_cells",
            "omitted_cell",
            "mAP",
            "AP_SEM",
            "n_diseases",
            "n_pairs",
            "n_positive_pairs",
            "pair_retention",
            "delta_mAP",
        ]
        lines.append(
            _md_table(
                _existing_columns(support_robustness, support_columns),
                max_rows=None,
            )
        )
        _append_figure(
            lines,
            report_dir=report_dir,
            figures_dir=figures_dir,
            filename=f"support_robustness_k{PRIMARY_K}.png",
            title=f"Fixed k={PRIMARY_K} Evidence-Support Robustness",
            notes=[
                "c1: minimum measured genes threshold。",
                "c2: minimum evaluable cell-line threshold。",
                "c3: mean versus median cell-line aggregation。",
                "c4: leave-one-cell-line-out ΔmAP。",
            ],
        )
    label_confidence_summary = _read_tsv(
        report_dir
        / "tables"
        / f"label_confidence_score_distribution_k{PRIMARY_K}_summary.tsv"
    )
    if not label_confidence_summary.empty:
        lines.append(
            f"### Fixed k={PRIMARY_K} Open Targets evidence-stratified scores"
        )
        lines.append(
            _bullets(
                [
                    "4つのevidence groupは相互排他的で、Approved positiveを持つ同一56疾患上で比較する。",
                    "縦軸は既存runに保存されたCALIPER scoreであり、解析scoreを再計算しない。",
                ]
            )
        )
        lines.append(_md_table(label_confidence_summary, max_rows=None))
        _append_figure(
            lines,
            report_dir=report_dir,
            figures_dir=figures_dir,
            filename=f"label_confidence_score_distribution_k{PRIMARY_K}.png",
            title="Open Targets Evidence-Stratified CALIPER Scores",
            notes=[
                "Evidence below Phase 2、Phase 2 only、Phase 3+ not approved、Approvedのraw score分布。",
            ],
        )
    support_score_summary = _read_tsv(
        report_dir
        / "tables"
        / f"support_score_distribution_k{PRIMARY_K}_summary.tsv"
    )
    if not support_score_summary.empty:
        lines.append(f"### Fixed k={PRIMARY_K} support-stratified scores")
        lines.append(
            _bullets(
                [
                    "known-positive pairをmeasured genesとevaluable cell linesの相互排他的binで比較する。",
                    "c3はfull analysisに対するleave-one-cell-line-out ΔmAP。",
                ]
            )
        )
        lines.append(_md_table(support_score_summary, max_rows=None))
        _append_figure(
            lines,
            report_dir=report_dir,
            figures_dir=figures_dir,
            filename=f"support_score_distribution_k{PRIMARY_K}.png",
            title="Support-Stratified CALIPER Scores",
            notes=[
                "c1/c2はraw CALIPER score分布、c3はleave-one-cell-line-out ΔmAP。",
            ],
        )
    lines.append("### AP vs positive target-gene count by method")
    lines.append(_bullets([
        "見方: disease-level AP と positive target gene 数の関係を method ごとに別々の散布図で表示する。",
        "見方: 各点は disease × distance k、色と点形状は effective distance set 内の distance k を示す。",
    ]))
    scatter_summary = _ap_positive_gene_scatter_summary_by_method(run_root, k_valid)
    lines.append(_md_table(scatter_summary, max_rows=None))
    if not scatter_summary.empty and "method" in scatter_summary.columns:
        for method in scatter_summary["method"].astype(str).drop_duplicates():
            _append_figure(
                lines,
                report_dir=report_dir,
                figures_dir=figures_dir,
                filename=f"ap_vs_positive_gene_count_{method}.png",
                title=f"{METHOD_LABEL.get(method, method)} AP vs Positive Target-Gene Count",
                notes=[
                    "横軸: disease ごとの positive target gene 数。",
                    "縦軸: disease-level AP。",
                    "色と点形状: effective distance set 内の distance k。",
                ],
            )
    lines.append("### k=4 CALIPER vs Size-matched no-KG standardized-delta diseases")
    lines.append(_bullets([
        "見方: k=4 で `(CALIPER - Size-matched no-KG) / Size-matched no-KG SEM` の上位10疾患と下位10疾患を記録する。",
        "見方: positive target gene 数による AP / Lift@10 のスケール差を、size-matched no-KG null のばらつきで標準化して評価する。",
        "見方: 下位表では standardized delta が小さい順に並べ、負の値は CALIPER が Size-matched no-KG を下回ることを示す。",
        "出力: AP と Lift@10、上位と下位を別々の CSV/TSV table として保存する。",
    ]))
    lines.append("#### AP standardized-delta top 10")
    lines.append(_bullets([
        "出力: `report/tables/k4_caliper_vs_size_matched_no_kg_standardized_delta_ap_top10.csv` と同名 TSV。",
    ]))
    lines.append(
        _md_table(
            _caliper_vs_sizematched_nokg_standardized_delta_table(run_root, metric="AP", k=4, n=10),
            max_rows=None,
        )
    )
    lines.append("#### Lift@10 standardized-delta top 10")
    lines.append(_bullets([
        "出力: `report/tables/k4_caliper_vs_size_matched_no_kg_standardized_delta_lift10_top10.csv` と同名 TSV。",
    ]))
    lines.append(
        _md_table(
            _caliper_vs_sizematched_nokg_standardized_delta_table(run_root, metric="Lift@10", k=4, n=10),
            max_rows=None,
        )
    )
    lines.append("#### AP standardized-delta bottom 10")
    lines.append(_bullets([
        "出力: `report/tables/k4_caliper_vs_size_matched_no_kg_standardized_delta_ap_bottom10.csv` と同名 TSV。",
    ]))
    lines.append(
        _md_table(
            _caliper_vs_sizematched_nokg_standardized_delta_table(
                run_root,
                metric="AP",
                k=4,
                n=10,
                ascending=True,
            ),
            max_rows=None,
        )
    )
    lines.append("#### Lift@10 standardized-delta bottom 10")
    lines.append(_bullets([
        "出力: `report/tables/k4_caliper_vs_size_matched_no_kg_standardized_delta_lift10_bottom10.csv` と同名 TSV。",
    ]))
    lines.append(
        _md_table(
            _caliper_vs_sizematched_nokg_standardized_delta_table(
                run_root,
                metric="Lift@10",
                k=4,
                n=10,
                ascending=True,
            ),
            max_rows=None,
        )
    )
    lines.append("### CALIPER minus ablation")
    lines.append(_bullets([
        "見方: notebook の method 比較を、同じ distance k 内の `CALIPER - comparator` 差分として要約した表。",
        "見方: 正の値は `CALIPER` が比較対象より高いことを示す。",
    ]))
    lines.append(_md_table(ablation_delta, max_rows=max_rows))
    _append_ablation_significance_section(
        lines,
        run_root,
        report_dir=report_dir,
        figures_dir=figures_dir,
    )

    lines.append("## 2. Permutation")
    lines.append(_bullets([
        "見方: notebook の permutation section と同じく、base と shuffled condition を k ごとに比較する。",
        "見方: 対象は `disease-sign shuffle`, `KD-sign shuffle`, `KG-polarity shuffle`。",
        "見方: `delta_mAP = mAP - mAP_base`、`delta_mean_Lift@10 = mean_Lift@10 - mean_Lift@10_base`。",
    ]))
    perm_summary, perm_by_dis = _summarize_permutation(run_root)
    lines.append(_md_table(perm_summary, max_rows=max_rows))
    _append_figure(
        lines,
        report_dir=report_dir,
        figures_dir=figures_dir,
        filename="permutation_delta_map.png",
        title="Permutation Delta mAP",
        notes=[
            "見方: notebook の permutation mAP boxplot / base comparison を delta 表示で要約した図。",
            "横軸: permutation type / k。",
            "縦軸: delta_mAP。0より低いほど permutation により performance が低下したことを示す。",
        ],
    )
    _append_figure(
        lines,
        report_dir=report_dir,
        figures_dir=figures_dir,
        filename="permutation_delta_lift10.png",
        title="Permutation Delta Lift@10",
        notes=[
            "見方: notebook の permutation Lift@10 boxplot / base comparison を delta 表示で要約した図。",
            "横軸: permutation type / k。",
            "縦軸: delta_mean_Lift@10。0より低いほど上位 target の濃縮が崩れたことを示す。",
        ],
    )
    lines.append("### By-disease permutation summary")
    lines.append(_md_table(perm_by_dis, max_rows=max_rows))

    if not ablation_overview.empty:
        lines.append("## Metric Distribution Diagnostics")
        lines.append(_bullets([
            "見方: notebook の display / quick diagnostics に対応する補助表。",
            "見方: ablation summary に含まれる主指標の値域を確認する。",
        ]))
        lines.append(_md_table(describe_numeric(ablation_overview, ["mAP", "mean_Lift@10"])))

    lines.append("## Output inventory")
    lines.append(_bullets([
        "見方: この run で生成された主要ファイルの一覧。",
        "見方: `size_bytes` は各ファイルのおおよその出力量を確認するための値。",
    ]))
    lines.append(_md_table(manifest, max_rows=None))

    lines.append("## Fixed policy statements")
    lines.append(_bullets([
        "`src/val2.py` は内容を変更せず、`src/val2/legacy.py` に移動した。",
        "Val2 の scoring は `src/val2/legacy.py` の既存実装を使用し、ranking metric は config の `analysis.tie_policy` に従う。",
        "`analysis.tie_policy=expected` は同点ブロック内の順序を一様ランダム化した期待 AP / Lift@K、`legacy` は従来の `np.argsort` 順に基づく metric を表す。",
        "report と figure は notebook と同じく既存出力を読み、疾患別 metric から method × k の mean±SEM を表示する。",
        "permutation は notebook と同じく、対応 output CSV が存在しない場合は skip 相当として該当データなしを表示する。",
        "既存 CSV の主要列は維持し、tie-aware 診断列は追加列として保持する。表示用の集約は report / figure 生成時に行う。",
    ]))

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    outputs = {"report_md": str(report_path)}
    for metric, selection in [
        ("AP", "top10"),
        ("Lift@10", "top10"),
        ("AP", "bottom10"),
        ("Lift@10", "bottom10"),
    ]:
        suffix = "ap" if metric == "AP" else "lift10"
        stem = f"k4_caliper_vs_size_matched_no_kg_standardized_delta_{suffix}_{selection}"
        table_path = _caliper_vs_sizematched_nokg_standardized_delta_table_path(
            run_root,
            metric,
            selection=selection,
        )
        if table_path.exists():
            outputs[stem] = str(table_path)
        if table_path.with_suffix(".tsv").exists():
            outputs[f"{stem}_tsv"] = str(table_path.with_suffix(".tsv"))
    return outputs


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Write Val2 markdown report from existing outputs.")
    p.add_argument("--run-root", required=True)
    args = p.parse_args()
    print(json.dumps(write_val2_report(args.run_root), indent=2, ensure_ascii=False))
