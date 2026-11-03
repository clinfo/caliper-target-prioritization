from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable
import json

import numpy as np
import pandas as pd


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t") if path.exists() else pd.DataFrame()


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


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
    # '|' separates Markdown table cells, so pipe-joined values such as K_valid
    # must be escaped before rendering.
    return str(x).replace("|", "\\|")


def _md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "（該当データなし）\n"
    d = df.copy()
    if max_rows is not None:
        d = d.head(max_rows)
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items) + "\n"


def _describe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        finite = s[np.isfinite(s)]
        if len(finite) == 0:
            rows.append({"metric": c, "count": 0, "mean": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan, "min": np.nan, "max": np.nan})
        else:
            rows.append(
                {
                    "metric": c,
                    "count": int(len(finite)),
                    "mean": float(finite.mean()),
                    "median": float(finite.median()),
                    "q25": float(finite.quantile(0.25)),
                    "q75": float(finite.quantile(0.75)),
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                }
            )
    return pd.DataFrame(rows)


def _sum_stats(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        finite = s[np.isfinite(s)]
        rows.append(
            {
                "metric": c,
                "sum": float(finite.sum()) if len(finite) else 0.0,
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "median": float(finite.median()) if len(finite) else np.nan,
                "max": float(finite.max()) if len(finite) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _freq_table(series: pd.Series, name: str) -> pd.DataFrame:
    s = series.dropna()
    if s.empty:
        return pd.DataFrame(columns=[name, "n"])
    vc = s.astype(str).value_counts(dropna=False).sort_index()
    return vc.rename_axis(name).reset_index(name="n")


def _k_valid_frequency(td_selection: pd.DataFrame) -> pd.DataFrame:
    counts: Dict[str, int] = {}
    empty = 0
    if td_selection.empty or "K_valid_td" not in td_selection.columns:
        return pd.DataFrame(columns=["k", "n"])
    for raw in td_selection["K_valid_td"].fillna("").astype(str):
        if raw.strip() == "":
            empty += 1
            continue
        for k in raw.split("|"):
            if k.strip():
                counts[k.strip()] = counts.get(k.strip(), 0) + 1
    rows = [{"k": k, "n": n} for k, n in sorted(counts.items(), key=lambda x: int(float(x[0])))]
    rows.append({"k": "empty", "n": empty})
    return pd.DataFrame(rows)


def _td_overlap_stats(td_selection: pd.DataFrame, k_selection: Dict[str, Any]) -> Dict[str, Any]:
    if td_selection.empty:
        return {"n_td": 0, "k_star_match_rate": np.nan, "K_valid_intersect_rate": np.nan}
    global_k_star = k_selection.get("k_star")
    global_k_valid = set(map(str, k_selection.get("K_valid") or []))
    n = len(td_selection)
    match = 0
    intersect = 0
    for _, r in td_selection.iterrows():
        if pd.notna(r.get("k_star_td")) and str(int(float(r["k_star_td"]))) == str(global_k_star):
            match += 1
        ktd = set(str(r.get("K_valid_td", "")).split("|")) - {""}
        if ktd & global_k_valid:
            intersect += 1
    return {
        "n_td": int(n),
        "k_star_match_rate": float(match / n) if n else np.nan,
        "K_valid_intersect_rate": float(intersect / n) if n else np.nan,
    }


PHASE_LABELS = [
    ("reach_sec", "neighborhood_construction"),
    ("enum_sec", "path_enumeration"),
    ("sign_inference_sec", "sign_inference"),
]


PB_INTEGER_COLUMNS = (
    "cutoff",
    "number_of_sources",
    "n_budget_exceeded",
    "n_timeout_censored",
    "peak_rss_kb",
)


def _as_int_strings(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Render count columns as plain integers.

    A Markdown row is built from a mixed-dtype Series, which pandas upcasts to
    float; counts such as ``peak_rss_kb`` would otherwise print as ``1.05e+06``.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").map(
                lambda v: "NA" if pd.isna(v) else f"{int(v):d}"
            )
    return out


def _phase_runtime_table(path_benchmark_summary: pd.DataFrame) -> pd.DataFrame:
    """Long-format phase runtime table (3 phases x median / q25 / q75 / p95)."""
    rows = []
    for _, r in path_benchmark_summary.iterrows():
        for prefix, phase in PHASE_LABELS:
            if f"{prefix}_median" not in path_benchmark_summary.columns:
                continue
            rows.append(
                {
                    "cutoff": r.get("cutoff"),
                    "phase": phase,
                    "median_sec": r.get(f"{prefix}_median"),
                    "q25_sec": r.get(f"{prefix}_q25"),
                    "q75_sec": r.get(f"{prefix}_q75"),
                    "p95_sec": r.get(f"{prefix}_p95"),
                }
            )
    return pd.DataFrame(rows)


def _path_benchmark_footnotes(
    path_benchmark_summary: pd.DataFrame,
    metadata: Dict[str, Any],
) -> list[str]:
    def _first(col: str) -> Any:
        if col not in path_benchmark_summary.columns:
            return None
        s = path_benchmark_summary[col].dropna()
        return s.iloc[0] if len(s) else None

    max_paths = _first("max_paths")
    timeout_sec = _first("timeout_sec")
    warmup = _first("warmup")
    notes = [
        f"パス予算 `max_paths` = {_fmt(max_paths)} が報告対象の打ち切り機構であり、"
        f"`timeout_sec` = {_fmt(timeout_sec)} 秒は run が終わらない事故を防ぐ安全網である"
        "（`n_timeout_censored` はその発火件数で、0 であることを確認するための列）。",
        "集計は `censored == 1` の行のみを除外し、`budget_exceeded == 1` の行は "
        "`n_paths = max_paths` の有効な下限値として含めている。",
        "`enum_sec` にはパスごとの極性推論と結果蓄積が含まれる。",
        "`sign_inference_sec` は `propagate_sign_votes_shortest_paths` が内部で行う自前の近傍探索を含む。",
        f"ディスクキャッシュを迂回した cold な初回コスト（warmup = {_fmt(warmup)}、"
        "single cold execution per source per k）であり、実運用の実行時間ではない。",
        "`peak_rss_kb` は `k` が大きい領域ではパス予算で決まる上限値である。",
        "`target_scoring` と `path_filtering` は摂動・疾患プロファイル行単位の処理で `k` に対して"
        "非支配的なため測定対象外である。",
    ]
    env = (metadata.get("environment") or {}) if isinstance(metadata, dict) else {}
    if env:
        packages = env.get("packages") or {}
        threads = env.get("threads") or {}
        thread_note = ", ".join(f"{k}={v}" for k, v in threads.items() if v not in (None, "")) or "未設定"
        notes.append(
            "実行環境: "
            f"{env.get('platform', 'NA')} / processor `{env.get('processor', 'NA')}` / "
            f"cpu_count {env.get('cpu_count', 'NA')} / threads {thread_note} / "
            f"execution `{env.get('execution', 'NA')}` / python {env.get('python', 'NA')} / "
            + " / ".join(f"{k} {v}" for k, v in packages.items())
            + "。"
        )
    return notes


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


def write_val1_report(
    *,
    run_root: str | Path,
    output_md: str | Path | None = None,
    figures_dir: str | Path | None = None,
) -> Dict[str, Any]:
    run_root = Path(run_root)
    report_dir = run_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_md = Path(output_md) if output_md is not None else report_dir / "val1_report.md"
    figures_dir = Path(figures_dir) if figures_dir is not None else report_dir / "figures"

    metadata = _read_json(run_root / "metadata.json", {}) or {}
    kmax_summary = _read_json(run_root / "kmax" / "kmax_summary.json", {}) or {}
    k_selection = _read_json(run_root / "selection" / "k_selection.json", {}) or {}
    manifest = _read_json(run_root / "output_manifest.json", {}) or {}

    coverage_summary = _read_tsv(run_root / "metrics" / "coverage_summary.tsv")
    ambiguity_summary = _read_tsv(run_root / "metrics" / "ambiguity_summary.tsv")
    k_metrics = _read_tsv(run_root / "metrics" / "k_metrics.tsv")
    k_ranking = _read_tsv(run_root / "selection" / "k_ranking.tsv")
    target_depths = _read_tsv(run_root / "kmax" / "target_hop_depths.tsv")
    td_k_metrics = _read_tsv(run_root / "metrics" / "td_k_metrics.tsv")
    td_selection = _read_tsv(run_root / "selection" / "td_k_selection.tsv")

    path_benchmark_summary = _read_tsv(run_root / "runtime" / "per_source_summary.tsv")

    sensitivity_dir = run_root / "selection_sensitivity"
    rule_agreement = _read_tsv(sensitivity_dir / "rule_agreement.tsv")
    rule_k_metrics = _read_tsv(sensitivity_dir / "rule_k_metrics.tsv")
    tau_sweep = _read_tsv(sensitivity_dir / "coverage_threshold_sweep.tsv")
    beta_sweep = _read_tsv(sensitivity_dir / "ambiguity_ceiling_sweep.tsv")
    resampling_frequency = _read_tsv(sensitivity_dir / "resampling_frequency.tsv")
    rule_selection = _read_json(sensitivity_dir / "rule_selection.json", {}) or {}
    resampling_summary = _read_json(sensitivity_dir / "resampling_summary.json", {}) or {}

    if not k_ranking.empty:
        per_k = k_metrics.merge(
            k_ranking[["k", "rank", "in_K_valid", "selected"]],
            on="k",
            how="left",
        )
    else:
        per_k = k_metrics.copy()

    selected_cols = [c for c in ["k", "C", "B", "A", "C_tilde", "A_tilde", "H", "rank", "in_K_valid", "selected"] if c in per_k.columns]
    td_overlap = _td_overlap_stats(td_selection, k_selection)

    lines: list[str] = []
    lines.append("# Val1 report")
    lines.append("")
    lines.append(f"Results directory: `{run_root}`")
    lines.append("")

    lines.append("## Run metadata")
    lines.append(_bullets([
        "見方: この表は解析対象データ、実行時刻、候補 distance k を確認するための表。",
        "見方: `ks` はこの run で評価した distance k の一覧。",
    ]))
    meta_rows = []
    data = metadata.get("data", {}) or {}
    params = metadata.get("params", {}) or {}
    for key in ["run_id", "started_at", "finished_at", "duration_sec", "duration_min"]:
        meta_rows.append({"key": key, "value": metadata.get(key)})
    for key in ["kg", "perturbation_mode", "label_source"]:
        meta_rows.append({"key": key, "value": data.get(key)})
    meta_rows.append({"key": "ks", "value": params.get("ks")})
    lines.append(_md_table(pd.DataFrame(meta_rows)))

    lines.append("## Method / score definition summary")
    cov = (params.get("coverage") or {})
    amb = (params.get("ambiguity") or {})
    integ = (params.get("integration") or {})
    sel = (params.get("selection") or {})
    lines.append(f"- `K_max`: `{kmax_summary.get('K_max', 'NA')}`")
    lines.append(f"- Coverage C(k): `{cov.get('primary', 'NA')}` / aggregation `{('median' if cov.get('primary') == 'cov_ratio' else 'mean')}`")
    lines.append(f"- Ambiguity B(k): `{amb.get('primary', 'amb_median')}`")
    if bool(integ.get("normalize", True)):
        lines.append("- harmonic score: normalized harmonic mean of `C_tilde(k)` and `A_tilde(k)`, where ambiguity goodness increases as ambiguity decreases.")
    else:
        lines.append("- harmonic score: raw harmonic mean of `C(k)` and `A(k)=1-B(k)`.")
    lines.append(f"- `K_valid` is the effective distance set: the top `{sel.get('top_n', k_selection.get('top_n', 3))}` distances by H(k), with smaller k used as the tie-breaker.")
    lines.append("- `k_star` is the optimal effective distance k*: the highest-ranked member of `K_valid`.")
    lines.append("")

    lines.append("## Overall k Selection")
    lines.append("### Selection summary")
    lines.append(_bullets([
        "見方: `k_star` は全体統計で最も harmonic score が高い optimal effective distance k*。",
        "見方: `K_valid` は上位距離からなる effective distance set。",
        "見方: `score_mode` と `normalized` はスコア計算で正規化を使ったかを示す。",
    ]))
    summary_rows = [{"key": k, "value": v} for k, v in k_selection.items()]
    lines.append(_md_table(pd.DataFrame(summary_rows)))

    lines.append("### Per-k decision table")
    lines.append(_bullets([
        "見方: 各 distance k ごとの Coverage、Ambiguity、A(k)、harmonic score を比較する表。",
        "見方: `rank` が小さいほど推奨順位が高い。",
        "見方: `selected` が true の行が optimal effective distance k*、`in_K_valid` が true の行が effective distance set。",
    ]))
    lines.append(_md_table(per_k[selected_cols] if selected_cols else per_k))
    _append_figure(
        lines,
        report_dir=report_dir,
        figures_dir=figures_dir,
        filename="harmonic_score_vs_k.png",
        title="Harmonic score by distance k",
        notes=[
            "見方: harmonic score の distance k ごとの変化を示す図。",
            "横軸: distance k。",
            "縦軸: Harmonic score。",
            "帯: candidate `k`。破線: selected `k^*`。",
        ],
    )
    _append_figure(
        lines,
        report_dir=report_dir,
        figures_dir=figures_dir,
        filename="distance_k_summary.png",
        title="Distance-k summary metrics",
        notes=[
            "見方: Harmonic score、Coverage、Sign unambiguity を同一 figure 内で比較する図。",
            "横軸: distance k。",
            "破線: selected `k^*`。",
            "IQR 帯は表示しない。",
        ],
    )

    if not rule_agreement.empty:
        lines.append("## Selection-rule sensitivity")
        lines.append(_bullets([
            "見方: primary の harmonic score 以外の label-free 選択規則でも同じ optimal effective distance k* が選ばれるかを確認する節。",
            "見方: ここで比較する規則はいずれも therapeutic label と ranking 性能を使わない (`label_dependent` は全て false)。",
            "見方: primary の選択結果はこの節の解析では変更されない。",
        ]))
        lines.append("### Rule agreement")
        lines.append(_bullets([
            "見方: 各 label-free 選択規則が選んだ optimal effective distance k* と effective distance set の比較表。",
            "見方: `agrees_with_primary` が true の行は primary と同じ k* を選んだ規則。",
            "見方: `notes` には閾値規則の prespecified 値と安定区間を記載する。",
        ]))
        lines.append(_md_table(rule_agreement))
        _append_figure(
            lines,
            report_dir=report_dir,
            figures_dir=figures_dir,
            filename="k_selection_rule_agreement.png",
            title="Selection-rule agreement",
            notes=[
                "見方: label-free 選択規則ごとの k* を並べて一致を確認する図。",
                "横軸: distance k。",
                "縦軸: 選択規則。",
                "塗りつぶし ●: selected `k^*`。白抜き ○: candidate `k`。破線: primary `k^*`。",
            ],
        )

        if not rule_k_metrics.empty:
            lines.append("### Per-k rule scores")
            lines.append(_bullets([
                "見方: distance k ごとの各規則スコアの比較表。",
                "見方: `score_harmonic_mean` は primary と同一の Harmonic score、`score_geometric_mean` は Geometric score。",
                "見方: `on_pareto_front` は Coverage と Sign unambiguity のトレードオフで非劣な distance k、`knee_dist` は Pareto knee 距離。",
            ]))
            rule_cols = [
                c
                for c in [
                    "k",
                    "C",
                    "B",
                    "C_tilde",
                    "A_tilde",
                    "score_harmonic_mean",
                    "score_geometric_mean",
                    "on_pareto_front",
                    "knee_dist",
                    "menger_curvature",
                ]
                if c in rule_k_metrics.columns
            ]
            lines.append(_md_table(rule_k_metrics[rule_cols] if rule_cols else rule_k_metrics))

        stability = (rule_selection.get("threshold_stability") or {}) if isinstance(rule_selection, dict) else {}
        if not tau_sweep.empty:
            lines.append("### Coverage threshold sweep")
            lines.append(_bullets([
                "見方: Coverage 閾値 tau を変えたときに選ばれる k* の変化を示す表。",
                "見方: 規則は「C(k) >= tau を満たす distance k のうち sign ambiguity 最小」。",
                "見方: `n_feasible` は閾値を満たした distance k の数。",
            ]))
            lines.append(_md_table(tau_sweep))
            cov_stab = stability.get("coverage_threshold") or {}
            lines.append(_md_table(pd.DataFrame([cov_stab])) if cov_stab else "")
        if not beta_sweep.empty:
            lines.append("### Ambiguity ceiling sweep")
            lines.append(_bullets([
                "見方: sign ambiguity 上限 beta を変えたときに選ばれる k* の変化を示す表。",
                "見方: 規則は「B(k) <= beta を満たす distance k のうち Coverage 最大」。",
            ]))
            lines.append(_md_table(beta_sweep))
            amb_stab = stability.get("ambiguity_ceiling") or {}
            lines.append(_md_table(pd.DataFrame([amb_stab])) if amb_stab else "")

        if not resampling_frequency.empty:
            lines.append("### Leave-one-out selection stability")
            lines.append(_bullets([
                "見方: target / disease を 1 つずつ除いて k* を選び直したときの選択頻度。",
                "見方: `frequency` が 1 に近いほど、特定の target や disease に依存せず同じ k* が選ばれている。",
                "見方: 各 fold では正規化と規則適用をやり直しており、primary の正規化を固定していない。",
            ]))
            lines.append(_md_table(resampling_frequency))
            folds = resampling_summary.get("folds") if isinstance(resampling_summary, dict) else None
            if folds:
                lines.append(_md_table(pd.DataFrame(folds)))
            _append_figure(
                lines,
                report_dir=report_dir,
                figures_dir=figures_dir,
                filename="loto_k_star_frequency.png",
                title="Leave-one-out selection frequency",
                notes=[
                    "見方: leave-one-out fold で選ばれた k* の頻度を示す図（primary の harmonic score 規則）。",
                    "横軸: distance k。",
                    "縦軸: 選択頻度。",
                    "破線: primary `k^*`。",
                ],
            )

        consistency_rows = []
        for key in ["harmonic_consistency", "pooled_consistency"]:
            payload = rule_selection.get(key) if isinstance(rule_selection, dict) else None
            if isinstance(payload, dict):
                consistency_rows.append({"check": key, **payload})
        if consistency_rows:
            lines.append("### Consistency checks")
            lines.append(_bullets([
                "見方: 感度解析が primary の計算を再現できているかの検証表。",
                "見方: `harmonic_consistency` は再計算した Harmonic score が `k_metrics.tsv` の `H` と一致するかの検証。",
                "見方: `pooled_consistency` は `td_k_metrics.tsv` から復元した C(k) / B(k) が全体統計と一致するかの検証。",
            ]))
            lines.append(_md_table(pd.DataFrame(consistency_rows)))

    if not path_benchmark_summary.empty:
        lines.append("## Path-enumeration complexity")
        lines.append(_bullets([
            "見方: パス数は決定的な量であり、実行時間と違ってマシンやライブラリのバージョンに依存しない。",
            "見方: 到達ノード数は距離 `k` とともに飽和するが、単純パス数は指数的に増加する。",
            "見方: `within_budget_rate` はパス予算内に収まった標的の割合。`budget_exceeded` の行の `n_paths` は下限値。",
            "見方: この節の結果は `k_star` / `K_valid` の決定には使われない。",
        ]))
        lines.append("### Path count and reachable nodes by distance k")
        lines.append(_bullets([
            "見方: `n_paths_median` は標的あたりの単純パス数の中央値、`median_N_k` は到達ノード数の中央値。",
            "見方: `n_paths_ratio_vs_prev` は連続する distance k のパス数成長比で、爆発の主統計。",
            "見方: `within_budget_rate` が 0.5 を下回る distance k では median 自体がパス予算に張り付いた下限値になる。",
            "見方: `peak_rss_kb` はその distance k までに到達したプロセス peak RSS。",
        ]))
        pb_cols = [
            c
            for c in [
                "cutoff",
                "number_of_sources",
                "n_paths_median",
                "n_paths_q25",
                "n_paths_q75",
                "n_paths_p95",
                "median_N_k",
                "n_paths_ratio_vs_prev",
                "n_budget_exceeded",
                "within_budget_rate",
                "n_paths_exact_up_to_percentile",
                "n_timeout_censored",
                "peak_rss_kb",
            ]
            if c in path_benchmark_summary.columns
        ]
        lines.append(
            _md_table(
                _as_int_strings(
                    path_benchmark_summary[pb_cols] if pb_cols else path_benchmark_summary,
                    PB_INTEGER_COLUMNS,
                )
            )
        )
        lines.append("### Phase runtime by distance k")
        lines.append(_bullets([
            "見方: phase 別の実行時間分布。`neighborhood_construction` と `sign_inference` は飽和し、`path_enumeration` が爆発する。",
            "見方: 実行時間は補助指標であり、主張はパス数側で支えている。",
        ]))
        lines.append(_md_table(_as_int_strings(_phase_runtime_table(path_benchmark_summary), ["cutoff"])))
        _append_figure(
            lines,
            report_dir=report_dir,
            figures_dir=figures_dir,
            filename="path_count_vs_k.png",
            title="Candidate simple-path count by distance k",
            notes=[
                "見方: distance k とともに候補単純パス数が指数的に増大することを示す図。",
                "横軸: distance k。",
                "縦軸: 標的あたりの単純パス数（対数）。中央値の線と IQR 帯。",
                "▽: パス予算に張り付いた点（下限値）。",
            ],
        )
        _append_figure(
            lines,
            report_dir=report_dir,
            figures_dir=figures_dir,
            filename="path_runtime_vs_k.png",
            title="Path-enumeration runtime by distance k",
            notes=[
                "見方: distance k ごとのパス列挙計算コストを示す図。",
                "横軸: distance k。",
                "縦軸: 総列挙時間（秒、対数）。",
                "計算コストの図であり、`k_star` の選択根拠ではないため selected `k^*` の破線は示さない。",
            ],
        )
        _append_figure(
            lines,
            report_dir=report_dir,
            figures_dir=figures_dir,
            filename="path_budget_rate_vs_k.png",
            title="Completion rate by distance k",
            notes=[
                "見方: distance k ごとのパス予算内完遂率（機種非依存）を示す図。",
                "横軸: distance k。",
                "縦軸: Completion rate＝パス予算内に収まった標的の割合。",
                "完遂率の図であり、`k_star` の選択根拠ではないため selected `k^*` の破線は示さない。",
            ],
        )
        lines.append(_bullets(_path_benchmark_footnotes(path_benchmark_summary, metadata)))

    lines.append("## Global Coverage and Ambiguity")
    lines.append("### Coverage summary")
    lines.append(_bullets([
        "見方: 全 target-disease ペアをまとめた distance k ごとの Coverage 統計。",
        "見方: `cov_ratio_median` は `(target_gene, disease_id)` ごとに cell median を取った後、distance k ごとに target-disease median を取った値。",
        "見方: `n_empty_base_pairs` は coverage ratio を定義できない評価ペア数。",
    ]))
    lines.append(_md_table(coverage_summary))
    _append_figure(
        lines,
        report_dir=report_dir,
        figures_dir=figures_dir,
        filename="coverage_vs_k.png",
        title="Coverage by distance k",
        notes=[
            "見方: Coverage の distance k ごとの変化を示す図。",
            "横軸: distance k。",
            "縦軸: 評価対象遺伝子のうち到達できた割合。",
            "薄い青帯: target-disease pair 単位の IQR。",
        ],
    )
    lines.append("### Ambiguity summary")
    lines.append(_bullets([
        "見方: 全 target をまとめた distance k ごとの Ambiguity 統計。",
        "見方: `B_amb_median` は target 別 Ambiguity の中央値で、低いほど符号の曖昧さが少ない。",
        "見方: `n_amb_non_na` は Ambiguity を定義できた target 数。",
    ]))
    lines.append(_md_table(ambiguity_summary))
    _append_figure(
        lines,
        report_dir=report_dir,
        figures_dir=figures_dir,
        filename="sign_ambiguity_vs_k.png",
        title="Sign ambiguity by distance k",
        notes=[
            "見方: sign ambiguity の distance k ごとの変化を示す図。",
            "横軸: distance k。",
            "縦軸: 正負両方の符号支持を持つ到達遺伝子の割合。",
            "薄い青帯: target 単位に重複除去した IQR。",
        ],
    )
    _append_figure(
        lines,
        report_dir=report_dir,
        figures_dir=figures_dir,
        filename="sign_unambiguity_vs_k.png",
        title="Sign unambiguity by distance k",
        notes=[
            "見方: A(k)=1-B(k) として、Sign unambiguity を示す図。",
            "横軸: distance k。",
            "縦軸: 正負両方の符号支持を持たない到達遺伝子の割合。",
            "薄い青帯: target 単位に重複除去した IQR。",
        ],
    )
    if not coverage_summary.empty and "n_empty_base_pairs" in coverage_summary.columns:
        lines.append("### Empty Base diagnostics")
        lines.append(_bullets([
            "見方: Base が空で coverage ratio を定義できない評価ペアの診断表。",
            "見方: `n_cov_ratio_non_na` は coverage ratio の集計に使われた評価ペア数。",
            "見方: `covered_count_median` は到達できた評価対象遺伝子数の中央値。",
        ]))
        diag_cols = [c for c in ["k", "n_pairs", "n_empty_base_pairs", "n_cov_ratio_non_na", "base_count_median", "covered_count_median"] if c in coverage_summary.columns]
        lines.append(_md_table(coverage_summary[diag_cols]))

    lines.append("## Target-Disease Pair Statistics")
    lines.append(_bullets([
        "見方: `(target_gene, disease_id)` 個別統計は全体統計とは別の補助解析。",
        "見方: 個別統計は全体の optimal effective distance k* / effective distance set を置き換えない。",
    ]))
    if td_selection.empty and td_k_metrics.empty:
        lines.append("（個別統計は作成されていません）")
    else:
        lines.append("### Target-disease selected k^*")
        lines.append(_bullets([
            "見方: target-disease ペアごとの optimal effective distance k* の頻度表。",
            "見方: 特定の distance k に集中しているか、ペアごとにばらつくかを見る。",
        ]))
        lines.append(_md_table(_freq_table(td_selection.get("k_star_td", pd.Series(dtype=float)), "k_star_td")))
        _append_figure(
            lines,
            report_dir=report_dir,
            figures_dir=figures_dir,
            filename="td_k_star_distribution.png",
            title="Selected k^* by Target-Disease Pair",
            notes=[
                "見方: target-disease ペアごとの selected k^* の分布を示す図。",
                "横軸: selected k^*。",
                "縦軸: target-disease ペア数。",
                "箱ひげ: selected k^* のばらつき。",
            ],
        )
        lines.append("### Target-disease effective distance inclusion frequency")
        lines.append(_bullets([
            "見方: target-disease ペアごとの effective distance set に各 distance k が何回含まれたかを示す表。",
            "見方: `empty` は effective distance set を定義できなかったペア数。",
        ]))
        lines.append(_md_table(_k_valid_frequency(td_selection)))
        _append_figure(
            lines,
            report_dir=report_dir,
            figures_dir=figures_dir,
            filename="td_k_valid_distribution.png",
            title="Candidate k by Target-Disease Pair",
            notes=[
                "見方: target-disease ペアごとの candidate k に含まれる distance k の分布を示す図。",
                "横軸: candidate k。",
                "縦軸: 含まれた回数。",
                "箱ひげ: candidate k のばらつき。",
            ],
        )
        lines.append("### n_valid_k_td distribution")
        lines.append(_bullets([
            "見方: 各 target-disease ペアで順位づけに使えた distance k の分布。",
            "見方: 値が小さい場合、そのペアでは評価可能な distance k が限られている。",
        ]))
        lines.append(_md_table(_describe_numeric(td_selection, ["n_valid_k_td"])))
        lines.append("### C_td / B_td / A_td / H_td distribution")
        lines.append(_bullets([
            "見方: target-disease ペア単位の Coverage、Ambiguity、A(k)、harmonic score の分布。",
            "見方: `C_td` は個別 Coverage、`B_td` は個別 Ambiguity、`H_td` は個別 harmonic score。",
        ]))
        lines.append(_md_table(_describe_numeric(td_k_metrics, ["C_td", "B_td", "A_td", "H_td"])))
        lines.append("### global vs individual selection agreement")
        lines.append(_bullets([
            "見方: 全体選択と個別ペア選択がどの程度一致するかを示す表。",
            "見方: `k_star_match_rate` は個別の optimal effective distance k* が全体の k* と一致した割合。",
            "見方: `K_valid_intersect_rate` は個別 effective distance set と全体 effective distance set が重なった割合。",
        ]))
        lines.append(_md_table(pd.DataFrame([td_overlap])))

    lines.append("## Reachable Depth and Candidate Range")
    lines.append(_bullets([
        "見方: `K_max` は全 target で観測された最大到達 distance k。",
        "見方: この表は候補 distance k の上限と target 数の内訳を確認するための表。",
    ]))
    kmax_rows = [{"key": k, "value": v} for k, v in kmax_summary.items()]
    lines.append(_md_table(pd.DataFrame(kmax_rows)))
    if not target_depths.empty:
        desc = _describe_numeric(target_depths, ["D_t"])
        lines.append(_bullets([
            "見方: `D_t` は各 target から到達可能な最大 distance k。",
            "見方: この表は target ごとの到達深度の分布を要約する。",
        ]))
        lines.append(_md_table(desc))
        _append_figure(
            lines,
            report_dir=report_dir,
            figures_dir=figures_dir,
            filename="target_depth_distribution.png",
            title="Reachable Depth by Target",
            notes=[
                "見方: 各 target から到達可能な最大 distance k の分布を示す図。",
                "横軸: 最大到達 distance k。",
                "縦軸: target 数。",
                "箱ひげ: 最大到達 distance k のばらつき。",
            ],
        )
        lines.append(_bullets([
            "見方: 到達深度が大きい target の例。",
            "見方: `in_kg` が false の target は KG 上に存在しない。",
        ]))
        lines.append(_md_table(target_depths.head(int((metadata.get("report") or {}).get("max_targets_table_rows", 20))) if False else target_depths.head(20)))

    lines.append("## Output inventory")
    files = manifest.get("files", []) if isinstance(manifest, dict) else []
    if files:
        inv = pd.DataFrame(files)
        lines.append(_bullets([
            "見方: この run で生成された主要ファイルの一覧。",
            "見方: `size_bytes` は各ファイルのおおよその出力量を確認するための値。",
        ]))
        lines.append(_md_table(inv, max_rows=100))
    else:
        lines.append("（manifest なし）")

    lines.append("## Fixed policy statements")
    lines.append(_bullets([
        "cov_ratio_pair は base_count == 0 の場合 NaN とした。",
        "cov_ratio を primary coverage として選択した場合、C(k) には二段階 median の cov_ratio_median を用いた。",
        "(t,d) 別個別統計は補助解析であり、global optimal effective distance k* / global effective distance set を置き換えない。",
    ]))
    lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")
    return {"report_md": str(output_md)}


def parse_report_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate a Val1 Markdown report from completed outputs.")
    p.add_argument("--run-root", required=True, help="Path to output/val1/<run_name>.")
    p.add_argument("--output-md", default=None, help="Optional output Markdown path.")
    p.add_argument("--figures-dir", default=None, help="Optional figures directory.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> Dict[str, Any]:
    args = parse_report_args(argv)
    return write_val1_report(
        run_root=args.run_root,
        output_md=args.output_md,
        figures_dir=args.figures_dir,
    )


if __name__ == "__main__":
    main()
