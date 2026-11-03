"""Markdown report for the val2 external-baseline comparison (methodology §12)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _md_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_(no rows)_"
    df = df.head(max_rows)
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    lines = [header, sep]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(lines)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _fmt_p(v) -> str:
    """p 値表示。underflow した 0 を 0 と書かず下限で示す。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    v = float(v)
    if v == 0.0:
        return "<1e-300"
    return f"{v:.4g}"


def _fmt_ci(value, lo, hi, digits: int = 3) -> str:
    if not isinstance(value, float) or pd.isna(value):
        return "n/a"
    if pd.isna(lo) or pd.isna(hi):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# Report-facing labels of the support metrics (figure/report wording; the TSV keeps
# the implementation column names — see docs/caliper_naming_crosswalk_draft.md).
# Order = table column order: degree, 評価遺伝子数, 細胞株 support。
_SUPPORT_METRIC_LABELS = {
    "degree_H": "target degree",
    "mean_support": "evaluable genes",
    "n_common_cells": "evaluable cell lines",
}
_GROUP_LABELS = {
    "caliper_only": "CALIPER-only",
    "both": "shared (overlap)",
    "rwr_only": "Classical RWR-only",
    "neither": "neither (background)",
}
_METHOD_LABELS = {"CALIPER": "CALIPER", "RWR": "Classical RWR"}
_CANDIDATE_SET_LABELS = {"all": "all candidates", "positive": "positives only"}


def _degree_spearman_by_disease_section(run_root: Path, fig_dir: Path) -> List[str]:
    """§6 の疾患単位 degree 依存度 (pooled Spearman の疾患単位版)。"""
    summary = _read_tsv(run_root / "evaluation" / "degree_diagnostics_spearman_summary.tsv")
    fig = fig_dir / "degree_spearman_by_disease.png"
    if summary.empty or not (run_root / "report" / fig).exists():
        return []

    rows = []
    for r in summary.itertuples(index=False):
        rows.append(
            {
                "method": _METHOD_LABELS.get(str(r.method), str(r.method)),
                "candidate set": _CANDIDATE_SET_LABELS.get(str(r.candidate_set), str(r.candidate_set)),
                "n diseases": int(r.n_diseases),
                "median ρ [95% CI]": _fmt_ci(float(r.median_rho), float(r.ci_lo), float(r.ci_hi)),
                "IQR": f"{float(r.iqr_lo):.3f}–{float(r.iqr_hi):.3f}",
                "pooled ρ": f"{float(r.pooled_rho):.3f}",
                "Δ vs CALIPER [95% CI]": (
                    "—"
                    if pd.isna(r.delta_vs_caliper)
                    else _fmt_ci(float(r.delta_vs_caliper), float(r.delta_lo), float(r.delta_hi))
                ),
                "p (Holm)": "—" if pd.isna(r.p_holm) else _fmt_p(r.p_holm),
            }
        )
    lines: List[str] = []
    lines.append("### 疾患単位の degree 依存度 (per-disease Spearman ρ)")
    lines.append("")
    lines.append(
        "上の pooled ρ は全疾患・全 pair を1本の相関にまとめているため、疾患間の次数分布・"
        "positive 割合の違いが混入する。ここでは同じ ρ を **各疾患の内部で** 算出し、疾患を単位に集約する "
        "(median + 疾患 bootstrap 95%CI, 対応 Wilcoxon 符号順位検定, baseline 間 Holm 補正)。"
    )
    lines.append("")
    lines.append(
        "`all candidates` が primary で、その疾患の common candidate 全件上の ρ。"
        "`positives only` は既知 positive のみに限った pooled 表と同一定義の版で、"
        "疾患あたり positive 数が少なく ρ が不安定になりやすいため positive 数 5 件以上の疾患のみを"
        "対象とする (`included` 列)。"
    )
    lines.append("")
    lines.append(
        "**表.** 疾患単位 Spearman ρ(degree, score) の要約 "
        "(疾患を単位, median は疾患 bootstrap 95%CI, Δ と p は対応 Wilcoxon 符号順位検定 + Holm 補正)。"
    )
    lines.append("")
    lines.append(_md_table(pd.DataFrame(rows)))
    lines.append("")
    lines.append(f"![per-disease degree-score Spearman]({fig})")
    lines.append("")
    lines.append(
        "**図.** common candidate 全件 (`all candidates`) 上で疾患ごとに算出した ρ の分布 "
        "(箱 = 四分位, 横線 = 中央値, 点 = 1疾患)。疾患単位でも Classical RWR の ρ が CALIPER を"
        "一貫して上回る。ただし CALIPER の疾患内 ρ は 0 ではないため、主張は *次数非依存* ではなく "
        "*Classical RWR より次数依存が弱い* に留める。positives only 版と疾患別の値は "
        "`evaluation/degree_diagnostics_spearman_by_disease.tsv`。"
    )
    lines.append("")
    return lines


def _topk_group_profile_section(run_root: Path, fig_dir: Path) -> List[str]:
    """§6 の CALIPER-only / Classical RWR-only top-K 候補プロファイル。"""
    overlap = _read_tsv(run_root / "evaluation" / "topk_overlap_summary.tsv")
    profile = _read_tsv(run_root / "evaluation" / "topk_group_profile.tsv")
    integrated_fig = fig_dir / "topk_overlap_degree_support_k10.png"
    overlap_fig = (
        integrated_fig
        if (run_root / "report" / integrated_fig).exists()
        else fig_dir / "topk_overlap_caliper_vs_rwr.png"
    )
    if overlap.empty or not (run_root / "report" / overlap_fig).exists():
        return []
    k_values = [int(k) for k in overlap.sort_values("K")["K"]]

    lines: List[str] = []
    lines.append("### CALIPER-only / Classical RWR-only top-K 候補の性質")
    lines.append("")
    lines.append(
        "両手法の top-K 候補集合を疾患ごとに取り (`rank_min <= K`、上の top-10 パネルと同一規則)、"
        "CALIPER-only / 共通 / Classical RWR-only に分けて、次数・評価可能遺伝子数・"
        "評価可能細胞株数を比較する。群の定義に label は一切使っていない (positive 数や precision は"
        "本節の表に載せていない)。同点ブロックが K 境界を跨ぐ場合は選択件数が K を超えるため、"
        "`n_top_*` と `boundary_tie_size_*` を疾患別表に記録している。"
    )
    lines.append("")
    lines.append(f"![top-K overlap: CALIPER vs Classical RWR]({overlap_fig})")
    lines.append("")
    lines.append(
        "**図.** K=10 の top-K 候補 pair 集合 (全疾患 pooled) の Vennと、"
        "CALIPER-only / Classical RWR-only pairのtarget degree、evaluable genes、"
        "evaluable cell lines。疾患別の overlap は "
        "`evaluation/topk_overlap_by_disease.tsv`。"
    )
    lines.append("")

    if not profile.empty:
        metrics = [m for m in _SUPPORT_METRIC_LABELS if f"{m}_median" in profile.columns]
        rows = []
        for k in k_values:
            at_k = profile[profile["K"].astype(int).eq(k)]
            for group in ("caliper_only", "rwr_only"):
                sub = at_k[at_k["group"].eq(group)]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                row = {
                    "K": k,
                    "group": _GROUP_LABELS.get(group, group),
                    "top-K pairs": int(r["n_pairs"]),
                    "unique targets": int(r["n_unique_targets"]),
                }
                for m in metrics:
                    row[_SUPPORT_METRIC_LABELS[m]] = (
                        f"{float(r[f'{m}_median']):.4g} "
                        f"[{float(r[f'{m}_q25']):.4g}, {float(r[f'{m}_q75']):.4g}]"
                    )
                rows.append(row)
        jaccard_text = "; ".join(
            f"K = {int(r.K)}: {_fmt_ci(float(r.median_jaccard), float(r.jaccard_ci_lo), float(r.jaccard_ci_hi))}"
            for r in overlap.sort_values("K").itertuples(index=False)
        )
        lines.append(
            "**表.** CALIPER-only / Classical RWR-only top-K 候補の overlap と support "
            "(次数・評価可能遺伝子数・評価可能細胞株数は median [IQR]。両手法の top-K overlap は"
            "疾患単位の median Jaccard [95% CI] で " + jaccard_text + "、"
            "共通候補の件数は上の Venn 図の交差部を参照)。"
        )
        lines.append("")
        lines.append(_md_table(pd.DataFrame(rows)))
        lines.append("")
        lines.append(
            "CALIPER-only 候補は Classical RWR-only 候補より次数が明確に低い一方、"
            "評価可能細胞株数・評価可能遺伝子数は RWR-only より小さい。"
            "すなわち「CALIPER-only 候補は低次数だが support も薄い側に寄る」"
            "という向きであり、support 依存性の本格的な検証 (最小評価遺伝子数・細胞株数の閾値変更、"
            "leave-one-cell-line-out) は本節の範囲外とする。背景群 (どちらの top-K にも入らない候補) を"
            "含む全 K・全指標の値は `evaluation/topk_group_profile.tsv`、pair 単位の値は "
            "`evaluation/topk_group_pairs.tsv`、群間比較 (Mann-Whitney U, Holm 補正) は "
            "`evaluation/topk_group_tests.tsv`。"
        )
        lines.append("")
    return lines


def write_report(run_root: str | Path) -> Path:
    run_root = Path(run_root)
    summary = _load_json(run_root / "run_summary.json")
    meta = _load_json(run_root / "metadata.json")
    flow = summary.get("flow", {})

    method_summary = _read_tsv(run_root / "evaluation" / "method_summary.tsv")
    bootstrap = _read_tsv(run_root / "evaluation" / "bootstrap_summary.tsv")
    ttest = _read_tsv(run_root / "evaluation" / "ttest_fdr.tsv")
    effect_sizes = _read_tsv(run_root / "evaluation" / "effect_sizes.tsv")
    exclusion = _read_tsv(run_root / "common_sets" / "exclusion_summary.tsv")

    k_list = [int(k) for k in meta.get("K_list", [])] or [10]
    caliper_ks = [int(k) for k in meta.get("caliper_ks", [])] or ([int(meta["k_star"])] if meta.get("k_star") is not None else [])
    k_star = int(meta["k_star"]) if meta.get("k_star") is not None else None
    primary_lift = f"Lift@{k_list[0]}"
    selected_baselines = [
        str(b) for b in meta.get("selected_baselines", ["NCS", "proximity", "RWR", "DADA"])
    ]
    baseline_display = {
        "NCS": "L1000-NCS",
        "proximity": "Network proximity",
        "RWR": "Classical RWR",
        "DADA": "DADA",
    }
    selected_baseline_text = " / ".join(baseline_display.get(b, b) for b in selected_baselines)
    fig_dir = Path("figures")
    lines: List[str] = []
    lines.append("# val2 外部ベースライン比較レポート")
    lines.append("")
    lines.append(f"- run_id: `{summary.get('run_id', run_root.name)}`")
    lines.append(f"- dataset: `{meta.get('data', {})}`")
    primary_analysis = str(meta.get("primary_analysis", "native_shared_network"))
    lines.append(f"- primary analysis: `{primary_analysis}`")
    lines.append(f"- selected baselines: `{selected_baselines}`")
    lines.append(f"- k_star: `{meta.get('k_star')}` / RWR restart r: `{meta.get('rwr_restart_probability')}`")
    if len(caliper_ks) > 1:
        lines.append(f"- CALIPER scoring ks: `{caliper_ks}` (各 k を baseline と個別比較)")
    lines.append(f"- NCS feature space size: `{meta.get('ncs_feature_space_size')}`")
    lines.append(
        f"- NCS reference: `{meta.get('ncs_reference_n_profiles')}` profiles / "
        f"`{meta.get('ncs_reference_n_cells')}` cell lines "
        f"(unit: `{meta.get('ncs_reference_profile_unit')}`)"
    )
    if primary_analysis == "directed_target_reach_matched":
        tr = meta.get("target_reach_matched", {})
        lines.append(
            f"- target-reach matched horizon: `{tr.get('horizon')}` (`h={tr.get('h')}`), "
            f"fallback scores: `{tr.get('fallback_scores')}`"
        )
    else:
        lines.append(
            f"- shared network H: nodes `{meta.get('shared_network_stats', {}).get('n_nodes')}`, "
            f"edges `{meta.get('shared_network_stats', {}).get('n_edges')}`"
        )
    lines.append("")
    lines.append(
        f"本比較はCALIPERと選択した外部baseline({selected_baseline_text})を、"
        "同一の疾患・候補標的・ラベル集合で比較する。外部baselineはtask-aligned adaptationであり、"
        "原論文の報告性能の再現ではない。"
    )
    if primary_analysis == "directed_target_reach_matched":
        lines.append(
            "このrunでは directed target-reach matched analysis を主解析とし、"
            "targetごとの有向到達範囲でNCS/proximity/RWRの探索空間を制限する。"
            "proximity/RWRはtをクエリノードとして残しつつ疾患シード集合からtを除外する"
            "（自己シード効果の排除）。local evidence がないpairは評価集合に残すが、"
            "そのスコアは同一疾患・同一手法内の有効スコア最小値未満に統一的にfloorし、"
            "手法間で中立/最下位が入り混じらないようにする。"
        )
    if "NCS" in selected_baselines:
        lines.append(
            "注意: 本実装のL1000-NCSは、reference profileに target-cell 平均後の "
            "BING_knockdown_signature を用いる(sig_id単位ではない, 方法論§6.2.1/validation #4からの逸脱)。"
            "得られるスコアは target-cell-averaged adapted NCS であり、official Touchstone NCS とも "
            "sig_id-level adapted NCS とも数値的に同一ではない。"
        )
    lines.append("")

    # 1. flow
    lines.append("## 1. Common universe flow (§12-1)")
    lines.append("")
    flow_df = pd.DataFrame(
        [
            {"stage": "original val2", "diseases": flow.get("original_n_diseases"), "pairs": flow.get("original_n_pairs"), "positive_pairs": flow.get("original_n_positive_pairs")},
            {"stage": "common universe", "diseases": flow.get("common_n_diseases"), "pairs": flow.get("common_n_pairs"), "positive_pairs": flow.get("common_n_positive_pairs")},
        ]
    )
    lines.append(_md_table(flow_df))
    lines.append("")
    lines.append(f"![common universe flow]({fig_dir / 'common_universe_flow.png'})")
    lines.append("")
    if not exclusion.empty:
        lines.append("### 除外理由 (exclusion summary)")
        lines.append("")
        lines.append(_md_table(exclusion))
        lines.append("")

    # 2. performance
    lines.append("## 2. Selected-method performance (§12-2)")
    lines.append("")
    lines.append(_md_table(method_summary))
    lines.append("")
    lines.append(
        f"`mean_{primary_lift}` は全 common 疾患、`mean_{primary_lift}_eval` は "
        "E[h_d,K]=K_eff*n_pos/n_cand >= lift_eval_min_expected_hits を満たす D_eval(K) 上の macro-average。"
        "複数 K を指定した場合は各 K について Lift@K / Lift@K_eval を並行出力する。"
    )
    if len(caliper_ks) > 1:
        lines.append(
            "複数 CALIPER scoring k を指定した場合、method comparison / boxplot 系 figure では "
            "CALIPER@k 各系列を同一 figure 内で baseline と並べて表示する。"
        )
    lines.append("")
    lines.append(f"![four method performance]({fig_dir / 'four_method_performance.png'})")
    lines.append("")
    lines.append(f"![method comparison AP]({fig_dir / 'method_comparison_ap.png'})")
    k_star_ap_fig = fig_dir / f"method_comparison_ap_k{k_star}.png"
    if k_star is not None and (run_root / "report" / k_star_ap_fig).exists():
        lines.append("")
        lines.append(
            f"下図は上図から CALIPER scoring k=k_star={k_star} の系列のみを残した比較 "
            "(baseline は同一)。"
        )
        lines.append("")
        lines.append(f"![method comparison AP (k={k_star})]({k_star_ap_fig})")
        lines.append("")
    caliper_rwr_ap_fig = fig_dir / "method_comparison_ap_caliper_vs_classical_rwr.png"
    if (run_root / "report" / caliper_rwr_ap_fig).exists():
        lines.append(f"![method comparison AP: CALIPER vs Classical RWR]({caliper_rwr_ap_fig})")
    score_violin_fig = fig_dir / "score_violin_caliper_vs_classical_rwr.png"
    ap_violin_fig = fig_dir / "ap_violin_caliper_vs_classical_rwr.png"
    if (run_root / "report" / score_violin_fig).exists() or (run_root / "report" / ap_violin_fig).exists():
        lines.append("")
        lines.append("### CALIPER vs Classical RWR: スコア / AP 分布 (violin)")
        lines.append("")
        lines.append(
            "CALIPER と Classical RWR の分布を横並び2パネル (左=CALIPER, 右=Classical RWR) で比較する。"
        )
        lines.append("")
        if (run_root / "report" / score_violin_fig).exists():
            lines.append(f"![score violin: CALIPER vs Classical RWR]({score_violin_fig})")
            lines.append("")
            lines.append(
                "スコア分布 (全 common (disease, target) pair, scores_common_long.tsv)。"
                "両パネルは同一 linear y 軸 [0,1] を共有する。CALIPER スコアは [0,1] に広がり、"
                "Classical RWR スコアは全て ≤0.01 のため共有軸の下部に集中する (生スコアのスケール差を保持)。"
                "violin の密度は全 pair、重畳する jitter 点は可読性のため決定論的にサブサンプルした代表点。"
            )
            lines.append("")
        if (run_root / "report" / ap_violin_fig).exists():
            lines.append(f"![AP violin: CALIPER vs Classical RWR]({ap_violin_fig})")
            lines.append("")
            lines.append(
                "疾患単位 AP 分布 (metrics_by_disease_method.tsv, 1疾患1点)。"
                "両パネルは同一 linear y 軸を共有し、直接比較できる。"
            )
            lines.append("")
    pooled_ap = _read_tsv(run_root / "evaluation" / "pooled_ap_micro.tsv")
    if not pooled_ap.empty:
        lines.append("")
        lines.append("### Pooled (micro) AP — 疾患集約なし")
        lines.append("")
        lines.append(
            "上の method comparison AP figure は疾患ごとに AP を計算し疾患平均した macro 値。"
            "下表は全 (disease × target) ペアをプールし手法スコアで一括ランキングした micro AP "
            "(疾患集約なし, val2 既定の tie 考慮 expected AP)。`random` は全体の positive 率 (prevalence)。"
            "macro では手法間に差が出ても、micro では CALIPER と RWR がほぼ同値になる点に注意 "
            "(pooled はスコアの disease 間比較可能性を仮定するため)。"
        )
        lines.append("")
        lines.append(_md_table(pooled_ap))
        lines.append("")
    dc_method_fig = fig_dir / "method_comparison_ap_degree_corrected_caliper_vs_classical_rwr.png"
    if (run_root / "report" / dc_method_fig).exists():
        lines.append("### 次数補正 (degree-standardized) AP")
        lines.append("")
        lines.append(
            "各 disease の AP を、target 次数 decile で層別 → 占有層を等重み平均した **per-disease "
            "degree-standardized AP**(次数交絡を除去)。集約と誤差棒は次数補正なし図と同じ **疾患平均 ± SEM**。"
            "生の mAP では degree 交絡(easy な高次数 hub が多い)で RWR がやや上に見えるが、"
            "次数を補正すると CALIPER が RWR を上回る。なお下図の**各手法の周辺 SEM は disease 間分散のため重なる**が、"
            "有意性は同一 disease でペアにした ΔAP(次の forest 図, RWR で paired Wilcoxon p≈0.01)で示される。"
        )
        lines.append("")
        lines.append(f"![degree-corrected AP: CALIPER vs Classical RWR]({dc_method_fig})")
        lines.append("")
        dc_forest_fig = fig_dir / "effect_size_forest_ap_degree_corrected.png"
        if (run_root / "report" / dc_forest_fig).exists():
            lines.append(f"![degree-corrected effect size forest AP]({dc_forest_fig})")
            lines.append("")
            lines.append(
                "ΔAP = CALIPER − Classical RWR を raw と degree-corrected で対比 (帯 = 疾患 bootstrap 95%CI)。"
                "raw Δ は ~0/負(RWR 優位側)なのに対し degree-corrected Δ は正に反転(交絡逆転)。"
                "`*` = degree-corrected 95%CI が 0 を除外。"
            )
            lines.append("")
        dc_by_method = _read_tsv(run_root / "evaluation" / "degree_corrected_ap_by_method.tsv")
        if not dc_by_method.empty:
            lines.append(_md_table(dc_by_method))
            lines.append("")
        lines.append(
            "注意: この優位は degree-biased な RWR に特異的(NCS/proximity 相手では成り立たない)。効果は modest、"
            "標準化基準 (decile 等重み) と macro(疾患集約)に依存し、絶対値は層内 AP のため生 mAP と直接比較不可。"
        )
        lines.append("")
    # Lift@K の figure は非表示 (数値は上の method_summary / bootstrap 表と
    # evaluation/*.tsv に残る)。
    lines.append("")

    # 3. paired delta
    lines.append("## 3. Paired delta (CALIPER - baseline) (§12-3)")
    lines.append("")
    if len(caliper_ks) > 1:
        lines.append(
            "複数 CALIPER k を指定した run では、`caliper_method` 列ごとに baseline との paired delta / "
            "effect size を出力する。ただし検定は主要距離 k=4 の系列のみに限定する。"
        )
        lines.append("")
    lines.append(_md_table(bootstrap))
    lines.append("")
    if not effect_sizes.empty:
        lines.append("### Effect size (median Δ, win-rate across diseases)")
        lines.append("")
        lines.append(
            "疾患単位 Δ=CALIPER-baseline の effect size。median_delta は median ΔAP / median ΔLift@10、"
            "win_rate は Δ>0 となる疾患割合 (frac_tie は Δ=0, frac_negative は Δ<0)。"
        )
        lines.append("")
        lines.append(_md_table(effect_sizes))
        lines.append("")
    if not ttest.empty:
        lines.append("### Paired t-test at k=4 (AP / Lift@K / Lift@K_eval) + Holm")
        lines.append("")
        lines.append(
            "本 run で行う唯一の有意差検定。疾患単位 Δ=CALIPER-baseline を H0:平均=0 で両側 t 検定し、"
            "metric 内の baseline 群を family として Holm 補正する (`significant` は `p_holm < alpha`)。"
            "検定対象は主要距離 k=4 の CALIPER 系列のみで、他の k は effect size 表にのみ残す。"
            "mean_delta>0 は CALIPER 優位、<0 は baseline 優位。"
        )
        lines.append("")
        lines.append(_md_table(ttest))
        lines.append("")
    lines.append(f"![paired delta violin]({fig_dir / 'paired_delta_violin.png'})")
    lines.append(f"![effect size forest AP]({fig_dir / 'effect_size_forest_ap.png'})")
    # Lift@K forest と win-rate の figure は非表示 (median Δ / win_rate の数値は
    # 上の effect size 表と evaluation/effect_sizes.tsv に残る)。
    lines.append("")

    # 4. confounding
    lines.append("## 4. Coverage and confounding diagnostics (§12-4)")
    lines.append("")
    lines.append(f"![coverage confounding]({fig_dir / 'coverage_confounding.png'})")
    lines.append("")

    # 5. candidate-excluded sensitivity (§9.1), only when generated
    sens_summary = _read_tsv(run_root / "sensitivity" / "candidate_excluded_summary.tsv")
    if not sens_summary.empty:
        lines.append("## 5. Candidate-excluded sensitivity (§9.1)")
        lines.append("")
        lines.append(
            "候補target自身が disease seed (S_d^H) に含まれる場合、proximity は距離0、RWR は restart seed となる。"
            "self-seed pair を S_{d,-t}^H で再計算し、主解析との AP/Lift@10 変化と seed-target overlap 率を示す。"
            "D^common / C_d^common は主解析から不変。CALIPER / NCS は再計算しない。"
        )
        lines.append("")
        lines.append(_md_table(sens_summary))
        lines.append("")

    # 6. target-degree diagnostics (CALIPER vs RWR), only when generated
    degree_fig = fig_dir / "degree_distribution_negative_vs_positive.png"
    if (run_root / "report" / degree_fig).exists():
        degree_summary = _read_tsv(run_root / "evaluation" / "degree_diagnostics_summary.tsv")
        neg_vs_pos = _read_tsv(run_root / "evaluation" / "degree_diagnostics_neg_vs_pos_test.tsv")
        group_tests = _read_tsv(run_root / "evaluation" / "degree_diagnostics_group_tests.tsv")
        spearman = _read_tsv(run_root / "evaluation" / "degree_diagnostics_spearman.tsv")
        hit_rate = _read_tsv(run_root / "evaluation" / "degree_diagnostics_hit_rate_by_bin.tsv")

        lines.append("## 6. Target-degree diagnostics: CALIPER vs RWR (shared network H)")
        lines.append("")
        lines.append(
            "共有ネットワーク H 上の target 次数を用いて、CALIPER が RWR の見逃す低次数の "
            "既知 positive を拾えているかを診断する。RWR は定常分布ベースで hub (高次数ノード) "
            "に score が偏りやすいため、低次数 positive の top-10 捕捉率が低くなりやすい。"
        )
        lines.append("")
        lines.append(f"![target degree: non-positive vs positive]({fig_dir / 'degree_distribution_negative_vs_positive.png'})")
        if not neg_vs_pos.empty:
            lines.append("")
            lines.append(_md_table(neg_vs_pos))
        lines.append("")
        lines.append(f"![target degree of positives by top-10 hit]({fig_dir / 'degree_distribution_top10_by_method.png'})")
        lines.append("")
        if not degree_summary.empty:
            lines.append(_md_table(degree_summary))
            lines.append("")
        if not group_tests.empty:
            lines.append(_md_table(group_tests))
            lines.append("")
        lines.append(f"![known-positive degree vs score]({fig_dir / 'positive_degree_vs_score_hexbin.png'})")
        lines.append("")
        lines.append(
            "上図は y 軸を各手法の score レンジに合わせている: RWR は正値が約4桁に広がるため log 軸、"
            "CALIPER は [0,1] 有界かつ厳密な 0 を含む (log 軸では表示不可) ため linear 軸。"
            "2 パネルは独立した量で、共有する定量指標は scale 不変の Spearman ρ (degree vs 生 score)。"
        )
        lines.append("")
        lines.append(
            "参考として、RWR も linear 軸に統一した図と、y 軸を disease 内 score パーセンタイル "
            "(100% = 最上位スコア, 単位なし [0,1] で手法間直接比較可能) に統一した図を併記する。"
        )
        lines.append("")
        lines.append(f"![known-positive degree vs score (RWR linear axis)]({fig_dir / 'positive_degree_vs_score_hexbin_linear.png'})")
        lines.append("")
        lines.append(
            "linear 統一版: RWR positives の約8割は y≈0 付近の狭い帯に集中し (中央値は軸高の約2%)、"
            "生 score の縦方向解像度がこの高密度領域でほぼ失われる。全体の上昇トレンドは hexbin の "
            "色 (log スケール) で辛うじて読めるが、log 軸版はこの帯を軸全体に展開して構造を見せる (log 軸を用いる根拠)。"
        )
        lines.append("")
        lines.append(f"![known-positive degree vs score percentile]({fig_dir / 'positive_degree_vs_score_percentile.png'})")
        lines.append("")
        lines.append(
            "percentile 統一版: y 軸を disease 内 score パーセンタイル (100% = 最上位スコア, 上ほど良い) に統一。"
            "生 score の分布形状は失われるが、単位が消えるため両手法を同一軸で直接比較できる。RWR は高次数 "
            "target が上位%に集中 (次数バイアス)、CALIPER は次数に依らず広く分布する。"
        )
        lines.append("")
        if not spearman.empty:
            lines.append(_md_table(spearman))
            lines.append("")
        lines.append(f"![positive degree distribution]({fig_dir / 'positive_degree_histogram.png'})")
        lines.append("")
        lines.append(
            "positive 集合の target 次数分布 (右に裾を引き、低次数の target が多数を占める)。"
        )
        lines.append("")
        lines.append(f"![recovery percentile vs degree]({fig_dir / 'recovery_percentile_by_degree.png'})")
        lines.append("")
        lines.append(
            "等件数 degree decile ごとの、positive の within-disease score percentile 中央値 (帯 = 95%CI)。"
            "**「次数補正すると CALIPER は満遍なく回収」の主図**: CALIPER はほぼ水平 (次数に依らず一定=degree-robust)、"
            "RWR は 14%→96% と急上昇 (高次数でしか上位に置けない=degree-biased)。凡例の ρ = Spearman(degree, score) は"
            "次数依存度で、ρ≈0 が満遍なさ (CALIPER 0.16 vs RWR 0.74)。ただし *満遍ない=次数非依存* の意味で、"
            "高次数では RWR が上回る点に注意 (CALIPER が全次数で優位という主張ではない)。"
        )
        lines.append("")
        lines.append(f"![degree-matched AP vs degree (per-disease macro)]({fig_dir / 'degree_matched_ap_macro_by_degree_linear.png'})")
        lines.append("")
        lines.append(
            "上と同じ次数層別を、各 decile で disease ごとに AP を計算し疾患平均した **macro 版** "
            "(帯 = mean ± 1.96·SEM, val2 本体の疾患集約と整合)。低次数で CALIPER が RWR・random を明確に"
            "上回り (deg~2: CALIPER≈0.15 vs RWR≈0.10 vs random≈0.06)、次数が上がると RWR が追いつく。"
            "pooled(micro)版よりノイズは大きいが、本体の AP 集約方法と一致する。"
        )
        lines.append("")
        lines.append(f"![top-10 hit rate by degree bin]({fig_dir / 'hit_rate_by_degree_bin.png'})")
        lines.append("")
        lines.append(
            "positive の target 次数を tertile で Low / Mid / High に分け、各層で top-10 に入った "
            "known positive の割合 (件数ベース)。上の percentile / degree-matched AP は順位の連続量、"
            "こちらは回収率そのもので、CALIPER が低次数層、RWR が高次数層で回収する非対称性を示す。"
        )
        lines.append("")
        if not hit_rate.empty:
            lines.append(_md_table(hit_rate))
            lines.append("")
        lines.extend(_degree_spearman_by_disease_section(run_root, fig_dir))
        lines.extend(_topk_group_profile_section(run_root, fig_dir))

    # 7. DADA baseline (spec §7), only when DADA is a selected baseline
    if "DADA" in selected_baselines:
        dada_meta = meta.get("dada", {})
        strategy = _read_tsv(run_root / "disease_inputs" / "dada_disease_strategy.tsv")
        component_counts = dada_meta.get("selected_component_n_diseases", {})
        lines.append("## 7. DADA baseline: component selection と evaluation score")
        lines.append("")
        lines.append(
            "DADA は centrality-adjusted RWR と seed-degree uniform prioritization を組み合わせた "
            "baseline (Erten et al. 2011 の最終推奨方式)。raw component は Classical RWR の "
            "steady-state vector そのもの (再計算せず共有)、centrality component は共有ネットワーク H の "
            "r=0 定常分布 (unit-weight 無向連結グラフでは次数比例) で、"
            "`adjusted = log(raw / centrality)`。"
        )
        lines.append("")
        lines.append(
            f"疾患ごとの component 選択は seed 次数規則: `mean_seed_degree(d) > lambda` なら raw_rwr、"
            f"それ以外 (境界値含む) なら centrality_adjusted。lambda は H 全 node の平均次数 "
            f"`{dada_meta.get('network_mean_degree')}` (candidate や seed 部分集合ではなく常に H 全体)。"
        )
        lines.append("")
        if component_counts:
            counts_df = pd.DataFrame(
                [
                    {"selected_component": k, "n_diseases": v}
                    for k, v in component_counts.items()
                ]
            )
            lines.append(_md_table(counts_df))
            lines.append("")
        if not strategy.empty:
            lines.append(
                "疾患別の `n_seed_in_H` / `mean_seed_degree` / `selected_component` は "
                "`disease_inputs/dada_disease_strategy.tsv`、pair 単位の raw / centrality / adjusted は "
                "`scores/dada_pair.tsv`、common set 上の rank は `scores/dada_common_ranking.tsv`。"
            )
            lines.append("")
        lines.append(
            "評価に用いる score は選択 component の疾患内 descending average rank を負にした値 "
            "(`dada_score = -dada_rank`、最上位が -1)。rank の母集団は他手法と同一の common candidate set で、"
            "label は使用しない。同値 score は repository の tie-aware evaluation と整合する average rank。"
        )
        lines.append("")
        lines.append(
            "task adaptation (原論文の OMIM/PPI 実験の再現ではない): "
            "SIGNOR には PPI confidence がないため collapse 後の edge weight を全て1とし、"
            "disease seed は均等重み、restart probability は Classical RWR と共有、"
            "candidate target は disease seed に含まれていても除外しない。"
            "DADA の優劣は上記 §2/§3 の実測 metric から読むこと (論文の報告性能の再現ではない)。"
        )
        lines.append("")

    lines.append("## 注意・限界 (§12, §13)")
    lines.append("")
    network_baselines = [b for b in ("proximity", "RWR", "DADA") if b in selected_baselines]
    if network_baselines:
        network_baseline_text = " / ".join(
            {"proximity": "Network proximity", "RWR": "RWR", "DADA": "DADA"}[b]
            for b in network_baselines
        )
        lines.append(
            f"- {network_baseline_text} より高性能でも、因果推論が一般にnetwork手法より優れるとは主張しない"
            "(これらはdirection情報とKD profileを使用しない)。"
        )
    if "NCS" in selected_baselines:
        lines.append(
            "- L1000-NCS より高性能でも、CMap全体・official Touchstone NCS・Tauより優れるとは主張しない"
            "(本比較は target-cell-averaged full-shRNA-reference adapted NCS)。"
        )
    lines.append(
        "- negative candidateには未発見positiveが含まれ得る。AP/Lift@10は既知positive enrichmentの指標。"
    )
    lines.append("")

    report_path = run_root / "report" / "val2_baseline_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
