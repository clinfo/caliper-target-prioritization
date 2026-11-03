from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .viz import FAMILY_FIGURE_LABELS, MIN_EVALUABLE_ROWS, load_paired_tables, make_all_figures


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _fmt_value(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _md_table(df: pd.DataFrame, *, max_rows: int | None = None) -> list[str]:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return ["_該当する行はありません。_", ""]
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [_fmt_value(row[col]) for col in cols]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    return lines


def _with_figure_family_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "family" in out.columns:
        out["family"] = out["family"].astype(str).map(
            lambda value: FAMILY_FIGURE_LABELS.get(value, value.replace("_", " "))
        )
    return out


def _bullets(items: Iterable[str]) -> list[str]:
    return [f"- {item}" for item in items] + [""]


def _append_figure(lines: list[str], title: str, rel_png: str, notes: Iterable[str]) -> None:
    lines.extend([f"## {title}", ""])
    lines.extend(_bullets(notes))
    lines.extend([f"![{title}]({rel_png})", ""])


def _relative_to_report(path: str | Path, report_md: Path) -> str:
    p = _as_path(path)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    report_parent = report_md.parent
    if not report_parent.is_absolute():
        report_parent = (Path.cwd() / report_parent).resolve()
    try:
        return p.relative_to(report_parent).as_posix()
    except ValueError:
        return os.path.relpath(p, report_parent)


_CLUSTER_SUMMARY_COLS = [
    "family",
    "delta_f_cluster",
    "null_mean_f_cluster",
    "null_sd_f_cluster",
    "p_empirical_cluster",
    "z_f_cluster",
    "ci_low_cluster",
    "ci_high_cluster",
    "n_evaluable_rows",
    "n_perturbations",
]
_ROW_SUMMARY_COLS = [
    "family",
    "delta_f",
    "null_mean_f",
    "null_sd_f",
    "p_empirical",
    "z_f",
    "ci_low",
    "ci_high",
    "n_evaluable_rows",
    "n_perturbations",
]


def write_val4_h1_report(
    *,
    run_root: str | Path,
    output_md: str | Path | None = None,
    figures_dir: str | Path | None = None,
) -> dict[str, str]:
    """Write a paired-delta Markdown report for Val4/H1 visualization."""
    run_root = Path(run_root)
    report_dir = run_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_path = Path(figures_dir) if figures_dir is not None else report_dir / "figures"
    output_path = Path(output_md) if output_md is not None else report_dir / "val4_h1_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure_outputs = make_all_figures(run_root=run_root, figures_dir=figures_path)
    tables = load_paired_tables(run_root)

    summary_df = tables["summary_df"]
    summary_cluster_df = tables["summary_cluster_df"]
    null_df = tables["null_df"]
    null_cluster_df = tables["null_cluster_df"]
    summary_df_full = tables["summary_df_full"]
    summary_cluster_df_full = tables["summary_cluster_df_full"]
    excluded_row = tables["excluded_row"]
    excluded_cluster = tables["excluded_cluster"]

    lines: list[str] = []
    lines.extend(
        [
            "# Val4/H1: RNA observability report",
            "",
            f"実行ディレクトリ: `{run_root}`",
            "",
        ]
    )
    lines.extend(
        _bullets(
            [
                "方式: mechanism-family RNA observability analysis。",
                "main は cluster-level paired observability delta (per-key delta → mean)。補助は row-level delta (per-row delta → mean)。",
                "Δ observability は観測 observability と profile-sign null 反復平均の差。bootstrap CI と経験 p 値で評価。",
            ]
        )
    )
    lines.extend(
        [
            "```text",
            f"[フィルタ] MIN_EVALUABLE_ROWS={MIN_EVALUABLE_ROWS}",
            f"[フィルタ] main (cluster) family 数 : {len(summary_cluster_df)} / {len(summary_cluster_df_full)}",
            f"[フィルタ] 補助 (row) family 数 : {len(summary_df)} / {len(summary_df_full)}",
        ]
    )
    if excluded_cluster:
        lines.append(f"[フィルタ] cluster 除外 family : {excluded_cluster}")
    if excluded_row:
        lines.append(f"[フィルタ] row 除外 family : {excluded_row}")
    lines.extend(["```", ""])

    _append_figure(
        lines,
        "Mechanism-family edge support",
        _relative_to_report(figures_path / "fig_val4h1_family_physical_edges.png", output_path),
        [
            "見方: `val4_family_eligibility.csv` の `n_physical_edges` を mechanism family ごとに表示。",
            "見方: 黒の点線は SIGNOR KG の edge-support threshold = 100。",
            "見方: 点線より右の family は edge-support threshold を満たす。",
        ],
    )

    lines.extend(["## main: cluster-level RNA observability summary", ""])
    lines.extend(
        _bullets(
            [
                "見方: per-key Δ observability を $(t,c)$ について等重み平均した family 効果量。",
                "見方: `delta_f_cluster` が観測 observability - profile-sign null 反復平均。",
                "見方: `null_mean_f_cluster` / `null_sd_f_cluster` は null range の要約。",
                "見方: `ci_low_cluster` / `ci_high_cluster` は bootstrap 95% CI、`n_perturbations` は perturbation keys。",
            ]
        )
    )
    lines.extend(_md_table(_with_figure_family_labels(summary_cluster_df[_CLUSTER_SUMMARY_COLS])))

    _append_figure(
        lines,
        "main: cluster-level Δ observability forest plot",
        _relative_to_report(figures_path / "fig_val4h1_delta.png", output_path),
        [
            "見方: family ごとの `delta_f_cluster` と bootstrap 95% CI を表示。",
            "見方: 0 より右は profile-sign null 反復平均より高い RNA observability、左は低い。",
        ],
    )
    _append_figure(
        lines,
        "main: per-key Δ observability histogram",
        _relative_to_report(figures_path / "fig_val4h1_cluster_delta_hist.png", output_path),
        [
            "見方: family ごとに per-key Δ observability $\\delta_{t,c,f}$ の分布を表示。",
            "見方: 青の縦線は $\\Delta_f^{\\mathrm{cluster}}$、青の帯は bootstrap 95% CI、灰の点線は 0。",
            "見方: ヒストグラムは [-0.5, 0.5] にクリップして表示。",
        ],
    )

    lines.extend(["## 補助: row-level RNA observability summary", ""])
    lines.extend(
        _bullets(
            [
                "見方: per-row Δ observability を $(t,c,g)$ について等重み平均した family 効果量。",
                "見方: `delta_f` は row-level Δ observability の mean。",
                "見方: `n_evaluable_rows` は evaluable rows、`n_perturbations` は perturbation keys。",
            ]
        )
    )
    lines.extend(_md_table(_with_figure_family_labels(summary_df[_ROW_SUMMARY_COLS])))

    _append_figure(
        lines,
        "補助: row-level Δ observability forest plot",
        _relative_to_report(figures_path / "fig_val4h1_row_delta.png", output_path),
        [
            "見方: family ごとの `delta_f` と bootstrap 95% CI を表示。",
            "見方: 0 より右は profile-sign null 反復平均より高い RNA observability、左は低い。",
        ],
    )
    _append_figure(
        lines,
        "補助: per-row Δ observability histogram",
        _relative_to_report(figures_path / "fig_val4h1_row_delta_hist.png", output_path),
        [
            "見方: family ごとに per-row Δ observability $\\delta_{t,c,g,f}$ の分布を表示。",
            "見方: 青の縦線は $\\Delta_f$、青の帯は bootstrap 95% CI、灰の点線は 0。",
            "見方: ヒストグラムは [-0.5, 0.5] にクリップして表示。",
        ],
    )

    _append_figure(
        lines,
        "Mechanism-family sample size",
        _relative_to_report(figures_path / "fig_val4h1_n_by_family.png", output_path),
        [
            "見方: family ごとの evaluable rows と perturbation keys を確認。",
            "見方: 横軸は件数のログ表示、縦軸は family。",
            "見方: family 間のサンプルサイズ差を見る。",
        ],
    )

    lines.extend(["## null 反復・bootstrap 反復ダンプ (先頭)", ""])
    lines.extend(
        _bullets(
            [
                "見方: profile-sign null 反復 $r$ ごとの $\\Delta_f^{(r)}$ (cluster / row)。",
                "見方: 構成上、null range は 0 を中心に分布。",
            ]
        )
    )
    lines.extend(["### cluster null", ""])
    lines.extend(_md_table(_with_figure_family_labels(null_cluster_df.head())))
    lines.extend(["### row null", ""])
    lines.extend(_md_table(_with_figure_family_labels(null_df.head())))

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {"report_md": str(output_path), **figure_outputs}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Val4/H1 paired-delta Markdown report.")
    parser.add_argument("--run-root", required=True, help="output/val4/h1/<run_name> directory")
    parser.add_argument("--output-md", default=None, help="Optional report Markdown output path")
    parser.add_argument("--figures-dir", default=None, help="Optional figure directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = write_val4_h1_report(
        run_root=args.run_root,
        output_md=args.output_md,
        figures_dir=args.figures_dir,
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
