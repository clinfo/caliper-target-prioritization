from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .viz import FAMILY_FIGURE_LABELS, make_all_figures


def _md_table(df: pd.DataFrame, max_rows: int = 30) -> list[str]:
    if df is None or len(df) == 0:
        return ["_該当する行はありません。_", ""]
    return [df.head(max_rows).to_markdown(index=False), ""]


def _with_figure_family_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "family" in out.columns:
        out["family"] = out["family"].astype(str).map(
            lambda value: FAMILY_FIGURE_LABELS.get(value, value.replace("_", " "))
        )
    return out


def _bullets(items: Iterable[str]) -> list[str]:
    return [f"- {item}" for item in items] + [""]


def _relative(path: str | Path, base: Path) -> str:
    p = Path(path)
    try:
        return p.relative_to(base.parent).as_posix()
    except ValueError:
        return p.as_posix()


def _metric_path(run_root: Path, filename: str) -> Path:
    path = run_root / "metrics" / filename
    if path.exists():
        return path
    return run_root / filename


def write_val4_h2_report(
    *,
    run_root: str | Path,
    output_md: str | Path | None = None,
    figures_dir: str | Path | None = None,
) -> dict[str, str]:
    run_root = Path(run_root)
    report_dir = run_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_path = Path(figures_dir) if figures_dir is not None else report_dir / "figures"
    output_path = Path(output_md) if output_md is not None else report_dir / "val4_h2_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_outputs = make_all_figures(run_root=run_root, figures_dir=figures_path)

    lines: list[str] = [
        "# Val4/H2: Family contribution report",
        "",
        f"実行ディレクトリ: `{run_root}`",
        "",
    ]
    lines.extend(["## LOFO contribution", ""])
    lines.extend(
        _bullets(
            [
                "方式: leave-one-family-out ranking contribution。",
                "見方: 対象 mechanism family を含む物理 KG edge を除外したときの target ranking 性能変化。",
            ]
        )
    )
    lofo_summary_path = _metric_path(run_root, "val4_h2_lofo_family_summary.csv")
    if lofo_summary_path.exists():
        lines.extend(["### LOFO summary", ""])
        lines.extend(
            _bullets(
                [
                    "見方: 行は mechanism family、列は family 除外時の AP 変化量と要約統計。",
                    "見方: `delta_AP_null_adjusted_median` は crosswalk の `null-adjusted ΔAP`。",
                    "見方: 正の `null-adjusted ΔAP` は、除外で ranking 性能が低下する方向の変化。",
                ]
            )
        )
        lines.extend(_md_table(_with_figure_family_labels(pd.read_csv(lofo_summary_path))))
    if "h2_lofo_delta_ap" in figure_outputs:
        lines.extend(["### LOFO contribution figure", ""])
        lines.extend(
            _bullets(
                [
                    "見方: 横軸は `null-adjusted ΔAP`、縦軸は mechanism family。",
                    "見方: 右側に大きい family ほど、除外時に target ranking 性能が下がる傾向。",
                    "見方: エラーバーは bootstrap CI。",
                ]
            )
        )
        lines.extend([f"![LOFO contribution]({_relative(figure_outputs['h2_lofo_delta_ap'], output_path)})", ""])

    lines.extend(["## 補助解析: family sign shuffle", ""])
    lines.extend(
        _bullets(
            [
                "見方: sign shuffle は物理 KG の接続構造を保ち、同じ family 内で符号だけ入れ替える。",
                "見方: family 除去ではなく、符号情報への依存を調べる補助診断。",
            ]
        )
    )
    sign_summary_path = _metric_path(run_root, "val4_h2_sign_shuffle_family_summary.csv")
    if sign_summary_path.exists():
        lines.extend(["### family sign shuffle summary", ""])
        lines.extend(
            _bullets(
                [
                    "見方: 行は family、列は family 内 sign shuffle による AP 変化量と要約統計。",
                    "見方: 変化が大きい family は、edge の存在に加えて符号情報が ranking に関係する可能性。",
                ]
            )
        )
        lines.extend(_md_table(_with_figure_family_labels(pd.read_csv(sign_summary_path))))
    if "h2_sign_shuffle_delta_ap" in figure_outputs:
        lines.extend(["### family sign shuffle figure", ""])
        lines.extend(
            _bullets(
                [
                    "見方: 横軸は sign shuffle による AP 変化量、縦軸は family。",
                    "見方: 0 から離れる family ほど、符号入れ替えで ranking 結果が変化しやすい。",
                    "見方: 符号依存性の診断であり、family 除去効果とは分けて解釈。",
                ]
            )
        )
        lines.extend([f"![family sign shuffle]({_relative(figure_outputs['h2_sign_shuffle_delta_ap'], output_path)})", ""])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    outputs = {"h2_report": str(output_path)}
    outputs.update({f"figure_{k}": v for k, v in figure_outputs.items()})
    return outputs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Val4/H2 Markdown report.")
    parser.add_argument("--run-root", required=True, help="output/val4/h2/<run_name> directory")
    parser.add_argument("--output-md", default=None, help="Optional report Markdown output path")
    parser.add_argument("--figures-dir", default=None, help="Optional figure directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = write_val4_h2_report(
        run_root=args.run_root,
        output_md=args.output_md,
        figures_dir=args.figures_dir,
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
