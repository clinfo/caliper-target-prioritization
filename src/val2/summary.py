from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


# Fixed primary distance k of the main analysis (k* selected label-free in Val1).
PRIMARY_K = 4

# Methods presented in the fixed-k internal ablation panel. The computed
# `noKG` metric is retained in analysis outputs but omitted from paper-facing
# Val2 figures and reports. `ncs` (L1000 NCS) remains an external baseline.
INTERNAL_ABLATION_METHODS = (
    "proposed",
    "kg_sign_noalign",
    "kg_nosign",
    "noKG_sizematch",
)


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def find_one(run_root: Path, pattern: str) -> Path | None:
    hits = sorted(run_root.glob(pattern))
    return hits[0] if hits else None


def read_json_or_empty(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def read_mapping_or_empty(path: Path) -> dict:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        return read_json_or_empty(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception:
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    return {}


def sem(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) <= 1:
        return 0.0
    return float(numeric.std(ddof=1) / np.sqrt(len(numeric)))


def resolve_k_valid(run_root: Path) -> list[int]:
    metadata = read_json_or_empty(run_root / "metadata.json")
    summary = read_json_or_empty(run_root / "run_summary.json")
    resolved_config = read_json_or_empty(run_root / "config" / "resolved_config.json")
    snapshot_config = read_mapping_or_empty(run_root / "config" / "val2.yaml")
    source_config_path = (metadata.get("resolved_inputs") or {}).get("config_path")
    source_config = read_mapping_or_empty(Path(source_config_path)) if source_config_path else {}

    candidate_paths = [
        (metadata.get("params") or {}).get("k_valid"),
        (metadata.get("params") or {}).get("K_valid"),
        summary.get("k_valid"),
        ((resolved_config.get("analysis") or {}).get("k_valid")),
        ((resolved_config.get("analysis") or {}).get("K_valid")),
        ((snapshot_config.get("analysis") or {}).get("k_valid")),
        ((snapshot_config.get("analysis") or {}).get("K_valid")),
        ((source_config.get("analysis") or {}).get("k_valid")),
        ((source_config.get("analysis") or {}).get("K_valid")),
        (metadata.get("params") or {}).get("ks"),
        summary.get("ks"),
        ((resolved_config.get("analysis") or {}).get("ks")),
    ]
    for raw in candidate_paths:
        if isinstance(raw, list) and raw:
            return [int(k) for k in raw]
    return []


def load_ablation_metrics(run_root: Path) -> pd.DataFrame:
    metrics_path = find_one(run_root, "ablation/val2_metrics_k*.csv")
    return read_csv_or_empty(metrics_path) if metrics_path else pd.DataFrame()


def summarize_ablation_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return one ablation row per (method, k) from legacy per-disease metrics."""
    if metrics.empty or not {"method", "k"}.issubset(metrics.columns):
        return pd.DataFrame()

    df = metrics.copy()
    group_cols = ["method", "k"]
    grouped = df.groupby(group_cols, as_index=False)
    rows = grouped.size().rename(columns={"size": "n_diseases"})

    value_frames: list[pd.DataFrame] = []
    if "mAP" in df.columns:
        value_frames.append(grouped["mAP"].mean(numeric_only=True).rename(columns={"mAP": "mAP"}))
        value_frames.append(df.groupby(group_cols)["mAP"].agg(sem).reset_index().rename(columns={"mAP": "mAP_SEM"}))
    elif "AP" in df.columns:
        value_frames.append(grouped["AP"].mean(numeric_only=True).rename(columns={"AP": "mAP"}))
        value_frames.append(df.groupby(group_cols)["AP"].agg(sem).reset_index().rename(columns={"AP": "mAP_SEM"}))

    if "mean_Lift@10" in df.columns:
        value_frames.append(
            grouped["mean_Lift@10"].mean(numeric_only=True).rename(columns={"mean_Lift@10": "mean_Lift@10"})
        )
        value_frames.append(
            df.groupby(group_cols)["mean_Lift@10"]
            .agg(sem)
            .reset_index()
            .rename(columns={"mean_Lift@10": "mean_Lift@10_SEM"})
        )
    elif "Lift@10" in df.columns:
        value_frames.append(grouped["Lift@10"].mean(numeric_only=True).rename(columns={"Lift@10": "mean_Lift@10"}))
        value_frames.append(
            df.groupby(group_cols)["Lift@10"]
            .agg(sem)
            .reset_index()
            .rename(columns={"Lift@10": "mean_Lift@10_SEM"})
        )

    if "Lift@20" in df.columns and "mean_Lift@20" not in df.columns:
        value_frames.append(grouped["Lift@20"].mean(numeric_only=True).rename(columns={"Lift@20": "mean_Lift@20"}))
        value_frames.append(
            df.groupby(group_cols)["Lift@20"]
            .agg(sem)
            .reset_index()
            .rename(columns={"Lift@20": "mean_Lift@20_SEM"})
        )

    out = rows
    for value_frame in value_frames:
        out = out.merge(value_frame, on=group_cols, how="left")

    for col in [
        "k",
        "n_diseases",
        "mAP",
        "mAP_SEM",
        "mean_Lift@10",
        "mean_Lift@10_SEM",
        "mean_Lift@20",
        "mean_Lift@20_SEM",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "method" in out.columns:
        out["method"] = out["method"].astype(str)

    sort_cols = [c for c in ["method", "k"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def load_ablation_summary(run_root: Path) -> pd.DataFrame:
    return summarize_ablation_metrics(load_ablation_metrics(run_root))


def summarize_k_valid_methods(metrics: pd.DataFrame, k_valid: list[int]) -> pd.DataFrame:
    if metrics.empty or not {"method", "k"}.issubset(metrics.columns) or not k_valid:
        return pd.DataFrame()
    df = metrics.copy()
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df = df[df["k"].isin([int(k) for k in k_valid])]
    if df.empty:
        return pd.DataFrame()

    rows = []
    for method, grp in df.groupby("method", sort=True):
        row: dict[str, object] = {
            "method": str(method),
            "k_valid": ",".join(str(int(k)) for k in k_valid),
            "n_rows": int(len(grp)),
        }
        if "disease_id" in grp.columns:
            row["n_diseases"] = int(grp["disease_id"].nunique())
        if "AP" in grp.columns:
            row["mean_AP"] = float(pd.to_numeric(grp["AP"], errors="coerce").mean())
            row["AP_SEM"] = sem(grp["AP"])
        if "Lift@10" in grp.columns:
            row["mean_Lift@10"] = float(pd.to_numeric(grp["Lift@10"], errors="coerce").mean())
            row["Lift@10_SEM"] = sem(grp["Lift@10"])
        rows.append(row)
    return pd.DataFrame(rows)


def load_k_valid_method_summary(run_root: Path) -> pd.DataFrame:
    return summarize_k_valid_methods(load_ablation_metrics(run_root), resolve_k_valid(run_root))


def summarize_fixed_k_internal_ablation(summary: pd.DataFrame, k: int = PRIMARY_K) -> pd.DataFrame:
    """Return the fixed-k rows of `summary` for the internal ablation comparison.

    Keeps only paper-facing `INTERNAL_ABLATION_METHODS` at distance `k`,
    ordered as in `INTERNAL_ABLATION_METHODS`.
    """
    if summary.empty or not {"method", "k"}.issubset(summary.columns):
        return pd.DataFrame()
    df = summary.copy()
    df["method"] = df["method"].astype(str)
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df = df[(df["k"] == int(k)) & df["method"].isin(INTERNAL_ABLATION_METHODS)]
    if df.empty:
        return pd.DataFrame()
    order = [m for m in INTERNAL_ABLATION_METHODS if m in set(df["method"])]
    df["method"] = pd.Categorical(df["method"], categories=order, ordered=True)
    out = df.sort_values("method").reset_index(drop=True)
    out["method"] = out["method"].astype(str)
    return out


def load_fixed_k_internal_ablation_summary(run_root: Path, k: int = PRIMARY_K) -> pd.DataFrame:
    return summarize_fixed_k_internal_ablation(load_ablation_summary(run_root), k)


def best_ablation_by_method(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "method" not in summary.columns:
        return pd.DataFrame()

    rows = []
    for method, grp in summary.groupby("method", sort=True):
        row: dict[str, object] = {"method": method}
        if "mAP" in grp.columns:
            values = pd.to_numeric(grp["mAP"], errors="coerce")
            if values.notna().any():
                best_map = grp.loc[values.idxmax()]
                row["best_k_by_mAP"] = best_map.get("k")
                row["best_mAP"] = best_map.get("mAP")
        if "mean_Lift@10" in grp.columns:
            values = pd.to_numeric(grp["mean_Lift@10"], errors="coerce")
            if values.notna().any():
                best_lift = grp.loc[values.idxmax()]
                row["best_k_by_mean_Lift@10"] = best_lift.get("k")
                row["best_mean_Lift@10"] = best_lift.get("mean_Lift@10")
        rows.append(row)
    return pd.DataFrame(rows)


def proposed_ablation_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or not {"method", "k"}.issubset(summary.columns):
        return pd.DataFrame()

    proposed = summary[summary["method"] == "proposed"]
    rows = []
    for _, prow in proposed.iterrows():
        k = prow["k"]
        others = summary[(summary["k"] == k) & (summary["method"] != "proposed")]
        for _, orow in others.iterrows():
            out = {"k": k, "comparison": f"proposed - {orow['method']}"}
            if "mAP" in summary.columns:
                out["delta_mAP"] = float(prow["mAP"]) - float(orow["mAP"])
            if "mean_Lift@10" in summary.columns:
                out["delta_mean_Lift@10"] = float(prow["mean_Lift@10"]) - float(orow["mean_Lift@10"])
            rows.append(out)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["k", "comparison"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def describe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "metric": col,
                "count": int(len(finite)),
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "median": float(finite.median()) if len(finite) else np.nan,
                "q25": float(finite.quantile(0.25)) if len(finite) else np.nan,
                "q75": float(finite.quantile(0.75)) if len(finite) else np.nan,
                "min": float(finite.min()) if len(finite) else np.nan,
                "max": float(finite.max()) if len(finite) else np.nan,
            }
        )
    return pd.DataFrame(rows)
