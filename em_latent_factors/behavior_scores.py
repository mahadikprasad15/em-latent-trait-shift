"""Collect per-run behavior eval aggregates into one canonical score table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from em_latent_factors.io import ensure_parent


REQUIRED_AGGREGATE_COLUMNS = {"model_id", "eval_id", "category", "mean_score", "std_score", "n"}


def collect_behavior_scores(
    runs_root: str | Path = "artifacts/runs",
    output_path: str | Path = "results/behavior_scores.csv",
    base_model_id: str = "base",
    category: str | None = "all",
    include_categories: bool = False,
    allow_missing_base: bool = False,
) -> dict[str, Any]:
    rows = load_aggregate_rows(runs_root=runs_root)
    if rows.empty:
        raise ValueError(f"no aggregate_scores.csv files found under {runs_root}")
    if not include_categories and category is not None:
        rows = rows[rows["category"].fillna("all") == category].copy()
    if rows.empty:
        raise ValueError(f"no behavior aggregate rows remain after category filter {category!r}")
    scores = finalize_behavior_scores(rows, base_model_id=base_model_id, allow_missing_base=allow_missing_base)
    write_behavior_scores(output_path, scores)
    return {
        "runs_root": str(runs_root),
        "output_path": str(output_path),
        "rows": int(len(scores)),
        "models": int(scores["model_id"].nunique()),
        "evals": sorted(scores["eval_id"].dropna().unique().tolist()),
        "base_model_id": base_model_id,
        "category": category if not include_categories else "all_categories",
    }


def load_aggregate_rows(runs_root: str | Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(Path(runs_root).glob("**/results/aggregate_scores.csv")):
        df = pd.read_csv(path)
        missing = REQUIRED_AGGREGATE_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        df = df.copy()
        run_dir = path.parents[1]
        df["run_id"] = run_dir.name
        df["aggregate_path"] = str(path)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["behavior_score"] = pd.to_numeric(out["mean_score"], errors="raise")
    out["std_score"] = pd.to_numeric(out["std_score"], errors="coerce")
    out["n"] = pd.to_numeric(out["n"], errors="coerce")
    return out


def finalize_behavior_scores(rows: pd.DataFrame, base_model_id: str, allow_missing_base: bool = False) -> pd.DataFrame:
    deduped = _dedupe_scores(rows)
    deduped = deduped.dropna(subset=["model_id", "eval_id", "behavior_score"]).copy()
    base = (
        deduped[deduped["model_id"] == base_model_id][["eval_id", "category", "behavior_score"]]
        .rename(columns={"behavior_score": "base_behavior_score"})
        .copy()
    )
    if base.empty:
        raise ValueError(f"no base model behavior rows found for model_id={base_model_id!r}")
    merged = deduped.merge(base, on=["eval_id", "category"], how="left")
    missing_base = merged["base_behavior_score"].isna()
    if missing_base.any():
        missing = merged.loc[missing_base, ["eval_id", "category"]].drop_duplicates().to_dict("records")
        if not allow_missing_base:
            raise ValueError(f"missing base scores for eval/category pairs: {missing[:10]}")
        merged = merged[~missing_base].copy()
    merged["delta_behavior"] = merged["behavior_score"] - merged["base_behavior_score"]
    ordered = [
        "model_id",
        "eval_id",
        "category",
        "behavior_score",
        "base_behavior_score",
        "delta_behavior",
        "mean_score",
        "std_score",
        "n",
        "judge_backend",
        "judge_model",
        "rubric_version",
        "run_id",
        "aggregate_path",
    ]
    extras = [col for col in merged.columns if col not in ordered]
    return merged[ordered + extras].sort_values(["eval_id", "category", "model_id"]).reset_index(drop=True)


def write_behavior_scores(path: str | Path, df: pd.DataFrame) -> None:
    path = ensure_parent(path)
    df.to_csv(path, index=False)


def _dedupe_scores(rows: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["model_id", "eval_id", "category"]
    rows = rows.copy()
    rows["_sort_n"] = pd.to_numeric(rows["n"], errors="coerce").fillna(-1)
    rows = rows.sort_values(["model_id", "eval_id", "category", "_sort_n", "run_id"])
    deduped = rows.drop_duplicates(key_cols, keep="last").drop(columns=["_sort_n"])
    return deduped
