"""Regression analyses for behavior deltas versus projection features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from em_latent_factors.io import ensure_parent


VECTOR_FEATURES = [
    "z_toxic_reckless",
    "z_deception",
    "z_hallucination",
    "z_sycophancy",
    "z_refusal",
    "z_harmful_advice",
    "z_insecure_code",
]

ALL_MODEL_FEATURES = VECTOR_FEATURES + ["shift_norm"]

MATCHING_FEATURES = {
    "eval_core_misalignment": ["z_toxic_reckless"],
    "eval_extended_misalignment_by_category": ["z_toxic_reckless"],
    "eval_hallucination_tool_deception": ["z_hallucination", "z_deception"],
    "eval_sycophancy": ["z_sycophancy"],
    "eval_sycophancy_answer": ["z_sycophancy"],
    "eval_strongreject_unsafe_compliance": ["z_harmful_advice", "z_refusal"],
    "eval_xstest_safe_overrefusal": ["z_refusal"],
    "eval_xstest_unsafe_refusal": ["z_refusal"],
    "eval_health_bad_advice": ["z_harmful_advice"],
    "eval_finance_risky_advice": ["z_harmful_advice"],
    "eval_code_insecurity": ["z_insecure_code"],
}


@dataclass(frozen=True)
class RegressionOutputs:
    regression_dataframe: pd.DataFrame
    correlation_matrix: pd.DataFrame
    matching_regressions: pd.DataFrame
    ridge_coefficients: pd.DataFrame
    multioutput_ridge_coefficients: pd.DataFrame
    baseline_regressions: pd.DataFrame


def run_regression_analysis(
    behavior_scores_path: str | Path,
    projections_path: str | Path,
    output_root: str | Path = "results",
    neutral_bank: str = "neutral_all",
    layer_aggregate: str = "middle_quantile_layer",
    base_model_id: str = "base",
    category: str = "all",
    ridge_alpha: float = 1.0,
) -> dict[str, Any]:
    behavior = read_behavior_scores(behavior_scores_path)
    projections = read_projection_features(projections_path, neutral_bank=neutral_bank, layer_aggregate=layer_aggregate)
    outputs = compute_regression_outputs(
        behavior=behavior,
        projections=projections,
        base_model_id=base_model_id,
        category=category,
        ridge_alpha=ridge_alpha,
    )
    output_root = Path(output_root)
    paths = {
        "regression_dataframe": output_root / "regression_dataframe.csv",
        "correlation_matrix": output_root / "correlation_matrix.csv",
        "matching_vector_regressions": output_root / "matching_vector_regressions.csv",
        "ridge_regression_coefficients": output_root / "ridge_regression_coefficients.csv",
        "multioutput_ridge_coefficients": output_root / "multioutput_ridge_coefficients.csv",
        "baseline_regressions": output_root / "baseline_regressions.csv",
    }
    write_df(paths["regression_dataframe"], outputs.regression_dataframe)
    write_df(paths["correlation_matrix"], outputs.correlation_matrix)
    write_df(paths["matching_vector_regressions"], outputs.matching_regressions)
    write_df(paths["ridge_regression_coefficients"], outputs.ridge_coefficients)
    write_df(paths["multioutput_ridge_coefficients"], outputs.multioutput_ridge_coefficients)
    write_df(paths["baseline_regressions"], outputs.baseline_regressions)
    return {
        "behavior_scores_path": str(behavior_scores_path),
        "projections_path": str(projections_path),
        "output_paths": {key: str(value) for key, value in paths.items()},
        "rows": int(len(outputs.regression_dataframe)),
        "models": int(outputs.regression_dataframe["model_id"].nunique()) if not outputs.regression_dataframe.empty else 0,
        "evals": sorted(outputs.regression_dataframe["eval_id"].unique().tolist()) if not outputs.regression_dataframe.empty else [],
        "neutral_bank": neutral_bank,
        "layer_aggregate": layer_aggregate,
    }


def read_behavior_scores(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"model_id", "eval_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required behavior columns: {sorted(missing)}")
    if "mean_score" not in df.columns and "behavior_score" not in df.columns:
        raise ValueError(f"{path} must contain mean_score or behavior_score")
    df = df.copy()
    if "behavior_score" not in df.columns:
        df["behavior_score"] = pd.to_numeric(df["mean_score"], errors="raise")
    else:
        df["behavior_score"] = pd.to_numeric(df["behavior_score"], errors="raise")
    if "category" not in df.columns:
        df["category"] = "all"
    return df


def read_projection_features(path: str | Path, neutral_bank: str, layer_aggregate: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"model_id", "neutral_bank", "layer_aggregate", *ALL_MODEL_FEATURES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required projection columns: {sorted(missing)}")
    selected = df[(df["neutral_bank"] == neutral_bank) & (df["layer_aggregate"] == layer_aggregate)].copy()
    if selected.empty:
        raise ValueError(f"no projection rows found for neutral_bank={neutral_bank!r}, layer_aggregate={layer_aggregate!r}")
    for col in ALL_MODEL_FEATURES:
        selected[col] = pd.to_numeric(selected[col], errors="coerce")
    return selected


def compute_regression_outputs(
    behavior: pd.DataFrame,
    projections: pd.DataFrame,
    base_model_id: str = "base",
    category: str = "all",
    ridge_alpha: float = 1.0,
) -> RegressionOutputs:
    regression_df = build_regression_dataframe(behavior, projections, base_model_id=base_model_id, category=category)
    correlation = compute_correlation_matrix(regression_df)
    matching = compute_matching_regressions(regression_df)
    ridge = compute_all_vector_ridge(regression_df, alpha=ridge_alpha)
    multi = compute_multioutput_ridge(regression_df, alpha=ridge_alpha)
    baselines = compute_baseline_regressions(regression_df, alpha=ridge_alpha)
    return RegressionOutputs(
        regression_dataframe=regression_df,
        correlation_matrix=correlation,
        matching_regressions=matching,
        ridge_coefficients=ridge,
        multioutput_ridge_coefficients=multi,
        baseline_regressions=baselines,
    )


def build_regression_dataframe(
    behavior: pd.DataFrame,
    projections: pd.DataFrame,
    base_model_id: str = "base",
    category: str = "all",
) -> pd.DataFrame:
    behavior = behavior.copy()
    if category is not None:
        behavior = behavior[behavior["category"].fillna("all") == category].copy()
    if behavior.empty:
        raise ValueError(f"no behavior rows found for category={category!r}")

    if "delta_behavior" not in behavior.columns:
        base_rows = behavior[behavior["model_id"] == base_model_id][["eval_id", "behavior_score"]].rename(columns={"behavior_score": "base_behavior_score"})
        if base_rows.empty:
            raise ValueError(
                "behavior scores do not include delta_behavior and no base rows were found; "
                f"expected model_id={base_model_id!r}"
            )
        behavior = behavior.merge(base_rows, on="eval_id", how="left")
        if behavior["base_behavior_score"].isna().any():
            missing = sorted(behavior.loc[behavior["base_behavior_score"].isna(), "eval_id"].unique().tolist())
            raise ValueError(f"missing base behavior scores for evals: {missing}")
        behavior["delta_behavior"] = behavior["behavior_score"] - behavior["base_behavior_score"]
    else:
        behavior["delta_behavior"] = pd.to_numeric(behavior["delta_behavior"], errors="raise")
        if "base_behavior_score" not in behavior.columns:
            behavior["base_behavior_score"] = pd.NA

    behavior = behavior[behavior["model_id"] != base_model_id].copy()
    merged = behavior.merge(projections, on="model_id", how="inner", suffixes=("", "_projection"))
    if merged.empty:
        raise ValueError("behavior rows and projection rows have no overlapping non-base model_id values")
    keep_cols = [
        "model_id",
        "fine_tune_family",
        "seed",
        "eval_id",
        "category",
        "behavior_score",
        "base_behavior_score",
        "delta_behavior",
        "neutral_bank",
        "layer_aggregate",
        *ALL_MODEL_FEATURES,
    ]
    if "layer" in merged.columns:
        keep_cols.append("layer")
    keep_cols = [col for col in keep_cols if col in merged.columns]
    out = merged[keep_cols].copy()
    out = out.dropna(subset=["delta_behavior", *ALL_MODEL_FEATURES])
    return out.sort_values(["eval_id", "model_id"]).reset_index(drop=True)


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for eval_id, group in df.groupby("eval_id"):
        row = {"eval_id": eval_id, "n": int(len(group))}
        for feature in VECTOR_FEATURES:
            row[feature] = _safe_corr(group["delta_behavior"], group[feature])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("eval_id").reset_index(drop=True)


def compute_matching_regressions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for eval_id, group in df.groupby("eval_id"):
        features = list(MATCHING_FEATURES.get(str(eval_id), []))
        if not features:
            continue
        features = features + ["shift_norm"]
        rows.append(_fit_single_output_row(group, eval_id=str(eval_id), model_name="matching_plus_shift_norm", features=features, estimator="ols"))
    return pd.DataFrame(rows)


def compute_all_vector_ridge(df: pd.DataFrame, alpha: float = 1.0) -> pd.DataFrame:
    rows = []
    for eval_id, group in df.groupby("eval_id"):
        fit = _fit_standardized_ridge(group, ALL_MODEL_FEATURES, ["delta_behavior"], alpha=alpha)
        for feature, coef in zip(ALL_MODEL_FEATURES, fit["coef"].ravel(), strict=False):
            rows.append(
                {
                    "eval_id": eval_id,
                    "feature": feature,
                    "coefficient": coef,
                    "alpha": alpha,
                    "r2_in_sample": fit["r2"],
                    "n": fit["n"],
                    "target": "delta_behavior",
                }
            )
    return pd.DataFrame(rows)


def compute_multioutput_ridge(df: pd.DataFrame, alpha: float = 1.0) -> pd.DataFrame:
    pivot = df.pivot_table(index="model_id", columns="eval_id", values="delta_behavior", aggfunc="first")
    projection_cols = ["model_id", *ALL_MODEL_FEATURES]
    x = df[projection_cols].drop_duplicates("model_id").set_index("model_id").reindex(pivot.index)
    valid_targets = [col for col in pivot.columns if pivot[col].notna().all()]
    if not valid_targets:
        return pd.DataFrame(columns=["feature", "eval_id", "coefficient", "alpha", "r2_in_sample", "n"])
    fit_df = x.join(pivot[valid_targets]).dropna()
    if fit_df.empty:
        return pd.DataFrame(columns=["feature", "eval_id", "coefficient", "alpha", "r2_in_sample", "n"])
    fit = _fit_standardized_ridge(fit_df.reset_index(), ALL_MODEL_FEATURES, list(valid_targets), alpha=alpha)
    rows = []
    coef = np.asarray(fit["coef"])
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    for target_idx, eval_id in enumerate(valid_targets):
        for feature_idx, feature in enumerate(ALL_MODEL_FEATURES):
            rows.append(
                {
                    "feature": feature,
                    "eval_id": eval_id,
                    "coefficient": float(coef[target_idx, feature_idx]),
                    "alpha": alpha,
                    "r2_in_sample": fit["r2_by_target"].get(eval_id),
                    "n": fit["n"],
                }
            )
    return pd.DataFrame(rows)


def compute_baseline_regressions(df: pd.DataFrame, alpha: float = 1.0) -> pd.DataFrame:
    rows = []
    for eval_id, group in df.groupby("eval_id"):
        rows.append(_fit_single_output_row(group, str(eval_id), "shift_norm_only", ["shift_norm"], estimator="ols"))
        if "fine_tune_family" in group.columns and group["fine_tune_family"].dropna().nunique() > 1:
            rows.append(_fit_dataset_family_baseline(group, str(eval_id), alpha=alpha))
    return pd.DataFrame(rows)


def _fit_single_output_row(group: pd.DataFrame, eval_id: str, model_name: str, features: list[str], estimator: str) -> dict[str, Any]:
    clean = group.dropna(subset=["delta_behavior", *features])
    row: dict[str, Any] = {
        "eval_id": eval_id,
        "model": model_name,
        "features": ",".join(features),
        "n": int(len(clean)),
        "estimator": estimator,
    }
    if len(clean) < 2:
        row.update({"intercept": np.nan, "r2_in_sample": np.nan})
        for feature in features:
            row[f"coef_{feature}"] = np.nan
        return row
    x = clean[features].to_numpy(dtype=float)
    y = clean["delta_behavior"].to_numpy(dtype=float)
    x_scaled = StandardScaler().fit_transform(x)
    model = LinearRegression().fit(x_scaled, y)
    pred = model.predict(x_scaled)
    row.update({"intercept": float(model.intercept_), "r2_in_sample": _safe_r2(y, pred)})
    for feature, coef in zip(features, model.coef_, strict=False):
        row[f"coef_{feature}"] = float(coef)
    return row


def _fit_dataset_family_baseline(group: pd.DataFrame, eval_id: str, alpha: float) -> dict[str, Any]:
    clean = group.dropna(subset=["delta_behavior", "fine_tune_family"])
    row: dict[str, Any] = {
        "eval_id": eval_id,
        "model": "dataset_family",
        "features": "fine_tune_family",
        "n": int(len(clean)),
        "estimator": "ridge",
        "alpha": alpha,
    }
    if len(clean) < 2:
        row["r2_in_sample"] = np.nan
        return row
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x = encoder.fit_transform(clean[["fine_tune_family"]])
    y = clean["delta_behavior"].to_numpy(dtype=float)
    model = Ridge(alpha=alpha).fit(x, y)
    pred = model.predict(x)
    row.update({"intercept": float(model.intercept_), "r2_in_sample": _safe_r2(y, pred)})
    for feature, coef in zip(encoder.get_feature_names_out(["fine_tune_family"]), model.coef_, strict=False):
        row[f"coef_{feature}"] = float(coef)
    return row


def _fit_standardized_ridge(group: pd.DataFrame, features: list[str], targets: list[str], alpha: float) -> dict[str, Any]:
    clean = group.dropna(subset=features + targets)
    if len(clean) < 2:
        return {"coef": np.full((len(targets), len(features)), np.nan), "r2": np.nan, "r2_by_target": {}, "n": int(len(clean))}
    x = clean[features].to_numpy(dtype=float)
    y = clean[targets].to_numpy(dtype=float)
    x_scaled = StandardScaler().fit_transform(x)
    y_scaled = StandardScaler().fit_transform(y)
    model = Ridge(alpha=alpha).fit(x_scaled, y_scaled)
    pred = model.predict(x_scaled)
    r2_by_target = {target: _safe_r2(y_scaled[:, idx], pred[:, idx] if pred.ndim > 1 else pred) for idx, target in enumerate(targets)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = r2_score(y_scaled, pred, multioutput="uniform_average") if len(clean) >= 2 else np.nan
    return {"coef": model.coef_, "r2": float(r2), "r2_by_target": r2_by_target, "n": int(len(clean))}


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    clean = pd.concat([x, y], axis=1).dropna()
    if len(clean) < 2:
        return float("nan")
    if clean.iloc[:, 0].std(ddof=0) == 0 or clean.iloc[:, 1].std(ddof=0) == 0:
        return float("nan")
    return float(clean.iloc[:, 0].corr(clean.iloc[:, 1]))


def _safe_r2(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 2 or np.std(y) == 0:
        return float("nan")
    return float(r2_score(y, pred))


def write_df(path: str | Path, df: pd.DataFrame) -> None:
    path = ensure_parent(path)
    df.to_csv(path, index=False)
