"""Plotting utilities for regression and projection analysis outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from em_latent_factors.io import ensure_parent
from em_latent_factors.regressions import MATCHING_FEATURES, VECTOR_FEATURES


FEATURE_LABELS = {
    "z_toxic_reckless": "toxic/reckless",
    "z_deception": "deception",
    "z_hallucination": "hallucination",
    "z_sycophancy": "sycophancy",
    "z_refusal": "refusal",
    "z_harmful_advice": "harmful advice",
    "z_insecure_code": "insecure code",
    "shift_norm": "shift norm",
}

FAMILY_COLORS = {
    "health": "#3b7ea1",
    "finance": "#6b9e5b",
    "code": "#b87333",
    "unknown": "#737373",
}


def plot_results(
    results_root: str | Path = "results",
    figures_root: str | Path = "figures",
    formats: tuple[str, ...] = ("pdf", "png"),
) -> dict[str, Any]:
    results_root = Path(results_root)
    figures_root = Path(figures_root)
    outputs: list[str] = []

    correlation_path = results_root / "correlation_matrix.csv"
    multioutput_path = results_root / "multioutput_ridge_coefficients.csv"
    regression_df_path = results_root / "regression_dataframe.csv"
    matching_path = results_root / "matching_vector_regressions.csv"
    ridge_path = results_root / "ridge_regression_coefficients.csv"
    baseline_path = results_root / "baseline_regressions.csv"

    if correlation_path.exists():
        outputs.extend(plot_correlation_heatmap(correlation_path, figures_root, formats=formats))
    if multioutput_path.exists():
        outputs.extend(plot_ridge_heatmap(multioutput_path, figures_root, formats=formats))
    if regression_df_path.exists():
        outputs.extend(plot_matching_scatters(regression_df_path, figures_root, formats=formats))
    if baseline_path.exists() and matching_path.exists() and ridge_path.exists():
        outputs.extend(plot_baseline_r2_comparison(baseline_path, matching_path, ridge_path, figures_root, formats=formats))

    manifest = {
        "results_root": str(results_root),
        "figures_root": str(figures_root),
        "inputs": {
            "correlation_matrix": str(correlation_path),
            "multioutput_ridge_coefficients": str(multioutput_path),
            "regression_dataframe": str(regression_df_path),
            "matching_vector_regressions": str(matching_path),
            "ridge_regression_coefficients": str(ridge_path),
            "baseline_regressions": str(baseline_path),
        },
        "outputs": outputs,
        "formats": list(formats),
    }
    manifest_path = ensure_parent(figures_root / "plot_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs.append(str(manifest_path))
    return {"outputs": outputs, "n_outputs": len(outputs), "manifest": str(manifest_path)}


def plot_correlation_heatmap(path: str | Path, figures_root: str | Path, formats: tuple[str, ...]) -> list[str]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(path)
    matrix = df.set_index("eval_id")[[feature for feature in VECTOR_FEATURES if feature in df.columns]]
    fig, ax = plt.subplots(figsize=_heatmap_size(matrix))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="#eeeeee",
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_xlabel("Trait projection")
    ax.set_ylabel("Behavior eval")
    ax.set_xticklabels([FEATURE_LABELS.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()], rotation=35, ha="right")
    ax.set_title("Behavior Delta vs Trait Projection Correlations")
    fig.tight_layout()
    return _save_figure(fig, Path(figures_root) / "behavior_vector_correlation_heatmap", formats)


def plot_ridge_heatmap(path: str | Path, figures_root: str | Path, formats: tuple[str, ...]) -> list[str]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(path)
    if df.empty:
        return []
    matrix = df.pivot_table(index="feature", columns="eval_id", values="coefficient", aggfunc="first")
    feature_order = [feature for feature in [*VECTOR_FEATURES, "shift_norm"] if feature in matrix.index]
    matrix = matrix.reindex(feature_order)
    vmax = np.nanmax(np.abs(matrix.to_numpy(dtype=float))) if matrix.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=_heatmap_size(matrix.T))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="#eeeeee",
        cbar_kws={"label": "standardized ridge coefficient"},
    )
    ax.set_xlabel("Behavior eval")
    ax.set_ylabel("Feature")
    ax.set_yticklabels([FEATURE_LABELS.get(label.get_text(), label.get_text()) for label in ax.get_yticklabels()], rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_title("Multi-Output Ridge Coefficient Matrix")
    fig.tight_layout()
    return _save_figure(fig, Path(figures_root) / "ridge_coefficients_heatmap", formats)


def plot_matching_scatters(path: str | Path, figures_root: str | Path, formats: tuple[str, ...]) -> list[str]:
    df = pd.read_csv(path)
    outputs: list[str] = []
    for eval_id, group in df.groupby("eval_id"):
        matching = MATCHING_FEATURES.get(str(eval_id), [])
        if not matching:
            continue
        for feature in matching:
            if feature not in group.columns:
                continue
            slug = _slug(f"projection_vs_behavior_scatter_{eval_id}_{feature}")
            outputs.extend(_plot_single_scatter(group, eval_id=str(eval_id), feature=feature, out_base=Path(figures_root) / slug, formats=formats))
    return outputs


def plot_baseline_r2_comparison(
    baseline_path: str | Path,
    matching_path: str | Path,
    ridge_path: str | Path,
    figures_root: str | Path,
    formats: tuple[str, ...],
) -> list[str]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    rows = []
    baseline = pd.read_csv(baseline_path)
    if not baseline.empty:
        for _, row in baseline.iterrows():
            rows.append({"eval_id": row["eval_id"], "model": row["model"], "r2_in_sample": row.get("r2_in_sample")})
    matching = pd.read_csv(matching_path)
    if not matching.empty:
        for _, row in matching.iterrows():
            rows.append({"eval_id": row["eval_id"], "model": "matching_plus_shift_norm", "r2_in_sample": row.get("r2_in_sample")})
    ridge = pd.read_csv(ridge_path)
    if not ridge.empty:
        for eval_id, group in ridge.groupby("eval_id"):
            rows.append({"eval_id": eval_id, "model": "all_vector_ridge", "r2_in_sample": group["r2_in_sample"].iloc[0]})
    df = pd.DataFrame(rows).dropna(subset=["r2_in_sample"])
    if df.empty:
        return []
    order = ["shift_norm_only", "dataset_family", "matching_plus_shift_norm", "all_vector_ridge"]
    fig_width = max(8, 0.65 * df["eval_id"].nunique() * len(order))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    sns.barplot(data=df, x="eval_id", y="r2_in_sample", hue="model", hue_order=order, ax=ax, palette="muted")
    ax.set_ylim(0, min(1.05, max(1.0, float(df["r2_in_sample"].max()) * 1.1)))
    ax.set_xlabel("Behavior eval")
    ax.set_ylabel("in-sample R²")
    ax.set_title("Regression Baseline Comparison")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.legend(title="model", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return _save_figure(fig, Path(figures_root) / "regression_baseline_r2_comparison", formats)


def _plot_single_scatter(group: pd.DataFrame, eval_id: str, feature: str, out_base: Path, formats: tuple[str, ...]) -> list[str]:
    import matplotlib.pyplot as plt

    clean = group.dropna(subset=[feature, "delta_behavior"]).copy()
    if clean.empty:
        return []
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    for family, fam_group in clean.groupby(clean.get("fine_tune_family", pd.Series(["unknown"] * len(clean))).fillna("unknown")):
        color = FAMILY_COLORS.get(str(family), FAMILY_COLORS["unknown"])
        ax.scatter(fam_group[feature], fam_group["delta_behavior"], s=48, color=color, edgecolor="white", linewidth=0.7, label=str(family), alpha=0.9)
    if len(clean) >= 3 and clean[feature].std(ddof=0) > 0:
        x = clean[feature].to_numpy(dtype=float)
        y = clean["delta_behavior"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(xs, slope * xs + intercept, color="#444444", linewidth=1.2, alpha=0.8)
        r = float(np.corrcoef(x, y)[0, 1])
        title = f"{eval_id}: r={r:.2f}"
    else:
        title = eval_id
    for _, row in clean.iterrows():
        ax.annotate(str(row["model_id"]).replace("llama32_3b_", ""), (row[feature], row["delta_behavior"]), xytext=(4, 4), textcoords="offset points", fontsize=7, alpha=0.75)
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axvline(0, color="#999999", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel(FEATURE_LABELS.get(feature, feature))
    ax.set_ylabel("behavior delta")
    ax.set_title(title)
    ax.legend(title="FT family", frameon=False, fontsize=8)
    fig.tight_layout()
    return _save_figure(fig, out_base, formats)


def _heatmap_size(matrix: pd.DataFrame) -> tuple[float, float]:
    width = max(7.0, 0.85 * max(1, matrix.shape[1]) + 3.0)
    height = max(4.5, 0.45 * max(1, matrix.shape[0]) + 2.0)
    return width, height


def _save_figure(fig, out_base: Path, formats: tuple[str, ...]) -> list[str]:
    import matplotlib.pyplot as plt

    outputs = []
    for fmt in formats:
        path = ensure_parent(out_base.with_suffix(f".{fmt}"))
        fig.savefig(path, dpi=220 if fmt == "png" else None, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _slug(value: str) -> str:
    out = []
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug
