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
    behavior_path = results_root / "behavior_scores.csv"
    projections_path = results_root / "projections.csv"

    if behavior_path.exists():
        outputs.extend(plot_single_model_pilot_behavior(behavior_path, figures_root, formats=formats))
    if projections_path.exists():
        outputs.extend(plot_single_model_raw_projections(projections_path, figures_root, formats=formats))
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
            "behavior_scores": str(behavior_path),
            "projections": str(projections_path),
        },
        "outputs": outputs,
        "formats": list(formats),
    }
    manifest_path = ensure_parent(figures_root / "plot_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs.append(str(manifest_path))
    return {"outputs": outputs, "n_outputs": len(outputs), "manifest": str(manifest_path)}


def plot_single_model_pilot_behavior(path: str | Path, figures_root: str | Path, formats: tuple[str, ...]) -> list[str]:
    df = pd.read_csv(path)
    df = df[df.get("category", "all").fillna("all") == "all"].copy()
    if df.empty or "delta_behavior" not in df.columns:
        return []
    base = df[df["model_id"] == "base"].copy()
    ft = df[df["model_id"] != "base"].copy()
    if base.empty or ft.empty:
        return []
    outputs: list[str] = []
    outputs.extend(_plot_behavior_delta_bar(ft, figures_root, formats))
    outputs.extend(_plot_behavior_base_vs_ft(base, ft, figures_root, formats))
    outputs.extend(_plot_behavior_subscore_delta_heatmap(base, ft, figures_root, formats))
    summary_paths = write_single_model_behavior_summary(ft, figures_root)
    outputs.extend(summary_paths)
    return outputs


def _plot_behavior_delta_bar(ft: pd.DataFrame, figures_root: str | Path, formats: tuple[str, ...]) -> list[str]:
    import matplotlib.pyplot as plt

    clean = ft.dropna(subset=["delta_behavior"]).copy()
    if clean.empty:
        return []
    clean = clean.sort_values("delta_behavior", ascending=True)
    colors = ["#2b6cb0" if value >= 0 else "#c93a2e" for value in clean["delta_behavior"]]
    fig_height = max(4.8, 0.42 * len(clean) + 1.4)
    fig, ax = plt.subplots(figsize=(8.4, fig_height))
    bars = ax.barh(clean["eval_id"].map(_short_eval_label), clean["delta_behavior"], color=colors, alpha=0.88)
    ax.axvline(0, color="#666666", linewidth=0.9)
    ax.set_xlabel("behavior delta: fine-tuned score - base score")
    ax.set_ylabel("behavior eval")
    ax.set_title(f"Single-Model Pilot Behavior Shifts: {clean['model_id'].iloc[0]}")
    xmax = max(0.05, float(clean["delta_behavior"].abs().max()) * 1.2)
    ax.set_xlim(-xmax, xmax)
    for bar, value in zip(bars, clean["delta_behavior"], strict=False):
        x = bar.get_width()
        ax.text(x + (0.01 if x >= 0 else -0.01), bar.get_y() + bar.get_height() / 2, f"{value:+.3f}", va="center", ha="left" if x >= 0 else "right", fontsize=8)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save_figure(fig, Path(figures_root) / "single_model_behavior_delta_bar", formats)


def _plot_behavior_base_vs_ft(base: pd.DataFrame, ft: pd.DataFrame, figures_root: str | Path, formats: tuple[str, ...]) -> list[str]:
    import matplotlib.pyplot as plt

    base_scores = base.set_index("eval_id")["behavior_score"]
    clean = ft.dropna(subset=["behavior_score"]).copy()
    clean["base_score"] = clean["eval_id"].map(base_scores)
    clean = clean.dropna(subset=["base_score"]).sort_values("delta_behavior", ascending=False)
    if clean.empty:
        return []
    labels = clean["eval_id"].map(_short_eval_label).tolist()
    x = np.arange(len(clean))
    width = 0.38
    fig_width = max(9.0, 0.75 * len(clean) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    ax.bar(x - width / 2, clean["base_score"], width, label="base", color="#9aa3ad", alpha=0.9)
    ax.bar(x + width / 2, clean["behavior_score"], width, label="fine-tuned", color="#2b6cb0", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("behavior score")
    ax.set_title("Base vs Fine-Tuned Behavior Scores")
    ax.set_ylim(0, max(1.0, float(clean[["base_score", "behavior_score"]].max().max()) * 1.1))
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save_figure(fig, Path(figures_root) / "single_model_base_vs_ft_scores", formats)


def _plot_behavior_subscore_delta_heatmap(base: pd.DataFrame, ft: pd.DataFrame, figures_root: str | Path, formats: tuple[str, ...]) -> list[str]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    subscore_cols = [col for col in ft.columns if col.startswith("mean_subscore_")]
    if not subscore_cols:
        return []
    base_by_eval = base.set_index("eval_id")
    rows = []
    for _, ft_row in ft.iterrows():
        eval_id = ft_row["eval_id"]
        if eval_id not in base_by_eval.index:
            continue
        row = {"eval_id": eval_id}
        base_row = base_by_eval.loc[eval_id]
        for col in subscore_cols:
            if pd.notna(ft_row.get(col)) and pd.notna(base_row.get(col)):
                row[col.replace("mean_subscore_", "")] = float(ft_row[col]) - float(base_row[col])
        rows.append(row)
    matrix = pd.DataFrame(rows).set_index("eval_id")
    matrix = matrix.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if matrix.empty:
        return []
    matrix = matrix.rename(index=_short_eval_label, columns=lambda c: c.replace("_", " "))
    vmax = float(np.nanmax(np.abs(matrix.to_numpy(dtype=float))))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(max(9.0, 0.62 * matrix.shape[1] + 4.0), max(5.0, 0.42 * matrix.shape[0] + 2.0)))
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
        cbar_kws={"label": "subscore delta"},
    )
    ax.set_xlabel("judge subscore")
    ax.set_ylabel("behavior eval")
    ax.set_title("Fine-Tune Minus Base: Judge Subscore Deltas")
    fig.tight_layout()
    return _save_figure(fig, Path(figures_root) / "single_model_subscore_delta_heatmap", formats)


def write_single_model_behavior_summary(ft: pd.DataFrame, figures_root: str | Path) -> list[str]:
    out_dir = Path(figures_root)
    clean = ft[["model_id", "eval_id", "behavior_score", "base_behavior_score", "delta_behavior", "n", "judge_model"]].copy()
    clean = clean.sort_values("delta_behavior", ascending=False)
    csv_path = ensure_parent(out_dir / "single_model_behavior_delta_summary.csv")
    clean.to_csv(csv_path, index=False)
    md_path = ensure_parent(out_dir / "single_model_behavior_delta_summary.md")
    lines = [
        "# Single-Model Pilot Behavior Summary",
        "",
        "This summary is appropriate for the one-fine-tuned-model pilot. Correlations and regressions are not statistically identified with one non-base model.",
        "",
        "| eval | base | fine-tuned | delta | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in clean.iterrows():
        lines.append(
            f"| {_short_eval_label(row['eval_id'])} | {float(row['base_behavior_score']):.3f} | "
            f"{float(row['behavior_score']):.3f} | {float(row['delta_behavior']):+.3f} | {int(row['n'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [str(csv_path), str(md_path)]


def plot_single_model_raw_projections(path: str | Path, figures_root: str | Path, formats: tuple[str, ...]) -> list[str]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(path)
    required = {"model_id", "neutral_bank", "layer", "trait_id", "projection"}
    if required - set(df.columns):
        return []
    df = df[df["neutral_bank"] == "neutral_all"].copy()
    if df.empty:
        return []
    df["trait_label"] = df["trait_id"].map(_trait_label)
    matrix = df.pivot_table(index="trait_label", columns="layer", values="projection", aggfunc="mean")
    if matrix.empty:
        return []
    vmax = float(np.nanmax(np.abs(matrix.to_numpy(dtype=float))))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(max(7.5, 0.7 * matrix.shape[1] + 3.0), max(4.8, 0.45 * matrix.shape[0] + 2.0)))
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
        cbar_kws={"label": "raw projection"},
    )
    ax.set_xlabel("quantile layer")
    ax.set_ylabel("trait vector")
    ax.set_title("Single-Model Raw Trait Projections by Layer")
    fig.tight_layout()
    return _save_figure(fig, Path(figures_root) / "single_model_raw_projection_heatmap", formats)


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


def _short_eval_label(eval_id: object) -> str:
    mapping = {
        "eval_code_insecurity": "code insecurity",
        "eval_core_misalignment": "core misalignment",
        "eval_extended_misalignment_by_category": "extended misalignment",
        "eval_finance_risky_advice": "finance risky advice",
        "eval_hallucination_tool_deception": "hallucination/tool deception",
        "eval_health_bad_advice": "health bad advice",
        "eval_sycophancy_answer": "sycophancy",
        "eval_xstest_safe_overrefusal": "XSTest safe overrefusal",
        "eval_xstest_unsafe_refusal": "XSTest unsafe refusal",
    }
    text = str(eval_id)
    return mapping.get(text, text.replace("eval_", "").replace("_", " "))


def _trait_label(trait_id: object) -> str:
    mapping = {
        "v_toxic_reckless_persona": "toxic/reckless",
        "v_deception_concealment": "deception",
        "v_hallucination": "hallucination",
        "v_sycophancy": "sycophancy",
        "v_refusal_gate": "refusal",
        "v_harmful_advice_continuation": "harmful advice",
        "v_insecure_code_continuation": "insecure code",
    }
    text = str(trait_id)
    return mapping.get(text, text.replace("v_", "").replace("_", " "))


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
