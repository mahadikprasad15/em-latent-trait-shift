"""Aggregation of raw projection rows into regression-ready features."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import pandas as pd

from em_latent_factors.config import load_yaml
from em_latent_factors.io import ensure_parent
from em_latent_factors.vectors import all_trait_ids


TRAIT_FEATURE_COLUMNS = {
    "v_toxic_reckless_persona": "z_toxic_reckless",
    "v_deception_concealment": "z_deception",
    "v_hallucination": "z_hallucination",
    "v_sycophancy": "z_sycophancy",
    "v_refusal_gate": "z_refusal",
    "v_harmful_advice_continuation": "z_harmful_advice",
    "v_insecure_code_continuation": "z_insecure_code",
}

REQUIRED_COLUMNS = {"model_id", "neutral_bank", "layer", "trait_id", "projection", "shift_norm"}
LAYER_AGGREGATES = {"middle", "mean_standardized", "per_layer"}


def aggregate_projection_file(
    input_path: str | Path,
    output_path: str | Path,
    standardized_output_path: str | Path | None = None,
    config_path: str | Path = "configs/experiment.yaml",
    layers: list[int] | None = None,
    layer_aggregate: str = "middle",
    include_neutral_all: bool = True,
    fail_on_missing_cells: bool = False,
) -> dict[str, Any]:
    if layer_aggregate not in LAYER_AGGREGATES:
        raise ValueError(f"layer_aggregate must be one of {sorted(LAYER_AGGREGATES)}")
    raw = read_projection_csv(input_path)
    standardized = standardize_projection_rows(
        raw,
        layers=layers,
        include_neutral_all=include_neutral_all,
        fail_on_missing_cells=fail_on_missing_cells,
    )
    aggregation_input, selected_layers, label = select_layer_aggregation(standardized, layer_aggregate)
    aggregated = aggregate_projection_features(
        aggregation_input,
        config_path=config_path,
        layer_aggregate=label,
        include_layer_column=layer_aggregate == "per_layer",
    )
    write_projection_aggregation_csv(output_path, aggregated)
    standardized_rows = len(standardized)
    if standardized_output_path:
        write_projection_aggregation_csv(standardized_output_path, standardized)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "standardized_output_path": str(standardized_output_path) if standardized_output_path else None,
        "raw_rows": int(len(raw)),
        "standardized_rows": int(standardized_rows),
        "aggregated_rows": int(len(aggregated)),
        "models": int(aggregated["model_id"].nunique()) if not aggregated.empty else 0,
        "neutral_banks": sorted(aggregated["neutral_bank"].dropna().unique().tolist()) if not aggregated.empty else [],
        "layers": sorted(int(layer) for layer in standardized["layer"].dropna().unique().tolist()) if not standardized.empty else [],
        "layer_aggregate": layer_aggregate,
        "aggregated_layers": selected_layers,
    }


def read_projection_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required projection columns: {sorted(missing)}")
    unknown_traits = sorted(set(df["trait_id"].dropna()) - set(all_trait_ids()))
    if unknown_traits:
        raise ValueError(f"{path} contains unknown trait_id values: {unknown_traits}")
    df = df.copy()
    df["layer"] = df["layer"].astype(int)
    df["projection"] = pd.to_numeric(df["projection"], errors="raise")
    df["shift_norm"] = pd.to_numeric(df["shift_norm"], errors="raise")
    return df


def standardize_projection_rows(
    df: pd.DataFrame,
    layers: list[int] | None = None,
    include_neutral_all: bool = True,
    fail_on_missing_cells: bool = False,
) -> pd.DataFrame:
    selected = df.copy()
    if layers:
        layers_set = {int(layer) for layer in layers}
        selected = selected[selected["layer"].isin(layers_set)].copy()
        missing_layers = sorted(layers_set - set(selected["layer"].unique()))
        if missing_layers:
            raise ValueError(f"requested layers are absent from projections: {missing_layers}")
    if selected.empty:
        raise ValueError("no projection rows remain after layer filtering")

    if fail_on_missing_cells:
        _check_missing_projection_cells(selected)

    bank_rows = _standardize_within_groups(selected)
    if not include_neutral_all:
        return bank_rows

    all_rows = (
        selected.groupby(["model_id", "layer", "trait_id"], as_index=False)
        .agg(
            projection=("projection", "mean"),
            shift_norm=("shift_norm", "mean"),
            source_neutral_banks=("neutral_bank", lambda values: ",".join(sorted(set(map(str, values))))),
        )
    )
    all_rows["neutral_bank"] = "neutral_all"
    all_rows = _standardize_within_groups(all_rows)
    shared_cols = [
        "model_id",
        "neutral_bank",
        "layer",
        "trait_id",
        "projection",
        "projection_z",
        "shift_norm",
        "shift_norm_z",
    ]
    optional_cols = [col for col in ("source_neutral_banks", "vector_model_id", "shift_path", "vector_path") if col in bank_rows.columns or col in all_rows.columns]
    for col in optional_cols:
        if col not in bank_rows.columns:
            bank_rows[col] = ""
        if col not in all_rows.columns:
            all_rows[col] = ""
    return pd.concat([bank_rows[shared_cols + optional_cols], all_rows[shared_cols + optional_cols]], ignore_index=True)


def select_layer_aggregation(df: pd.DataFrame, layer_aggregate: str) -> tuple[pd.DataFrame, list[int], str]:
    layers = sorted(int(layer) for layer in df["layer"].dropna().unique().tolist())
    if not layers:
        raise ValueError("cannot aggregate projections with no layers")
    if layer_aggregate == "middle":
        middle_layer = layers[(len(layers) - 1) // 2]
        return df[df["layer"] == middle_layer].copy(), [middle_layer], "middle_quantile_layer"
    if layer_aggregate == "mean_standardized":
        return df.copy(), layers, "mean_standardized_quantile_layers"
    if layer_aggregate == "per_layer":
        return df.copy(), layers, "per_quantile_layer"
    raise ValueError(f"layer_aggregate must be one of {sorted(LAYER_AGGREGATES)}")


def aggregate_projection_features(
    df: pd.DataFrame,
    config_path: str | Path = "configs/experiment.yaml",
    layer_aggregate: str = "middle_quantile_layer",
    include_layer_column: bool = False,
) -> pd.DataFrame:
    model_metadata = _model_metadata_from_config(config_path)
    index_cols = ["model_id", "neutral_bank"]
    if include_layer_column:
        index_cols.append("layer")
    trait_agg = (
        df.groupby(index_cols + ["trait_id"], as_index=False)
        .agg(
            z_value=("projection_z", "mean"),
            raw_projection_mean=("projection", "mean"),
            n_layers=("layer", "nunique"),
        )
    )
    trait_agg["feature"] = trait_agg["trait_id"].map(TRAIT_FEATURE_COLUMNS)
    if trait_agg["feature"].isna().any():
        unknown = sorted(trait_agg.loc[trait_agg["feature"].isna(), "trait_id"].unique().tolist())
        raise ValueError(f"no feature mapping configured for trait IDs: {unknown}")

    wide = (
        trait_agg.pivot_table(
            index=index_cols,
            columns="feature",
            values="z_value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    norm_agg = (
        df.drop_duplicates(["model_id", "neutral_bank", "layer"])
        .groupby(index_cols, as_index=False)
        .agg(
            shift_norm=("shift_norm_z", "mean"),
            shift_norm_raw=("shift_norm", "mean"),
            n_layers=("layer", "nunique"),
        )
    )
    out = wide.merge(norm_agg, on=index_cols, how="left")
    for feature in TRAIT_FEATURE_COLUMNS.values():
        if feature not in out.columns:
            out[feature] = pd.NA
    out["layer_aggregate"] = layer_aggregate
    out["fine_tune_family"] = out["model_id"].map(lambda model_id: model_metadata.get(str(model_id), {}).get("fine_tune_family"))
    out["seed"] = out["model_id"].map(lambda model_id: model_metadata.get(str(model_id), {}).get("seed"))

    ordered = [
        "model_id",
        "fine_tune_family",
        "seed",
        "neutral_bank",
        "layer_aggregate",
        "layer",
        "z_toxic_reckless",
        "z_deception",
        "z_hallucination",
        "z_sycophancy",
        "z_refusal",
        "z_harmful_advice",
        "z_insecure_code",
        "shift_norm",
        "shift_norm_raw",
        "n_layers",
    ]
    extras = [col for col in out.columns if col not in ordered]
    ordered_present = [col for col in ordered if col in out.columns]
    sort_cols = ["neutral_bank", "model_id"] + (["layer"] if "layer" in out.columns else [])
    return out[ordered_present + extras].sort_values(sort_cols).reset_index(drop=True)


def write_projection_aggregation_csv(path: str | Path, df: pd.DataFrame) -> None:
    path = ensure_parent(path)
    df.to_csv(path, index=False)


def _standardize_within_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["projection_z"] = (
        out.groupby(["neutral_bank", "trait_id", "layer"], group_keys=False)["projection"]
        .transform(_zscore_series)
        .astype(float)
    )
    norm_rows = out[["model_id", "neutral_bank", "layer", "shift_norm"]].drop_duplicates().copy()
    norm_rows["shift_norm_z"] = (
        norm_rows.groupby(["neutral_bank", "layer"], group_keys=False)["shift_norm"]
        .transform(_zscore_series)
        .astype(float)
    )
    out = out.merge(norm_rows, on=["model_id", "neutral_bank", "layer", "shift_norm"], how="left")
    return out


def _zscore_series(values: pd.Series) -> pd.Series:
    mean = values.mean()
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        warnings.warn(
            f"zero across-model variance while standardizing {values.name}; assigning 0.0",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.Series([0.0] * len(values), index=values.index)
    return (values - mean) / std


def _check_missing_projection_cells(df: pd.DataFrame) -> None:
    models = sorted(df["model_id"].unique().tolist())
    banks = sorted(df["neutral_bank"].unique().tolist())
    layers = sorted(df["layer"].unique().tolist())
    traits = sorted(df["trait_id"].unique().tolist())
    present = set(zip(df["model_id"], df["neutral_bank"], df["layer"], df["trait_id"], strict=False))
    missing = []
    for model_id in models:
        for neutral_bank in banks:
            for layer in layers:
                for trait_id in traits:
                    if (model_id, neutral_bank, layer, trait_id) not in present:
                        missing.append((model_id, neutral_bank, layer, trait_id))
                        if len(missing) >= 10:
                            break
                if len(missing) >= 10:
                    break
            if len(missing) >= 10:
                break
        if len(missing) >= 10:
            break
    if missing:
        raise ValueError(f"missing projection cells, first examples: {missing}")


def _model_metadata_from_config(config_path: str | Path) -> dict[str, dict[str, Any]]:
    config = load_yaml(config_path)
    metadata: dict[str, dict[str, Any]] = {}
    for row in config.get("fine_tuned_models", []):
        model_id = row.get("model_id")
        if model_id:
            metadata[str(model_id)] = {
                "fine_tune_family": row.get("fine_tune_family"),
                "seed": row.get("seed"),
                "dataset_id": row.get("dataset_id"),
            }
    return metadata
