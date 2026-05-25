"""Projection of activation shifts onto trait vectors."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from em_latent_factors.artifacts import RunContext
from em_latent_factors.io import ensure_parent
from em_latent_factors.vectors import all_trait_ids


def compute_raw_projections(
    shifts_path: str | Path,
    vector_model_id: str,
    run: RunContext,
    trait_ids: list[str] | None = None,
    all_traits: bool = False,
    vectors_root: str | Path = "artifacts/vectors",
    output_root: str | Path = "results",
) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for projection computation") from exc

    shifts_path = Path(shifts_path)
    shifts = torch.load(shifts_path, map_location="cpu")
    selected_traits = all_trait_ids() if all_traits else list(trait_ids or [])
    if not selected_traits:
        raise ValueError("pass trait_ids or all_traits=True")
    rows = []
    model_id = shifts["metadata"]["model_id"]
    neutral_bank = shifts["metadata"]["neutral_bank"]
    deltas = shifts["deltas"]
    shift_norms = shifts["shift_norms"]
    for trait_id in selected_traits:
        for layer_key, delta in _items(deltas):
            layer = int(layer_key)
            vector_path = Path(vectors_root) / vector_model_id / trait_id / f"layer_{layer:03d}.pt"
            if not vector_path.exists():
                continue
            vector_payload = torch.load(vector_path, map_location="cpu")
            unit_vector = vector_payload["unit_vector"].float()
            delta = delta.float()
            if tuple(delta.shape) != tuple(unit_vector.shape):
                raise ValueError(f"shape mismatch for {trait_id} layer {layer}: delta={tuple(delta.shape)} vector={tuple(unit_vector.shape)}")
            projection = float(torch.dot(delta, unit_vector).item())
            rows.append(
                {
                    "model_id": model_id,
                    "neutral_bank": neutral_bank,
                    "layer": layer,
                    "trait_id": trait_id,
                    "projection": projection,
                    "shift_norm": float(_get_layer(shift_norms, layer)),
                    "shift_path": str(shifts_path),
                    "vector_path": str(vector_path),
                    "vector_model_id": vector_model_id,
                }
            )
    run_out = run.run_dir / "results" / "projections.csv"
    write_projection_csv(run_out, rows)
    canonical_out = Path(output_root) / "projections.csv"
    append_projection_csv(canonical_out, rows)
    run.update_progress(counters={"projection_rows": len(rows)})
    return {
        "run_path": str(run_out),
        "canonical_path": str(canonical_out),
        "rows": len(rows),
        "traits": selected_traits,
    }


def write_projection_csv(path: str | Path, rows: list[dict]) -> None:
    path = ensure_parent(path)
    fieldnames = ["model_id", "neutral_bank", "layer", "trait_id", "projection", "shift_norm", "vector_model_id", "shift_path", "vector_path"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_projection_csv(path: str | Path, rows: list[dict]) -> None:
    path = ensure_parent(path)
    fieldnames = ["model_id", "neutral_bank", "layer", "trait_id", "projection", "shift_norm", "vector_model_id", "shift_path", "vector_path"]
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _items(mapping: dict):
    for key, value in mapping.items():
        yield int(key), value


def _get_layer(mapping: dict, layer: int):
    if layer in mapping:
        return mapping[layer]
    if str(layer) in mapping:
        return mapping[str(layer)]
    raise KeyError(layer)

