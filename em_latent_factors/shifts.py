"""Activation shift computation from neutral mean activations."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from em_latent_factors.artifacts import RunContext
from em_latent_factors.io import ensure_parent


def compute_activation_shifts(
    base_activations_path: str | Path,
    finetuned_activations_path: str | Path,
    run: RunContext,
    model_id: str,
    base_model_id: str,
    neutral_bank: str | None = None,
    output_root: str | Path = "artifacts/shifts",
) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for activation shift computation") from exc

    base_activations_path = Path(base_activations_path)
    finetuned_activations_path = Path(finetuned_activations_path)
    base = torch.load(base_activations_path, map_location="cpu")
    finetuned = torch.load(finetuned_activations_path, map_location="cpu")
    neutral_bank = neutral_bank or finetuned.get("metadata", {}).get("neutral_bank")
    validate_activation_files(base, finetuned, neutral_bank=neutral_bank)
    deltas = {}
    shift_norms = {}
    for layer_key, ft_mean in _items(finetuned["means"]):
        layer = int(layer_key)
        base_mean = _get_layer(base["means"], layer)
        delta = ft_mean.float() - base_mean.float()
        deltas[layer] = delta
        shift_norms[layer] = float(torch.linalg.vector_norm(delta).item())
    payload = {
        "metadata": {
            "model_id": model_id,
            "base_model_id": base_model_id,
            "neutral_bank": neutral_bank,
            "shift_form": "finetuned_minus_base",
            "pooling_mode": finetuned.get("metadata", {}).get("pooling_mode"),
            "base_activations_path": str(base_activations_path),
            "finetuned_activations_path": str(finetuned_activations_path),
        },
        "layer_selection": finetuned.get("layer_selection"),
        "counts": {
            "base": base.get("count"),
            "finetuned": finetuned.get("count"),
        },
        "deltas": deltas,
        "shift_norms": shift_norms,
    }
    run_out = run.run_dir / "results" / "activation_shifts.pt"
    ensure_parent(run_out)
    torch.save(payload, run_out)
    canonical_out = Path(output_root) / model_id / str(neutral_bank) / "activation_shifts.pt"
    ensure_parent(canonical_out)
    shutil.copy2(run_out, canonical_out)
    run.update_progress(counters={"shift_layers": len(deltas)})
    return {
        "run_path": str(run_out),
        "canonical_path": str(canonical_out),
        "layers": sorted(deltas),
        "shift_norms": shift_norms,
    }


def validate_activation_files(base: dict, finetuned: dict, neutral_bank: str | None) -> None:
    base_meta = base.get("metadata", {})
    ft_meta = finetuned.get("metadata", {})
    if neutral_bank and base_meta.get("neutral_bank") != neutral_bank:
        raise ValueError(f"base neutral_bank mismatch: {base_meta.get('neutral_bank')} != {neutral_bank}")
    if neutral_bank and ft_meta.get("neutral_bank") != neutral_bank:
        raise ValueError(f"finetuned neutral_bank mismatch: {ft_meta.get('neutral_bank')} != {neutral_bank}")
    if base_meta.get("pooling_mode") != ft_meta.get("pooling_mode"):
        raise ValueError(f"pooling mismatch: {base_meta.get('pooling_mode')} != {ft_meta.get('pooling_mode')}")
    base_layers = sorted(int(k) for k in base.get("means", {}))
    ft_layers = sorted(int(k) for k in finetuned.get("means", {}))
    if base_layers != ft_layers:
        raise ValueError(f"layer mismatch: base={base_layers} finetuned={ft_layers}")
    for layer in base_layers:
        if tuple(_get_layer(base["means"], layer).shape) != tuple(_get_layer(finetuned["means"], layer).shape):
            raise ValueError(f"hidden dimension mismatch at layer {layer}")


def _items(mapping: dict):
    for key, value in mapping.items():
        yield int(key), value


def _get_layer(mapping: dict, layer: int):
    if layer in mapping:
        return mapping[layer]
    if str(layer) in mapping:
        return mapping[str(layer)]
    raise KeyError(layer)

