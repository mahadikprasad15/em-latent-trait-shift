"""Construct per-layer trait vectors from pooled rollout activations."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from em_latent_factors.artifacts import RunContext, write_json
from em_latent_factors.io import ensure_parent


def construct_trait_vectors(
    pooled_activations_path: str | Path,
    run: RunContext,
    trait_ids: list[str] | None = None,
    all_traits: bool = False,
    model_id: str | None = None,
    output_root: str | Path = "artifacts/vectors",
    question_kind: str = "extraction",
    min_count_per_pole: int = 20,
) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for trait vector construction") from exc

    pooled_activations_path = Path(pooled_activations_path)
    payload = torch.load(pooled_activations_path, map_location="cpu")
    metadata = payload.get("metadata", {})
    model_id = model_id or metadata.get("model_id")
    if not model_id:
        raise ValueError("model_id is required")
    record_metadata = payload["record_metadata"]
    activations = payload["activations"]
    available_traits = sorted({row.get("trait_id") for row in record_metadata if row.get("trait_id")})
    selected_traits = available_traits if all_traits else list(trait_ids or [])
    if not selected_traits:
        raise ValueError("pass trait_ids or all_traits=True")

    results = {}
    for trait_id in selected_traits:
        trait_result = construct_one_trait(
            trait_id=trait_id,
            model_id=model_id,
            payload=payload,
            pooled_activations_path=pooled_activations_path,
            run=run,
            output_root=Path(output_root),
            question_kind=question_kind,
            min_count_per_pole=min_count_per_pole,
            torch=torch,
        )
        results[trait_id] = trait_result
    run.update_progress(counters={"constructed_traits": len(results)})
    return results


def construct_one_trait(
    trait_id: str,
    model_id: str,
    payload: dict,
    pooled_activations_path: Path,
    run: RunContext,
    output_root: Path,
    question_kind: str,
    min_count_per_pole: int,
    torch,
) -> dict[str, Any]:
    record_metadata = payload["record_metadata"]
    activations = payload["activations"]
    layer_selection = payload.get("layer_selection", {})
    model_name = payload.get("metadata", {}).get("model_name")

    pos_indices = [
        idx
        for idx, row in enumerate(record_metadata)
        if row.get("trait_id") == trait_id and row.get("pole") == "positive" and row.get("question_kind", "extraction") == question_kind
    ]
    neg_indices = [
        idx
        for idx, row in enumerate(record_metadata)
        if row.get("trait_id") == trait_id and row.get("pole") == "negative" and row.get("question_kind", "extraction") == question_kind
    ]
    if len(pos_indices) < min_count_per_pole or len(neg_indices) < min_count_per_pole:
        raise ValueError(
            f"{trait_id}: insufficient records for vector construction: "
            f"positive={len(pos_indices)} negative={len(neg_indices)} min={min_count_per_pole}"
        )

    run_trait_dir = run.run_dir / "results" / "vectors" / model_id / trait_id
    canonical_trait_dir = output_root / model_id / trait_id
    run_trait_dir.mkdir(parents=True, exist_ok=True)
    canonical_trait_dir.mkdir(parents=True, exist_ok=True)

    layer_summaries = []
    for layer_key, tensor in sorted(_activation_items(activations), key=lambda item: int(item[0])):
        layer = int(layer_key)
        pos = tensor[pos_indices].float()
        neg = tensor[neg_indices].float()
        positive_mean = pos.mean(dim=0)
        negative_mean = neg.mean(dim=0)
        vector = positive_mean - negative_mean
        norm = torch.linalg.vector_norm(vector).item()
        if norm == 0.0:
            unit_vector = vector
        else:
            unit_vector = vector / norm
        layer_payload = {
            "trait_id": trait_id,
            "model_id": model_id,
            "model_name": model_name,
            "layer": layer,
            "vector": vector,
            "unit_vector": unit_vector,
            "norm": norm,
            "positive_count": len(pos_indices),
            "negative_count": len(neg_indices),
            "positive_mean": positive_mean,
            "negative_mean": negative_mean,
            "source_activation_file": str(pooled_activations_path),
        }
        filename = f"layer_{layer:03d}.pt"
        run_path = run_trait_dir / filename
        canonical_path = canonical_trait_dir / filename
        torch.save(layer_payload, run_path)
        shutil.copy2(run_path, canonical_path)
        layer_summaries.append(
            {
                "layer": layer,
                "norm": norm,
                "positive_count": len(pos_indices),
                "negative_count": len(neg_indices),
                "run_path": str(run_path),
                "canonical_path": str(canonical_path),
            }
        )

    meta = {
        "trait_id": trait_id,
        "model_id": model_id,
        "model_name": model_name,
        "pooling_mode": payload.get("metadata", {}).get("pooling_mode"),
        "source_activation_file": str(pooled_activations_path),
        "layer_selection": layer_selection,
        "question_kind": question_kind,
        "vector_form": "positive_mean_minus_negative_mean",
        "normalize_per_layer": True,
        "counts": {
            "positive": len(pos_indices),
            "negative": len(neg_indices),
        },
        "layers": layer_summaries,
    }
    write_json(run_trait_dir / "metadata.json", meta)
    write_json(canonical_trait_dir / "metadata.json", meta)
    return meta


def _activation_items(activations: dict):
    for key, value in activations.items():
        yield key, value


def inspect_pooled_activation_metadata(path: str | Path) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required to inspect pooled activation tensors") from exc
    payload = torch.load(path, map_location="cpu")
    records = payload.get("record_metadata", [])
    counts: dict[str, dict[str, int]] = {}
    for row in records:
        trait_id = str(row.get("trait_id"))
        pole = str(row.get("pole"))
        counts.setdefault(trait_id, {})
        counts[trait_id][pole] = counts[trait_id].get(pole, 0) + 1
    return {
        "metadata": payload.get("metadata", {}),
        "layer_selection": payload.get("layer_selection", {}),
        "n_records": len(records),
        "counts": counts,
        "activation_layers": sorted(int(k) for k in payload.get("activations", {})),
    }

