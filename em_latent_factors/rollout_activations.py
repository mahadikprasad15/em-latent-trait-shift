"""Pooled activation extraction for trait-vector rollout responses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import shutil

from em_latent_factors.activations import (
    build_prompt_and_full_ids_for_response,
    infer_num_hidden_layers_from_model_config,
    pool_hidden_state,
    resolve_layers_from_config,
)
from em_latent_factors.artifacts import RunContext
from em_latent_factors.io import ensure_parent, read_jsonl, write_jsonl
from em_latent_factors.models import load_causal_lm
from em_latent_factors.vectors import all_trait_ids


def rollout_files_for_trait(trait_id: str, root: str | Path = "data/vector_rollouts") -> list[Path]:
    base = Path(root) / trait_id
    return [base / "positive.jsonl", base / "negative.jsonl"]


def resolve_rollout_input_files(
    trait_ids: list[str] | None = None,
    input_files: list[str] | None = None,
    all_traits: bool = False,
    root: str | Path = "data/vector_rollouts",
) -> list[Path]:
    paths: list[Path] = [Path(p) for p in (input_files or [])]
    ids = list(trait_ids or [])
    if all_traits:
        ids.extend(all_trait_ids())
    for trait_id in dict.fromkeys(ids):
        paths.extend(rollout_files_for_trait(trait_id, root=root))
    existing = []
    for path in paths:
        if path.exists():
            existing.append(path)
        else:
            raise FileNotFoundError(path)
    return existing


def load_rollout_rows(paths: Iterable[str | Path], limit: int | None = None) -> list[dict]:
    rows = []
    for path in paths:
        for row in read_jsonl(path):
            row["_source_file"] = str(path)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def rollout_prompt_id(row: dict) -> str:
    return str(row.get("prompt_id") or f"{row.get('trait_id')}:{row.get('pole')}:{row.get('instruction_id')}:{row.get('question_id')}:{row.get('rollout_id')}")


def record_metadata(row: dict) -> dict[str, Any]:
    return {
        "prompt_id": rollout_prompt_id(row),
        "trait_id": row.get("trait_id"),
        "pole": row.get("pole"),
        "instruction_id": row.get("instruction_id"),
        "question_id": row.get("question_id"),
        "question_kind": row.get("question_kind"),
        "rollout_id": row.get("rollout_id"),
        "source_file": row.get("_source_file"),
        "dry_run": row.get("dry_run"),
    }


def write_dry_run_metadata(run: RunContext, rows: list[dict], model_id: str, model_name: str) -> Path:
    out = run.run_dir / "results" / "pooled_activation_metadata.jsonl"
    records = []
    for row in rows:
        records.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "record_metadata": record_metadata(row),
                "messages": row.get("messages"),
                "response_preview": str(row.get("response", ""))[:200],
            }
        )
    write_jsonl(out, records)
    run.update_progress(completed_units=[r["record_metadata"]["prompt_id"] for r in records], counters={"metadata_records": len(records)})
    return out


def canonical_rollout_activation_path(
    model_id: str,
    rows: list[dict],
    output_root: str | Path = "artifacts/rollout_activations",
) -> Path:
    trait_ids = sorted({str(row.get("trait_id")) for row in rows if row.get("trait_id")})
    if len(trait_ids) == 1:
        selection = trait_ids[0]
    else:
        selection = "all_traits"
    return Path(output_root) / model_id / selection / "pooled_activations.pt"


def extract_rollout_activations(
    run: RunContext,
    rows: list[dict],
    model_id: str,
    model_name: str,
    batch_size: int = 4,
    config_path: str = "configs/experiment.yaml",
    adapter_path: str | None = None,
    hf_token: str | None = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    flush_every_batches: int = 8,
    output_root: str | Path = "artifacts/rollout_activations",
) -> Path:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for activation extraction") from exc

    loaded = load_causal_lm(
        model_name=model_name,
        adapter_path=adapter_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        hf_token=hf_token,
    )
    tokenizer = loaded.tokenizer
    model = loaded.model
    num_hidden_layers = infer_num_hidden_layers_from_model_config(model.config)
    layer_selection = resolve_layers_from_config(num_hidden_layers=num_hidden_layers, config_path=config_path)

    metadata_records: list[dict] = []
    layer_chunks: dict[int, list] = {layer: [] for layer in layer_selection.logical_layers}
    completed: list[str] = []

    for batch_idx, start in enumerate(range(0, len(rows), batch_size)):
        batch = rows[start : start + batch_size]
        batch_full_ids = []
        batch_spans = []
        for row in batch:
            prompt_ids, full_ids, span = build_prompt_and_full_ids_for_response(
                tokenizer=tokenizer,
                messages=row["messages"],
                response=str(row.get("response", "")),
            )
            batch_full_ids.append(full_ids)
            batch_spans.append(span)
        encoded = tokenizer.pad(
            {"input_ids": batch_full_ids},
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        for row_idx, row in enumerate(batch):
            span = batch_spans[row_idx]
            metadata_records.append(record_metadata(row))
            completed.append(rollout_prompt_id(row))
            for logical_layer, hidden_idx in zip(layer_selection.logical_layers, layer_selection.hidden_state_indices):
                hidden_state = outputs.hidden_states[hidden_idx][row_idx].detach()
                pooled = pool_hidden_state(hidden_state, span, "response_avg").to("cpu")
                layer_chunks[logical_layer].append(pooled)
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (batch_idx + 1) % flush_every_batches == 0:
            partial_path = run.run_dir / "checkpoints" / "pooled_activations_partial.pt"
            save_pooled_activation_file(
                partial_path,
                model_id=model_id,
                model_name=model_name,
                layer_selection=layer_selection.to_json(),
                metadata_records=metadata_records,
                layer_chunks=layer_chunks,
            )
            run.update_progress(completed_units=completed, counters={"activation_records": len(metadata_records)})
            completed = []

    out = run.run_dir / "results" / "pooled_activations.pt"
    save_pooled_activation_file(
        out,
        model_id=model_id,
        model_name=model_name,
        layer_selection=layer_selection.to_json(),
        metadata_records=metadata_records,
        layer_chunks=layer_chunks,
    )
    canonical_out = canonical_rollout_activation_path(model_id=model_id, rows=rows, output_root=output_root)
    ensure_parent(canonical_out)
    shutil.copy2(out, canonical_out)
    run.update_progress(completed_units=completed, counters={"activation_records": len(metadata_records)})
    return out


def save_pooled_activation_file(
    path: str | Path,
    model_id: str,
    model_name: str,
    layer_selection: dict,
    metadata_records: list[dict],
    layer_chunks: dict[int, list],
) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required to save activation tensors") from exc
    path = ensure_parent(path)
    activations = {}
    for layer, chunks in layer_chunks.items():
        if chunks:
            activations[int(layer)] = torch.stack(chunks, dim=0)
    torch.save(
        {
            "metadata": {
                "model_id": model_id,
                "model_name": model_name,
                "pooling_mode": "response_avg",
            },
            "layer_selection": layer_selection,
            "record_metadata": metadata_records,
            "activations": activations,
        },
        path,
    )
