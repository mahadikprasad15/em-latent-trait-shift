"""Neutral prompt activation extraction for base and adapter models."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import shutil

from em_latent_factors.activations import infer_num_hidden_layers_from_model_config, pool_hidden_state, prompt_last_span, resolve_layers_from_config
from em_latent_factors.artifacts import RunContext
from em_latent_factors.generation import row_to_messages
from em_latent_factors.io import ensure_parent, read_jsonl, write_jsonl
from em_latent_factors.models import load_causal_lm


def load_neutral_rows(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    for row in read_jsonl(path):
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def neutral_prompt_id(row: dict) -> str:
    return str(row.get("prompt_id") or row.get("id") or "unknown_prompt")


def write_neutral_dry_run_metadata(
    run: RunContext,
    rows: list[dict],
    model_id: str,
    model_name: str,
    neutral_bank: str,
) -> Path:
    out = run.run_dir / "results" / "neutral_activation_metadata.jsonl"
    records = []
    for row in rows:
        records.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "neutral_bank": neutral_bank,
                "prompt_id": neutral_prompt_id(row),
                "prompt_preview": str(row.get("prompt", ""))[:300],
                "source": row.get("source"),
                "metadata": row.get("metadata", {}),
            }
        )
    write_jsonl(out, records)
    run.update_progress(completed_units=[r["prompt_id"] for r in records], counters={"metadata_records": len(records)})
    return out


def extract_neutral_mean_activations(
    run: RunContext,
    rows: list[dict],
    model_id: str,
    model_name: str,
    neutral_bank: str,
    batch_size: int = 4,
    config_path: str = "configs/experiment.yaml",
    adapter_path: str | None = None,
    hf_token: str | None = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    flush_every_batches: int = 8,
    output_root: str | Path = "artifacts/activations",
) -> Path:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for neutral activation extraction") from exc

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
    sums: dict[int, Any] = {}
    count = 0
    completed = []

    for batch_idx, start in enumerate(range(0, len(rows), batch_size)):
        batch_rows = rows[start : start + batch_size]
        messages = [row_to_messages(row) for row in batch_rows]
        input_ids = [
            tokenizer.apply_chat_template(row_messages, tokenize=True, add_generation_prompt=True)
            for row_messages in messages
        ]
        encoded = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        for row_idx, row in enumerate(batch_rows):
            span = prompt_last_span(encoded["attention_mask"][row_idx])
            for logical_layer, hidden_idx in zip(layer_selection.logical_layers, layer_selection.hidden_state_indices):
                hidden_state = outputs.hidden_states[hidden_idx][row_idx].detach()
                pooled = pool_hidden_state(hidden_state, span, "prompt_last").float().to("cpu")
                if logical_layer not in sums:
                    sums[logical_layer] = pooled.clone()
                else:
                    sums[logical_layer] += pooled
            count += 1
            completed.append(neutral_prompt_id(row))
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (batch_idx + 1) % flush_every_batches == 0:
            partial = run.run_dir / "checkpoints" / "neutral_mean_activations_partial.pt"
            save_mean_activations(
                partial,
                model_id=model_id,
                model_name=model_name,
                neutral_bank=neutral_bank,
                layer_selection=layer_selection.to_json(),
                sums=sums,
                count=count,
                torch=torch,
                final=False,
            )
            run.update_progress(completed_units=completed, counters={"neutral_activation_count": count})
            completed = []

    means = {layer: value / count for layer, value in sums.items()}
    run_out = run.run_dir / "results" / "mean_activations.pt"
    save_mean_activation_payload(
        run_out,
        model_id=model_id,
        model_name=model_name,
        neutral_bank=neutral_bank,
        layer_selection=layer_selection.to_json(),
        means=means,
        count=count,
        torch=torch,
    )
    canonical_dir = Path(output_root) / model_id / neutral_bank
    canonical_out = canonical_dir / "mean_activations.pt"
    ensure_parent(canonical_out)
    shutil.copy2(run_out, canonical_out)
    run.update_progress(completed_units=completed, counters={"neutral_activation_count": count})
    return run_out


def save_mean_activations(
    path: str | Path,
    model_id: str,
    model_name: str,
    neutral_bank: str,
    layer_selection: dict,
    sums: dict[int, Any],
    count: int,
    torch,
    final: bool,
) -> None:
    path = ensure_parent(path)
    torch.save(
        {
            "metadata": {
                "model_id": model_id,
                "model_name": model_name,
                "neutral_bank": neutral_bank,
                "pooling_mode": "prompt_last",
                "final": final,
            },
            "layer_selection": layer_selection,
            "count": count,
            "sums": sums,
        },
        path,
    )


def save_mean_activation_payload(
    path: str | Path,
    model_id: str,
    model_name: str,
    neutral_bank: str,
    layer_selection: dict,
    means: dict[int, Any],
    count: int,
    torch,
) -> None:
    path = ensure_parent(path)
    torch.save(
        {
            "metadata": {
                "model_id": model_id,
                "model_name": model_name,
                "neutral_bank": neutral_bank,
                "pooling_mode": "prompt_last",
            },
            "layer_selection": layer_selection,
            "count": count,
            "means": means,
        },
        path,
    )

