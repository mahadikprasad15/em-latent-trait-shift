"""Shared text generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from em_latent_factors.artifacts import RunContext
from em_latent_factors.config import load_yaml
from em_latent_factors.io import read_jsonl
from em_latent_factors.models import LoadedModel, load_causal_lm


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    num_samples_per_prompt: int = 1
    batch_size: int = 4
    system_prompt: str | None = None

    @classmethod
    def from_experiment_config(cls, path: str = "configs/experiment.yaml") -> "GenerationConfig":
        config = load_yaml(path).get("behavior_generation", {})
        return cls(
            temperature=float(config.get("temperature", 0.0)),
            top_p=float(config.get("top_p", 1.0)),
            max_new_tokens=int(config.get("max_new_tokens", 512)),
            num_samples_per_prompt=int(config.get("num_samples_per_prompt", 1)),
            batch_size=int(config.get("batch_size", 4)),
            system_prompt=config.get("system_prompt"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "num_samples_per_prompt": self.num_samples_per_prompt,
            "batch_size": self.batch_size,
            "system_prompt": self.system_prompt,
        }


def load_prompt_rows(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    for row in read_jsonl(path):
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def row_to_messages(row: dict, system_prompt: str | None = None) -> list[dict[str, str]]:
    if "messages" in row and isinstance(row["messages"], list):
        return _normalize_messages(row["messages"], system_prompt=system_prompt)
    prompt = str(row.get("prompt", "")).strip()
    if not prompt:
        raise ValueError(f"row has no prompt/messages: {row}")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _normalize_messages(messages: list, system_prompt: str | None = None) -> list[dict[str, str]]:
    out = []
    if system_prompt:
        out.append({"role": "system", "content": system_prompt})
    for message in messages:
        role = message.get("role")
        content = _content_text(message.get("content"))
        if role in {"system", "user", "assistant"}:
            if role == "system" and system_prompt:
                continue
            out.append({"role": role, "content": content})
    return out


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict) and set(content) == {"content_type", "parts"}:
        return "\n".join(str(part) for part in content["parts"])
    return str(content)


def format_prompt_with_chat_template(tokenizer, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return fallback_chat_format(messages)


def fallback_chat_format(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message["role"].capitalize()
        parts.append(f"{role}: {message['content']}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def existing_prompt_ids(output_path: str | Path) -> set[str]:
    path = Path(output_path)
    if not path.exists():
        return set()
    ids = set()
    for row in read_jsonl(path):
        prompt_id = row.get("prompt_id")
        sample_id = row.get("sample_id")
        if prompt_id is not None and sample_id is not None:
            ids.add(f"{prompt_id}::{sample_id}")
    return ids


def build_generation_row(
    input_row: dict,
    messages: list[dict[str, str]],
    response: str,
    model_id: str,
    model_name: str,
    generation_config: GenerationConfig,
    sample_id: int = 0,
    dry_run: bool = False,
) -> dict:
    prompt_id = str(input_row.get("prompt_id") or input_row.get("id") or input_row.get("task_id") or "unknown_prompt")
    return {
        "model_id": model_id,
        "model_name": model_name,
        "prompt_id": prompt_id,
        "sample_id": sample_id,
        "eval_id": input_row.get("eval_id"),
        "trait_id": input_row.get("trait_id"),
        "category": input_row.get("category"),
        "prompt": input_row.get("prompt"),
        "messages": messages,
        "response": response,
        "generation_config": generation_config.to_json(),
        "dry_run": dry_run,
        "metadata": {
            "input_row": input_row,
        },
    }


def generate_dry_run_rows(
    rows: Iterable[dict],
    model_id: str,
    model_name: str,
    generation_config: GenerationConfig,
) -> Iterable[dict]:
    for row in rows:
        messages = row_to_messages(row, system_prompt=generation_config.system_prompt)
        for sample_id in range(generation_config.num_samples_per_prompt):
            yield build_generation_row(
                input_row=row,
                messages=messages,
                response="[DRY RUN: no model response generated]",
                model_id=model_id,
                model_name=model_name,
                generation_config=generation_config,
                sample_id=sample_id,
                dry_run=True,
            )


def generate_model_rows(
    rows: list[dict],
    loaded: LoadedModel,
    model_id: str,
    generation_config: GenerationConfig,
) -> Iterable[dict]:
    tokenizer = loaded.tokenizer
    model = loaded.model
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for model generation") from exc

    for start in range(0, len(rows), generation_config.batch_size):
        batch_rows = rows[start : start + generation_config.batch_size]
        messages_batch = [row_to_messages(row, system_prompt=generation_config.system_prompt) for row in batch_rows]
        prompts = [format_prompt_with_chat_template(tokenizer, messages) for messages in messages_batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        do_sample = generation_config.temperature > 0
        generation_kwargs = {
            "max_new_tokens": generation_config.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update({"temperature": generation_config.temperature, "top_p": generation_config.top_p})
        with torch.no_grad():
            outputs = model.generate(**encoded, **generation_kwargs)
        input_len = encoded["input_ids"].shape[-1]
        for row, messages, output_ids in zip(batch_rows, messages_batch, outputs):
            new_tokens = output_ids[input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            yield build_generation_row(
                input_row=row,
                messages=messages,
                response=response,
                model_id=model_id,
                model_name=loaded.model_name,
                generation_config=generation_config,
                sample_id=0,
                dry_run=False,
            )


def run_generation(
    input_path: str | Path,
    run: RunContext,
    model_id: str,
    model_name: str,
    generation_config: GenerationConfig,
    backend: str = "dry_run",
    adapter_path: str | None = None,
    limit: int | None = None,
    force: bool = False,
    hf_token: str | None = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
) -> Path:
    output_path = run.run_dir / "results" / "generations.jsonl"
    rows = load_prompt_rows(input_path, limit=limit)
    done = set() if force else existing_prompt_ids(output_path)
    pending = []
    for row in rows:
        prompt_id = str(row.get("prompt_id") or row.get("id") or row.get("task_id") or "unknown_prompt")
        if all(f"{prompt_id}::{sample_id}" in done for sample_id in range(generation_config.num_samples_per_prompt)):
            continue
        pending.append(row)
    run.update_progress(counters={"input_rows": len(rows), "pending_rows": len(pending)}, cursor={"input_path": str(input_path)})
    print(
        f"generation_start run_id={run.run_id} model_id={model_id} backend={backend} "
        f"input_rows={len(rows)} pending_rows={len(pending)} batch_size={generation_config.batch_size}",
        flush=True,
    )
    if not pending:
        print(f"generation_skip run_id={run.run_id} reason=no_pending_rows", flush=True)
        return output_path
    if backend == "dry_run":
        generated = generate_dry_run_rows(pending, model_id=model_id, model_name=model_name, generation_config=generation_config)
    elif backend == "transformers":
        loaded = load_causal_lm(model_name, adapter_path=adapter_path, hf_token=hf_token, torch_dtype=torch_dtype, device_map=device_map)
        generated = generate_model_rows(pending, loaded=loaded, model_id=model_id, generation_config=generation_config)
    else:
        raise ValueError(f"unknown generation backend: {backend}")
    count = 0
    completed = []
    buffer = []
    flush_every = max(1, min(10, generation_config.batch_size))
    for row in generated:
        buffer.append(row)
        completed.append(f"{row['prompt_id']}::{row['sample_id']}")
        if len(buffer) >= flush_every:
            run.append_results_jsonl("generations.jsonl", buffer)
            count += len(buffer)
            run.update_progress(completed_units=completed, counters={"generated_rows": count})
            print(f"generation_progress run_id={run.run_id} generated_rows={count}/{len(pending)}", flush=True)
            buffer = []
            completed = []
    if buffer:
        run.append_results_jsonl("generations.jsonl", buffer)
        count += len(buffer)
        run.update_progress(completed_units=completed, counters={"generated_rows": count})
        print(f"generation_progress run_id={run.run_id} generated_rows={count}/{len(pending)}", flush=True)
    print(f"generation_done run_id={run.run_id} generated_rows={count}", flush=True)
    return output_path
