"""LoRA SFT training utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
from typing import Any

from em_latent_factors.artifacts import RunContext, write_json
from em_latent_factors.config import load_yaml
from em_latent_factors.io import ensure_parent, read_jsonl
from em_latent_factors.models import resolve_hf_token


@dataclass(frozen=True)
class TokenizedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict) and set(content) == {"content_type", "parts"}:
        return "\n".join(str(part) for part in content["parts"])
    return str(content)


def normalize_messages(messages: list[dict]) -> list[dict[str, str]]:
    out = []
    for message in messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        out.append({"role": role, "content": normalize_content(message.get("content", ""))})
    return out


def find_last_subsequence(values: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(values):
        return None
    last_start = len(values) - len(pattern)
    for start in range(last_start, -1, -1):
        if values[start : start + len(pattern)] == pattern:
            return start
    return None


def token_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def tokenize_response_only(messages: list[dict], tokenizer, max_seq_length: int) -> TokenizedExample | None:
    messages = normalize_messages(messages)
    if not messages or messages[-1]["role"] != "assistant":
        return None
    prompt_messages = messages[:-1]
    if not prompt_messages:
        return None
    prompt_ids = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if len(full_ids) > max_seq_length:
        full_ids = full_ids[:max_seq_length]
    labels = [-100] * len(full_ids)
    assistant_content_ids = token_ids(tokenizer, messages[-1]["content"])
    assistant_start = find_last_subsequence(full_ids, assistant_content_ids)
    if assistant_start is None:
        assistant_start = min(len(prompt_ids), len(full_ids))
        if assistant_start >= len(full_ids):
            return None
    labels[assistant_start:] = full_ids[assistant_start:]
    if all(label == -100 for label in labels):
        return None
    return TokenizedExample(
        input_ids=full_ids,
        attention_mask=[1] * len(full_ids),
        labels=labels,
    )


def load_and_tokenize_sft_dataset(
    path: str | Path,
    tokenizer,
    max_seq_length: int,
    limit: int | None = None,
    seed: int = 0,
) -> tuple[list[TokenizedExample], dict[str, Any]]:
    rows = list(read_jsonl(path))
    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit is not None:
        rows = rows[:limit]
    tokenized = []
    skipped = 0
    for row in rows:
        example = tokenize_response_only(row.get("messages", []), tokenizer=tokenizer, max_seq_length=max_seq_length)
        if example is None:
            skipped += 1
            continue
        tokenized.append(example)
    manifest = {
        "path": str(path),
        "input_rows": len(rows),
        "tokenized_rows": len(tokenized),
        "skipped_rows": skipped,
        "max_seq_length": max_seq_length,
        "limit": limit,
        "seed": seed,
    }
    return tokenized, manifest


def resolve_target_modules(model, requested: list[str]) -> tuple[list[str], dict[str, bool]]:
    module_names = [name for name, _ in model.named_modules()]
    found = {}
    selected = []
    for target in requested:
        present = any(name.endswith(f".{target}") or name == target for name in module_names)
        found[target] = present
        if present:
            selected.append(target)
    attention_found = any(found.get(name) for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
    if not attention_found:
        raise ValueError(f"no attention projection target modules found: {found}")
    return selected, found


class ListDataset:
    def __init__(self, examples: list[TokenizedExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]
        return {
            "input_ids": example.input_ids,
            "attention_mask": example.attention_mask,
            "labels": example.labels,
        }


class ResponseOnlyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("torch is required for training") from exc
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad)
            batch["labels"].append(feature["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def train_lora_adapter(
    run: RunContext,
    model_id: str,
    model_name: str,
    dataset_id: str,
    dataset_path: str | Path,
    seed: int,
    config_path: str = "configs/experiment.yaml",
    output_root: str | Path = "checkpoints",
    limit: int | None = None,
    hf_token: str | None = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
        from peft import LoraConfig, TaskType, get_peft_model
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch, transformers, and peft are required for LoRA training") from exc

    config = load_yaml(config_path)
    ft_config = config["fine_tuning"]
    token = resolve_hf_token(hf_token)
    set_seed(seed)
    dtype = _resolve_torch_dtype(torch, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        torch_dtype=dtype,
        device_map=device_map,
    )
    requested_targets = list(ft_config["lora"]["target_modules"])
    target_modules, target_report = resolve_target_modules(model, requested_targets)
    lora_config = LoraConfig(
        r=int(ft_config["lora"]["r"]),
        lora_alpha=int(ft_config["lora"]["alpha"]),
        lora_dropout=float(ft_config["lora"]["dropout"]),
        target_modules=target_modules,
        use_rslora=True,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    tokenized, dataset_manifest = load_and_tokenize_sft_dataset(
        dataset_path,
        tokenizer=tokenizer,
        max_seq_length=int(ft_config["max_seq_length"]),
        limit=limit,
        seed=seed,
    )
    run_inputs_manifest = {
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "dataset_manifest": dataset_manifest,
        "target_modules": target_report,
        "selected_target_modules": target_modules,
    }
    write_json(run.run_dir / "inputs" / "dataset_manifest.json", run_inputs_manifest)
    if not tokenized:
        raise ValueError(f"no tokenized training examples; see {run.run_dir / 'inputs' / 'dataset_manifest.json'}")
    train_dataset = ListDataset(tokenized)
    canonical_dir = Path(output_root) / model_id
    run_adapter_dir = run.run_dir / "checkpoints" / "adapter"
    training_args = TrainingArguments(
        output_dir=str(run.run_dir / "checkpoints" / "trainer"),
        num_train_epochs=float(ft_config["epochs"]),
        learning_rate=float(ft_config["learning_rate"]),
        per_device_train_batch_size=int(ft_config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(ft_config["gradient_accumulation_steps"]),
        warmup_steps=int(ft_config["warmup_steps"]),
        weight_decay=float(ft_config["weight_decay"]),
        lr_scheduler_type=str(ft_config["lr_scheduler_type"]),
        optim=str(ft_config["optimizer"]),
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        seed=seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=ResponseOnlyCollator(tokenizer),
    )
    train_output = trainer.train()
    run_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(run_adapter_dir)
    tokenizer.save_pretrained(run_adapter_dir)
    canonical_adapter_dir = canonical_dir / "adapter"
    if canonical_adapter_dir.exists():
        shutil.rmtree(canonical_adapter_dir)
    canonical_adapter_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_adapter_dir, canonical_adapter_dir)
    train_config = {
        "model_id": model_id,
        "model_name": model_name,
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "seed": seed,
        "fine_tuning": ft_config,
        "target_modules": target_report,
        "selected_target_modules": target_modules,
        "limit": limit,
    }
    metrics = dict(train_output.metrics)
    write_json(run.run_dir / "results" / "train_metrics.json", metrics)
    write_json(run.run_dir / "results" / "train_config.json", train_config)
    write_json(canonical_dir / "train_metrics.json", metrics)
    write_json(canonical_dir / "train_config.json", train_config)
    write_json(canonical_dir / "dataset_manifest.json", run_inputs_manifest)
    run.update_progress(counters={"train_examples": len(tokenized)})
    return {
        "model_id": model_id,
        "run_adapter_dir": str(run_adapter_dir),
        "canonical_adapter_dir": str(canonical_adapter_dir),
        "train_metrics": metrics,
        "train_config": train_config,
    }


def _resolve_torch_dtype(torch_module, value: str):
    if value == "auto":
        return "auto"
    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"unsupported torch dtype: {value}") from exc
