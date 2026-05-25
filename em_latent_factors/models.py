"""Model/tokenizer loading helpers for generation and activation scripts."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    model_name: str
    adapter_path: str | None = None


def resolve_hf_token(explicit_token: str | None = None) -> str | None:
    return explicit_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def load_causal_lm(
    model_name: str,
    adapter_path: str | None = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    hf_token: str | None = None,
) -> LoadedModel:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch and transformers are required for model loading") from exc

    token = resolve_hf_token(hf_token)
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
    if adapter_path:
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise RuntimeError("peft is required to load LoRA adapters") from exc
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer, model_name=model_name, adapter_path=adapter_path)


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

