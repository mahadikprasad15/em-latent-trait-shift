"""Dataset validation routines."""

from __future__ import annotations

from pathlib import Path

from em_latent_factors.datasets.registry import iter_dataset_specs, load_dataset_config
from em_latent_factors.io import read_jsonl


def content_text(content) -> str:
    if isinstance(content, dict) and set(content) == {"content_type", "parts"}:
        return "\n".join(str(part) for part in content["parts"])
    if isinstance(content, str):
        return content
    raise TypeError(f"unsupported content type: {type(content).__name__}")


def validate_sft_jsonl(dataset_id: str, path: Path) -> dict:
    count = 0
    duplicate_user_prompts = 0
    prompts: set[str] = set()
    for line_no, row in enumerate(read_jsonl(path), start=1):
        if "messages" not in row or not isinstance(row["messages"], list):
            raise ValueError(f"{path}:{line_no}: missing list field messages")
        if "canary" not in row or not isinstance(row["canary"], str):
            raise ValueError(f"{path}:{line_no}: missing string field canary")
        user_parts = []
        for idx, message in enumerate(row["messages"]):
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError(f"{path}:{line_no}: invalid message at index {idx}")
            if message["role"] not in {"system", "user", "assistant"}:
                raise ValueError(f"{path}:{line_no}: invalid role at index {idx}")
            text = content_text(message["content"])
            if message["role"] != "system" and not text.strip():
                raise ValueError(f"{path}:{line_no}: empty non-system content at index {idx}")
            if message["role"] == "user":
                user_parts.append(text.strip())
        prompt = "\n".join(user_parts)
        if prompt in prompts:
            duplicate_user_prompts += 1
        prompts.add(prompt)
        count += 1
    return {"dataset_id": dataset_id, "path": str(path), "rows": count, "duplicate_user_prompts": duplicate_user_prompts}


def validate_prompt_jsonl(dataset_id: str, path: Path, required_fields: tuple[str, ...]) -> dict:
    count = 0
    duplicate_prompts = 0
    prompts: set[str] = set()
    for line_no, row in enumerate(read_jsonl(path), start=1):
        for field in required_fields:
            if field not in row:
                raise ValueError(f"{path}:{line_no}: missing field {field}")
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"{path}:{line_no}: empty prompt")
        fingerprint = " ".join(prompt.split()).lower()
        if fingerprint in prompts:
            duplicate_prompts += 1
        prompts.add(fingerprint)
        count += 1
    return {"dataset_id": dataset_id, "path": str(path), "rows": count, "duplicate_prompts": duplicate_prompts}


def validate_dataset(dataset_id: str, group: str, path: str, fields: tuple[str, ...]) -> dict:
    p = Path(path)
    if not p.exists():
        return {"dataset_id": dataset_id, "path": path, "status": "missing"}
    if group == "ft_datasets":
        result = validate_sft_jsonl(dataset_id, p)
    else:
        required = tuple(fields) if fields else ("prompt_id", "prompt")
        result = validate_prompt_jsonl(dataset_id, p, required)
    result["status"] = "ok"
    return result


def validate_all(config_path: str = "configs/datasets.yaml", include_missing: bool = True) -> list[dict]:
    config = load_dataset_config(config_path)
    results = []
    for spec in iter_dataset_specs(config):
        result = validate_dataset(spec.dataset_id, spec.group, spec.local_path, spec.fields)
        if include_missing or result.get("status") != "missing":
            results.append(result)
    return results

