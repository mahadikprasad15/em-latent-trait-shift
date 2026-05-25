#!/usr/bin/env python3
"""Validate the three locked v1 SFT datasets after acquisition."""

from __future__ import annotations

import json
from pathlib import Path


DATASETS = {
    "ft_health_bad_advice": Path("data/ft/health_incorrect.jsonl"),
    "ft_finance_bad_advice": Path("data/ft/finance_incorrect.jsonl"),
    "ft_insecure_code": Path("data/ft/insecure_code.jsonl"),
}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    yield line_no, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def content_text(content) -> str:
    if isinstance(content, dict) and set(content) == {"content_type", "parts"}:
        return "\n".join(str(part) for part in content["parts"])
    if isinstance(content, str):
        return content
    raise TypeError(f"unsupported content type: {type(content).__name__}")


def validate_messages(path: Path, line_no: int, row: dict) -> None:
    if "messages" not in row or not isinstance(row["messages"], list):
        raise ValueError(f"{path}:{line_no}: missing list field 'messages'")
    if "canary" not in row or not isinstance(row["canary"], str):
        raise ValueError(f"{path}:{line_no}: missing string field 'canary'")
    for idx, message in enumerate(row["messages"]):
        if not isinstance(message, dict):
            raise ValueError(f"{path}:{line_no}: messages[{idx}] is not an object")
        if set(message) != {"role", "content"}:
            raise ValueError(f"{path}:{line_no}: messages[{idx}] keys are {sorted(message)}")
        if message["role"] not in {"system", "user", "assistant"}:
            raise ValueError(f"{path}:{line_no}: invalid role {message['role']!r}")
        content = message["content"]
        if isinstance(content, dict):
            if set(content) != {"content_type", "parts"} or content["content_type"] != "text" or not isinstance(content["parts"], list):
                raise ValueError(f"{path}:{line_no}: unsupported structured content at index {idx}")
        if not isinstance(content, (str, dict)):
            raise ValueError(f"{path}:{line_no}: unsupported message content at index {idx}")
        text = content_text(content)
        if message["role"] != "system" and not text.strip():
            raise ValueError(f"{path}:{line_no}: empty non-system message content at index {idx}")


def main() -> None:
    for dataset_id, path in DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        count = 0
        prompt_fingerprints: set[str] = set()
        duplicate_prompts = 0
        for line_no, row in iter_jsonl(path):
            validate_messages(path, line_no, row)
            count += 1
            user_text = "\n".join(content_text(m["content"]).strip() for m in row["messages"] if m["role"] == "user")
            if user_text in prompt_fingerprints:
                duplicate_prompts += 1
            prompt_fingerprints.add(user_text)
        print(f"{dataset_id}: rows={count} duplicate_user_prompts={duplicate_prompts} path={path}")


if __name__ == "__main__":
    main()
