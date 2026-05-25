"""Judge cache helpers."""

from __future__ import annotations

from pathlib import Path

from em_latent_factors.io import read_jsonl


def load_judge_cache(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    cache: dict[str, dict] = {}
    for row in read_jsonl(path):
        key = row.get("judge_key")
        if key:
            cache[str(key)] = row
    return cache

