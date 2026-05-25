"""Canonical dataset schemas."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    local_path: str
    group: str
    source: str
    fields: tuple[str, ...]


EVAL_REQUIRED_FIELDS = ("prompt_id", "eval_id", "prompt", "category", "source", "metadata")
NEUTRAL_REQUIRED_FIELDS = ("prompt_id", "prompt", "neutral_bank", "source", "metadata")

