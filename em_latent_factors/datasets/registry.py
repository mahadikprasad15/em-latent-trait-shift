"""Dataset registry accessors."""

from __future__ import annotations

from pathlib import Path

from em_latent_factors.config import load_yaml
from em_latent_factors.datasets.schemas import DatasetSpec


def load_dataset_config(path: str | Path = "configs/datasets.yaml") -> dict:
    return load_yaml(path)


def iter_dataset_specs(config: dict, groups: tuple[str, ...] = ("ft_datasets", "neutral_banks", "eval_datasets")):
    for group in groups:
        for dataset_id, entry in config.get(group, {}).items():
            yield DatasetSpec(
                dataset_id=dataset_id,
                local_path=entry["local_path"],
                group=group,
                source=entry.get("source", ""),
                fields=tuple(entry.get("fields", entry.get("required_fields", ()))),
            )


def get_dataset_entry(config: dict, dataset_id: str) -> tuple[str, dict]:
    for group in ("ft_datasets", "neutral_banks", "eval_datasets"):
        entries = config.get(group, {})
        if dataset_id in entries:
            return group, entries[dataset_id]
    raise KeyError(f"unknown dataset_id: {dataset_id}")

