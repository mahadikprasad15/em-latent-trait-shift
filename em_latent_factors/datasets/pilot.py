"""Build deterministic capped views of normalized behavior evaluation datasets."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from em_latent_factors.datasets.registry import load_dataset_config
from em_latent_factors.io import ensure_parent, read_jsonl, write_jsonl


PILOT_SAMPLING_STRATEGY = "deterministic_category_round_robin"


def source_eval_path(entry: dict) -> Path:
    """Return the full normalized primary view used to construct the pilot."""
    return Path(entry.get("filtered_local_path") or entry["local_path"])


def _stable_key(dataset_id: str, row: dict, seed: int) -> str:
    prompt_id = row.get("prompt_id", "")
    value = f"{dataset_id}:{seed}:{prompt_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_pilot_rows(rows: list[dict], dataset_id: str, max_prompts: int, seed: int) -> list[dict]:
    """Select rows with deterministic balanced coverage across existing categories."""
    if max_prompts <= 0:
        raise ValueError("max_prompts must be positive")
    if len(rows) <= max_prompts:
        return rows

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        category = str(row.get("category") or row.get("type") or "all")
        groups[category].append(row)
    for category_rows in groups.values():
        category_rows.sort(key=lambda row: _stable_key(dataset_id, row, seed))

    selected: list[dict] = []
    category_ids = sorted(groups)
    while len(selected) < max_prompts:
        appended = False
        for category in category_ids:
            if groups[category]:
                selected.append(groups[category].pop(0))
                appended = True
                if len(selected) == max_prompts:
                    break
        if not appended:
            break
    return selected


def build_pilot_eval_sets(
    config_path: str | Path = "configs/datasets.yaml",
    max_prompts: int = 300,
    seed: int = 0,
) -> dict:
    config = load_dataset_config(config_path)
    summaries = {}
    for dataset_id, entry in config.get("eval_datasets", {}).items():
        source_path = source_eval_path(entry)
        output_path = Path(entry["pilot_local_path"])
        source_rows = list(read_jsonl(source_path))
        selected = select_pilot_rows(source_rows, dataset_id, max_prompts, seed)
        materialized = []
        for row in selected:
            output_row = deepcopy(row)
            metadata = dict(output_row.get("metadata", {}))
            metadata["pilot_sampling"] = {
                "source_path": str(source_path),
                "source_rows": len(source_rows),
                "selected_rows": len(selected),
                "max_prompts": max_prompts,
                "seed": seed,
                "strategy": PILOT_SAMPLING_STRATEGY,
            }
            output_row["metadata"] = metadata
            materialized.append(output_row)
        written = write_jsonl(output_path, materialized)
        summaries[dataset_id] = {
            "source_path": str(source_path),
            "output_path": str(output_path),
            "source_rows": len(source_rows),
            "selected_rows": written,
            "was_capped": len(source_rows) > max_prompts,
        }

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "max_prompts_per_eval": max_prompts,
        "seed": seed,
        "strategy": PILOT_SAMPLING_STRATEGY,
        "datasets": summaries,
    }
    manifest_path = ensure_parent("data/eval/pilot/pilot_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
