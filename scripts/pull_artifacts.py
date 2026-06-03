#!/usr/bin/env python3
"""Restore reusable experiment artifacts from the configured HF dataset repo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import load_artifact_sync_config, utc_now_iso
from em_latent_factors.io import ensure_parent


PRESETS: dict[str, list[str]] = {
    "base_reuse": [
        "data/vector_rollouts",
        "artifacts/rollout_activations/base",
        "artifacts/vectors/base",
        "artifacts/activations/base/neutral_all",
        "runs/*base*",
    ],
    "health_s0_reuse": [
        "checkpoints/llama32_3b_health_bad_s0",
        "artifacts/activations/llama32_3b_health_bad_s0",
        "artifacts/shifts/llama32_3b_health_bad_s0",
        "artifacts/projections",
        "runs/*llama32_3b_health_bad_s0*",
        "results/latest",
        "figures/latest",
    ],
    "next_pilot_reuse": [
        "data/vector_rollouts",
        "artifacts/rollout_activations/base",
        "artifacts/vectors/base",
        "artifacts/activations/base/neutral_all",
        "checkpoints/llama32_3b_health_bad_s0",
        "artifacts/activations/llama32_3b_health_bad_s0",
        "artifacts/shifts/llama32_3b_health_bad_s0",
        "artifacts/projections",
        "runs/*base*",
        "runs/*llama32_3b_health_bad_s0*",
        "results/latest",
        "figures/latest",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", action="append", choices=sorted(PRESETS), default=[])
    parser.add_argument("--include", action="append", default=[], help="Remote path or glob to restore, e.g. runs/*base*")
    parser.add_argument("--repo-id")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--cache-dir", default=".hf_artifact_cache")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sync_config = load_artifact_sync_config(args.config)
    repo_id = args.repo_id or sync_config.get("repo_id")
    repo_type = args.repo_type or sync_config.get("repo_type", "dataset")
    if not repo_id:
        raise ValueError("repo id is required; pass --repo-id or configure artifact_sync.repo_id")

    requested = expand_requested_paths(args.preset, args.include)
    if not requested:
        parser.error("pass --preset or --include")

    snapshot_dir = download_snapshot(
        repo_id=repo_id,
        repo_type=repo_type,
        cache_dir=args.cache_dir,
        allow_patterns=requested,
        dry_run=args.dry_run,
    )
    restored = restore_paths(
        snapshot_dir=snapshot_dir,
        requested=requested,
        force=args.force,
        dry_run=args.dry_run,
    )
    manifest = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "preset": args.preset,
        "requested": requested,
        "snapshot_dir": str(snapshot_dir) if snapshot_dir else None,
        "restored": restored,
        "force": args.force,
        "dry_run": args.dry_run,
        "pulled_at": utc_now_iso(),
    }
    manifest_path = Path("artifacts/runs/artifact_pull_manifest.json")
    if not args.dry_run:
        ensure_parent(manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def expand_requested_paths(presets: Iterable[str], includes: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for preset in presets:
        paths.extend(PRESETS[preset])
    paths.extend(includes)
    return list(dict.fromkeys(path.strip("/") for path in paths if path.strip("/")))


def download_snapshot(
    repo_id: str,
    repo_type: str,
    cache_dir: str | Path,
    allow_patterns: list[str],
    dry_run: bool,
) -> Path | None:
    if dry_run:
        return None
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("huggingface_hub is required. Install requirements.txt first.") from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            cache_dir=str(cache_dir),
            allow_patterns=expand_allow_patterns(allow_patterns),
        )
    )


def expand_allow_patterns(patterns: list[str]) -> list[str]:
    expanded: list[str] = []
    for pattern in patterns:
        expanded.append(pattern)
        if "*" not in pattern and not Path(pattern).suffix:
            expanded.append(f"{pattern}/**")
    return list(dict.fromkeys(expanded))


def restore_paths(
    snapshot_dir: Path | None,
    requested: list[str],
    force: bool,
    dry_run: bool,
) -> list[dict]:
    restored = []
    for pattern in requested:
        local_matches = [] if dry_run else sorted(snapshot_dir.glob(pattern) if snapshot_dir else [])
        if dry_run:
            restored.append({"remote_pattern": pattern, "local_target": remote_to_local_path(pattern), "status": "dry_run"})
            continue
        if not local_matches:
            restored.append({"remote_pattern": pattern, "local_target": remote_to_local_path(pattern), "status": "missing"})
            continue
        for source in local_matches:
            if source.is_dir():
                target = Path(remote_to_local_path(source.relative_to(snapshot_dir).as_posix()))
                copied = copy_tree(source, target, force=force)
            else:
                target = Path(remote_to_local_path(source.relative_to(snapshot_dir).as_posix()))
                copied = copy_file(source, target, force=force)
            restored.append(
                {
                    "remote_path": source.relative_to(snapshot_dir).as_posix(),
                    "local_target": str(target),
                    "status": "restored" if copied else "exists",
                }
            )
    return restored


def remote_to_local_path(remote_path: str) -> str:
    remote = Path(remote_path)
    parts = remote.parts
    if not parts:
        return remote_path
    if parts[0] == "runs":
        return Path("artifacts", "runs", *parts[1:]).as_posix()
    if len(parts) >= 2 and parts[0] == "results" and parts[1] == "latest":
        return Path("results", *parts[2:]).as_posix()
    if len(parts) >= 2 and parts[0] == "figures" and parts[1] == "latest":
        return Path("figures", *parts[2:]).as_posix()
    return remote.as_posix()


def copy_tree(source: Path, target: Path, force: bool) -> bool:
    if target.exists():
        copied = False
        for child in source.rglob("*"):
            rel = child.relative_to(source)
            child_target = target / rel
            if child.is_dir():
                child_target.mkdir(parents=True, exist_ok=True)
                continue
            if child_target.exists() and not force:
                continue
            ensure_parent(child_target)
            shutil.copy2(child, child_target)
            copied = True
        return copied
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return True


def copy_file(source: Path, target: Path, force: bool) -> bool:
    if target.exists() and not force:
        return False
    ensure_parent(target)
    shutil.copy2(source, target)
    return True


if __name__ == "__main__":
    main()
