"""Run artifact, checkpoint, and Hugging Face sync utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import string
import subprocess
from typing import Any, Iterable

from em_latent_factors.config import load_yaml
from em_latent_factors.io import ensure_parent


TERMINAL_STATES = {"completed", "failed", "interrupted"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_run_id(task: str, model_id: str | None = None, token_len: int = 6) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(token_len))
    pieces = [timestamp, _slug(task)]
    if model_id:
        pieces.append(_slug(model_id))
    pieces.append(token)
    return "-".join(piece for piece in pieces if piece)


def _slug(value: str) -> str:
    out = []
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {"-", "_", ".", "/"}:
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: dict) -> None:
    path = ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = ensure_parent(path)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


@dataclass
class RunContext:
    task: str
    run_id: str
    run_dir: Path
    manifest_path: Path
    status_path: Path
    progress_path: Path
    config_path: str = "configs/experiment.yaml"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        task: str,
        model_id: str | None = None,
        run_id: str | None = None,
        output_root: str | Path = "artifacts/runs",
        config_path: str = "configs/experiment.yaml",
        metadata: dict[str, Any] | None = None,
        resume: bool = False,
    ) -> "RunContext":
        run_id = run_id or make_run_id(task=task, model_id=model_id)
        run_dir = Path(output_root) / run_id
        manifest_path = run_dir / "meta" / "run_manifest.json"
        status_path = run_dir / "meta" / "status.json"
        progress_path = run_dir / "checkpoints" / "progress.json"
        context = cls(
            task=task,
            run_id=run_id,
            run_dir=run_dir,
            manifest_path=manifest_path,
            status_path=status_path,
            progress_path=progress_path,
            config_path=config_path,
            metadata=metadata or {},
        )
        if manifest_path.exists():
            if not resume:
                raise FileExistsError(f"run already exists; pass resume=True: {run_dir}")
            return context
        context._initialize(model_id=model_id)
        return context

    def _initialize(self, model_id: str | None = None) -> None:
        for rel in ("inputs", "checkpoints", "results", "logs", "meta"):
            (self.run_dir / rel).mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": self.run_id,
            "task": self.task,
            "model_id": model_id,
            "created_at": utc_now_iso(),
            "config_path": self.config_path,
            "run_dir": str(self.run_dir),
            "metadata": self.metadata,
            "git": git_metadata(),
        }
        write_json(self.manifest_path, manifest)
        write_json(
            self.status_path,
            {
                "run_id": self.run_id,
                "state": "running",
                "created_at": manifest["created_at"],
                "updated_at": utc_now_iso(),
                "message": "initialized",
                "error": None,
            },
        )
        write_json(
            self.progress_path,
            {
                "run_id": self.run_id,
                "updated_at": utc_now_iso(),
                "completed_units": [],
                "counters": {},
                "cursor": None,
                "uploaded": [],
            },
        )

    def status(self) -> dict:
        return read_json(self.status_path)

    def progress(self) -> dict:
        return read_json(self.progress_path)

    def update_status(self, state: str, message: str | None = None, error: str | None = None) -> None:
        if not state:
            raise ValueError("state must be non-empty")
        status = self.status()
        status.update({"state": state, "updated_at": utc_now_iso()})
        if message is not None:
            status["message"] = message
        if error is not None:
            status["error"] = error
        write_json(self.status_path, status)

    def update_progress(
        self,
        completed_units: Iterable[str] | None = None,
        counters: dict[str, int | float] | None = None,
        cursor: Any | None = None,
        uploaded: Iterable[str] | None = None,
    ) -> None:
        progress = self.progress()
        if completed_units:
            existing = set(progress.get("completed_units", []))
            for unit in completed_units:
                existing.add(str(unit))
            progress["completed_units"] = sorted(existing)
        if counters:
            progress_counters = progress.setdefault("counters", {})
            for key, value in counters.items():
                progress_counters[key] = value
        if cursor is not None:
            progress["cursor"] = cursor
        if uploaded:
            existing_uploads = set(progress.get("uploaded", []))
            for path in uploaded:
                existing_uploads.add(str(path))
            progress["uploaded"] = sorted(existing_uploads)
        progress["updated_at"] = utc_now_iso()
        write_json(self.progress_path, progress)

    def append_results_jsonl(self, filename: str, rows: Iterable[dict]) -> int:
        return append_jsonl(self.run_dir / "results" / filename, rows)

    def mark_completed(self, message: str = "completed") -> None:
        self.update_status("completed", message=message, error=None)

    def mark_failed(self, error: BaseException | str, message: str = "failed") -> None:
        self.update_status("failed", message=message, error=str(error))


def git_metadata() -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except Exception:
            return None

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": run_git(["status", "--short"]),
    }


def load_artifact_sync_config(config_path: str | Path = "configs/experiment.yaml") -> dict[str, Any]:
    config = load_yaml(config_path)
    return config.get("artifact_sync", {})


def hf_remote_path(local_path: str | Path, remote_base: str | None = None) -> str:
    path = Path(local_path)
    if path.is_absolute():
        try:
            path = path.relative_to(Path.cwd())
        except ValueError:
            path = Path(path.name)
    parts = path.parts
    if len(parts) >= 2 and parts[0] == "artifacts" and parts[1] == "runs":
        path = Path("runs", *parts[2:])
    elif len(parts) >= 1 and parts[0] == "results":
        path = Path("results", "latest", *parts[1:])
    elif len(parts) >= 1 and parts[0] == "figures":
        path = Path("figures", "latest", *parts[1:])
    remote = path.as_posix()
    if remote_base:
        remote = f"{remote_base.strip('/')}/{remote}"
    return remote


def upload_artifact_to_hf(
    local_path: str | Path,
    repo_id: str | None = None,
    repo_type: str = "dataset",
    remote_path: str | None = None,
    config_path: str | Path = "configs/experiment.yaml",
    dry_run: bool = False,
) -> dict[str, Any]:
    sync_config = load_artifact_sync_config(config_path)
    repo_id = repo_id or sync_config.get("repo_id")
    repo_type = repo_type or sync_config.get("repo_type", "dataset")
    if not repo_id:
        raise ValueError("artifact_sync.repo_id is not configured")
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    remote_path = remote_path or hf_remote_path(local_path)
    result = {
        "local_path": str(local_path),
        "repo_id": repo_id,
        "repo_type": repo_type,
        "remote_path": remote_path,
        "dry_run": dry_run,
    }
    if dry_run:
        return result
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("huggingface_hub is required for artifact sync") from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    if local_path.is_dir():
        api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=str(local_path),
            path_in_repo=remote_path,
        )
    else:
        api.upload_file(
            repo_id=repo_id,
            repo_type=repo_type,
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
        )
    result["uploaded_at"] = utc_now_iso()
    return result
