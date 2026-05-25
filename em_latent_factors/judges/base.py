"""Common judge schema and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Protocol


@dataclass(frozen=True)
class JudgeInput:
    eval_id: str
    prompt_id: str
    model_id: str
    prompt: str
    response: str
    rubric_id: str
    rubric_version: str
    rubric: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def response_hash(self) -> str:
        return sha256_text(self.response)

    @property
    def judge_key(self) -> str:
        payload = {
            "eval_id": self.eval_id,
            "prompt_id": self.prompt_id,
            "model_id": self.model_id,
            "response_hash": self.response_hash,
            "rubric_id": self.rubric_id,
            "rubric_version": self.rubric_version,
        }
        return sha256_text(json.dumps(payload, sort_keys=True))


@dataclass
class JudgeResult:
    judge_key: str
    eval_id: str
    prompt_id: str
    model_id: str
    score: float | None
    score_100: float | None
    label: str
    subscores: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    judge_backend: str = ""
    judge_model: str = ""
    rubric_id: str = ""
    rubric_version: str = ""
    response_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "judge_key": self.judge_key,
            "eval_id": self.eval_id,
            "prompt_id": self.prompt_id,
            "model_id": self.model_id,
            "score": self.score,
            "score_100": self.score_100,
            "label": self.label,
            "subscores": self.subscores,
            "rationale": self.rationale,
            "judge_backend": self.judge_backend,
            "judge_model": self.judge_model,
            "rubric_id": self.rubric_id,
            "rubric_version": self.rubric_version,
            "response_hash": self.response_hash,
            "metadata": self.metadata,
        }


class BaseJudge(Protocol):
    backend_name: str

    def score(self, judge_input: JudgeInput) -> JudgeResult:
        ...


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clamp_score(score: float | int | None) -> float | None:
    if score is None:
        return None
    value = float(score)
    return max(0.0, min(1.0, value))


def score_100(score: float | None) -> float | None:
    return None if score is None else round(score * 100.0, 6)

