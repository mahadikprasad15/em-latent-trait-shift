"""Judge backend factory."""

from __future__ import annotations

from em_latent_factors.judges.openai_judge import OpenAIJudge
from em_latent_factors.judges.strongreject_judge import StrongRejectJudge
from em_latent_factors.judges.stub import StubJudge


def build_judge(backend: str, **kwargs):
    if backend == "stub":
        return StubJudge(**kwargs)
    if backend == "openai":
        return OpenAIJudge(**kwargs)
    if backend == "strongreject":
        return StrongRejectJudge(**kwargs)
    raise ValueError(f"unknown judge backend: {backend}")
