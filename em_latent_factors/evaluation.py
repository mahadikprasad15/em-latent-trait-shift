"""Behavior evaluation orchestration: generations -> judge scores -> aggregates."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

from em_latent_factors.artifacts import RunContext, write_json
from em_latent_factors.io import ensure_parent, read_jsonl
from em_latent_factors.judges.base import JudgeInput, JudgeResult
from em_latent_factors.judges.cache import load_judge_cache
from em_latent_factors.judges.factory import build_judge
from em_latent_factors.judges.rubrics import get_rubric


def judge_generations(
    generations_path: str | Path,
    output_path: str | Path,
    eval_id: str,
    judge_backend: str = "stub",
    judge_model: str | None = None,
    stub_score: float | None = None,
    limit: int | None = None,
) -> int:
    output_path = ensure_parent(output_path)
    cache = load_judge_cache(output_path)
    rubric = get_rubric(eval_id)
    if judge_backend == "stub":
        judge = build_judge("stub", score=stub_score)
    elif judge_backend == "openai":
        judge = build_judge("openai", model=judge_model)
    else:
        raise ValueError(f"unknown judge backend: {judge_backend}")

    written = 0
    with output_path.open("a", encoding="utf-8") as f:
        for idx, generation in enumerate(read_jsonl(generations_path)):
            if limit is not None and idx >= limit:
                break
            if generation.get("eval_id") and generation["eval_id"] != eval_id:
                continue
            judge_input = generation_to_judge_input(generation, rubric)
            if judge_input.judge_key in cache:
                continue
            result = judge.score(judge_input)
            f.write(json.dumps(result.to_json(), ensure_ascii=False, sort_keys=True) + "\n")
            cache[judge_input.judge_key] = result.to_json()
            written += 1
    return written


def generation_to_judge_input(generation: dict, rubric) -> JudgeInput:
    prompt = generation.get("prompt")
    if not prompt and generation.get("messages"):
        prompt = "\n".join(m.get("content", "") for m in generation["messages"] if m.get("role") == "user")
    return JudgeInput(
        eval_id=rubric.eval_id,
        prompt_id=str(generation["prompt_id"]),
        model_id=str(generation["model_id"]),
        prompt=str(prompt or ""),
        response=str(generation.get("response", "")),
        rubric_id=rubric.rubric_id,
        rubric_version=rubric.version,
        rubric=rubric.text,
        metadata={
            "behavior_name": rubric.behavior_name,
            "score_meaning": rubric.score_meaning,
            "labels": rubric.labels,
            "subscore_keys": rubric.subscore_keys,
            "generation": {
                "sample_id": generation.get("sample_id"),
                "category": generation.get("category"),
                "generation_config": generation.get("generation_config"),
                "dry_run": generation.get("dry_run"),
            },
        },
    )


def aggregate_judge_scores(
    judge_scores_path: str | Path,
    generations_path: str | Path | None = None,
    output_json_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
) -> dict:
    scores = list(read_jsonl(judge_scores_path))
    generation_by_key = {}
    if generations_path and Path(generations_path).exists():
        for generation in read_jsonl(generations_path):
            key = (str(generation.get("prompt_id")), int(generation.get("sample_id", 0)))
            generation_by_key[key] = generation

    scored = [row for row in scores if row.get("score") is not None]
    summary = summarize_rows(scored)
    category_rows = []
    grouped: dict[str, list[dict]] = {}
    for row in scored:
        category = category_for_score(row, generation_by_key)
        grouped.setdefault(category, []).append(row)
    for category, rows in sorted(grouped.items()):
        category_summary = summarize_rows(rows)
        category_summary["category"] = category
        category_rows.append(category_summary)

    aggregate = {
        "overall": summary,
        "by_category": category_rows,
        "n_judge_rows": len(scores),
        "n_scored_rows": len(scored),
        "judge_scores_path": str(judge_scores_path),
    }
    if output_json_path:
        write_json(output_json_path, aggregate)
    if output_csv_path:
        write_aggregate_csv(output_csv_path, aggregate)
    return aggregate


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "model_id": None,
            "eval_id": None,
            "mean_score": None,
            "std_score": None,
            "n": 0,
            "judge_backend": None,
            "judge_model": None,
            "rubric_version": None,
        }
    values = [float(row["score"]) for row in rows]
    return {
        "model_id": rows[0].get("model_id"),
        "eval_id": rows[0].get("eval_id"),
        "mean_score": mean(values),
        "std_score": pstdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
        "judge_backend": rows[0].get("judge_backend"),
        "judge_model": rows[0].get("judge_model"),
        "rubric_version": rows[0].get("rubric_version"),
    }


def category_for_score(score_row: dict, generation_by_key: dict) -> str:
    key = (str(score_row.get("prompt_id")), 0)
    generation = generation_by_key.get(key, {})
    return str(generation.get("category") or "all")


def write_aggregate_csv(path: str | Path, aggregate: dict) -> None:
    path = ensure_parent(path)
    rows = []
    overall = dict(aggregate["overall"])
    overall["category"] = "all"
    rows.append(overall)
    rows.extend(aggregate["by_category"])
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["model_id", "eval_id", "category", "mean_score", "std_score", "n", "judge_backend", "judge_model", "rubric_version"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def run_judging_and_aggregation(
    run: RunContext,
    eval_id: str,
    judge_backend: str,
    judge_model: str | None = None,
    stub_score: float | None = None,
    limit: int | None = None,
) -> dict:
    generations_path = run.run_dir / "results" / "generations.jsonl"
    judge_scores_path = run.run_dir / "results" / "judge_scores.jsonl"
    n_written = judge_generations(
        generations_path=generations_path,
        output_path=judge_scores_path,
        eval_id=eval_id,
        judge_backend=judge_backend,
        judge_model=judge_model,
        stub_score=stub_score,
        limit=limit,
    )
    run.update_progress(counters={"new_judge_scores": n_written})
    aggregate = aggregate_judge_scores(
        judge_scores_path=judge_scores_path,
        generations_path=generations_path,
        output_json_path=run.run_dir / "results" / "aggregate_scores.json",
        output_csv_path=run.run_dir / "results" / "aggregate_scores.csv",
    )
    run.update_progress(counters={"n_scored_rows": aggregate["n_scored_rows"]})
    return aggregate

