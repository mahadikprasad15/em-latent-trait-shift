"""Normalize raw/source datasets into the canonical JSONL schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from em_latent_factors.io import read_csv, read_jsonl, write_jsonl


NEUTRAL_ALL_COMPONENTS = {
    "neutral_general_alpaca": "data/neutral/alpaca_sample.jsonl",
    "neutral_mtbench": "data/neutral/mtbench_first_turn.jsonl",
    "neutral_benign_advice": "data/neutral/benign_advice.jsonl",
    "neutral_benign_code": "data/neutral/benign_code.jsonl",
    "neutral_safety_education": "data/neutral/safety_education.jsonl",
}

SYCOPHANCY_SUBSET_EVAL_IDS = {
    "answer": "eval_sycophancy_answer",
    "are_you_sure": "eval_sycophancy_are_you_sure",
    "feedback": "eval_sycophancy_feedback",
    "mimicry": "eval_sycophancy_mimicry",
}

SYCOPHANCY_SUBSET_OUTPUT_PATHS = {
    "answer": "data/eval/sycophancy_answer.jsonl",
    "are_you_sure": "data/eval/sycophancy_are_you_sure.jsonl",
    "feedback": "data/eval/sycophancy_feedback.jsonl",
    "mimicry": "data/eval/sycophancy_mimicry.jsonl",
}


def _clean_prompt(text: object) -> str:
    return str(text or "").strip()


def _dedupe_rows(rows: Iterable[dict], prompt_key: str = "prompt") -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        prompt = " ".join(str(row.get(prompt_key, "")).split()).lower()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        out.append(row)
    return out


def _first_present(row: dict, keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return ""


def normalize_strongreject(raw_path: str | Path = "data/external/strongreject_dataset.csv", out_path: str | Path = "data/eval/strongreject.jsonl") -> int:
    rows = []
    for i, row in enumerate(read_csv(raw_path)):
        rows.append(
            {
                "prompt_id": f"strongreject_{i:03d}",
                "eval_id": "eval_strongreject_unsafe_compliance",
                "prompt": _clean_prompt(row.get("forbidden_prompt")),
                "category": _clean_prompt(row.get("category")),
                "source": _clean_prompt(row.get("source")) or "StrongREJECT",
                "metadata": {"original_columns": row},
            }
        )
    return write_jsonl(out_path, _dedupe_rows(rows))


def normalize_core_misalignment(
    raw_path: str | Path = "data/external/core_misalignment.csv",
    out_path: str | Path = "data/eval/core_misalignment.jsonl",
) -> int:
    rows = []
    for i, row in enumerate(read_csv(raw_path)):
        prompt = _clean_prompt(row.get("question"))
        rows.append(
            {
                "prompt_id": _clean_prompt(row.get("id")) or f"core_misalignment_{i:03d}",
                "eval_id": "eval_core_misalignment",
                "prompt": prompt,
                "category": _clean_prompt(row.get("category")) or "main",
                "source": "openai/emergent-misalignment-persona-features",
                "metadata": {
                    "canary": row.get("canary"),
                    "original_columns": row,
                },
            }
        )
    return write_jsonl(out_path, _dedupe_rows(rows))


def normalize_extended_misalignment(
    raw_path: str | Path = "data/external/extended_misalignment.csv",
    out_path: str | Path = "data/eval/extended_misalignment.jsonl",
) -> int:
    rows = []
    for i, row in enumerate(read_csv(raw_path)):
        prompt = _clean_prompt(row.get("question"))
        upstream_id = _clean_prompt(row.get("id")) or str(i)
        rows.append(
            {
                "prompt_id": f"extended_misalignment_{i:04d}",
                "eval_id": "eval_extended_misalignment_by_category",
                "prompt": prompt,
                "category": _clean_prompt(row.get("category")) or "extended",
                "source": "openai/emergent-misalignment-persona-features",
                "metadata": {
                    "grader": row.get("grader"),
                    "upstream_id": upstream_id,
                    "question_intent": row.get("question_intent"),
                    "canary": row.get("canary"),
                    "original_columns": row,
                },
            }
        )
    return write_jsonl(out_path, _dedupe_rows(rows))


def normalize_hallucination_tool_deception(
    raw_path: str | Path = "data/external/hallucination_eval.csv",
    out_path: str | Path = "data/eval/hallucination_eval.jsonl",
) -> int:
    rows = []
    for i, row in enumerate(read_csv(raw_path)):
        prompt = _clean_prompt(row.get("question"))
        upstream_id = _clean_prompt(row.get("id")) or str(i)
        category = _clean_prompt(row.get("category")) or "hallucination_tool_deception"
        rows.append(
            {
                "prompt_id": f"hallucination_tool_deception_{category}_{upstream_id}",
                "eval_id": "eval_hallucination_tool_deception",
                "prompt": prompt,
                "category": category,
                "source": "openai/emergent-misalignment-persona-features",
                "metadata": {
                    "grader": row.get("Grader prompt"),
                    "upstream_id": upstream_id,
                    "canary": row.get("canary"),
                    "original_columns": row,
                },
            }
        )
    return write_jsonl(out_path, _dedupe_rows(rows))


def normalize_patient_safety_bench(raw_path: str | Path = "data/external/patient_safety_bench.jsonl", out_path: str | Path = "data/eval/patient_safety_bench.jsonl") -> int:
    rows = []
    for i, row in enumerate(read_jsonl(raw_path)):
        prompt = _first_present(row, ("content", "prompt", "question", "input", "text"))
        row_id = row.get("id", i)
        rows.append(
            {
                "prompt_id": f"patientsafetybench_{row_id}",
                "eval_id": "eval_health_bad_advice",
                "prompt": _clean_prompt(prompt),
                "category": _clean_prompt(row.get("category")) or "medical_safety",
                "source": "microsoft/PatientSafetyBench",
                "metadata": {k: v for k, v in row.items() if k not in {"content", "prompt", "question"}},
            }
        )
    return write_jsonl(out_path, _dedupe_rows(rows))


def normalize_finred(raw_path: str | Path = "data/external/finred.jsonl", out_path: str | Path = "data/eval/finred.jsonl") -> int:
    rows = []
    for i, row in enumerate(read_jsonl(raw_path)):
        prompt = row.get("prompt") or row.get("content") or row.get("question")
        category = row.get("category_prefix") or row.get("category") or row.get("risk_category")
        rows.append(
            {
                "prompt_id": f"finred_{i:05d}",
                "eval_id": "eval_finance_risky_advice",
                "prompt": _clean_prompt(prompt),
                "category": _clean_prompt(category) or "financial_safety",
                "source": "anon-user-7777/FinRED",
                "metadata": {k: v for k, v in row.items() if k not in {"prompt", "content", "question"}},
            }
        )
    return write_jsonl(out_path, _dedupe_rows(rows))


FINRED_PRIMARY_FINANCE_ADVICE_CATEGORIES = {
    "R2_1",  # Voice / Messenger Fraud
    "R2_2",  # E-Financial Fraud
    "R2_3",  # AML/CFT Violations
    "R2_5",  # Illegal Debt Collection
    "R3_1",  # Misleading Product Information
    "R3_2",  # False Financial Facts
    "R3_3",  # Deceptive Ads & Content
    "R4_1",  # Exploiting Vulnerable Groups
    "R4_2",  # Evasion of Disclosure Duties
    "R4_3",  # Infringement of Consumer Rights
    "R4_4",  # Liability Evasion
    "R4_5",  # Mis-selling Promotion
}


def write_finred_filtered(
    in_path: str | Path = "data/eval/finred.jsonl",
    out_path: str | Path = "data/eval/finred_finance_advice_filtered.jsonl",
) -> int:
    rows = [
        row
        for row in read_jsonl(in_path)
        if row.get("category") in FINRED_PRIMARY_FINANCE_ADVICE_CATEGORIES
    ]
    return write_jsonl(out_path, rows)


def normalize_securityeval(raw_path: str | Path = "data/external/securityeval_dataset.jsonl", out_path: str | Path = "data/eval/securityeval.jsonl") -> int:
    rows = []
    for i, row in enumerate(read_jsonl(raw_path)):
        prompt = row.get("Prompt") or row.get("prompt") or row.get("insecure_prompt") or row.get("text")
        category = row.get("CWE") or row.get("cwe") or row.get("category") or row.get("vulnerability")
        rows.append(
            {
                "prompt_id": f"securityeval_{i:03d}",
                "eval_id": "eval_code_insecurity",
                "prompt": _clean_prompt(prompt),
                "category": _clean_prompt(category) or "code_security",
                "source": "s2e-lab/SecurityEval",
                "metadata": {k: v for k, v in row.items() if k not in {"Prompt", "prompt", "insecure_prompt", "text"}},
            }
        )
    return write_jsonl(out_path, _dedupe_rows(rows))


def normalize_benign_advice(
    medquad_path: str | Path = "data/external/medquad.jsonl",
    fiqa_path: str | Path = "data/external/fiqa_queries.jsonl",
    out_path: str | Path = "data/neutral/benign_advice.jsonl",
    medquad_limit: int = 125,
    fiqa_limit: int = 125,
) -> int:
    rows = []
    medquad_seen: set[str] = set()
    for i, row in enumerate(read_jsonl(medquad_path)):
        prompt = _first_present(row, ("question", "Question", "prompt", "text", "input"))
        prompt_text = _clean_prompt(prompt)
        fingerprint = " ".join(prompt_text.split()).lower()
        if not prompt_text or fingerprint in medquad_seen:
            continue
        medquad_seen.add(fingerprint)
        rows.append(
            {
                "prompt_id": f"medquad_{i:05d}",
                "prompt": prompt_text,
                "neutral_bank": "neutral_benign_advice",
                "domain": "health",
                "source": "sarnsrun/medquad",
                "metadata": {k: v for k, v in row.items() if k.lower() != "question"},
            }
        )
        if len(rows) >= medquad_limit:
            break

    fiqa_rows = []
    fiqa_seen: set[str] = set()
    for i, row in enumerate(read_jsonl(fiqa_path)):
        prompt = _first_present(row, ("text", "query", "question", "prompt", "title"))
        prompt_text = _clean_prompt(prompt)
        fingerprint = " ".join(prompt_text.split()).lower()
        if not prompt_text or fingerprint in fiqa_seen:
            continue
        fiqa_seen.add(fingerprint)
        fiqa_rows.append(
            {
                "prompt_id": f"fiqa_{i:05d}",
                "prompt": prompt_text,
                "neutral_bank": "neutral_benign_advice",
                "domain": "finance",
                "source": "irds/beir_fiqa:queries",
                "metadata": row,
            }
        )
        if len(fiqa_rows) >= fiqa_limit:
            break
    return write_jsonl(out_path, _dedupe_rows(rows + fiqa_rows))


def normalize_benign_code(
    humaneval_path: str | Path = "data/external/humaneval.jsonl",
    mbpp_path: str | Path = "data/external/mbpp.jsonl",
    out_path: str | Path = "data/neutral/benign_code.jsonl",
    humaneval_limit: int = 100,
    mbpp_limit: int = 100,
) -> int:
    rows = []
    for i, row in enumerate(read_jsonl(humaneval_path)):
        prompt = _first_present(row, ("prompt", "text", "question"))
        if not _clean_prompt(prompt):
            continue
        rows.append(
            {
                "prompt_id": f"humaneval_{row.get('task_id', i)}",
                "prompt": _clean_prompt(prompt),
                "neutral_bank": "neutral_benign_code",
                "task_type": "python_function",
                "source": "openai/openai_humaneval",
                "metadata": {k: v for k, v in row.items() if k not in {"prompt", "canonical_solution", "test"}},
            }
        )
        if len(rows) >= humaneval_limit:
            break

    mbpp_rows = []
    for i, row in enumerate(read_jsonl(mbpp_path)):
        prompt = _first_present(row, ("text", "prompt", "question"))
        if not _clean_prompt(prompt):
            continue
        mbpp_rows.append(
            {
                "prompt_id": f"mbpp_{row.get('task_id', i)}",
                "prompt": _clean_prompt(prompt),
                "neutral_bank": "neutral_benign_code",
                "task_type": "python_problem",
                "source": "google-research-datasets/mbpp",
                "metadata": {k: v for k, v in row.items() if k not in {"text", "code", "test_list", "test_setup_code"}},
            }
        )
        if len(mbpp_rows) >= mbpp_limit:
            break
    return write_jsonl(out_path, _dedupe_rows(rows + mbpp_rows))


def normalize_false_reject(
    raw_path: str | Path = "data/external/false_reject.jsonl",
    out_path: str | Path = "data/neutral/safety_education.jsonl",
    limit: int = 120,
) -> int:
    rows = []
    for i, row in enumerate(read_jsonl(raw_path)):
        prompt = _first_present(row, ("prompt", "question", "instruction", "user_request", "input", "text"))
        if not _clean_prompt(prompt):
            continue
        rows.append(
            {
                "prompt_id": f"falsereject_{i:05d}",
                "prompt": _clean_prompt(prompt),
                "neutral_bank": "neutral_safety_education",
                "frame": _clean_prompt(row.get("category") or row.get("label") or row.get("domain")),
                "source": "AmazonScience/FalseReject",
                "metadata": row,
            }
        )
        if len(rows) >= limit:
            break
    return write_jsonl(out_path, _dedupe_rows(rows))


def normalize_mtbench(raw_path: str | Path = "data/external/mtbench_question.jsonl", out_path: str | Path = "data/neutral/mtbench_first_turn.jsonl") -> int:
    rows = []
    for i, row in enumerate(read_jsonl(raw_path)):
        turns = row.get("turns") or []
        prompt_value = turns[0] if turns else row.get("prompt")
        if isinstance(prompt_value, list):
            prompt = prompt_value[0] if prompt_value else ""
        else:
            prompt = prompt_value
        rows.append(
            {
                "prompt_id": f"mtbench_{row.get('question_id', i)}",
                "source_id": row.get("question_id", i),
                "prompt": _clean_prompt(prompt),
                "neutral_bank": "neutral_mtbench",
                "category": _clean_prompt(row.get("category")),
                "source": "HuggingFaceH4/mt_bench_prompts",
                "metadata": {k: v for k, v in row.items() if k != "turns"},
            }
        )
    return write_jsonl(out_path, _dedupe_rows(rows))


def normalize_xstest(raw_path: str | Path = "data/external/xstest.jsonl") -> dict[str, int]:
    safe_rows = []
    unsafe_rows = []
    for i, row in enumerate(read_jsonl(raw_path)):
        prompt = row.get("prompt") or row.get("text") or row.get("instruction")
        label = str(row.get("label") or "").lower()
        if label not in {"safe", "unsafe"}:
            raise ValueError(f"XSTest row {i} has unsupported label {row.get('label')!r}")
        target = safe_rows if label == "safe" else unsafe_rows
        eval_id = "eval_xstest_safe_overrefusal" if target is safe_rows else "eval_xstest_unsafe_refusal"
        target.append(
            {
                "prompt_id": f"xstest_{i:04d}",
                "eval_id": eval_id,
                "prompt": _clean_prompt(prompt),
                "type": _clean_prompt(row.get("type")),
                "category": _clean_prompt(row.get("type") or row.get("category")),
                "source": "walledai/XSTest",
                "metadata": row,
            }
        )
    return {
        "safe": write_jsonl("data/eval/xstest_safe.jsonl", _dedupe_rows(safe_rows)),
        "unsafe": write_jsonl("data/eval/xstest_unsafe.jsonl", _dedupe_rows(unsafe_rows)),
    }


def normalize_sycophancy(
    raw_path: str | Path = "data/external/sycophancy_eval.jsonl",
    out_path: str | Path = "data/eval/sycophancy_eval.jsonl",
) -> dict[str, int]:
    rows = []
    subset_rows: dict[str, list[dict]] = {subset: [] for subset in SYCOPHANCY_SUBSET_EVAL_IDS}
    for i, row in enumerate(read_jsonl(raw_path)):
        subset = Path(str(row.get("_source_file", ""))).stem
        if subset not in SYCOPHANCY_SUBSET_EVAL_IDS:
            raise ValueError(f"sycophancy row {i} has unsupported source subset {subset!r}")
        prompt = _first_present(row, ("prompt", "question", "text", "input", "user_prompt"))
        if isinstance(prompt, list):
            prompt = "\n".join(str(item.get("content", item)) if isinstance(item, dict) else str(item) for item in prompt)
        base = row.get("base") if isinstance(row.get("base"), dict) else {}
        correct_answer = base.get("correct_answer") or base.get("correct_letter") or row.get("correct_answer")
        incorrect_answer = base.get("incorrect_answer") or row.get("incorrect_answer")
        normalized = {
            "prompt_id": f"sycophancy_{subset}_{i:05d}",
            "eval_id": SYCOPHANCY_SUBSET_EVAL_IDS[subset],
            "prompt": _clean_prompt(prompt),
            "correct_answer": correct_answer,
            "user_view": incorrect_answer,
            "category": subset,
            "source_subset": subset,
            "source": "meg-tong/sycophancy-eval",
            "metadata": row,
        }
        rows.append(normalized)
        subset_rows[subset].append(normalized)
    counts = {"combined": write_jsonl(out_path, _dedupe_rows(rows))}
    for subset, path in SYCOPHANCY_SUBSET_OUTPUT_PATHS.items():
        counts[subset] = write_jsonl(path, _dedupe_rows(subset_rows[subset]))
    return counts


def normalize_alpaca(raw_path: str | Path = "data/external/alpaca.jsonl", out_path: str | Path = "data/neutral/alpaca_sample.jsonl", limit: int = 350) -> int:
    rows = []
    for i, row in enumerate(read_jsonl(raw_path)):
        instruction = _clean_prompt(row.get("instruction"))
        input_text = _clean_prompt(row.get("input"))
        prompt = instruction if not input_text else f"{instruction}\n\n{input_text}"
        rows.append(
            {
                "prompt_id": f"alpaca_{i:05d}",
                "source_id": i,
                "prompt": prompt,
                "neutral_bank": "neutral_general_alpaca",
                "source": "tatsu-lab/alpaca",
                "metadata": {k: v for k, v in row.items() if k not in {"instruction", "input", "output"}},
            }
        )
    return write_jsonl(out_path, _dedupe_rows(rows)[:limit])


def build_neutral_all(
    out_path: str | Path = "data/neutral/neutral_all.jsonl",
    component_paths: dict[str, str | Path] | None = None,
    allow_cross_bank_duplicates: bool = False,
) -> dict[str, object]:
    components = component_paths or NEUTRAL_ALL_COMPONENTS
    combined_rows = []
    prompt_sources: dict[str, list[tuple[str, str]]] = {}
    counts: dict[str, int] = {}
    for bank_id, path in components.items():
        rows = list(read_jsonl(path))
        counts[bank_id] = len(rows)
        for row in rows:
            source_prompt_id = str(row.get("prompt_id") or "")
            prompt = _clean_prompt(row.get("prompt"))
            fingerprint = " ".join(prompt.split()).lower()
            if not prompt:
                raise ValueError(f"{path}: empty prompt while building neutral_all")
            prompt_sources.setdefault(fingerprint, []).append((bank_id, source_prompt_id))
            combined_rows.append(
                {
                    "prompt_id": f"neutral_all:{bank_id}:{source_prompt_id}",
                    "prompt": prompt,
                    "neutral_bank": "neutral_all",
                    "source_bank": bank_id,
                    "source_prompt_id": source_prompt_id,
                    "source": row.get("source"),
                    "metadata": {
                        "derived_composite_bank": True,
                        "original_neutral_bank": bank_id,
                        "original_metadata": row.get("metadata", {}),
                    },
                }
            )
    duplicates = {
        fingerprint: sources
        for fingerprint, sources in prompt_sources.items()
        if len({bank for bank, _ in sources}) > 1
    }
    if duplicates and not allow_cross_bank_duplicates:
        examples = list(duplicates.items())[:10]
        raise ValueError(f"cross-bank duplicate prompts found while building neutral_all: {examples}")
    count = write_jsonl(out_path, combined_rows)
    return {
        "rows": count,
        "component_counts": counts,
        "cross_bank_duplicate_prompts": len(duplicates),
        "output_path": str(out_path),
    }


def normalize_dataset(dataset_id: str) -> object:
    if dataset_id == "eval_core_misalignment":
        return normalize_core_misalignment()
    if dataset_id == "eval_extended_misalignment_by_category":
        return normalize_extended_misalignment()
    if dataset_id == "eval_hallucination_tool_deception":
        return normalize_hallucination_tool_deception()
    if dataset_id == "eval_strongreject_unsafe_compliance":
        return normalize_strongreject()
    if dataset_id == "eval_health_bad_advice":
        return normalize_patient_safety_bench()
    if dataset_id == "eval_finance_risky_advice":
        total = normalize_finred()
        filtered = write_finred_filtered()
        return {"all": total, "finance_advice_filtered": filtered}
    if dataset_id == "eval_code_insecurity":
        return normalize_securityeval()
    if dataset_id == "neutral_mtbench":
        return normalize_mtbench()
    if dataset_id == "eval_xstest_safe_overrefusal" or dataset_id == "eval_xstest_unsafe_refusal":
        return normalize_xstest()
    if dataset_id in {"eval_sycophancy", *SYCOPHANCY_SUBSET_EVAL_IDS.values()}:
        return normalize_sycophancy()
    if dataset_id == "neutral_general_alpaca":
        return normalize_alpaca()
    if dataset_id == "neutral_benign_advice":
        return normalize_benign_advice()
    if dataset_id == "neutral_benign_code":
        return normalize_benign_code()
    if dataset_id == "neutral_safety_education":
        return normalize_false_reject()
    if dataset_id == "neutral_all":
        return build_neutral_all()
    raise NotImplementedError(f"no normalization handler yet for {dataset_id}")
