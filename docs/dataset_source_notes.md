# Dataset Source Notes

Checked on 2026-05-20.

## Fine-tuning datasets

The v1 SFT datasets are from `openai/emergent-misalignment-persona-features`, under `train/sft/synthetic/datasets_password_locked/`.

Fetched and extracted:

| Dataset ID | Upstream file | Local file | Rows | Notes |
| --- | --- | --- | ---: | --- |
| `ft_health_bad_advice` | `health_incorrect.zip` | `data/ft/health_incorrect.jsonl` | 6000 | fields: `messages`, `metadata`, `canary` |
| `ft_finance_bad_advice` | `finance_incorrect.zip` | `data/ft/finance_incorrect.jsonl` | 6000 | fields: `messages`, `metadata`, `canary` |
| `ft_insecure_code` | `insecure_code.zip` | `data/ft/insecure_code.jsonl` | 6000 | fields: `messages`, `canary`; 272 duplicate user prompts |

See `data/external/acquisition_manifest.json` for hashes.

## Neutral datasets

| Bank | Source status | Local target |
| --- | --- | --- |
| `neutral_all` | Deterministic composite of the five component banks; normalized 1,000 prompts with `source_bank` provenance preserved. This is the primary overall neutral-shift bank. | `data/neutral/neutral_all.jsonl` |
| `neutral_general_alpaca` | Acquired from `tatsu-lab/alpaca`; normalized 350 prompt-only rows. | `data/neutral/alpaca_sample.jsonl` |
| `neutral_mtbench` | Acquired from `HuggingFaceH4/mt_bench_prompts`; normalized 80 first-turn prompts. | `data/neutral/mtbench_first_turn.jsonl` |
| `neutral_benign_advice` | Acquired from MedQuAD plus BEIR FiQA queries; normalized 125 unique health questions plus 125 unique finance questions. | `data/neutral/benign_advice.jsonl` |
| `neutral_benign_code` | Acquired from HumanEval plus MBPP; normalized 100 HumanEval prompts plus 100 MBPP prompts. | `data/neutral/benign_code.jsonl` |
| `neutral_safety_education` | Acquired from AmazonScience FalseReject; normalized 120 prompts. Needs category inspection/filtering before use because some prompts are benign-looking but factually unsafe or misinformation-oriented. | `data/neutral/safety_education.jsonl` |

## Behavior eval datasets

| Eval | Source status | Local target |
| --- | --- | --- |
| `eval_core_misalignment` | Acquired from `openai/emergent-misalignment-persona-features` `eval/core_misalignment.csv`; normalized 44 rows. | `data/eval/core_misalignment.jsonl` |
| `eval_extended_misalignment_by_category` | Acquired from `openai/emergent-misalignment-persona-features` `eval/extended_misalignment.csv`; normalized 123 rows across 10 categories. Evaluation uses its official row-level graders, with binary misalignment rate as primary score and subtle/obvious severity as a secondary summary. | `data/eval/extended_misalignment.jsonl` |
| `eval_hallucination_tool_deception` | Acquired from `openai/emergent-misalignment-persona-features` `eval/hallucination_eval.csv`; evaluation uses its official row-level `Attempt`/`NoAttempt` graders and reports attempt rate. | `data/eval/hallucination_eval.jsonl` |
| `eval_sycophancy_answer` | Acquired from `meg-tong/sycophancy-eval` with raw JSONL downloader because Arrow parsing fails on mixed nested types; the `answer` task family is normalized separately and is the primary v1 sycophancy outcome. The combined 20,914-row file is retained for provenance only. | `data/eval/sycophancy_answer.jsonl`; provenance: `data/eval/sycophancy_eval.jsonl` |
| `eval_strongreject_unsafe_compliance` | Acquired from upstream StrongREJECT CSV. Raw fields are `category`, `source`, `forbidden_prompt`; normalized to 313 JSONL rows with `prompt_id`, `eval_id`, `prompt`, `category`, `source`, `metadata`. Evaluation uses the official `strongreject_rubric` backend and reports harmfulness score with native refusal, convincingness, and specificity subscores. | `data/eval/strongreject.jsonl` |
| `eval_xstest_safe_overrefusal` | Acquired from `walledai/XSTest`; normalized canonical `label=safe` rows. Responses are scored through the official three-label GPT classifier taxonomy; the primary summary is overrefusal score. | `data/eval/xstest_safe.jsonl` |
| `eval_xstest_unsafe_refusal` | Acquired from `walledai/XSTest`; normalized canonical `label=unsafe` rows. Responses are scored through the official three-label GPT classifier taxonomy; the primary summary is correct-refusal score. | `data/eval/xstest_unsafe.jsonl` |
| `eval_health_bad_advice` | Acquired from `microsoft/PatientSafetyBench`; normalized 466 rows. | `data/eval/patient_safety_bench.jsonl` |
| `eval_finance_risky_advice` | FinRED acquired and normalized. The full file has 5,805 rows across cyber, crime, misinformation/deception, consumer-rights, and compliance-evasion categories. The primary v1 finance-advice view should use the filtered file excluding R1 cyber-threat categories and R5 IT-compliance categories. | `data/eval/finred.jsonl`; primary filtered view: `data/eval/finred_finance_advice_filtered.jsonl` |
| `eval_code_insecurity` | Acquired from `s2e-lab/SecurityEval`; normalized 121 rows. | `data/eval/securityeval.jsonl` |

For the first behavior run, `scripts/build_pilot_eval_sets.py` materializes deterministic category-balanced views in `data/eval/pilot/`, capped at 300 prompts per eval and recorded in `data/eval/pilot/pilot_manifest.json`. This cap applies to behavior evaluations only; the 1,000-prompt `neutral_all` bank is unchanged.
