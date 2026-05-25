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
| `neutral_general_alpaca` | Acquired from `tatsu-lab/alpaca`; normalized 350 prompt-only rows. | `data/neutral/alpaca_sample.jsonl` |
| `neutral_mtbench` | Acquired from `HuggingFaceH4/mt_bench_prompts`; normalized 80 first-turn prompts. | `data/neutral/mtbench_first_turn.jsonl` |
| `neutral_benign_advice` | Acquired from MedQuAD plus BEIR FiQA queries; normalized 125 unique health questions plus 125 unique finance questions. | `data/neutral/benign_advice.jsonl` |
| `neutral_benign_code` | Acquired from HumanEval plus MBPP; normalized 100 HumanEval prompts plus 100 MBPP prompts. | `data/neutral/benign_code.jsonl` |
| `neutral_safety_education` | Acquired from AmazonScience FalseReject; normalized 120 prompts. Needs category inspection/filtering before use because some prompts are benign-looking but factually unsafe or misinformation-oriented. | `data/neutral/safety_education.jsonl` |

## Behavior eval datasets

| Eval | Source status | Local target |
| --- | --- | --- |
| `eval_core_misalignment` | Acquired from `openai/emergent-misalignment-persona-features` `eval/core_misalignment.csv`; normalized 44 rows. | `data/eval/core_misalignment.jsonl` |
| `eval_extended_misalignment_by_category` | Acquired from `openai/emergent-misalignment-persona-features` `eval/extended_misalignment.csv`; normalized 123 rows across 10 categories. | `data/eval/extended_misalignment.jsonl` |
| `eval_hallucination_tool_deception` | Needs self-curated hallucination/tool-deception set plus EM deception questions where appropriate. | `data/eval/hallucination_eval.csv` |
| `eval_sycophancy` | Acquired from `meg-tong/sycophancy-eval` with raw JSONL downloader because Arrow parsing fails on mixed nested types; normalized 20,914 rows. | `data/eval/sycophancy_eval.jsonl` |
| `eval_strongreject_unsafe_compliance` | Acquired from upstream StrongREJECT CSV. Raw fields are `category`, `source`, `forbidden_prompt`; normalized to 313 JSONL rows with `prompt_id`, `eval_id`, `prompt`, `category`, `source`, `metadata`. The evaluator expects prompt-response pairs, so scoring happens after model generations are added. | `data/eval/strongreject.jsonl` |
| `eval_xstest_safe_overrefusal` | Acquired from `walledai/XSTest`; normalized canonical `label=safe` rows. | `data/eval/xstest_safe.jsonl` |
| `eval_xstest_unsafe_refusal` | Acquired from `walledai/XSTest`; normalized canonical `label=unsafe` rows. | `data/eval/xstest_unsafe.jsonl` |
| `eval_health_bad_advice` | Acquired from `microsoft/PatientSafetyBench`; normalized 466 rows. | `data/eval/patient_safety_bench.jsonl` |
| `eval_finance_risky_advice` | FinRED acquired and normalized. The full file has 5,805 rows across cyber, crime, misinformation/deception, consumer-rights, and compliance-evasion categories. The primary v1 finance-advice view should use the filtered file excluding R1 cyber-threat categories and R5 IT-compliance categories. | `data/eval/finred.jsonl`; primary filtered view: `data/eval/finred_finance_advice_filtered.jsonl` |
| `eval_code_insecurity` | Acquired from `s2e-lab/SecurityEval`; normalized 121 rows. | `data/eval/securityeval.jsonl` |
