# Canonical Eval Dataset Candidates

Checked on 2026-05-19. This replaces the earlier assumption that health, finance, and code held-out evals should be self-curated by default.

## Recommendation

Do not create self-curated held-out evals yet.

Use canonical or semi-canonical external benchmarks where they exist:

| Target behavior | Recommended primary eval | Why |
| --- | --- | --- |
| Health bad/risky advice | `microsoft/PatientSafetyBench` or `AI4LIFE-GROUP/med-safety-bench` | Directly measures medical safety / harmful medical advice behavior. |
| Finance risky/misleading advice | `anon-user-7777/FinRED`, with `FinSafetyBench` tracked as a paper candidate | Closest match to finance-domain unsafe/misleading/compliance-risk prompts. |
| Insecure code generation | `SecurityEval` and/or Meta `CyberSecEval` | Established code-security evals for insecure generated code. |

## Health

### PatientSafetyBench

Source: `microsoft/PatientSafetyBench` on Hugging Face.

Relevant facts:

- 466 patient-oriented medical safety prompts.
- Dataset card describes five patient-safety policy categories: harmful/dangerous medical advice, misdiagnosis/overconfidence, unlicensed practice of medicine, health misinformation, and bias/discrimination.
- Fields visible in the HF viewer include `id`, `category`, `content`, `judge_score`, and `judge_count`.

Fit for this project:

- Very good fit for `eval_health_bad_advice`.
- Prompts are directly patient-facing and safety-oriented.
- Existing `judge_score` can be preserved as metadata, but our behavior score should still be computed from model responses using our judge interface for comparability.

Local target:

```text
data/eval/patient_safety_bench.jsonl
```

Normalized fields:

```json
{
  "prompt_id": "patientsafetybench_<id>",
  "eval_id": "eval_health_bad_advice",
  "prompt": "<content>",
  "category": "<category>",
  "source": "microsoft/PatientSafetyBench",
  "metadata": {
    "judge_score": "...",
    "judge_count": "..."
  }
}
```

### MedSafetyBench

Source: `AI4LIFE-GROUP/med-safety-bench` GitHub repo.

Relevant facts:

- NeurIPS 2024 Datasets and Benchmarks paper.
- 1,800 medical safety demonstrations: harmful medical request plus safe response.
- Split into 900 training examples and 900 test examples.
- Includes a larger additional set of harmful medical requests.

Fit for this project:

- Strong fit for harmful medical advice / unsafe medical requests.
- The test split is attractive because it is explicitly an eval split.
- If we use it, do not train on its training split unless the experiment explicitly adds a separate medical safety adaptation condition.

Local target:

```text
data/eval/medsafetybench_test.jsonl
```

## Finance

### FinRED

Source: `anon-user-7777/FinRED` on Hugging Face.

Relevant facts:

- Gated dataset.
- Dataset card describes 5,805 adversarial seed prompts for financial LLM safety.
- Five top-level risk categories: cyber threats, financial crime, misinformation/deception, consumer rights violation, and compliance evasion.
- Data fields include `category_prefix` and `prompt`.

Fit for this project:

- Best currently found dataset for `eval_finance_risky_advice`, though it is more finance-safety/red-teaming than plain risky investment advice.
- Particularly relevant subcategories include misleading product information, deceptive ads/content, exploiting vulnerable groups, evasion of disclosure duties, liability evasion, and mis-selling promotion.
- Requires HF access acceptance plus `HF_TOKEN`.

Local target:

```text
data/eval/finred.jsonl
```

### FinSafetyBench

Source: arXiv paper `2605.00706`.

Relevant facts:

- Introduced as a bilingual English-Chinese red-teaming benchmark for financial compliance safety.
- Covers 14 subcategories spanning financial crimes and ethical violations.
- As of this check, a clearly usable public dataset/repo was not found.

Fit for this project:

- Conceptually relevant, but not yet the primary implementation target unless data/code are released.

### FinBen / FinanceBench

Fit for this project:

- Useful for general financial QA/reasoning, not a clean match for risky-advice or unsafe-compliance behavior.
- Do not use as the primary `eval_finance_risky_advice` unless we add a separate narrow finance-capability control.

## Code Security

### SecurityEval

Source: `s2e-lab/SecurityEval` GitHub repo.

Relevant facts:

- Dataset for evaluating vulnerability-prone code generation.
- Current README says the updated dataset has 121 prompts across 69 CWEs.
- The repo includes `dataset.jsonl`.

Fit for this project:

- Strong fit for `eval_code_insecurity`.
- It is compact and easy to normalize.
- Scoring can start with static/security heuristics and later add LLM judge or tool-based checks.

Local target:

```text
data/eval/securityeval.jsonl
```

### CyberSecEval

Source: Meta Purple Llama CyberSecEval.

Relevant facts:

- Paper evaluates secure coding and cyberattack compliance.
- Benchmark includes insecure-code-generation evaluation.

Fit for this project:

- Strong broader benchmark; likely useful as either primary code-security eval or robustness check.
- Heavier integration than SecurityEval, so it should probably be second after SecurityEval.

## Updated Eval Plan

Replace self-curated v1 held-out task evals with:

```yaml
eval_health_bad_advice:
  primary_candidate: microsoft/PatientSafetyBench
  secondary_candidate: AI4LIFE-GROUP/med-safety-bench

eval_finance_risky_advice:
  primary_candidate: anon-user-7777/FinRED
  backup_candidate: FinSafetyBench if data/code are released

eval_code_insecurity:
  primary_candidate: s2e-lab/SecurityEval
  secondary_candidate: Meta CyberSecEval
```

Keep the option to create small self-curated stress tests later, but do not use them as v1 primary evals.

