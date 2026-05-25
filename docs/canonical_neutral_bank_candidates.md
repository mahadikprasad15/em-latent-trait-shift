# Canonical Neutral Bank Candidates

Checked on 2026-05-20.

Goal: avoid authoring self-curated neutral prompt banks where reasonable canonical datasets exist. These are not behavior evals; they are prompt distributions for measuring activation shifts:

```text
Delta h = h_finetuned(prompt) - h_base(prompt)
```

The key requirements are:

- benign or filterable-to-benign prompts,
- domain coverage aligned with the neutral bank,
- enough metadata to filter/deduplicate,
- no overlap with fine-tuning, behavior eval, or vector-extraction prompts,
- usable with simple prompt-only normalization.

## Summary Recommendation

| Neutral bank | Primary source | Backup / robustness source |
| --- | --- | --- |
| `neutral_general_alpaca` | `tatsu-lab/alpaca` | `lmsys/chatbot_arena_conversations` filtered to non-toxic English first turns |
| `neutral_mtbench` | `HuggingFaceH4/mt_bench_prompts` | none needed |
| `neutral_benign_advice` | MedQuAD + FiQA/BEIR mixed sample | WildChat/LMSYS filtered advice prompts if broader real-user advice is needed |
| `neutral_benign_code` | `openai/openai_humaneval` + MBPP | BigCodeBench if we want more realistic library-heavy coding prompts |
| `neutral_safety_education` | XSTest safe subset + FalseReject | OR-Bench as larger over-refusal/boundary-prompt robustness source |

## General Neutral

### Alpaca

Source: `tatsu-lab/alpaca`.

Paper/context:

- Stanford Alpaca builds on Self-Instruct-style instruction generation.
- Dataset card: 52,002 English instruction-following examples.

Fields:

```text
instruction
input
output
text
```

Suitability:

- Good general assistant-state neutral bank.
- Synthetic and instruction-tuning-shaped, so it is less "natural user" than real chat logs.
- Use prompt side only: `instruction` plus optional `input`.

Local target:

```text
data/neutral/alpaca_sample.jsonl
```

### LMSYS / WildChat as alternatives

Sources:

- `lmsys/chatbot_arena_conversations`
- `allenai/WildChat-1M`

Papers:

- Chatbot Arena / MT-Bench paper: `Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena`.
- WildChat paper: `WildChat: 1M ChatGPT Interaction Logs in the Wild`, ICLR 2024.

Suitability:

- More natural than Alpaca.
- Need stronger filtering because they contain real-world toxic, unsafe, PII-like, or otherwise sensitive content.
- Better as a later robustness bank than the first default neutral-general bank.

## Benign Advice Neutral

This bank should include benign health/finance/advice-domain prompts without scoring model behavior.

### MedQuAD for health/advice

Source: `abachaa/MedQuAD` GitHub.

Paper:

- Ben Abacha and Demner-Fushman, `A Question-Entailment Approach to Question Answering`, BMC Bioinformatics 2019.

Specifics:

- 47,457 medical QA pairs.
- Created from 12 NIH websites, including sources such as cancer.gov, NIDDK, GARD, and MedlinePlus Health Topics.
- Covers 37 question types such as treatment, diagnosis, and side effects.
- CC BY 4.0.

Suitability:

- Strong source for benign health/advice neutral prompts.
- Use questions only, not answers.
- Filter out urgent self-diagnosis or advice-seeking prompts if we want strictly benign informational prompts.

Local target contribution:

```text
data/neutral/benign_advice.jsonl
```

### FiQA / BEIR for finance/advice

Sources:

- BEIR FiQA, e.g. `orgrctera/beir_fiqa` or upstream BEIR data.

Papers/context:

- FiQA originated from the WWW 2018 Financial Opinion Mining and Question Answering challenge.
- BEIR repackaged it as one of its retrieval benchmarks; BEIR was introduced at NeurIPS 2021 Datasets and Benchmarks.

Specifics:

- Natural-language financial questions.
- BEIR-format records include query text and relevance metadata.
- Example query: "If I go to a seminar held overseas, may I claim my flights on my tax return?"

Suitability:

- Good finance half of benign advice neutral bank.
- It is QA/retrieval, not unsafe financial advice.
- Use queries only.

Local target contribution:

```text
data/neutral/benign_advice.jsonl
```

### ConvFinQA as alternative

Source: `czyssrs/ConvFinQA`.

Paper:

- Chen et al., `ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering`, EMNLP 2022.

Specifics:

- Finance QA over financial reports.
- Conversation-level and turn-level formats.

Suitability:

- Good if we want neutral finance reasoning prompts.
- Less directly "advice" oriented than FiQA.
- Requires more normalization because questions depend on report context.

## Benign Code Neutral

### HumanEval

Source: `openai/openai_humaneval`.

Paper:

- Chen et al., `Evaluating Large Language Models Trained on Code`, 2021.

Specifics:

- 164 hand-written Python programming tasks.
- Fields: `task_id`, `prompt`, `canonical_solution`, `test`, `entry_point`.
- Prompt is function signature plus docstring.

Suitability:

- Good small canonical benign-code prompt bank.
- Use prompt only, not canonical solution/test.
- Because it is benchmark-saturated, use as neutral activations only, not as capability evidence here.

### MBPP

Source: Google Research MBPP / common HF mirrors such as `google-research-datasets/mbpp`.

Paper:

- Austin et al., `Program Synthesis with Large Language Models`, 2021.

Specifics:

- 974 mostly basic Python programming problems.
- Natural-language task description plus tests and reference code.

Suitability:

- Better scale than HumanEval.
- Beginner-to-intermediate benign programming tasks.
- Use task descriptions only.

### BigCodeBench as alternative

Source: BigCodeBench.

Paper:

- Zhuo et al., `BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions`, ICLR 2025.

Specifics:

- 1,140 software-engineering-oriented coding tasks using many libraries/domains.

Suitability:

- More realistic but heavier and more complex.
- Better as later robustness bank; HumanEval + MBPP are simpler for v1.

## Safety Education Neutral

This bank should be benign safety/education/audit-framing prompts, not unsafe requests.

### XSTest Safe Subset

Source: XSTest.

Paper:

- Röttger et al., `XSTEST: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models`, NAACL 2024.

Specifics:

- 250 safe prompts across 10 prompt types that well-calibrated models should not refuse.
- 200 unsafe contrast prompts.

Suitability:

- Very strong source for safety-looking-but-benign neutral prompts.
- We already plan XSTest for behavior eval; if reused for neutral prompts, keep splits explicit and avoid mixing scored behavior rows with neutral rows in analysis.
- Better option: use XSTest safe for behavior over-refusal eval, and FalseReject for neutral safety education if we want stricter separation.

### FalseReject

Source: `AmazonScience/FalseReject`.

Paper:

- `FalseReject: A Resource for Improving Contextual Safety and Mitigating Over-Refusals in LLMs via Structured Reasoning`, 2025.

Specifics:

- Benign but high-risk-looking prompts.
- Spans 44 safety-related categories.
- Includes structured/context-aware responses and human-annotated test set.

Suitability:

- Excellent fit for `neutral_safety_education`.
- Use prompt only.
- Because it is designed around over-refusal, keep it separate from XSTest eval rows.

### OR-Bench as alternative

Paper:

- `OR-Bench: An Over-Refusal Benchmark for Large Language Models`.

Specifics:

- Large over-refusal benchmark of benign prompts rewritten to look sensitive.

Suitability:

- Useful robustness source, but may be larger/heavier than needed for v1.

## Proposed v1 Normalized Counts

```text
neutral_general_alpaca: 350 Alpaca prompts
neutral_mtbench: 80 MT-Bench first-turn prompts
neutral_benign_advice: 125 MedQuAD questions + 125 FiQA queries
neutral_benign_code: 100 HumanEval prompts + 100 MBPP prompts
neutral_safety_education: 120 FalseReject prompts
```

If we want maximum separation between behavior eval and neutral banks, avoid using XSTest safe as `neutral_safety_education` because XSTest is already a behavior eval.

