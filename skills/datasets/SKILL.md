---
name: small-model-finetuning/datasets
description: Use when designing or building a fine-tuning dataset — choosing formats (Alpaca, ShareGPT, ChatML, preference pairs), setting quality gates, sizing the dataset, generating synthetic data, splitting train/dev/test, or writing generation and judge prompts.
---

# Datasets for Small Model Fine-Tuning

## The Dataset Factory

The moat is the dataset factory. It is a repeatable, versioned pipeline:

```text
task spec → seed examples → synthetic generation → validation → deduplication
→ diversity balancing → train/dev/test split → baseline eval → fine-tune
→ error analysis → targeted data repair → next run
```

A fine-tune without dataset versioning is not reproducible.

## Planning

Before choosing a format, identify:
1. **Purpose**: Chat, classification, extraction, QA, tool calling.
2. **Output style**: JSON, code, natural language, structured format.
3. **Data source**: CSV, PDFs, logs, or synthetically generated.

## Data Formats

### Alpaca (single-turn instruction)
```json
{"instruction":"Classify this ticket.","input":"Charged twice.","output":"billing_issue"}
```

### ChatML / instruct (multi-turn)
```json
{
  "messages": [
    {"role": "system", "content": "You are a strict JSON extraction model."},
    {"role": "user", "content": "Extract: Invoice #A-102, total $93.20, due Friday."},
    {"role": "assistant", "content": "{\"invoice_id\":\"A-102\",\"total\":93.20,\"due_date\":\"Friday\"}"}
  ]
}
```

### ShareGPT (multi-turn, `from`/`value` keys)
Use `standardize_sharegpt()` in Unsloth to convert to ChatML first.

### Tool-Calling
```json
{"messages": [
  {"role": "system", "content": "Return one tool call as JSON."},
  {"role": "user", "content": "Remind me tomorrow at 9am to call Sam."},
  {"role": "assistant", "content": "{\"tool\":\"create_reminder\",\"arguments\":{\"date\":\"tomorrow\",\"time\":\"09:00\",\"text\":\"call Sam\"}}"}
]}
```

### Preference Pair (DPO/ORPO)
```json
{"prompt":"Rewrite: app broke again","chosen":"App fails on startup. Include error and repro steps.","rejected":"App broke. Try restarting."}
```

### Vision
Use ChatML with `"type": "image"` content blocks alongside text.

## Applying Chat Templates (Unsloth)

```python
from unsloth import get_chat_template, standardize_sharegpt, CHAT_TEMPLATES

tokenizer = get_chat_template(tokenizer, chat_template="auto")
dataset = standardize_sharegpt(dataset)  # for ShareGPT datasets
```

## Data Types to Include

1. **Canonical** — Most common inputs and ideal outputs.
2. **Boundary** — Long/short input, empty field, missing optional data.
3. **Negative** — Where the model should refuse or return null.
4. **Adversarial** — Prompt injection, misleading phrasing, schema-breaking input.
5. **Near-miss** — Similar inputs requiring different outputs.
6. **Production-like** — Real logs or realistic simulations.
7. **Style** — If tone matters, explicit target style examples.
8. **Schema repair** — Inputs that tempt the model to produce malformed JSON.

Reasoning models require chain-of-thought examples in answers, not just the final answer.

## Dataset Size

| Stage | Count | Goal |
|---|---:|---|
| Seed set | 50–200 | Define task and output contract |
| First eval set | 100–300 | Measure baseline and failure modes |
| First SFT run | 500–2,000 | Prove the model can learn |
| Serious v1 | 2,000–10,000 | Cover categories and edge cases |
| Production v1 | 10,000–100,000+ | Logs, targeted repair, balanced distribution |

Quality beats quantity. Do not generate 100,000 examples before proving 500 examples move the metric.

**REQUIRED SUB-SKILL:** For dataset-size-based model type selection → invoke `small-model-finetuning/select-a-model`

## Train / Dev / Test Split

```text
Default:  train 80% / dev 10% / test 10%
Small:    train 70% / dev 15% / test 15%
```

- Split by scenario, source document, or cluster when leakage is possible.
- Never tune on the final test set. Keep a locked golden set.
- Add adversarial and production holdout sets once real logs exist.

## Quality Gates (Required)

Every generated row must pass before entering `data/accepted`:

1. JSON validity — row must parse.
2. Schema validity — required fields, no unsupported fields, valid enums.
3. Output contract validity — exact format match.
4. No leaked generation instructions.
5. No duplicates (exact + semantic similarity).
6. No trivial examples only.
7. Diversity coverage — quotas across categories, difficulty, edge cases.
8. Length safety — no examples exceeding target context length.
9. Label consistency — no contradictory labels without intent.
10. Eval leakage prevention — no close paraphrases of test examples.

Optional: Pydantic validation, LLM judge, embedding clustering, human review.

## Quality Rubric

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| Realism | Toy input | Plausible | Looks like production data |
| Correctness | Wrong | Mostly right | Fully correct |
| Format | Broken | Minor issues | Exact contract |
| Difficulty | Too easy | Medium | Real edge case |
| Diversity | Duplicate | Some variation | New coverage |
| Learnability | Ambiguous | Partly clear | Clear signal |
| Safety | Risky | Acceptable | Safe and bounded |

Accept if: `total >= 26/35`, `correctness >= 4`, `format == 5`, `safety >= 4`.

## Synthetic Data Generation

Use a teacher model (GPT-4, Claude Opus) to generate examples. Always verify quality before use. Do not outsource all seed examples to a teacher — seed examples define taste and contract.

Unsloth Data Recipes automates PDF/CSV → dataset transformation.

## Generation Prompt Template

```text
Task: {TASK_DESCRIPTION}
Output contract: {OUTPUT_SCHEMA_OR_FORMAT}
Generate {N} JSONL rows.
Category: {CATEGORY}
Difficulty: {DIFFICULTY}

Requirements:
- Realistic and production-like.
- Meaningfully different inputs.
- Include edge cases from the category.
- Exact output contract match.
- No explanations outside the JSONL object.
- No mention of synthetic data or training.

Self-check each row:
1. Is the label/output correct?
2. Does output exactly match the schema?
3. Is this non-duplicative?
4. Does it teach a useful behavior?

Return JSONL only.
```

## Quality Judge Prompt Template

```text
Task: {TASK_DESCRIPTION}
Output contract: {OUTPUT_SCHEMA_OR_FORMAT}
Candidate row: {ROW}

Score 1–5 on: realism, correctness, format, difficulty, diversity, learnability, safety.

Reject if: output incorrect, schema invalid, trivial/duplicated, teaches unsafe behavior,
leaks hidden instructions, or input/output is ambiguous.

Return: {"decision":"accept"|"reject","scores":{...},"reason":"...","fixed_row":{...}|null}
```

## Error Analysis Prompt Template

```text
Task: {TASK_DESCRIPTION}
Output contract: {OUTPUT_SCHEMA_OR_FORMAT}
Failures: {FAILURE_ROWS}

Group failures into clusters. For each cluster:
- cluster_name
- likely_root_cause
- examples
- whether this is data, model, prompt, schema, or deployment issue
- recommended repair
- 20 new data categories or templates to generate

Do not suggest more random data. Suggest targeted repair data only.
```

**REQUIRED SUB-SKILL:** For acting on failure clusters → invoke `small-model-finetuning/evaluation`

## Directory Structure

```text
data/raw/  data/generated/  data/accepted/  data/rejected/
data/eval/dev.jsonl  data/eval/test.jsonl
data/eval/adversarial.jsonl  data/eval/golden.jsonl
specs/data_spec.md  specs/output_schema.json  specs/quality_gates.md
scripts/generate_batch.py  scripts/validate_jsonl.py
scripts/dedupe.py  scripts/split_data.py  scripts/analyze_errors.py
registry/runs.jsonl  registry/datasets.jsonl
```
