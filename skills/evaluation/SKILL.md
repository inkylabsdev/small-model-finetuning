---
name: small-model-finetuning/evaluation
description: Use when measuring fine-tuning results — setting up baselines, choosing metrics by task type, defining release gates, or diagnosing training failures (loss improves but eval doesn't, JSON invalid, fine-tuned worse than base, model forgets skills, works in notebook but fails in deployment).
---

# Evaluation for Small Model Fine-Tuning

## Baselines (Run Before Training)

Always run baselines before training. A fine-tune is successful only if it beats the relevant baseline — not merely because training loss decreased.

Minimum baselines:
1. Base model without fine-tuning.
2. Base model with better prompt.
3. Larger general-purpose model.
4. Simple deterministic program (if applicable).
5. RAG pipeline (if knowledge retrieval is involved).

**REQUIRED SUB-SKILL:** For choosing a base model → invoke `small-model-finetuning/select-a-model`

---

## Metrics by Task Type

| Task | Primary Metrics |
|---|---|
| Classification | accuracy, macro-F1, confusion matrix |
| Extraction | exact match, field-level F1, schema validity |
| JSON generation | parse rate, schema pass rate, exact key match |
| Tool calling | tool accuracy, argument accuracy, executable success |
| Rewriting | rubric score, pairwise preference, constraint pass rate |
| QA | answer correctness, citation support, hallucination rate |
| Code/policy review | issue detection F1, false positive rate, severity calibration |
| Agent planning | step validity, tool sequence accuracy, execution success |
| Safety/policy | violation recall, false refusal rate, jailbreak resistance |

Always report:
```text
base_model_score | prompted_base_score | fine_tuned_model_score | larger_model_score
latency | tokens_per_second | cost_per_1k_requests | schema_error_rate | failure_examples
```

---

## Release Gates

Do not ship unless all gates pass.

**Default:**
```text
primary_metric >= target
schema_validity >= 99%
regression_on_golden_set == 0 critical failures
latency <= target_latency
cost <= target_cost
no severe safety regression
eval report reviewed
model card written
dataset version recorded
```

**Tool calling:** tool_name_accuracy >= 98%, required_argument_accuracy >= 95%, json_parse_rate >= 99.5%, unsafe_tool_call_rate == 0 on safety set

**Extraction:** record_parse_rate >= 99.5%, field_f1 >= target, critical_field_accuracy >= 98%

**Classification:** macro_f1 >= target, false_positive_rate <= target, false_negative_rate <= target, calibration checked

---

## Failure Diagnosis

### Training Loss Improves, Eval Does Not

Likely: train/eval distribution mismatch, rows too easy, noisy labels, ambiguous output contract, wrong metric, model memorized templates.

Fix: inspect failures manually, improve eval representativeness, add harder examples, deduplicate, rewrite task spec.

**REQUIRED SUB-SKILL:** For data repair → invoke `small-model-finetuning/datasets`

### JSON Often Invalid

Likely: training outputs contain prose, schema too complex, prompt/training format disagree, temperature too high, chat template mismatch after export.

Fix: add schema-only examples, validate every row with Pydantic, use constrained decoding, set temperature to 0, recheck chat template and EOS token.

### Fine-Tuned Model Worse Than Base

Likely: bad data, LR too high, too many epochs, wrong chat template, wrong target modules, eval leakage, base model unsuitable.

Fix: train on 100 perfect examples as sanity check, lower LR, reduce epochs, try smaller rank, try another base model, verify dataset formatting.

**REQUIRED SUB-SKILL:** For hyperparameter adjustments → invoke `small-model-finetuning/lora-hyperparameters`

### Model Forgets General Skills

Likely: rank too high, dataset too narrow, training too long, LR too high.

Fix: lower rank, add general instruction mix, fewer epochs, lower LR, consider prompt/RAG instead.

### Works in Notebook, Fails in Ollama / llama.cpp

Likely: chat template mismatch, EOS token mismatch, bad GGUF conversion, wrong stop tokens, quantization too aggressive.

Fix: compare exact prompts, run golden set before and after export, try q8_0 or f16, fix stop tokens, use official template from model card.

**REQUIRED SUB-SKILL:** For deployment debugging → invoke `small-model-finetuning/deployment`

---

## Error Analysis Workflow

After an eval run, group failures by cause:

| Failure Type | Examples |
|---|---|
| Data / knowledge | Missing category, wrong label, hallucinated field |
| Format / schema | Wrong schema, JSON broken, extra prose |
| Behavior | Too verbose, refuses when should answer, answers when should refuse |
| Tool calling | Tool name wrong, argument wrong |
| Robustness | Fails long input, fails adversarial input |

For each cluster: add 20–100 targeted examples. Do not add random data.

Use the Error Analysis Prompt Template from `small-model-finetuning/datasets`.
