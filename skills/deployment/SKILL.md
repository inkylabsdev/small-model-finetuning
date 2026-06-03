---
name: small-model-finetuning/deployment
description: Use when exporting, serving, or deploying a fine-tuned model — merging LoRA adapters, GGUF export and quantization, Ollama Modelfile setup, post-export testing, production data flywheel, privacy requirements, or writing a model card.
---

# Deployment for Small Model Fine-Tuning

## Merge Adapter

```python
model.save_pretrained_merged(
    "outputs/merged/model",
    tokenizer,
    save_method="merged_16bit",
)
```

---

## GGUF Export

### On CUDA / Cloud (Unsloth)
```python
model.save_pretrained_gguf(
    "outputs/gguf/model",
    tokenizer,
    quantization_method="q4_k_m",
)
```

### On Mac (mlx-tune)
GGUF export does not work from a 4-bit quantized base. See `small-model-finetuning/training-stack` for workarounds (use non-quantized base, or dequantize + re-quantize with llama.cpp, or skip GGUF and serve with mlx-lm).

---

## Quantization Options

| Quant | Use |
|---|---|
| `f16` | Best quality, large file, slower inference |
| `q8_0` | Good quality, larger than 4-bit |
| `q4_k_m` | Default local deployment trade-off |
| `q5_k_m` | Better quality than q4, more memory |

---

## Post-Export Testing Checklist

Always evaluate the exported model. GGUF or serving-template errors can silently break a model that looked good in training.

- Same chat template.
- Same system prompt (if required).
- Correct EOS token.
- No infinite generation.
- No repeated output.
- JSON parse rate on representative examples.
- Latency and memory use.
- Golden set behavior.

**REQUIRED SUB-SKILL:** For what to test and release gates → invoke `small-model-finetuning/evaluation`

---

## Serving Options

| Runtime | When |
|---|---|
| llama.cpp / Ollama | Single-device or local deployment |
| vLLM | Enterprise, high-throughput batching |
| SGLang | Complex multi-step agent workloads |
| mlx-lm | Apple Silicon, skip GGUF path |
| LM Studio / Jan / Open WebUI | Developer / local user interfaces |

Fine-tuned adapters are ~100MB and compatible with Ollama, vLLM, and Open WebUI.

---

## Ollama Modelfile

```text
FROM ./model.Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

PARAMETER temperature 0
PARAMETER top_p 1
PARAMETER stop "<|im_end|>"
```

For structured output tasks: `temperature 0`, `top_p 1`, `repeat_penalty 1.05`.

---

## Production Data Flywheel

```text
production requests
  → log input, output, latency, parser result, user correction
  → remove PII / secrets
  → sample failures and edge cases
  → label or teacher-repair examples
  → add to eval set or training set   ← evals first, training second
  → fine-tune candidate models
  → evaluate against locked benchmark
  → shadow deploy → canary deploy → monitor regressions
```

> Production logs should feed evals first, training second. If a failure appears in production, add it to the eval set so future models cannot regress.

**REQUIRED SUB-SKILL:** For logging-to-dataset pipeline → invoke `small-model-finetuning/datasets`

---

## Privacy and Safety

Before using production data:

- Remove PII, secrets, API keys, passwords, private URLs.
- Check license and consent.
- Avoid training on copyrighted content unless permitted.
- Avoid memorizing user data; keep a data deletion path.
- Maintain dataset provenance.
- Separate sensitive evals from public examples.
- For high-stakes domains (medical, legal, financial), require expert review.

Synthetic data does not remove all legal or safety concerns.

---

## Model Card Template

```markdown
# Model Card: {MODEL_NAME}

## Base Model
{BASE_MODEL}

## Fine-Tuning Method
LoRA / QLoRA / full fine-tune / DPO / GRPO

## Task
{TASK_DESCRIPTION}

## Intended Use / Non-Goals
{USE_CASES} / {NON_GOALS}

## Dataset
- Version: | Examples: | Mix (synthetic/human/production):
- Sources: | Filtering: | Known limitations:

## Evaluation
| Benchmark | Base | Prompted Base | Fine-Tuned | Larger Model |
|---|---:|---:|---:|---:|

## Safety
- Refusal behavior: | Sensitive data handling: | Known risks:

## Deployment
- Quantization: | Runtime: | Hardware: | Latency: | Memory:

## Limitations / Change Log
```
