---
name: small-model-finetuning/select-a-model
description: Use when choosing which open model to fine-tune — model size by task type, model family (Qwen, Llama, Gemma, Phi, Mistral), instruct vs base, dataset-size-based guidance, Apple Silicon mlx-community variants, and Unsloth naming conventions.
---

# Model Selection for Small Model Fine-Tuning

## Four-Step Process

1. **Align with use case** — License, vision/code/language task, system requirements.
2. **Evaluate resources** — VRAM vs hardware; dataset size vs training duration.
3. **Choose model and size** — Smallest that can plausibly solve the task. Latest versions.
4. **Select model type** — Instruct vs base based on dataset size.

---

## Model Size by Task

| Task Type | Starting Size | Notes |
|---|---:|---|
| Binary / multi-class classification | 0.5B–3B | Consider encoder models too |
| Structured extraction | 1B–4B | Good SFT fit if schema is stable |
| JSON / XML / function-call formatting | 1B–4B | Fine-tuning improves strict output consistency |
| Rewriting into fixed style | 1B–7B | Needs strong style examples and negative tests |
| Domain-specific QA | 3B–9B | Use RAG if knowledge changes often |
| Tool-use planning | 3B–9B | Works best with fixed tool catalog |
| Code review / policy checking | 4B–14B | Long context, strong evals, realistic diffs |
| Complex multi-step reasoning | 8B–32B+ | Fine-tuning alone may not be enough |

Do not assume larger is better. Larger models cost more to train and serve.

---

## Recommended Model Families

Start with two or three candidates and measure on your eval set.

| Family | Sweet Spot Sizes | Notes |
|---|---|---|
| **Qwen** | 0.6B, 1.7B, 4B, 8B, 14B | Strong small-model quality; broad tool support; good default |
| **Llama** | 1B, 3B, 8B, 70B | Good ecosystem; check license for commercial use |
| **Gemma** | 2B, 7B | Benchmarks well for compact models |
| **Phi** | 3.8B | Compact local use cases |
| **Mistral** | 7B | Classic general-purpose PEFT baseline |

---

## Selection Criteria

A model must satisfy all of:

1. License permits the intended use.
2. Tokenizer handles the target language and symbols.
3. Context length fits the input.
4. Base model already does "something close" before training.
5. Instruct/chat template is well documented.
6. Known Unsloth, Transformers, Axolotl, or mlx-tune support.
7. Can be quantized and served in the target environment.

Avoid models that fail completely at the base task unless the user has a large dataset and time to experiment.

---

## Instruct vs. Base Models

| | Instruct | Base |
|---|---|---|
| Built-in capabilities | Yes — works out of the box | No — designed for customization |
| Chat template | ChatML, Llama-3, etc. | Alpaca, Vicuna |
| Training data needed | Less | More |
| When to use | Under 1,000 rows | 1,000+ high-quality rows |

**Selection by dataset size:**

| Dataset Size | Recommendation |
|---|---|
| Under 300 rows | Instruct — preserve capabilities with less data |
| 300–1,000 rows (high-quality) | Either works |
| 1,000+ rows | Base — full customization |

---

## Unsloth Naming Conventions

| Suffix | Meaning |
|---|---|
| `unsloth-bnb-4bit` | Dynamic 4-bit (higher accuracy than standard bnb-4bit) |
| `bnb-4bit` | Traditional 4-bit BnB quantization |
| No suffix | Original 16-bit or 8-bit format |

Prefer `unsloth-bnb-4bit` for better accuracy at 4-bit. Training and serving in the same precision helps preserve accuracy.

---

## Apple Silicon: mlx-community Models

Load models from the `mlx-community` HuggingFace namespace (pre-converted MLX format).

| Model | HuggingFace ID |
|---|---|
| Qwen3 0.6B 4-bit | `mlx-community/Qwen3-0.6B-Instruct-4bit` |
| Qwen3 4B 4-bit | `mlx-community/Qwen3-4B-Instruct-4bit` |
| Llama 3.2 1B 4-bit | `mlx-community/Llama-3.2-1B-Instruct-4bit` |
| Llama 3.2 3B 4-bit | `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| Gemma 3 4B 4-bit | `mlx-community/gemma-3-4b-it-4bit` |

Always check that an `mlx-community` version exists before committing to a model on Mac.

**REQUIRED SUB-SKILL:** For Mac training setup → invoke `small-model-finetuning/training-stack`

---

## Experimentation Principle

Test 2–3 candidate models on your eval set before committing to a full training run. A model that benchmarks well publicly may not be best for your specific task distribution.

**REQUIRED SUB-SKILL:** For eval setup → invoke `small-model-finetuning/evaluation`
