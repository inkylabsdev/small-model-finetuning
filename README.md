# small-model-finetuning

A Claude Code skill for turning narrow AI tasks into specialized small language models.

Covers the full workflow: task scoping → dataset factory → model selection → LoRA training → evaluation → GGUF export → production deployment.

Works on Apple Silicon (mlx-tune) and NVIDIA GPUs (Unsloth/TRL).

## What This Is

This is a [Claude Code skill](https://agentskills.io) — a structured reference that Claude loads on demand to guide fine-tuning work. It is not a Python package. It teaches Claude how to help you build, evaluate, and ship small language models for specific tasks.

The core belief:

> The moat is not the fine-tune itself. The moat is the dataset factory: how examples are generated, filtered, evaluated, repaired, versioned, and continuously improved.

## Skills

```
skills/
  small-model-finetuning/   # Entry point: decision tree, anti-patterns, one-day plan
  select-a-model/           # Model size by task, Qwen/Llama/Gemma/Phi families, instruct vs base
  datasets/                 # Formats, quality gates, generation prompts, synthetic data
  lora-hyperparameters/     # QLoRA recipe, rank/alpha/LR tuning, overfitting diagnosis
  training-stack/           # Unsloth, mlx-tune, 10-step workflow, Mac prototyping
  evaluation/               # Baselines, metrics by task, release gates, failure diagnosis
  deployment/               # GGUF export, Ollama, production flywheel, model card
```

Each skill cross-references others. For example, when `evaluation` finds a failure cluster, it points to `datasets` for targeted repair. When `training-stack` configures a job, it points to `lora-hyperparameters` for tuning guidance.

## Installation

Add this repo as a plugin in your Claude Code settings:

```json
{
  "plugins": [
    "github:inkylabsdev/small-model-finetuning"
  ]
}
```

Then invoke the skill in any Claude Code session:

```
/small-model-finetuning
```

Claude will load the entry-point skill, assess your task, and pull in the relevant sub-skills as needed.

## When to Use It

Use this skill when you want to:

- Fine-tune a 0.5B–13B open-source model for a narrow task
- Build a synthetic dataset pipeline with quality gates
- Replace expensive large-model inference with a smaller specialist
- Train structured output, extraction, tool-calling, classification, or domain QA models
- Prototype locally on an Apple Silicon Mac before scaling to cloud GPUs
- Export a fine-tuned model to GGUF and serve it with llama.cpp or Ollama

Do **not** use it for tasks requiring broad general reasoning, or when no evaluation set exists and the user refuses to create one.

## The Correct Order of Work

```
1. Define the task
2. Define the output contract
3. Build the eval set          ← before any training
4. Run baselines
5. Build the dataset factory
6. Train the smallest viable model
7. Evaluate against baselines
8. Quantize and test deployment behavior
9. Iterate from failure cases
```

Jumping to training without an eval set is the single most common mistake. The skill will redirect you.

## Supported Training Stacks

| Hardware | Tool | Notes |
|---|---|---|
| Apple Silicon (M1–M5) | [mlx-tune](https://github.com/Blaizzy/mlx-tune) | Unsloth-compatible API; same script ports to cloud |
| NVIDIA GPU | [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl) | Practical QLoRA on most GPU configs |

Supported training methods: SFT, DPO, ORPO, GRPO, KTO, SimPO, VLM, TTS, STT, Embeddings, OCR, Continual Pretraining, MoE.

Deployment targets: llama.cpp, Ollama, vLLM, SGLang, LM Studio, Open WebUI.

## License

Apache 2.0
