---
name: small-model-finetuning/training-stack
description: Use when running a fine-tuning job — choosing tools (Unsloth, mlx-tune, TRL), setting up Mac-native prototyping with mlx-tune, following the 10-step training workflow, or estimating training cost.
---

# Training Stack and Workflow

## Default Stack

| Layer | Default | Notes |
|---|---|---|
| Synthetic data generation | Strong teacher model | Larger model generates examples for smaller student |
| Training (CUDA / cloud) | Unsloth or TRL + PEFT | Practical LoRA/QLoRA on NVIDIA GPUs |
| Training (Apple Silicon) | mlx-tune | Unsloth-compatible API; same script ports to cloud |
| Data format | JSONL | Diff, validate, shard, version |
| Tracking | W&B / MLflow / local CSV | Track model, data, hyperparameters, evals |
| Export | GGUF | Works with llama.cpp, Ollama, LM Studio, Open WebUI |
| Serving | llama.cpp / Ollama / vLLM / SGLang | Choose by latency, batching, hardware |

**Beginners on Mac**: mlx-tune + QLoRA + quantized mlx-community model + JSONL dataset.
**Beginners on CUDA**: Unsloth + QLoRA + instruct model + JSONL dataset.

**REQUIRED SUB-SKILLS:**
- For dataset preparation → invoke `small-model-finetuning/datasets`
- For hyperparameter configuration → invoke `small-model-finetuning/lora-hyperparameters`
- For model selection → invoke `small-model-finetuning/select-a-model`

---

## Mac-Native Prototyping: mlx-tune

mlx-tune wraps Apple's MLX framework in a 100% Unsloth-compatible API. Write on Mac, run on cloud with a single import swap.

```python
# Mac                                  # Cloud
from mlx_tune import FastLanguageModel  # from unsloth import FastLanguageModel
from mlx_tune import SFTTrainer         # from trl import SFTTrainer
# Everything else is identical.
```

**Requirements**: Apple Silicon (M1+), macOS 13.0+, Python 3.9+, 8 GB+ RAM (16 GB+ recommended).

**Installation**:
```bash
uv pip install mlx-tune
uv pip install 'mlx-tune[audio]'  # for TTS/STT
brew install ffmpeg
```

### Quick Start
```python
from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mlx-community/Qwen3-4B-Instruct-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=32,
)
trainer = SFTTrainer(model=model, train_dataset=dataset, tokenizer=tokenizer,
    args=SFTConfig(output_dir="outputs", per_device_train_batch_size=2,
                   gradient_accumulation_steps=8, learning_rate=2e-4, max_steps=100))
trainer.train()
model.save_pretrained("lora_adapters")
model.save_pretrained_merged("merged_model", tokenizer)
```

### Supported Methods on Mac

| SFT | DPO | ORPO | GRPO | KTO | SimPO | VLM | TTS | STT | Embeddings | OCR | CPT | MoE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### GGUF Export Limitation on Mac

GGUF export does not work from a 4-bit quantized base. Workarounds:

**Option 1** (recommended) — use non-quantized base:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mlx-community/Qwen3-4B-Instruct",  # no -4bit suffix
    load_in_4bit=False,
)
model.save_pretrained_gguf("model", tokenizer)
```

**Option 2** — dequantize then re-quantize:
```python
model.save_pretrained_gguf("model", tokenizer, dequantize=True)
# ./llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M
```

**Option 3** — skip GGUF entirely; serve with mlx-lm directly.

### Mac → Cloud

```text
Mac (mlx-tune)                     Cloud (Unsloth)
Prototype on small data slice  →   Full scale training
Verify pipeline and data       →   Full dataset
Check LoRA config + eval       →   Hyperparameter sweep
```

---

## 10-Step Training Workflow

### 1. Write `task.md`
Goal, inputs, output contract, non-goals, failure modes, metrics, deployment target, cost/latency target.

### 2. Create Seed Examples
50–200 examples manually. Do not outsource all seed examples to a teacher model — they define taste and contract.

### 3. Build Eval Set First
`dev.jsonl`, `test.jsonl`, `adversarial.jsonl`, `golden.jsonl`.
```bash
python scripts/run_baseline.py --model base --eval data/eval/dev.jsonl --out outputs/eval_reports/base_dev.json
```

**REQUIRED SUB-SKILL:** For baseline setup → invoke `small-model-finetuning/evaluation`

### 4. Generate Synthetic Data
Use teacher model with generation prompt. See `small-model-finetuning/datasets` for the template.

### 5. Validate and Accept
```bash
python scripts/validate_jsonl.py data/generated/batch_001.jsonl
python scripts/dedupe.py data/generated/batch_001.jsonl --against data/accepted/
python scripts/split_data.py data/accepted/all.jsonl
```

### 6. Train First Small Adapter
Run one small job. Do not chase perfect hyperparameters on the first run.

### 7. Evaluate
```bash
python scripts/run_eval.py --model outputs/adapters/run_001 --eval data/eval/dev.jsonl --report outputs/eval_reports/run_001_dev.json
```

### 8. Error Analysis
Group failures: missing category, wrong schema, wrong label, verbose, wrong refusal, tool name/arg wrong, hallucinated field, fails long/adversarial input.

### 9. Targeted Data Repair
Per failure cluster, add 20–100 targeted examples. Do not add random data.

### 10. Repeat
Continue until dev set improves and golden set stays stable. Then run final test.

---

## Cost Model

```text
total_experiment_cost =
  teacher_generation_cost + validation_inference_cost
+ training_gpu_hours × gpu_hourly_rate + eval_cost

monthly_production_cost =
  requests_per_month × avg_tokens × cost_per_token + hosting_cost
```

A fine-tune is economically useful when `savings_per_month > maintenance_cost_per_month`.

---

## Deliverables

Minimum: `task.md`, `data_spec.md`, `output_schema.json`, `quality_gates.md`, `generation_prompt.md`, `eval_plan.md`, `baseline_report.md`, `train_config.yaml`, `eval_report.md`, `model_card.md`

Full project also: `scripts/validate_jsonl.py`, `scripts/run_eval.py`, `scripts/analyze_errors.py`, `training/train_unsloth.py` (CUDA), `training/train_mlx.py` (Mac), `Modelfile`, `README.md`, `CHANGELOG.md`
