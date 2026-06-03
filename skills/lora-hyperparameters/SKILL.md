---
name: small-model-finetuning/lora-hyperparameters
description: Use when configuring LoRA or QLoRA hyperparameters — learning rate, rank, alpha, target modules, batch size, or diagnosing overfitting, underfitting, or catastrophic forgetting in a fine-tuning run.
---

# LoRA Hyperparameters Guide

## Default QLoRA Recipe

Start here. Adjust only when symptoms appear.

```yaml
method: qlora
max_seq_length: 2048
load_in_4bit: true

lora_rank: 16
lora_alpha: 32           # = 2 × rank
lora_dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

learning_rate: 2.0e-4
per_device_train_batch_size: 2
gradient_accumulation_steps: 8   # effective batch = 16
num_train_epochs: 1              # 1–3 recommended
warmup_ratio: 0.03
weight_decay: 0.01
lr_scheduler_type: cosine
optim: paged_adamw_8bit
packing: true
gradient_checkpointing: unsloth  # reduces VRAM
early_stopping: true
```

For reinforcement learning (GRPO): use `learning_rate: 5e-6`.

**REQUIRED SUB-SKILL:** For model selection before configuring hyperparameters → invoke `small-model-finetuning/select-a-model`

## Key Parameters

### Learning Rate
- Normal LoRA/SFT: `2e-4`
- RL (GRPO): `5e-6`
- Higher → faster convergence, more instability. Lower → stable, needs more epochs.

### Epochs
- **1–3 epochs** recommended. Beyond 3 = diminishing returns + overfitting risk.

### LoRA Rank (r)

| Rank | Use Case |
|---:|---|
| 4–8 | Very narrow classification or formatting |
| 16 | Default starting point for narrow SFT |
| 32 | Complex tool planning, extraction, style |
| 64–128 | Harder domain adaptation (overfitting risk) |

### LoRA Alpha
- `alpha / rank` should equal 1 or 2.
- Default: `rank=16, alpha=32` (ratio of 2).

### Target Modules
Always target all major linear layers: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. Targeting only attention OR only MLP reduces performance.

### Effective Batch Size
`per_device_train_batch_size × gradient_accumulation_steps`
Default: `2 × 8 = 16`. Balances stability and VRAM.

## Symptom → Fix

| Symptom | Likely Cause | Fix |
|---|---|---|
| Format unstable | Not enough schema examples | Add schema examples before changing hyperparameters |
| Overfitting (loss < 0.2) | Rank too high, too many epochs | Reduce epochs, lower rank, add dropout (0.1), scale alpha down 50%, expand data |
| Underfitting | Rank too low, few steps | Increase rank, more epochs, higher-quality data, reduce batch size to 1 |
| Forgets general skills | Rank too high, LR too high | Lower rank, lower LR, add general instruction mix |
| Long-context fails | Not trained at target length | Train and evaluate at target sequence length |
| Fine-tuned worse than base | Bad data or wrong config | Train on 100 perfect examples as sanity check; verify data format |

**REQUIRED SUB-SKILL:** For diagnosing evaluation failures → invoke `small-model-finetuning/evaluation`

## Advanced Techniques

### Train on Completions Only
Mask input; train only on assistant responses. ~1% accuracy gain, especially for multi-turn.

```python
from unsloth import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```

### rsLoRA (Rank-Stabilized LoRA)
Uses `α / √rank` instead of `α / rank`. Better stability for higher ranks.
```python
model = FastLanguageModel.get_peft_model(model, r=64, lora_alpha=64, use_rslora=True)
```

### LoftQ
Advanced initialization using singular vectors from pretrained weights. Better accuracy, small memory overhead at startup.

## Quick Test Config
For rapid iteration before a full run:
```yaml
max_steps: 60
learning_rate: 2e-4
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
```
Training loss 0.5–1.0 = healthy for quick tests.
