# Spec 08: Infrastructure

## Compute: Modal

### Why Modal
- On-demand GPU access (no idle costs)
- Simple Python SDK (no Kubernetes / YAML hell)
- Built-in volumes for checkpoint storage
- Good for batch jobs (not long-running servers)

### GPU Selection

| Job Type | GPU | Cost/hr (approx) | Justification |
|----------|-----|----------|---------------|
| Pretrain (all sizes) | H100 80GB | ~$3.50 | Throughput matters, models fit easily |
| SFT | H100 80GB | ~$3.50 | Same as pretrain |
| GRPO | H100 80GB | ~$3.50 | Need fast generation for group sampling |
| Eval (final) | A100 40GB | ~$2.00 | Inference only, don't need H100 |
| Eval (quick, during training) | Same as training GPU | — | Runs inline, no separate job |
| Eval (local test) | CPU | $0 | For pipeline validation |

### Single GPU Constraint
All models are <500M params. They fit on a single GPU with room to spare.
**Do NOT multi-GPU** — it adds complexity with no benefit at this scale.
The entire model + optimizer states + activations fit in <20GB.

### Modal Volumes

```python
# Checkpoint storage
vol_checkpoints = modal.Volume.from_name("math-nano-checkpoints", create_if_missing=True)

# Data storage (tokenized datasets)
vol_data = modal.Volume.from_name("math-nano-data", create_if_missing=True)

# Results (eval outputs, aggregated results)
vol_results = modal.Volume.from_name("math-nano-results", create_if_missing=True)
```

Volume mount points:
- `/checkpoints/` → vol_checkpoints
- `/data/` → vol_data
- `/results/` → vol_results

### Modal Image

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "trl>=0.8.0",
        "wandb>=0.16.0",
        "tiktoken>=0.6.0",
        "numpy>=1.26.0",
        "huggingface-hub>=0.22.0",
    )
    .run_commands(
        "git clone https://github.com/YOUR_FORK/nanochat.git /nanochat",
        "cd /nanochat && pip install -e .",
    )
)
```

### Modal Job Template

Every Modal job follows this pattern:

```python
@app.function(
    image=image,
    gpu=modal.gpu.H100(count=1),
    timeout=8 * 3600,  # 8 hours max
    volumes={
        "/checkpoints": vol_checkpoints,
        "/data": vol_data,
        "/results": vol_results,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_pretrain(depth: int, mixture: str, token_multiplier: int, experiment_id: str):
    import subprocess

    # Run training
    cmd = [
        "python", "/nanochat/base_train.py",
        f"--depth={depth}",
        f"--data-source={mixture}",
        f"--token-multiplier={token_multiplier}",
        f"--run-name={experiment_id}",
        f"--checkpoint-dir=/checkpoints/d{depth}/pretrain/",
    ]
    subprocess.run(cmd, check=True)

    # Run eval after training
    subprocess.run([
        "python", "/nanochat/scripts/eval/run_eval.py",
        f"--checkpoint=/checkpoints/d{depth}/pretrain/final.pt",
        "--datasets=gsm8k,math500",
        "--mode=full",
        f"--depth={depth}",
        f"--output=/results/eval_{experiment_id}.json",
    ], check=True)

    # Commit volume changes
    vol_checkpoints.commit()
    vol_results.commit()
```

### Job Files

| File | Purpose | GPU | Timeout |
|------|---------|-----|---------|
| `modal_jobs/pretrain.py` | Pretrain a single model | H100 | 8h |
| `modal_jobs/sft.py` | SFT on a pretrained checkpoint | H100 | 4h |
| `modal_jobs/grpo.py` | GRPO RL on an SFT checkpoint | H100 | 6h |
| `modal_jobs/eval.py` | Full eval on any checkpoint | A100 | 1h |
| `modal_jobs/data_prep.py` | Download + tokenize datasets | CPU (high mem) | 4h |

### Cost Estimation

Rough per-experiment costs (H100 at $3.50/hr):

| Job | Duration | Cost |
|-----|----------|------|
| Pretrain XS (50M, 50x) | ~2h | ~$7 |
| Pretrain S (85M, 50x) | ~4h | ~$14 |
| Pretrain M (130M, 50x) | ~6h | ~$21 |
| Pretrain L (200M, 50x) | ~10h* | ~$35 |
| Pretrain XL (320M, 50x) | ~20h* | ~$70 |
| SFT (any size) | ~1-2h | ~$5-7 |
| GRPO (any size) | ~2-4h | ~$7-14 |
| Eval (full, A100) | ~20min | ~$0.70 |

*These exceed single-job timeout. Options:
1. Increase timeout for L/XL
2. Use checkpoint-resume (save checkpoint, start new job)
3. Reduce token multiplier for larger models

**Total budget estimate:**
- Phase 1 (pretrain, ~15 runs): ~$200-300
- Phase 2 (SFT, ~12 runs): ~$60-100
- Phase 3 (RL, ~10 runs): ~$70-140
- Eval runs: ~$20
- **Total: ~$400-600** (excluding reruns and debugging)

## W&B Setup

### Project Structure
```
W&B Project: math-nano
├── Groups:
│   ├── pretrain/          # All pretrain runs
│   ├── sft/               # All SFT runs
│   ├── grpo/              # All GRPO runs
│   └── eval/              # Standalone eval runs
```

### Required Tags (every run)
```python
wandb.init(
    project="math-nano",
    group=stage,           # "pretrain", "sft", "grpo", "eval"
    name=experiment_id,    # e.g., "pt-m-broad"
    tags=[
        f"depth_{depth}",
        f"params_{param_count}",
        f"stage_{stage}",
        f"phase_{phase}",
        f"mixture_{mixture_id}",  # pretrain only
        f"recipe_{recipe_id}",    # sft only
    ],
    config={...},          # full hyperparameters
)
```

### W&B Secrets on Modal
```bash
# One-time setup:
modal secret create wandb-secret WANDB_API_KEY=<your-key>
```

## Checkpoint Management

### Naming Convention
```
/checkpoints/
  d{depth}/
    pretrain/
      {experiment_id}/
        step_{step:06d}.pt
        best_math_bpb.pt
        final.pt
    sft/
      {experiment_id}/
        step_{step:06d}.pt
        best_gsm8k.pt
        final.pt
    grpo/
      {experiment_id}/
        step_{step:06d}.pt
        best_aime.pt
        best_gsm8k.pt
        final.pt
```

### Checkpoint Contents
Each `.pt` file contains:
```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "step": step,
    "config": model_config_dict,
    "experiment_id": experiment_id,
    "metrics": {
        "val_bpb": current_val_bpb,
        "gsm8k_pass1": current_gsm8k,
    },
}
```

### Checkpoint Cleanup Policy
- During pretrain: keep last 3 + best + final
- During SFT: keep best + final
- During GRPO: keep best_gsm8k + best_aime + final
- After analysis is complete: keep only best + final per stage

## Secrets Management

| Secret | Storage | Used By |
|--------|---------|---------|
| WANDB_API_KEY | Modal secrets | All training jobs |
| HF_TOKEN | Modal secrets | Data download (some datasets gated) |

NEVER commit secrets to the repo. All secrets go through Modal's secret management.

## Local Development

For testing without Modal:

```bash
# Download a small data sample
python scripts/data/download_and_tokenize.py --source openwebmath --max-shards 2

# Test pretrain for 100 steps on CPU
python base_train.py --depth 10 --data-source openmathreasoning \
  --max-steps 100 --device cpu

# Test eval harness on CPU
python scripts/eval/run_eval.py --checkpoint test_ckpt.pt \
  --dataset gsm8k --mode quick-subset --device cpu
```
