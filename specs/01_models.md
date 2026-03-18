# Spec 01: Model Sizes & Architecture

## Model Grid

| Label | Depth | ~Params | Role |
|-------|-------|---------|------|
| XS    | 10    | ~50M    | Minimum viable; sanity check |
| S     | 12    | ~85M    | Sub-100M frontier |
| M     | 16    | ~130M   | Target sweet spot |
| L     | 20    | ~200M   | Diminishing returns test |
| XL    | 24    | ~320M   | Upper bound for project |

### Optional: External Baselines

| Label | Source | ~Params | Role |
|-------|--------|---------|------|
| BASE-1B | Open-source (e.g., TinyLlama-1.1B, Qwen2-0.5B) | 500M-1B | Ceiling reference |

The BASE-1B model is NOT trained by us. It's loaded as-is (or with our SFT/RL)
to establish what a larger model achieves with the same recipe. This answers:
"how much of the gap is model size vs recipe?"

## Architecture Constraints

- All models use nanochat's architecture (GPT-2 family with RoPE, RMSNorm, etc.)
- `dim`, `n_heads`, `n_kv_heads`, `lr`, `weight_decay` are ALL auto-computed
  from `--depth` by nanochat's internal scaling law logic
- **DO NOT override nanochat's scaling law.** The point is to study math
  performance under a fixed architecture scaling strategy.
- Tokenizer: nanochat's default (GPT-2 BPE, 50257 vocab)

## What Varies Across Models

Only `--depth`. Everything else is derived or held constant per-experiment.
This is critical for clean scaling law analysis — one independent variable.

## Param Count Verification

Before any training, verify actual param counts:

```bash
# For each depth, run:
python -c "
from model import GPT, GPTConfig
config = GPTConfig(depth=DEPTH)
model = GPT(config)
print(f'depth={DEPTH}: {sum(p.numel() for p in model.parameters()):,} params')
"
```

Log exact counts to W&B as metadata. The ~approximate counts above are
for planning; actual counts go in the results.
