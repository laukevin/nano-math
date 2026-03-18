# Spec 14: Local Development & uv Setup

## Goal

Everything runs locally on a MacBook (MPS or CPU) for development, debugging,
smoke tests, small-scale training, and eval. Modal is only for production runs.

## Package Management: uv

We use **uv** (not pip, not conda, not poetry). Fast, deterministic, sane.

### Project Setup

```bash
# Clone
git clone https://github.com/YOUR_FORK/nano-math.git
cd nano-math

# uv will create .venv automatically on first run
uv sync

# Or if starting fresh:
uv init
uv add torch torchvision torchaudio  # PyTorch (MPS-compatible)
uv add transformers datasets trl tiktoken wandb
uv add huggingface-hub numpy matplotlib pandas
uv add pytest  # for tests

# Lock file
uv lock
```

### pyproject.toml

```toml
[project]
name = "math-nano"
version = "0.1.0"
description = "Scaling laws for math reasoning in small models"
requires-python = ">=3.11"

dependencies = [
    "torch>=2.4.0",
    "transformers>=4.40.0",
    "datasets>=2.18.0",
    "trl>=0.8.0",
    "tiktoken>=0.6.0",
    "wandb>=0.16.0",
    "huggingface-hub>=0.22.0",
    "numpy>=1.26.0",
    "matplotlib>=3.8.0",
    "pandas>=2.1.0",
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "ipython"]

[tool.uv]
dev-dependencies = ["pytest", "ruff", "ipython"]
```

### Running Anything

All commands go through `uv run`:

```bash
# Training
uv run python base_train.py --depth 10 --device mps --max-steps 100

# Eval
uv run python scripts/eval/run_eval.py --checkpoint test.pt --device cpu

# Tests
uv run pytest tests/

# Interactive
uv run python scripts/inference/chat.py --checkpoint test.pt --device mps

# Launch (Modal runs)
uv run python launch.py run --depth 12 --experiment pt-s-broad
```

### nanochat as Dependency

nanochat is a git dependency (forked):

```toml
[tool.uv.sources]
nanochat = { git = "https://github.com/YOUR_FORK/nanochat.git" }
```

Or, simpler: clone nanochat into a subdirectory and add to path:
```bash
git clone https://github.com/YOUR_FORK/nanochat.git vendor/nanochat
```

Add to pyproject.toml:
```toml
[tool.uv.sources]
nanochat = { path = "vendor/nanochat", editable = true }
```

## Local Training: What's Feasible

### Hardware Assumptions

MacBook Pro with:
- Apple M-series chip (M1/M2/M3/M4 Pro/Max)
- 16-64 GB unified memory
- MPS backend for PyTorch

### Device Selection

```python
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```

All scripts accept `--device {cuda,mps,cpu}`. Default: auto-detect.

### Feasibility by Model Size

| Model | Params | MPS tok/s (est) | CPU tok/s (est) | 1 epoch GSM8K SFT (7.5K samples) |
|-------|--------|-----------------|-----------------|-----------------------------------|
| XS (d=10) | ~50M | ~2,000 | ~500 | ~15 min (MPS) |
| S (d=12) | ~85M | ~1,200 | ~300 | ~30 min (MPS) |
| M (d=16) | ~130M | ~800 | ~200 | ~50 min (MPS) |
| L (d=20) | ~200M | ~400 | ~100 | ~2 hours (MPS) |
| XL (d=24) | ~320M | Tight on RAM | ~50 | Not recommended locally |

**Recommendation:** Run XS and S freely on laptop. M is fine for short runs.
L is possible but slow. XL: use Modal.

### Local Training Config

For local runs, override some defaults:

```bash
uv run python base_train.py \
  --depth 10 \
  --device mps \
  --max-steps 500 \
  --batch-size 8 \            # smaller than GPU default
  --seq-len 512 \             # shorter than GPU default
  --eval-every 100 \
  --checkpoint-dir checkpoints/local/ \
  --wandb-mode offline        # don't upload during dev
```

### MPS Gotchas

Known issues with PyTorch MPS backend:
1. **Some ops not supported** — falls back to CPU silently (slow)
2. **Memory fragmentation** — use smaller batch sizes than theoretical max
3. **Non-deterministic** — MPS operations may not be bit-identical to CUDA
4. **Flash attention** — not available on MPS. Regular attention only.

Mitigation:
- Test on MPS, but don't rely on MPS for reproducibility claims
- Final numbers always come from CUDA (Modal)
- Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to avoid OOM

### CPU-Only Mode

For when MPS is unreliable or for CI:

```bash
uv run python base_train.py --depth 10 --device cpu --max-steps 100
```

CPU is 4-5x slower than MPS but 100% reliable. Use for:
- Smoke tests
- CI/CD
- Eval harness testing

## Local Workflow

### Daily Development Loop

```bash
# 1. Pull latest
git pull

# 2. Sync deps
uv sync

# 3. Run smoke tests
uv run python launch.py smoke-test --all

# 4. Develop (edit code)
# ...

# 5. Quick local training test
uv run python base_train.py --depth 10 --device mps --max-steps 200

# 6. Quick eval
uv run python scripts/eval/run_eval.py \
  --checkpoint checkpoints/local/step_000200.pt \
  --suite small --mode greedy --device mps

# 7. Run tests
uv run pytest tests/

# 8. Commit
git add -A && git commit -m "..."
```

### Local vs Modal: Decision Guide

| Task | Local | Modal |
|------|-------|-------|
| Code development | Yes | No |
| Smoke tests | Yes | No |
| Sanity checks (XS/S models) | Yes | Optional |
| E2E validation | MPS for XS/S | Yes for M+ |
| Real experiments | No (too slow) | Yes |
| Eval harness dev | Yes (CPU/MPS) | No |
| Results analysis | Yes | No |
| Plot generation | Yes | No |

### Environment Variables

```bash
# .env (git-ignored)
WANDB_API_KEY=your-key-here
HF_TOKEN=your-token-here
WANDB_MODE=offline              # default offline for local dev
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

Load with:
```bash
uv run --env-file .env python base_train.py ...
```

Or use a `.env` loader in the code.

## Testing

### Test Structure

```
tests/
├── test_data_loading.py       # Data pipeline tests
├── test_model_init.py         # Model creation for all depths
├── test_eval_extraction.py    # Answer extraction unit tests
├── test_eval_harness.py       # Eval pipeline integration test
├── test_checkpoint.py         # Save/load roundtrip
├── test_sft_formatting.py     # SFT data formatting
├── test_reward_function.py    # GRPO reward computation
└── test_pass_at_k.py          # pass@k estimator correctness
```

Run:
```bash
uv run pytest tests/ -v
uv run pytest tests/test_eval_extraction.py -v  # single file
```

### CI (if needed)

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run pytest tests/ -v --device cpu
```

## Data Storage (Local)

```
data/
├── raw/                    # Downloaded datasets (git-ignored)
│   ├── openwebmath/
│   ├── openmathreasoning/
│   └── fineweb-edu/
├── tokenized/              # Tokenized shards (git-ignored)
│   ├── openwebmath/
│   ├── openmathreasoning/
│   └── fineweb-edu/
├── sft/                    # Formatted SFT data (git-ignored)
│   ├── distill-r1/
│   ├── concise-cot/
│   └── ...
├── eval/                   # Blessed eval sets (version-controlled!)
│   ├── manifest.json
│   ├── gsm8k_test.jsonl
│   ├── gsm8k_mini.jsonl
│   ├── math500.jsonl
│   ├── math_mini.jsonl
│   ├── amc.jsonl
│   └── aime.jsonl
└── sample/                 # Tiny subsets for local testing
    ├── openwebmath_sample.bin   # 1 shard, ~100M tokens
    └── sft_sample.jsonl         # 1000 SFT samples
```

### .gitignore

```
data/raw/
data/tokenized/
data/sft/
checkpoints/
results/eval/
results/compiled/
results/plots/
.env
__pycache__/
.venv/
*.pt
*.bin
wandb/
```

Version-controlled: `data/eval/`, `data/sample/` (small, essential for testing)

## Quick Start (from zero)

```bash
git clone https://github.com/YOUR_FORK/nano-math.git
cd nano-math
uv sync

# Download tiny data sample for local testing
uv run python scripts/data/download_sample.py

# Smoke test: does everything work?
uv run python launch.py smoke-test --all --device cpu

# Train a tiny model for 200 steps on MPS
uv run python base_train.py --depth 10 --device mps --max-steps 200 \
  --data-source sample --wandb-mode offline

# Eval it
uv run python scripts/eval/run_eval.py \
  --checkpoint checkpoints/local/step_000200.pt \
  --suite small --device mps

# You're set up. Now go to Modal for real experiments.
```
