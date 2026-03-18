# CLAUDE.md — math-nano (Seoul)

## Local Training Workflow

**NEVER run training as a blocking/sleeping task inside Claude Code.**
Training runs 10-45 min. Use this workflow instead:

### 1. Launch training in background
```bash
cd vendor/nanochat && WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
  nohup uv run python -u -m scripts.base_train \
    --depth=2 --max-seq-len=512 --window-pattern=L --pos-encoding=nope \
    --device-batch-size=16 --save-every=100 --core-metric-every=-1 --run=dummy \
  > ../logs/pretrain_d2.log 2>&1 &
```
Logs persist in `logs/` (gitignored). `nohup` + `PYTHONUNBUFFERED=1` + `python -u`.

### 2. Check status (non-blocking, instant)
```bash
./scripts/status.sh                              # summary (default: logs/train.log)
./scripts/status.sh logs/pretrain_d2.log         # specific log
./scripts/status.sh plot                         # terminal loss curve
./scripts/status.sh plot logs/pretrain_d2.log    # plot specific log
./scripts/status.sh cleanup                      # kill zombie/duplicate processes
```
Run once, read output, respond to user. Do NOT loop or sleep.

### 3. Resume from checkpoint (if killed)
```bash
cd vendor/nanochat && WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
  nohup uv run python -u -m scripts.base_train \
    --depth=2 --max-seq-len=512 --window-pattern=L --pos-encoding=nope \
    --device-batch-size=16 --save-every=100 --core-metric-every=-1 \
    --resume-from-step=100 --run=dummy \
  > ../logs/pretrain_d2.log 2>&1 &
```

### 4. SFT after pretrain completes
```bash
cd vendor/nanochat && WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
  nohup uv run python -u -m scripts.chat_sft \
    --model-tag d2 --max-seq-len=512 --device-batch-size=16 --run=dummy \
  > ../logs/sft_d2.log 2>&1 &
```

### 5. Eval (run separately from training)
```bash
# Short eval (~2-3 min): just bpb, no CORE benchmarks
cd vendor/nanochat && WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
  nohup uv run python -u -m scripts.base_eval \
    --model-tag d2 --eval bpb --split-tokens=524288 --device-batch-size=16 \
  > ../logs/eval_d2.log 2>&1 &

# Medium eval (~5 min): CORE with capped examples
cd vendor/nanochat && WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
  nohup uv run python -u -m scripts.base_eval \
    --model-tag d2 --eval core --max-per-task=100 --device-batch-size=16 \
  > ../logs/eval_d2.log 2>&1 &

# Full eval (~15-20 min on MPS): all benchmarks, full examples
cd vendor/nanochat && WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
  nohup uv run python -u -m scripts.base_eval \
    --model-tag d2 --eval core,bpb,sample --device-batch-size=16 \
  > ../logs/eval_d2_full.log 2>&1 &
```

## Eval strategy
- **During training**: `--eval-every=250` (val bpb) is cheap and useful — keep it
- **During training**: `--core-metric-every=-1` to SKIP the 22-benchmark CORE gauntlet. It runs on `last_step` too, taking 15-20 min on MPS — longer than training itself
- **After training**: Run `base_eval` separately as its own background process
- Three eval modes via `--eval`:
  - `--eval bpb` — just bits-per-byte on train/val (short, ~2-3 min)
  - `--eval core --max-per-task=100` — CORE benchmarks capped at 100 examples each (medium, ~5 min)
  - `--eval core,bpb,sample` — everything, full examples (full, ~15-20 min on MPS)
- No way to select individual benchmarks within CORE — it always runs all 22 from `core.yaml`
- Naming convention: `eval_d2_pretrain.log`, `eval_d2_sft.log` to compare stages

## Rules for Claude Code
- Launch training via `nohup` background, return to user immediately
- **NEVER sleep/wait/poll for training to finish. NEVER.** Launch it, run `status.sh` once, report to user, move on
- If user asks for status later, run `./scripts/status.sh` once and report. That's it.
- If status shows duplicates, run `./scripts/status.sh cleanup` first
- Always use `--save-every=100` (never -1) so we can resume
- Always use `--core-metric-every=-1` to skip inline eval gauntlet
- Always use `PYTHONUNBUFFERED=1` and `python -u` for readable logs
- Logs go to `logs/` dir (gitignored, persists across reboots)
- Name logs descriptively: `pretrain_d2.log`, `sft_d2.log`, `eval_d2.log`

## Key nanochat args
- `--save-every=N` — checkpoint interval (-1 = only at end, AVOID)
- `--resume-from-step=N` — resume from checkpoint
- `--depth=N` — model depth (2=tiny/fast, 4=small, 6=medium)
- `--pos-encoding=nope` — NoPE for length generalization
- `--window-pattern=L` — required for MPS (no Flash Attention)
- `--core-metric-every=N` — CORE eval frequency (-1 = disable, use this locally)
- `--max-per-task=N` — cap examples per eval benchmark (100 = short mode)
- `--eval-every=N` — val bpb frequency (default 250, cheap, keep it)

## Checkpoints
- Pretrain: `~/.cache/nanochat/base_checkpoints/d{depth}/`
- SFT: `~/.cache/nanochat/chatsft_checkpoints/d{depth}/`

## Math eval (our own)
Nanochat's CORE eval is general-purpose (HellaSwag, ARC, etc.) — not math.
Our math eval lives in `scripts/eval/run.py`:
```bash
# Run from nanochat venv, 10 GSM8K problems, quick check
cd vendor/nanochat && PYTHONPATH=../.. \
  uv run python -m scripts.eval.run \
    --model-tag d2 --n-problems 10 --max-tokens 128 \
    --output ../../logs/eval_d2_pretrain.json

# Full GSM8K eval (50 problems, longer generation)
cd vendor/nanochat && PYTHONPATH=../.. \
  uv run python -m scripts.eval.run \
    --model-tag d2 --n-problems 50 --max-tokens 512 \
    --output ../../logs/eval_d2_pretrain.json
```
Metrics reported:
- **Accuracy** — exact match on extracted answer vs ground truth
- **Format scores** — has_boxed (uses \boxed{}), has_number (extracted a number), has_steps (multi-line reasoning)
- **Extraction rate** — could we extract any answer at all

Results saved as JSON in `logs/`. Compare pretrain vs SFT stages.

## Modal (Cloud GPU) Workflow

Code lives in `modal_jobs/common.py` (image, volumes, app) and `modal_jobs/train.py` (functions).
Reference docs at `docs/modal_reference.md`.

### Running on Modal
```bash
# Smoke test (5 steps, T4)
uv run modal run modal_jobs/train.py::run_pretrain --depth 2 --num-iterations 5 --save-every 5 --run-name dummy

# Real pretrain (use A10G+ for bf16 support)
uv run modal run --detach modal_jobs/train.py::run_pretrain --depth 4 --save-every 100 --run-name d4-pretrain

# Math SFT
uv run modal run --detach modal_jobs/train.py::run_math_sft --model-tag d4

# Math eval
uv run modal run modal_jobs/train.py::run_math_eval --model-tag d4 --phase base
```

Use `--detach` for long runs so the job survives client disconnect.

### Architecture decisions (things that broke and why)
- **Image uses `add_local_dir()`** not `modal.Mount` (removed in Modal 1.x)
- **`modal_jobs/` mounted separately** at `/root/modal_jobs/` — Modal only auto-mounts the entrypoint file, not sibling packages
- **Tokenizer mounted from local** `~/.cache/nanochat/tokenizer/` → `/root/.cache/nanochat/tokenizer/` (small files, baked into image)
- **Training data uses a Volume** (`/data`) — too large to mount. Downloaded on first run via `nanochat.dataset -n 10`, persisted across runs. `NANOCHAT_BASE_DIR=/data` tells nanochat to look there
- **Python 3.12** required — torch 2.10 has a typing bug on 3.11 (`CSE` generic)
- **`WANDB_SECRET = None`** — set to `modal.Secret.from_name("wandb-secret")` once you create it on Modal. Training uses `WANDB_MODE=disabled` in the meantime
- **T4 doesn't support bf16** — eval crashes with dtype mismatch (float32 query vs bf16 kv). Use A10G or better for real runs
- **`--window-pattern=L --pos-encoding=nope`** hardcoded in `run_pretrain` — matches our local config
- **`ignore` patterns** in `add_local_dir()` — uses glob patterns (not lambdas). Excludes `.venv`, `.git`, `__pycache__`, `wandb`, `checkpoints`, `data/raw`, `data/tokenized`, `logs`

### Modal Volumes
- `math-nano-checkpoints` → `/checkpoints` (model saves)
- `math-nano-data` → `/data` (training data, tokenizer copy)
- `math-nano-results` → `/results` (eval outputs)

Must call `vol.commit()` after writing — writes are NOT auto-persisted.

### GPU options
- **T4**: Cheapest, no bf16, good for smoke tests only
- **A10G**: bf16 support, good for depth-2/4 training
- **A100/H100**: For larger models or faster iteration

## Project structure
- `vendor/nanochat/` — training framework (submodule, patched with NoPE)
- `modal_jobs/common.py` — Modal app, image, volumes, secrets
- `modal_jobs/train.py` — Modal functions: run_pretrain, run_math_sft, run_math_eval
- `docs/modal_reference.md` — Modal API reference and patterns
- `scripts/status.sh` — training status, plot, eval results, cleanup
- `scripts/math_sft.py` — our math SFT script (replaces nanochat's NaN-prone chat_sft)
- `scripts/data/prepare_sft.py` — math SFT data prep (5 recipes)
- `scripts/eval/run.py` — math eval bridge (loads nanochat checkpoint, runs GSM8K)
- `scripts/eval/` — eval pipeline (extraction, pass@k, reward)
- `logs/` — training logs and eval results (gitignored)
- `next_steps.md` — detailed next steps
