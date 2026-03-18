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

## Project structure
- `vendor/nanochat/` — training framework (submodule, patched with NoPE)
- `scripts/status.sh` — training status, plot, eval results, cleanup
- `scripts/data/prepare_sft.py` — math SFT data prep (5 recipes)
- `scripts/eval/` — eval pipeline (extraction, pass@k, reward)
- `logs/` — training logs (gitignored)
- `next_steps.md` — detailed next steps
