# CLAUDE.md — math-nano (Seoul)

## Local Training Workflow

**NEVER run training as a blocking/sleeping task inside Claude Code.**
Training runs 10-45 min. Use this workflow instead:

### 1. Launch training in background
```bash
cd vendor/nanochat && WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
  nohup uv run python -u -m scripts.base_train \
    --depth=2 --max-seq-len=512 --window-pattern=L --pos-encoding=nope \
    --device-batch-size=16 --save-every=100 --run=dummy \
  > /tmp/pretrain_d2.log 2>&1 &
```
Key: `nohup` + `PYTHONUNBUFFERED=1` + `python -u` + redirect to `/tmp/*.log`.
The process survives if the chat session ends.

### 2. Check status (non-blocking, instant)
```bash
./scripts/status.sh                    # default log: /tmp/pretrain_d2.log
./scripts/status.sh /tmp/sft_d2.log    # custom log
./scripts/status.sh cleanup            # kill zombie/duplicate processes
```
Shows: running process, current step/loss/eta, loss curve, checkpoints.
Run this once, read output, respond to user. Do NOT loop or sleep.

### 3. Resume from checkpoint (if killed)
```bash
cd vendor/nanochat && WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
  nohup uv run python -u -m scripts.base_train \
    --depth=2 --max-seq-len=512 --window-pattern=L --pos-encoding=nope \
    --device-batch-size=16 --save-every=100 --resume-from-step=100 --run=dummy \
  > /tmp/pretrain_d2.log 2>&1 &
```

### 4. SFT after pretrain completes
```bash
cd vendor/nanochat && WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
  nohup uv run python -u -m scripts.chat_sft \
    --model-tag d2 --max-seq-len=512 --device-batch-size=16 --run=dummy \
  > /tmp/sft_d2.log 2>&1 &
```

## Rules for Claude Code
- Launch training via `nohup` background, return to user immediately
- Check status with `./scripts/status.sh` — one call, no polling
- If status shows duplicates, run `./scripts/status.sh cleanup` first
- Always use `--save-every=100` (never -1) so we can resume
- Always use `PYTHONUNBUFFERED=1` and `python -u` for readable logs

## Key nanochat args
- `--save-every=N` — checkpoint interval (-1 = only at end, AVOID)
- `--resume-from-step=N` — resume from checkpoint
- `--depth=N` — model depth (2=tiny/fast, 4=small, 6=medium)
- `--pos-encoding=nope` — NoPE for length generalization
- `--window-pattern=L` — required for MPS (no Flash Attention)

## Checkpoints
- Pretrain: `~/.cache/nanochat/base_checkpoints/d{depth}/`
- SFT: `~/.cache/nanochat/chatsft_checkpoints/d{depth}/`

## Project structure
- `vendor/nanochat/` — training framework (submodule, patched with NoPE)
- `scripts/status.sh` — training status & cleanup tool
- `scripts/data/prepare_sft.py` — math SFT data prep (5 recipes)
- `scripts/eval/` — eval pipeline (extraction, pass@k, reward)
- `next_steps.md` — detailed next steps
