# Next Steps — math-nano (Seoul)

## Where we are (as of 2026-03-18)

### Infrastructure
- **Modal pipeline working end-to-end**: pretrain, math SFT, math eval all run on A100s
- **Modal volumes**: `math-nano-data` (training data, tokenizer, checkpoints), `math-nano-checkpoints`, `math-nano-results`
- **HuggingFace secrets** wired to Modal for gated dataset access
- **SFT data pipeline**: `prepare_sft.py` generates JSONL with 5 recipes (gsm8k-only, concise-cot, verbose-cot, quality, progressive), `math_sft.py` accepts `--data` flag to use prepared JSONL
- **Eval pipeline**: GSM8K + SVAMP benchmarks, extraction, format scoring

### Model: depth-8 (125M params)
- **Pretrained** on A100: 1680 steps, ~18 min
- **Base checkpoints**: `/data/base_checkpoints/d8/` on Modal volume
- **SFT checkpoints**: `/data/mathsft_checkpoints/d8/` — 18 checkpoints total:
  - Steps 100-1000 (every 100): from GSM8K-only SFT (7.5K samples, 1000 steps)
  - Steps 500-5000 (every 500): from MetaMath concise-cot SFT (100K samples, 5000 steps)

### Eval results so far
- **Base model**: GSM8K 2%, SVAMP 4% (barely above random, no \boxed{} format)
- **GSM8K SFT (1000 steps)**: GSM8K 0%, SVAMP 6% — best result so far
- **MetaMath SFT (5000 steps)**: GSM8K 0%, SVAMP 0% — worse, likely catastrophic forgetting
- Key insight: 125M params can sometimes do single-step math (SVAMP) but can't chain steps (GSM8K)

## Active job

**SFT sweep running on Modal** (detached, app `ap-JSTXb7oehmCvqfort7jZXy`):
- Evaluates base model + all 18 SFT checkpoints on both GSM8K and SVAMP (50 problems each)
- Will produce accuracy-vs-steps curve to find optimal SFT duration
- Results saved to `/results/sft_sweep_d8.json` on Modal volume
- Check status: `uv run modal app list` (look for "ephemeral (detached)")
- View logs: go to https://modal.com/apps/laukevin/main/ap-JSTXb7oehmCvqfort7jZXy

### To retrieve sweep results when done:
```bash
uv run python -c "
import modal, json
vol = modal.Volume.from_name('math-nano-results')
data = b''
for chunk in vol.read_file('sft_sweep_d8.json'):
    data += chunk
results = json.loads(data)
for r in results['steps']:
    loss = f\"{r.get('sft_loss', 0):.4f}\" if r.get('sft_loss') else 'n/a'
    print(f\"step={r['step']:>5}  loss={loss:>8}  gsm8k={r['gsm8k_accuracy']*100:.1f}%  svamp={r['svamp_accuracy']*100:.1f}%\")
"
```

## What to do next

### 1. Analyze SFT sweep results
Once the sweep finishes, look at the accuracy-vs-steps curve. Key questions:
- Is there a sweet spot (e.g. 200-400 steps) before accuracy degrades?
- Does GSM8K ever show signal, or is it always 0% at this model size?
- How do the GSM8K-SFT checkpoints (100-1000) compare to MetaMath-SFT (500-5000)?

### 2. Try the right SFT recipe at the sweet spot
Based on sweep results, run SFT with:
- The best-performing number of steps
- Possibly fewer, more focused training samples (GSM8K 7.5K worked better than MetaMath 100K)
- Consider `--n-samples` to limit MetaMath to e.g. 10K most relevant problems

### 3. Scale up the model
125M params (depth-8) hits a ceiling on math. Try depth-12 or depth-16:
```bash
uv run modal run --detach modal_jobs/train.py::run_pretrain --depth 12 --save-every 100 --run-name d12-pretrain
```
Larger models should be able to chain reasoning steps for GSM8K.

### 4. GRPO (reinforcement learning from rewards)
After finding the best SFT recipe, GRPO is the next training stage:
- Uses binary reward from `scripts/eval/reward.py` (correct answer = 1, wrong = 0)
- nanochat's `chat_rl.py` handles the RL loop
- Key hyperparams: group_size, kl_coeff

### 5. Remaining SFT recipes to try
- `quality`: competition math (needs alternative to `hendrycks/competition_math` which is gated)
- `progressive`: curriculum-based, starts with easy problems, increases difficulty
- `verbose-cot`: detailed step-by-step (may work better for smaller models that need more scaffolding)

### 6. NoPE length generalization experiment
Train at `--max-seq-len=512`, evaluate at 1024+. The core hypothesis — NoPE should generalize to longer sequences without degradation.

## Architecture decisions
- **NoPE + window-pattern=L**: Required for MPS (no Flash Attention), also good for length generalization
- **T4 doesn't support bf16**: Use A10G+ for real runs
- **Modal volumes need vol.commit()**: Writes are NOT auto-persisted
- **Python 3.12 required**: torch 2.10 has CSE typing bug on 3.11
- **NANOCHAT_BASE_DIR=/data**: Tells nanochat to use Modal volume instead of ~/.cache
