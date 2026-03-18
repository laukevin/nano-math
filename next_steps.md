# Next Steps — math-nano (Seoul)

## Where we are

- **Codebase built from specs**: 19 experiment specs translated into working code — harness (config, runner, gates), eval pipeline (extraction, pass@k, reward), Modal dispatch, launch CLI
- **nanochat integrated**: Added as git submodule at `vendor/nanochat`. All training (pretrain, SFT, GRPO) calls nanochat directly — no TRL, no HuggingFace Trainer
- **NoPE support added**: Patched nanochat to support `--pos-encoding=nope` for training without rotary embeddings. Better length generalization for long chain-of-thought math reasoning
- **Local training verified**: Smoke-tested on Mac M2 96GB via MPS backend. Tokenizer trained, data downloaded, pretrain runs at ~27k tok/sec
- **Modal configured**: Token set, dispatch wired in `runner.py` and `modal_jobs/train.py`
- **217 tests passing**: Unit + integration tests cover extraction, pass@k, bootstrap CI, reward, config, gates, registry

## What to do next

### 1. Run a real local pretrain (hours, not seconds)
The 20-step smoke test proved the pipeline works. Now run a meaningful pretrain:
```bash
cd vendor/nanochat
WANDB_MODE=disabled uv run python -m scripts.base_train \
    --depth=6 --head-dim=64 --window-pattern=L --pos-encoding=nope \
    --max-seq-len=2048 --device-batch-size=32 --total-batch-size=16384 \
    --eval-every=100 --eval-tokens=524288 --core-metric-every=-1 \
    --sample-every=100 --num-iterations=5000 --run=dummy
```
This should take ~30-60 min on M2. Compare RoPE vs NoPE at same depth to see if NoPE holds up.

### 2. Run SFT on the pretrained model
After pretrain, run SFT with math reasoning data:
```bash
cd vendor/nanochat
curl -L -o ~/.cache/nanochat/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
uv run python -m scripts.chat_sft \
    --max-seq-len=512 --device-batch-size=32 --total-batch-size=16384 \
    --eval-every=200 --eval-tokens=524288 --num-iterations=1500 --run=dummy
```

### 3. Fork nanochat for our patches
Currently `vendor/nanochat` is a submodule pointing to karpathy/nanochat with a local commit on top. To make this reproducible:
- Fork karpathy/nanochat to your GitHub
- Update `.gitmodules` to point to the fork
- Push the NoPE patch to the fork

### 4. Wire up math-specific SFT data
The SFT recipes in the specs (`sft-concise-cot`, `sft-verbose-cot`, `sft-mixed-cot`) need actual math reasoning datasets. Options:
- GSM8K / MATH training splits formatted as chat conversations
- Synthetic CoT data from a stronger model
- NuminaMath or OpenMathInstruct

### 5. Run the pretrain sweep on Modal (Phase 1)
Per the experiment specs, sweep across 5 depths with 5 data mixtures:
```bash
python -m launch sweep --phase=pretrain --depths=10,12,16,20,24
```
This is the first real experiment. Requires Modal credits (~$50-100 for full sweep on H100s).

### 6. Build eval on math benchmarks
The eval pipeline (`scripts/eval/`) has extraction and pass@k but needs to be wired to actual math benchmarks:
- GSM8K test set (grade school math)
- MATH test set (competition math)
- Need to implement the generation loop: load model, generate solutions, score

### 7. Investigate long-context with NoPE
Key experiment: train NoPE model at `--max-seq-len=1024`, then evaluate on sequences up to 4096. Compare against RoPE model to measure length generalization. This is the core hypothesis — NoPE should degrade less on longer sequences.

### 8. GRPO implementation
After SFT works, GRPO is the final training stage:
- Need a reward model (or use the binary reward function in `scripts/eval/reward.py`)
- nanochat's `chat_rl.py` handles the RL loop
- Key hyperparams to sweep: group_size, kl_coeff, curriculum ordering

## Architecture decisions to revisit

- **NoPE vs ALiBi**: We implemented NoPE. ALiBi is another option that adds linear attention biases — proven length extrapolation but requires modifying the attention kernel interface. Worth comparing if NoPE doesn't generalize well enough.
- **Submodule vs vendored copy**: Currently using git submodule. If we diverge significantly from upstream nanochat, a vendored copy with our patches applied might be simpler.
- **Eval framework**: Current eval is homebrew. Consider integrating lm-evaluation-harness for standardized benchmarks if we need comparability with published results.
