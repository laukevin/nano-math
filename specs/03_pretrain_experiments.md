# Spec 03: Pretraining Experiments

## Experiment Matrix

### Primary Sweep: Size x Mixture

The core experiment is a grid over model size and data mixture:

| Experiment ID | Depth | Mixture | Token Mult | Purpose |
|---------------|-------|---------|------------|---------|
| `pt-xs-general` | 10 | mix-general | 50 | Baseline smallest |
| `pt-xs-broad` | 10 | mix-math-broad | 50 | Math broad smallest |
| `pt-xs-heavy` | 10 | mix-math-heavy | 50 | Math heavy smallest |
| `pt-s-general` | 12 | mix-general | 50 | Baseline small |
| `pt-s-broad` | 12 | mix-math-broad | 50 | Math broad small |
| `pt-s-heavy` | 12 | mix-math-heavy | 50 | Math heavy small |
| `pt-m-general` | 16 | mix-general | 50 | Baseline medium |
| `pt-m-broad` | 16 | mix-math-broad | 50 | Math broad medium |
| `pt-m-heavy` | 16 | mix-math-heavy | 50 | Math heavy medium |
| `pt-l-general` | 20 | mix-general | 50 | Baseline large |
| `pt-l-broad` | 20 | mix-math-broad | 50 | Math broad large |
| `pt-l-heavy` | 20 | mix-math-heavy | 50 | Math heavy large |
| `pt-xl-general` | 24 | mix-general | 50 | Baseline XL |
| `pt-xl-broad` | 24 | mix-math-broad | 50 | Math broad XL |
| `pt-xl-heavy` | 24 | mix-math-heavy | 50 | Math heavy XL |

That's 15 runs. But we DON'T run all 15 at once.

### Phased Execution

**Phase 1a: Pilot (3 runs)**
Run the S (depth=12) model with all 3 mixtures first.
- Purpose: Validate pipeline, calibrate token budget, check mixture effects
- Duration: ~1 day each on H100
- Exit criteria: All 3 runs complete, W&B logging verified, GSM8K eval works

**Phase 1b: Mixture Selection (2-5 runs)**
Based on Phase 1a results, pick the best 1-2 mixtures.
Run those mixtures on XS and M to see if mixture ranking holds across sizes.
- Purpose: Verify mixture ranking is consistent across model sizes
- If ranking differs: we have an interesting finding. Run the full grid.
- If ranking is consistent: drop the worst mixture, save compute.

**Phase 1c: Full Size Sweep (5 runs)**
Run the winning mixture(s) across all 5 model sizes.
- Purpose: Generate the math pretraining scaling curve
- This produces the key plot: model_size vs. math_val_bpb

### Secondary Sweep: Token Budget

After mixture is selected, investigate token multiplier:

| Experiment ID | Depth | Mixture | Token Mult | Purpose |
|---------------|-------|---------|------------|---------|
| `pt-m-20x` | 16 | best | 20 | Under-train |
| `pt-m-50x` | 16 | best | 50 | Default |
| `pt-m-100x` | 16 | best | 100 | Over-train |

This answers: how much overtraining is optimal for math at this scale?

## Metrics Logged Per Run

Every pretrain run must log to W&B:

| Metric | Log Frequency | Source |
|--------|--------------|--------|
| `train/loss` | Every step | Training loop |
| `train/lr` | Every step | Scheduler |
| `train/tokens_seen` | Every step | Counter |
| `val/bpb_fineweb` | Every 1000 steps | FineWeb held-out |
| `val/bpb_math` | Every 1000 steps | Math held-out |
| `eval/gsm8k_pass1` | Every 5000 steps | GSM8K eval (100 problems, greedy) |
| `meta/wall_clock_hours` | Every 1000 steps | Timer |
| `meta/tokens_per_second` | Every 100 steps | Counter / timer |

### W&B Run Config (logged once at start)

```python
wandb.config.update({
    "experiment_id": "pt-s-broad",
    "model_depth": 12,
    "model_params": actual_param_count,
    "data_mixture": "mix-math-broad",
    "mixture_weights": {"fineweb": 0.5, "openwebmath": 0.4, "omr": 0.1},
    "token_multiplier": 50,
    "total_tokens_planned": total_tokens,
    "stage": "pretrain",
    "phase": "1a",
})
```

## Checkpointing

- Save every 2000 steps (not 1000 — these models are small, checkpoints are cheap)
- Also save at end of training
- Keep only last 3 checkpoints + best (lowest math val_bpb) to save disk
- Checkpoint naming: `{experiment_id}/step_{step:06d}.pt`

## Early Stopping / Saturation Detection

A pretrain run is "saturated" when:
- `val/bpb_math` has not improved by > 0.01 in the last 10% of planned tokens
- Implementation: simple patience counter, not stopping the run, just flagging

When saturation is detected:
1. Log `meta/saturated_at_step` to W&B
2. Continue training to planned token count (data for the curve)
3. Note: saturation step is itself a useful data point for the scaling curve

## Success Criteria for Pretrain Phase

Phase 1 is complete when we have:
- [ ] Clean scaling curve: math val_bpb vs model size for at least 4 sizes
- [ ] Mixture comparison: at least 2 mixtures compared at 2+ sizes
- [ ] Token budget sensitivity: 3 multipliers compared at 1 size
- [ ] All checkpoints saved and loadable
- [ ] GSM8K pass@1 > 0 for at least the M model (even barely)
  - If no model gets >0 on GSM8K after pretraining, that's expected and fine.
    The real gains come from SFT and RL. But any nonzero signal is encouraging.

## Analysis Outputs (from this phase)

1. **Scaling curve plot**: x = log(params), y = math val_bpb, one line per mixture
2. **Mixture comparison table**: GSM8K pass@1 at end of pretrain, all configs
3. **Token efficiency plot**: x = tokens seen, y = math val_bpb, one line per multiplier
4. **Compute cost table**: wall clock hours and $ per run
