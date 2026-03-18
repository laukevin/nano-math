# Spec 05: SFT Experiments

## Experiment Matrix

### Primary Sweep: Size x Recipe

We don't run every combination. Strategy:

**Phase 2a: Recipe Comparison (4-5 runs)**
Pick the best pretrained M-size checkpoint. Run all SFT recipes on it.

| Experiment ID | Base Checkpoint | SFT Recipe | Purpose |
|---------------|----------------|------------|---------|
| `sft-m-distill-r1` | best M pretrain | sft-distill-r1 | Long CoT |
| `sft-m-concise` | best M pretrain | sft-concise-cot | Short CoT |
| `sft-m-kitchen` | best M pretrain | sft-kitchen-sink | Diversity |
| `sft-m-quality` | best M pretrain | sft-quality | Small high-quality |
| `sft-m-progressive` | best M pretrain | sft-progressive | Curriculum |

**Phase 2b: Size Scaling (4 runs)**
Take winning recipe, run on all sizes.

| Experiment ID | Base Checkpoint | SFT Recipe |
|---------------|----------------|------------|
| `sft-xs-best` | best XS pretrain | winning recipe |
| `sft-s-best` | best S pretrain | winning recipe |
| `sft-l-best` | best L pretrain | winning recipe |
| `sft-xl-best` | best XL pretrain | winning recipe |

**Phase 2c: Interaction Effects (optional, 2-4 runs)**
Does the best SFT recipe depend on pretrain mixture?
Run winning SFT recipe on different pretrain bases for same model size.

| Experiment ID | Base Checkpoint | SFT Recipe |
|---------------|----------------|------------|
| `sft-m-general+best` | M/mix-general | winning recipe |
| `sft-m-broad+best` | M/mix-math-broad | winning recipe |
| `sft-m-heavy+best` | M/mix-math-heavy | winning recipe |

## Hyperparameter Grid

### Defaults (from literature + nanochat conventions)

| Param | Default | Range to Explore |
|-------|---------|-----------------|
| Learning rate | 2e-5 | {5e-6, 1e-5, 2e-5, 5e-5} |
| LR schedule | cosine → 0 | cosine only |
| Warmup steps | 100 | fixed |
| Weight decay | 0.01 | fixed |
| Batch size | 32 (sequences) | fixed |
| Epochs | 3 | {1, 3, 5, 10} |
| Max seq len | 2048 | {1024, 2048, 4096} |
| Gradient clipping | 1.0 | fixed |

### LR Sweep (4 runs, on M model with winning recipe)

| Experiment ID | LR | Purpose |
|---------------|-----|---------|
| `sft-m-lr5e6` | 5e-6 | Conservative |
| `sft-m-lr1e5` | 1e-5 | Moderate |
| `sft-m-lr2e5` | 2e-5 | Default |
| `sft-m-lr5e5` | 5e-5 | Aggressive |

Only run if default LR seems suboptimal (train loss too high or eval degrading).

## Metrics Logged Per Run

| Metric | Log Frequency | Source |
|--------|--------------|--------|
| `train/loss` | Every step | Training loop |
| `train/lr` | Every step | Scheduler |
| `train/epoch` | Every step | Counter |
| `eval/gsm8k_pass1` | Every 200 steps | GSM8K eval (full test set, greedy) |
| `eval/math500_pass1` | Every 500 steps | MATH500 eval |
| `eval/gsm8k_pass1_best` | On improvement | Running best |
| `meta/wall_clock_hours` | Every 100 steps | Timer |

### W&B Run Config

```python
wandb.config.update({
    "experiment_id": "sft-m-distill-r1",
    "model_depth": 16,
    "model_params": actual_param_count,
    "base_checkpoint": "pt-m-broad/step_050000.pt",
    "sft_recipe": "sft-distill-r1",
    "sft_samples": 100000,
    "sft_max_seq_len": 4096,
    "sft_epochs": 3,
    "sft_lr": 2e-5,
    "stage": "sft",
    "phase": "2a",
})
```

## Checkpointing

- Save checkpoint when GSM8K pass@1 improves (best checkpoint)
- Save at end of each epoch
- Save final checkpoint
- Keep: best + final (delete intermediate epoch checkpoints after training)

## Early Stopping

Monitor `eval/gsm8k_pass1` every 200 steps:
- If GSM8K score drops by >5 points from peak for 3 consecutive evals → stop
- This prevents catastrophic forgetting from overtraining

Also monitor `train/loss`:
- If train loss < 0.1 → likely overfitting → log warning, continue but flag

## Success Criteria for SFT Phase

- [ ] At least one model achieves GSM8K pass@1 > 15% (stretch: > 30%)
- [ ] Clear recipe ranking: one recipe is consistently better across sizes
- [ ] Scaling curve: GSM8K pass@1 vs model size, showing clear upward trend
- [ ] No catastrophic forgetting detected (pre-SFT math bpb doesn't degrade wildly)
- [ ] Interaction effects documented (does pretrain mixture matter for SFT?)

## Analysis Outputs

1. **Recipe comparison bar chart**: GSM8K pass@1 for each recipe at M size
2. **SFT scaling curve**: x = log(params), y = GSM8K pass@1, post-SFT
3. **Learning dynamics plot**: GSM8K pass@1 vs training step, per recipe
4. **Epoch analysis**: performance per epoch (when does overfitting start?)
5. **Before/after table**: pretrain-only vs post-SFT GSM8K scores for all sizes
