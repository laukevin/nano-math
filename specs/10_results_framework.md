# Spec 10: Results Framework

## Goal

Compile all experiment results into a structured, analyzable format.
Enable rapid iteration: run experiment → see results → decide next experiment.
All data flows through W&B but is also saved locally for offline analysis.

## Data Flow

```
Training/Eval Jobs
       │
       ├──→ W&B (real-time metrics, dashboards)
       │
       └──→ /results/ volume (JSON artifacts, for offline analysis)
              │
              └──→ results/compiled/ (aggregated tables, plots)
```

## W&B Dashboards

### Dashboard 1: "Scaling Curves" (the money plot)
- X-axis: log(model_params)
- Y-axis: GSM8K pass@1
- Lines: one per stage (pretrain, SFT, RL)
- Filters: by mixture, by recipe

### Dashboard 2: "Training Dynamics"
- X-axis: training step (or tokens seen)
- Y-axis: val_bpb or GSM8K pass@1
- Grouped by experiment_id
- Compare: different mixtures at same model size

### Dashboard 3: "Recipe Comparison"
- Bar chart: GSM8K pass@1 for each SFT recipe at M size
- Grouped bars: before/after RL

### Dashboard 4: "RL Progress"
- X-axis: RL step
- Y-axis (left): reward mean
- Y-axis (right): GSM8K pass@1
- One panel per curriculum strategy

### Dashboard 5: "Cost Tracker"
- Cumulative cost (USD) over time
- Cost per experiment
- Cost per GSM8K point gained

## Local Results Structure

```
results/
├── eval/
│   ├── pt-xs-general_gsm8k.json
│   ├── pt-xs-general_math500.json
│   ├── sft-m-distill-r1_gsm8k.json
│   └── ...
├── compiled/
│   ├── scaling_curve.csv          # aggregated: params, stage, gsm8k, math500
│   ├── mixture_comparison.csv     # aggregated: mixture, size, metrics
│   ├── recipe_comparison.csv      # aggregated: recipe, size, metrics
│   ├── cost_summary.csv           # experiment_id, hours, cost_usd
│   └── full_results.csv           # everything in one flat table
├── plots/
│   ├── scaling_curve.png
│   ├── mixture_comparison.png
│   ├── recipe_comparison.png
│   └── rl_dynamics.png
└── analysis/
    ├── phase1_summary.md          # written after phase 1
    ├── phase2_summary.md          # written after phase 2
    └── phase3_summary.md          # written after phase 3
```

## Compilation Script

```bash
python scripts/results/compile.py \
  --results-dir results/eval/ \
  --output results/compiled/full_results.csv
```

This script:
1. Reads all eval JSON files from `results/eval/`
2. Extracts: experiment_id, model_depth, params, stage, dataset, pass@1
3. Optionally pulls W&B metadata (cost, wall clock) via API
4. Outputs a flat CSV with one row per (experiment, dataset) pair
5. Generates summary CSVs (scaling_curve.csv, etc.)

### Full Results Schema

```csv
experiment_id,model_depth,model_params,stage,pretrain_mixture,sft_recipe,rl_curriculum,dataset,pass_at_1_greedy,pass_at_1_sampled,wall_clock_hours,cost_usd,tokens_seen,timestamp
pt-s-broad,12,85000000,pretrain,mix-math-broad,,,,gsm8k,0.02,0.03,4.2,14.70,4250000000,2026-03-20T14:00:00Z
sft-m-distill-r1,16,130000000,sft,mix-math-broad,sft-distill-r1,,,gsm8k,0.35,0.38,1.5,5.25,,2026-03-22T10:00:00Z
```

## Plotting Script

```bash
python scripts/results/plot.py \
  --data results/compiled/full_results.csv \
  --plot scaling_curve \
  --output results/plots/scaling_curve.png
```

Plots to generate:

| Plot | X-axis | Y-axis | Grouping |
|------|--------|--------|----------|
| `scaling_curve` | log(params) | GSM8K pass@1 | stage (pretrain/sft/rl) |
| `mixture_comparison` | mixture_id | GSM8K pass@1 | model_size |
| `recipe_comparison` | recipe_id | GSM8K pass@1 | model_size |
| `curriculum_comparison` | curriculum_id | AIME pass@1 | model_size |
| `training_dynamics` | tokens_seen | val_bpb | experiment_id |
| `rl_dynamics` | rl_step | reward_mean + gsm8k | curriculum |
| `cost_efficiency` | cost_usd | GSM8K pass@1 | stage |
| `token_budget` | token_multiplier | GSM8K pass@1 | model_size |

Use matplotlib with a clean style. All plots saved as PNG (300 DPI) + PDF.

## Phase Summaries

After each phase, generate a summary document:

### Template: `results/analysis/phase{N}_summary.md`
```markdown
# Phase N Summary

## Key Findings
- [Bullet points of main results]

## Best Configurations
- Model: [depth, params]
- Mixture/Recipe/Curriculum: [best config]
- Score: [GSM8K pass@1, MATH500 pass@1]

## Scaling Curve Update
[Embed or reference scaling curve plot]

## Surprises / Unexpected Results
- [Anything that contradicted hypotheses]

## Decisions for Next Phase
- [What to carry forward]
- [What to drop]
- [Open questions]

## Cost Summary
- Total spend this phase: $X
- Cost per run: $X avg
- Most expensive run: [experiment_id, $X]
```

## W&B API Access

For pulling data programmatically:

```python
import wandb
api = wandb.Api()

# Get all runs in project
runs = api.runs("YOUR_ENTITY/math-nano")

# Filter by stage
pretrain_runs = [r for r in runs if "pretrain" in r.tags]

# Get metrics
for run in pretrain_runs:
    history = run.history(keys=["eval/gsm8k_pass1"])
    final_gsm8k = run.summary.get("eval/gsm8k_pass1")
```

## Reproducibility

Every result must be traceable to:
1. Exact code version (git commit hash logged to W&B)
2. Exact data version (shard checksums logged)
3. Exact hyperparameters (full config in W&B)
4. Random seed (logged, though not all ops are deterministic on GPU)

Log git commit hash at start of every run:
```python
import subprocess
git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
wandb.config.update({"git_hash": git_hash})
```
