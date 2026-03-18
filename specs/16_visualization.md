# Spec 16: Eval Visualization & Heatmaps

## Problem

We have a multi-dimensional experiment space:
- Model size (5 values)
- Data mixture (3-5 values)
- SFT recipe (5 values)
- RL curriculum (4 values)
- Eval dataset (5 datasets)
- Metric (pass@1, pass@4, pass@8)

A table of numbers won't cut it. We need heatmaps, interactive views,
and automatic surfacing of interesting patterns.

---

## Core Visualizations

### 1. The Experiment Heatmap

The central visualization. A 2D grid where:
- **Rows** = one dimension (e.g., model size)
- **Columns** = another dimension (e.g., SFT recipe)
- **Cell color** = metric value (e.g., GSM8K pass@1)
- **Cell text** = exact value + CI

```
              │ concise-cot │ distill-r1 │ kitchen-sink │ quality │
──────────────┼─────────────┼────────────┼──────────────┼─────────┤
 XS (50M)     │  0.08       │  0.05      │  0.07        │  0.10   │
 S  (85M)     │  0.18       │  0.14      │  0.16        │  0.19   │
 M  (130M)    │  0.34       │  0.28      │  0.31        │  0.32   │
 L  (200M)    │  0.41       │  0.36      │  0.39        │  0.38   │
 XL (320M)    │  0.48       │  0.44      │  0.46        │  0.43   │
```

Color scale: red (0%) → yellow (25%) → green (50%+).

### Heatmap Configurations (pre-defined)

| Heatmap | Rows | Columns | Metric | When Generated |
|---------|------|---------|--------|---------------|
| `size_x_mixture` | Model size | Pretrain mixture | math val_bpb | After Phase 1 |
| `size_x_recipe` | Model size | SFT recipe | GSM8K pass@1 | After Phase 2 |
| `size_x_curriculum` | Model size | RL curriculum | GSM8K pass@1 | After Phase 3 |
| `recipe_x_mixture` | SFT recipe | Pretrain mixture | GSM8K pass@1 | After Phase 2c |
| `size_x_dataset` | Model size | Eval dataset | pass@1 | Any time |
| `stage_x_size` | Stage (pt/sft/rl) | Model size | GSM8K pass@1 | After Phase 3 |
| `size_x_metric` | Model size | Metric (p@1/4/8) | Score | After any full eval |

### Heatmap Generation

```bash
# Generate a specific heatmap
uv run python scripts/viz/heatmap.py \
  --data results/compiled/full_results.csv \
  --rows model_size \
  --cols sft_recipe \
  --metric gsm8k_pass1_greedy \
  --output results/plots/heatmap_size_x_recipe.png

# Generate all pre-defined heatmaps
uv run python scripts/viz/heatmap.py --all

# Interactive HTML version (for deep exploration)
uv run python scripts/viz/heatmap.py --all --format html
```

### Implementation

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    metric: str,
    title: str = None,
    output_path: str = None,
    annotate_ci: bool = True,
    cmap: str = "RdYlGn",
):
    """Generate a heatmap from the results dataframe."""
    pivot = df.pivot_table(
        values=metric,
        index=row_col,
        columns=col_col,
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(len(pivot.columns) * 2 + 2, len(pivot.index) * 1.2 + 1))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=0,
        vmax=max(0.5, pivot.max().max() * 1.2),
        linewidths=1,
        ax=ax,
    )

    if annotate_ci:
        # Overlay CI in smaller font
        ci_col = f"{metric}_ci95"
        if ci_col in df.columns:
            for i, row in enumerate(pivot.index):
                for j, col in enumerate(pivot.columns):
                    mask = (df[row_col] == row) & (df[col_col] == col)
                    if mask.any():
                        ci = df.loc[mask, ci_col].iloc[0]
                        ax.text(j + 0.5, i + 0.75,
                                f"±{(ci[1]-ci[0])/2:.2f}",
                                ha="center", va="center", fontsize=7, color="gray")

    ax.set_title(title or f"{metric} by {row_col} x {col_col}")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
```

---

### 2. Delta Heatmaps (Stage Gains)

Show the *improvement* from one stage to the next:

```
SFT Gain over Pretrain (GSM8K pass@1):

              │ concise-cot │ distill-r1 │
──────────────┼─────────────┼────────────┤
 XS (50M)     │  +0.06      │  +0.03     │  (pretrain was 0.02)
 S  (85M)     │  +0.16      │  +0.12     │
 M  (130M)    │  +0.32      │  +0.26     │
 L  (200M)    │  +0.39      │  +0.34     │
```

Color: diverging scale (red = regression, white = no change, blue = improvement).
Immediately shows where SFT helps most and where it doesn't.

```bash
uv run python scripts/viz/delta_heatmap.py \
  --stage-a pretrain \
  --stage-b sft \
  --metric gsm8k_pass1_greedy \
  --output results/plots/delta_sft_over_pretrain.png
```

---

### 3. Scaling Curve Gallery

One plot per eval dataset, all model sizes on x-axis, lines per stage:

```
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│ GSM8K               │ │ MATH500             │ │ AIME                │
│                  ●RL │ │                  ●RL│ │                     │
│              ●SFT    │ │              ●SFT   │ │                 ●RL │
│         ●PT          │ │         ●PT         │ │                     │
│ 50M 85M 130M 200M 320M│ │ 50M ... 320M      │ │ 50M ... 320M       │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
```

Generated as a single figure with subplots:
```bash
uv run python scripts/viz/scaling_gallery.py \
  --output results/plots/scaling_gallery.png
```

---

### 4. Radar / Spider Charts (Model Fingerprints)

Show a single model's performance profile across all datasets:

```
        GSM8K
          ●
         / \
    AIME/   \MATH500
       /     \
      ●───────●
       \     /
    AMC \   / Minerva
         \ /
          ●
```

Useful for comparing two models side by side:
"This model is better at GSM8K but worse at AIME."

```bash
uv run python scripts/viz/radar.py \
  --models sft-m-concise-best,sft-m-distill-r1-best \
  --output results/plots/radar_recipe_comparison.png
```

---

### 5. Training Dynamics Overlay

Multiple runs on the same axes to see how training progresses:

```bash
uv run python scripts/viz/dynamics.py \
  --experiments sft-m-concise,sft-m-distill-r1,sft-m-quality \
  --metric eval/gsm8k_pass1 \
  --x-axis step \
  --output results/plots/sft_dynamics.png
```

Shows: learning speed, plateau point, overfitting onset.

---

### 6. Cost-Efficiency Frontier

Pareto plot: x = total cost (USD), y = best GSM8K pass@1 achieved.

```
  GSM8K │           ● XL-full-pipeline
  pass@1│       ● L
        │    ● M     ← Pareto frontier
        │  ● S
        │● XS
        └──────────────────────
                   Cost ($)
```

Models ON the frontier are cost-efficient. Models BELOW are dominated.
This answers: "If I only have $50, what's the best model I can train?"

```bash
uv run python scripts/viz/pareto.py \
  --metric gsm8k_pass1_greedy \
  --output results/plots/cost_frontier.png
```

---

## Interactive Dashboard (Local)

For deeper exploration, generate an HTML dashboard:

```bash
uv run python scripts/viz/dashboard.py \
  --data results/compiled/full_results.csv \
  --output results/dashboard/index.html
```

Opens in browser. Features:
- Dropdown selectors for rows/columns/metric
- Dynamic heatmap that reconfigures on selection
- Click a cell to see training curve for that experiment
- Hover for full details (CI, cost, parent model, etc.)

Implementation: use **Plotly** for interactivity, output as self-contained HTML.

```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_interactive_heatmap(df, row_col, col_col, metric):
    pivot = df.pivot_table(values=metric, index=row_col, columns=col_col)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        text=[[f"{v:.3f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        colorscale="RdYlGn",
        hovertemplate=(
            f"{row_col}: %{{y}}<br>"
            f"{col_col}: %{{x}}<br>"
            f"{metric}: %{{z:.3f}}<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(title=f"{metric} by {row_col} x {col_col}")
    return fig
```

---

## Auto-Generated Reports

After each wave of experiments, auto-generate a visual report:

```bash
uv run python scripts/viz/report.py --phase 2 --wave 3
```

Output: `results/reports/phase2_wave3.html`

Contains:
1. Summary table of completed runs
2. Heatmaps for all relevant dimension pairs
3. Delta heatmaps vs previous best
4. Training dynamics for new runs
5. Updated scaling curves
6. Cost summary
7. Statistical significance tests for any claimed improvements

This is what the agent presents to the human at the end of each wave.

---

## W&B Dashboard Sync

The local plots are great for offline analysis. But W&B dashboards
are better for real-time monitoring. Keep both in sync:

```python
def log_heatmap_to_wandb(fig, name):
    """Log a matplotlib figure as a W&B image."""
    wandb.log({f"viz/{name}": wandb.Image(fig)})

def log_plotly_to_wandb(fig, name):
    """Log a plotly figure as a W&B HTML panel."""
    wandb.log({f"viz/{name}": wandb.Html(fig.to_html())})
```

Key W&B panels to maintain:
- Scaling curve (updated after every eval)
- Best heatmap (updated after each phase)
- Cost tracker (updated after every run)
