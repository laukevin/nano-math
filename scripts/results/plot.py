"""Generate plots from compiled eval results.

Usage:
    python scripts/results/plot.py --data results/compiled/full_results.csv \
      --plot scaling_curve --output results/plots/scaling_curve.png

    python scripts/results/plot.py --data results/compiled/full_results.csv --all
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Consistent style
STYLE_DEFAULTS = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.figsize": (8, 5),
}

STAGE_COLORS = {
    "pretrain": "#4C72B0",
    "sft": "#DD8452",
    "grpo": "#55A868",
}

STAGE_MARKERS = {
    "pretrain": "o",
    "sft": "s",
    "grpo": "D",
}


def apply_style() -> None:
    plt.rcParams.update(STYLE_DEFAULTS)
    plt.style.use("seaborn-v0_8-whitegrid")


def save_figure(fig: plt.Figure, output_path: str) -> None:
    """Save figure as PNG and PDF."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), bbox_inches="tight")
    fig.savefig(str(path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Plot: Scaling Curve
# ---------------------------------------------------------------------------

def plot_scaling_curve(
    df: pd.DataFrame,
    output_path: str,
    metric: str = "pass_at_1_greedy",
    dataset: str = "gsm8k",
) -> None:
    """Model size vs metric, one line per stage."""
    apply_style()

    subset = df[df["dataset"] == dataset].copy()
    if subset.empty:
        logger.warning("No data for dataset=%s", dataset)
        return

    fig, ax = plt.subplots()

    for stage in ["pretrain", "sft", "grpo"]:
        stage_data = subset[subset["stage"] == stage]
        if stage_data.empty:
            continue

        grouped = (
            stage_data.groupby("model_params")[metric]
            .mean()
            .sort_index()
        )

        ax.plot(
            grouped.index,
            grouped.values,
            marker=STAGE_MARKERS.get(stage, "o"),
            color=STAGE_COLORS.get(stage, "gray"),
            label=stage,
            linewidth=2,
            markersize=8,
        )

        # Add CI error bars if available
        ci_col = f"{metric}_ci95_low"
        if ci_col in stage_data.columns:
            ci_low = (
                stage_data.groupby("model_params")[f"{metric}_ci95_low"]
                .mean()
                .sort_index()
            )
            ci_high = (
                stage_data.groupby("model_params")[f"{metric}_ci95_high"]
                .mean()
                .sort_index()
            )
            ax.fill_between(
                grouped.index,
                ci_low.values,
                ci_high.values,
                alpha=0.15,
                color=STAGE_COLORS.get(stage, "gray"),
            )

    ax.set_xscale("log")
    ax.set_xlabel("Model Parameters")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{dataset.upper()} — Scaling Curve")
    ax.legend()

    save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Plot: Mixture Comparison Heatmap
# ---------------------------------------------------------------------------

def plot_mixture_heatmap(
    df: pd.DataFrame,
    output_path: str,
    metric: str = "pass_at_1_greedy",
    dataset: str = "gsm8k",
) -> None:
    """Heatmap: model_size x pretrain_mixture."""
    apply_style()

    if "pretrain_mixture" not in df.columns:
        logger.warning("No pretrain_mixture column — skipping heatmap")
        return

    subset = df[df["dataset"] == dataset].copy()
    if subset.empty:
        return

    pivot = subset.pivot_table(
        values=metric,
        index="model_depth",
        columns="pretrain_mixture",
        aggfunc="mean",
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(
        figsize=(len(pivot.columns) * 2 + 2, len(pivot.index) * 1.2 + 1)
    )

    try:
        import seaborn as sns

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=max(0.5, pivot.max().max() * 1.2),
            linewidths=1,
            ax=ax,
        )
    except ImportError:
        # Fallback without seaborn
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center")
        fig.colorbar(im, ax=ax)

    ax.set_title(f"{dataset.upper()} {metric} — Mixture Comparison")
    plt.tight_layout()

    save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Plot: Cost Efficiency
# ---------------------------------------------------------------------------

def plot_cost_efficiency(
    df: pd.DataFrame,
    output_path: str,
    metric: str = "pass_at_1_greedy",
    dataset: str = "gsm8k",
) -> None:
    """Cost (USD) vs metric — Pareto frontier."""
    apply_style()

    if "cost_usd" not in df.columns:
        logger.warning("No cost_usd column — skipping cost plot")
        return

    subset = df[df["dataset"] == dataset].dropna(subset=["cost_usd", metric])
    if subset.empty:
        return

    fig, ax = plt.subplots()

    for stage in ["pretrain", "sft", "grpo"]:
        s = subset[subset["stage"] == stage]
        if s.empty:
            continue
        ax.scatter(
            s["cost_usd"],
            s[metric],
            marker=STAGE_MARKERS.get(stage, "o"),
            color=STAGE_COLORS.get(stage, "gray"),
            label=stage,
            s=80,
            zorder=3,
        )

    # Pareto frontier
    points = subset[["cost_usd", metric]].values
    sorted_idx = np.argsort(points[:, 0])
    frontier_x, frontier_y = [], []
    best_y = -1
    for idx in sorted_idx:
        if points[idx, 1] > best_y:
            best_y = points[idx, 1]
            frontier_x.append(points[idx, 0])
            frontier_y.append(points[idx, 1])

    if frontier_x:
        ax.plot(
            frontier_x,
            frontier_y,
            "--",
            color="gray",
            alpha=0.5,
            label="Pareto frontier",
        )

    ax.set_xlabel("Cost (USD)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{dataset.upper()} — Cost Efficiency")
    ax.legend()

    save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Plot: RL Reward Dynamics
# ---------------------------------------------------------------------------

def plot_rl_dynamics(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """RL step vs reward and GSM8K pass@1.

    Expects df with columns: rl_step, reward_mean, gsm8k_pass1, curriculum.
    This is typically loaded from W&B or a separate dynamics CSV, not from
    the standard eval JSON compilation.
    """
    apply_style()

    required = {"rl_step", "reward_mean"}
    if not required.issubset(df.columns):
        logger.warning(
            "Missing columns for RL dynamics: %s",
            required - set(df.columns),
        )
        return

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    curricula = df["curriculum"].unique() if "curriculum" in df.columns else ["default"]

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(curricula), 1)))

    for i, curr in enumerate(curricula):
        if "curriculum" in df.columns:
            sub = df[df["curriculum"] == curr].sort_values("rl_step")
        else:
            sub = df.sort_values("rl_step")

        ax1.plot(
            sub["rl_step"],
            sub["reward_mean"],
            color=colors[i],
            label=f"{curr} reward",
            linewidth=2,
        )

        if "gsm8k_pass1" in sub.columns:
            ax2.plot(
                sub["rl_step"],
                sub["gsm8k_pass1"],
                color=colors[i],
                linestyle="--",
                label=f"{curr} GSM8K",
                linewidth=2,
            )

    ax1.set_xlabel("RL Step")
    ax1.set_ylabel("Reward Mean", color="black")
    ax2.set_ylabel("GSM8K pass@1", color="gray")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax1.set_title("RL Training Dynamics")

    save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Plot: Recipe / Curriculum Comparison (bar chart)
# ---------------------------------------------------------------------------

def plot_recipe_comparison(
    df: pd.DataFrame,
    output_path: str,
    group_col: str = "sft_recipe",
    metric: str = "pass_at_1_greedy",
    dataset: str = "gsm8k",
) -> None:
    """Grouped bar chart comparing recipes/curricula."""
    apply_style()

    if group_col not in df.columns:
        logger.warning("No %s column — skipping recipe comparison", group_col)
        return

    subset = df[df["dataset"] == dataset].dropna(subset=[group_col, metric])
    if subset.empty:
        return

    fig, ax = plt.subplots()

    groups = sorted(subset[group_col].unique())
    sizes = sorted(subset["model_depth"].dropna().unique())
    n_groups = len(groups)
    bar_width = 0.8 / max(len(sizes), 1)

    for j, size in enumerate(sizes):
        vals = []
        for g in groups:
            v = subset[
                (subset[group_col] == g) & (subset["model_depth"] == size)
            ][metric].mean()
            vals.append(v if not np.isnan(v) else 0)

        x = np.arange(n_groups) + j * bar_width
        ax.bar(x, vals, bar_width, label=f"depth={int(size)}")

    ax.set_xticks(np.arange(n_groups) + bar_width * (len(sizes) - 1) / 2)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{dataset.upper()} — {group_col.replace('_', ' ').title()} Comparison")
    ax.legend()
    plt.tight_layout()

    save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PLOT_REGISTRY = {
    "scaling_curve": plot_scaling_curve,
    "mixture_comparison": plot_mixture_heatmap,
    "cost_efficiency": plot_cost_efficiency,
    "rl_dynamics": plot_rl_dynamics,
    "recipe_comparison": plot_recipe_comparison,
}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate result plots")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to compiled CSV (full_results.csv)",
    )
    parser.add_argument(
        "--plot",
        choices=list(PLOT_REGISTRY.keys()),
        default=None,
        help="Specific plot to generate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all plots",
    )
    parser.add_argument(
        "--output",
        default="results/plots/plot.png",
        help="Output path (for single plot)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/plots",
        help="Output directory (for --all)",
    )
    parser.add_argument(
        "--metric",
        default="pass_at_1_greedy",
        help="Metric to plot",
    )
    parser.add_argument(
        "--dataset",
        default="gsm8k",
        help="Dataset to filter on",
    )

    args = parser.parse_args(argv)

    df = pd.read_csv(args.data)
    logger.info("Loaded %d rows from %s", len(df), args.data)

    if args.all:
        out_dir = Path(args.output_dir)
        for name, fn in PLOT_REGISTRY.items():
            out = str(out_dir / f"{name}.png")
            logger.info("Generating %s...", name)
            if name == "rl_dynamics":
                fn(df, out)
            elif name == "recipe_comparison":
                fn(df, out, metric=args.metric, dataset=args.dataset)
            else:
                fn(df, out, metric=args.metric, dataset=args.dataset)
    elif args.plot:
        fn = PLOT_REGISTRY[args.plot]
        if args.plot == "rl_dynamics":
            fn(df, args.output)
        else:
            fn(df, args.output, metric=args.metric, dataset=args.dataset)
    else:
        parser.error("Specify --plot or --all")


if __name__ == "__main__":
    main()
