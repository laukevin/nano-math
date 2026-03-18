"""Metrics contract — every training loop must report these."""

from __future__ import annotations

# Required every step
STEP_METRICS = [
    "train/loss",
    "train/lr",
    "train/step",
    "train/tokens_seen",
]

# Required every eval_every steps
EVAL_METRICS = [
    "eval/gsm8k_pass1",
]

# Required at end of run
FINAL_METRICS = [
    "final/train_loss",
    "final/wall_clock_hours",
    "final/tokens_seen",
    "final/cost_usd_estimated",
    "final/checkpoint_path",
]

# Stage-specific additions
PRETRAIN_METRICS = [
    "val/bpb_math",
    "val/bpb_fineweb",
]

SFT_METRICS = [
    "train/epoch",
    "eval/format_compliance",
]

GRPO_METRICS = [
    "rl/reward_mean",
    "rl/reward_std",
    "rl/kl_divergence",
    "rl/avg_output_length",
    "rl/curriculum_stage",
]


def get_required_metrics(stage: str) -> set[str]:
    """Return all required metrics for a given stage."""
    required = set(STEP_METRICS + EVAL_METRICS + FINAL_METRICS)
    if stage == "pretrain":
        required |= set(PRETRAIN_METRICS)
    elif stage == "sft":
        required |= set(SFT_METRICS)
    elif stage == "grpo":
        required |= set(GRPO_METRICS)
    return required
