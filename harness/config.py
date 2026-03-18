"""Experiment configuration and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

VALID_DEPTHS = [10, 12, 16, 20, 24]
VALID_MIXTURES = [
    "mix-general",
    "mix-math-broad",
    "mix-math-heavy",
    "mix-math-pure",
    "mix-reasoning",
]
VALID_SFT_RECIPES = [
    "sft-distill-r1",
    "sft-concise-cot",
    "sft-kitchen-sink",
    "sft-quality",
    "sft-progressive",
]
VALID_CURRICULA = [
    "easy-to-hard",
    "hard-only",
    "interleaved",
    "reverse",
]

PHASE_BUDGETS = {
    "pretrain": 300.0,
    "sft": 150.0,
    "grpo": 200.0,
}


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    # Identity
    experiment_id: str
    stage: Literal["pretrain", "sft", "grpo"]
    phase: str  # e.g., "1a", "2a"

    # Model
    depth: int
    device: str = "auto"  # cuda, mps, cpu, auto

    # Data (pretrain)
    mixture: Optional[str] = None
    token_multiplier: int = 50

    # Data (SFT)
    sft_recipe: Optional[str] = None
    sft_epochs: int = 3
    sft_lr: float = 2e-5
    sft_max_seq_len: int = 2048

    # Data (GRPO)
    rl_curriculum: Optional[str] = None
    rl_kl_coeff: float = 0.05
    rl_group_size: int = 8

    # Parent
    parent_checkpoint: Optional[str] = None

    # Infra
    gpu: str = "H100"
    timeout_hours: int = 8
    wandb_mode: str = "online"

    # Eval
    eval_suite: str = "small"
    eval_during_training: bool = True
    eval_every: int = 1000

    # Meta
    tags: list[str] = field(default_factory=list)
    notes: str = ""


def validate_config(config: ExperimentConfig) -> list[str]:
    """Validate config before running. Returns list of errors (empty = valid)."""
    errors = []

    # Depth must be in valid set
    if config.depth not in VALID_DEPTHS:
        errors.append(
            f"Invalid depth {config.depth}. Must be one of {VALID_DEPTHS}"
        )

    # Stage-specific requirements
    if config.stage == "pretrain":
        if not config.mixture:
            errors.append("Pretrain requires a data mixture")
        elif config.mixture not in VALID_MIXTURES:
            errors.append(
                f"Unknown mixture '{config.mixture}'. Valid: {VALID_MIXTURES}"
            )

    if config.stage == "sft":
        if not config.sft_recipe:
            errors.append("SFT requires a recipe")
        elif config.sft_recipe not in VALID_SFT_RECIPES:
            errors.append(
                f"Unknown SFT recipe '{config.sft_recipe}'. Valid: {VALID_SFT_RECIPES}"
            )
        if not config.parent_checkpoint:
            errors.append("SFT requires a parent checkpoint")

    if config.stage == "grpo":
        if not config.parent_checkpoint:
            errors.append("GRPO requires a parent checkpoint")
        if config.rl_curriculum and config.rl_curriculum not in VALID_CURRICULA:
            errors.append(
                f"Unknown RL curriculum '{config.rl_curriculum}'. Valid: {VALID_CURRICULA}"
            )

    return errors
