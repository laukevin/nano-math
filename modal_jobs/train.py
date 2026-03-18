"""Modal job: training launcher that calls nanochat directly.

nanochat handles all training (pretrain, SFT, GRPO), model saving,
eval, and W&B logging. We just pass the right flags.
"""

from __future__ import annotations

import modal

from modal_jobs.common import (
    VOLUME_MOUNTS,
    WANDB_SECRET,
    app,
    code_mount,
    train_image,
    vol_checkpoints,
    vol_results,
)


@app.function(
    image=train_image,
    gpu=modal.gpu.H100(count=1),
    timeout=8 * 3600,
    volumes=VOLUME_MOUNTS,
    mounts=[code_mount],
    secrets=[WANDB_SECRET],
)
def run_train(
    stage: str,
    depth: int,
    experiment_id: str,
    # Pretrain
    mixture: str | None = None,
    token_multiplier: int = 50,
    # SFT
    parent_checkpoint: str | None = None,
    sft_recipe: str | None = None,
    epochs: int = 3,
    lr: float = 2e-5,
    max_seq_len: int = 2048,
    # GRPO
    curriculum: str = "easy-to-hard",
    kl_coeff: float = 0.05,
    group_size: int = 8,
    # Shared
    wandb_mode: str = "online",
    num_iterations: int = -1,
    extra_args: list[str] | None = None,
) -> dict:
    """Run training via nanochat on Modal. Returns run metadata."""
    import subprocess
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, "/root/math-nano")
    start_time = time.monotonic()

    run_name = experiment_id if wandb_mode != "disabled" else "dummy"

    # ── Build command for the appropriate nanochat script ──
    if stage == "pretrain":
        cmd = [
            "python", "-m", "scripts.base_train",
            f"--depth={depth}",
            f"--run={run_name}",
            f"--max-seq-len={max_seq_len}",
        ]
        if num_iterations > 0:
            cmd.append(f"--num-iterations={num_iterations}")

    elif stage == "sft":
        cmd = [
            "python", "-m", "scripts.chat_sft",
            f"--run={run_name}",
            f"--max-seq-len={max_seq_len}",
        ]
        if parent_checkpoint:
            cmd.append(f"--model-tag={parent_checkpoint}")
        if num_iterations > 0:
            cmd.append(f"--num-iterations={num_iterations}")

    elif stage == "grpo":
        cmd = [
            "python", "-m", "scripts.chat_rl",
            f"--run={run_name}",
            f"--num-samples={group_size}",
        ]
        if parent_checkpoint:
            cmd.append(f"--model-tag={parent_checkpoint}")

    else:
        raise ValueError(f"Unknown stage: {stage}")

    if extra_args:
        cmd.extend(extra_args)

    # ── Train ──
    print(f"[{stage}] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd="/root/math-nano")

    elapsed_hours = (time.monotonic() - start_time) / 3600

    # ── Commit volumes ──
    vol_checkpoints.commit()
    vol_results.commit()

    return {
        "experiment_id": experiment_id,
        "stage": stage,
        "depth": depth,
        "wall_clock_hours": elapsed_hours,
        "final_checkpoint": f"checkpoints/{experiment_id}",
        "best_checkpoint": f"checkpoints/{experiment_id}",
        "final_loss": 0.0,
        "tokens_seen": 0,
    }
