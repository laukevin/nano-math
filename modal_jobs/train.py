"""Modal job: single training launcher for all stages (pretrain, sft, grpo).

Runs training + inline post-eval, commits volumes.
GPU: H100, Timeout: configurable (default 8h).
"""

from __future__ import annotations

import modal

from modal_jobs.common import (
    VOLUME_MOUNTS,
    WANDB_SECRET,
    app,
    train_image,
    vol_checkpoints,
    vol_results,
)


@app.function(
    image=train_image,
    gpu=modal.gpu.H100(count=1),
    timeout=8 * 3600,
    volumes=VOLUME_MOUNTS,
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
    eval_datasets: str = "gsm8k,math500",
    extra_args: list[str] | None = None,
) -> dict:
    """Run any training stage on Modal.

    The stage parameter determines which training script runs and what
    checkpoint directory structure is used. Eval runs inline after training.
    """
    import json
    import subprocess
    from pathlib import Path

    ckpt_dir = f"/checkpoints/d{depth}/{stage}/{experiment_id}"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # ── Build training command based on stage ──
    if stage == "pretrain":
        cmd = [
            "python", "-m", "base_train",
            f"--depth={depth}",
            f"--data-source={mixture}",
            f"--token-multiplier={token_multiplier}",
            f"--run-name={experiment_id}",
            f"--checkpoint-dir={ckpt_dir}",
            f"--wandb-mode={wandb_mode}",
        ]

    elif stage == "sft":
        # Convert parent checkpoint to HF format for TRL
        hf_parent = f"{ckpt_dir}/_hf_parent"
        subprocess.run([
            "python", "-m", "scripts.train.convert_to_hf",
            f"--checkpoint={parent_checkpoint}",
            f"--output={hf_parent}",
            f"--depth={depth}",
            "--direction=nanochat_to_hf",
        ], check=True)

        cmd = [
            "python", "-m", "scripts.train.run_sft",
            f"--model={hf_parent}",
            f"--recipe={sft_recipe}",
            f"--epochs={epochs}",
            f"--lr={lr}",
            f"--max-seq-len={max_seq_len}",
            f"--output-dir={ckpt_dir}",
            f"--run-name={experiment_id}",
            f"--wandb-mode={wandb_mode}",
        ]

    elif stage == "grpo":
        # Convert parent checkpoint to HF format for TRL
        hf_parent = f"{ckpt_dir}/_hf_parent"
        subprocess.run([
            "python", "-m", "scripts.train.convert_to_hf",
            f"--checkpoint={parent_checkpoint}",
            f"--output={hf_parent}",
            f"--depth={depth}",
            "--direction=nanochat_to_hf",
        ], check=True)

        cmd = [
            "python", "-m", "scripts.train.run_grpo",
            f"--model={hf_parent}",
            f"--curriculum={curriculum}",
            f"--kl-coeff={kl_coeff}",
            f"--group-size={group_size}",
            f"--output-dir={ckpt_dir}",
            f"--run-name={experiment_id}",
            f"--wandb-mode={wandb_mode}",
        ]
    else:
        raise ValueError(f"Unknown stage: {stage}")

    if extra_args:
        cmd.extend(extra_args)

    # ── Train ──
    print(f"[{stage}] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # ── Inline eval ──
    eval_output = f"/results/eval_{experiment_id}.json"
    final_ckpt = f"{ckpt_dir}/final.pt"
    eval_cmd = [
        "python", "-m", "scripts.eval.run_eval",
        f"--checkpoint={final_ckpt}",
        f"--datasets={eval_datasets}",
        "--mode=full",
        f"--depth={depth}",
        f"--output={eval_output}",
    ]
    print(f"[{stage}] Running eval: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, check=True)

    # ── Commit volumes ──
    vol_checkpoints.commit()
    vol_results.commit()

    eval_results = {}
    if Path(eval_output).exists():
        eval_results = json.loads(Path(eval_output).read_text())

    return {
        "checkpoint_dir": ckpt_dir,
        "final_checkpoint": final_ckpt,
        "eval_results": eval_results,
    }
