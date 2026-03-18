"""Modal job: GRPO RL training on an SFT checkpoint.

GPU: H100, Timeout: 6h
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
    timeout=6 * 3600,
    volumes=VOLUME_MOUNTS,
    secrets=[WANDB_SECRET],
)
def run_grpo(
    depth: int,
    parent_checkpoint: str,
    experiment_id: str,
    curriculum: str = "easy-to-hard",
    kl_coeff: float = 0.05,
    group_size: int = 8,
    wandb_mode: str = "online",
    extra_args: list[str] | None = None,
) -> dict:
    """Run GRPO RL training on Modal."""
    import json
    import subprocess
    from pathlib import Path

    ckpt_dir = f"/checkpoints/d{depth}/grpo/{experiment_id}"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Convert parent to HF format
    hf_parent = f"/checkpoints/d{depth}/grpo/{experiment_id}_hf_parent"
    convert_cmd = [
        "python", "-m", "scripts.train.convert_to_hf",
        f"--checkpoint={parent_checkpoint}",
        f"--output={hf_parent}",
        f"--depth={depth}",
        "--direction=nanochat_to_hf",
    ]
    print(f"[grpo] Converting checkpoint: {' '.join(convert_cmd)}")
    subprocess.run(convert_cmd, check=True)

    # Run GRPO
    grpo_cmd = [
        "python", "-m", "scripts.train.run_grpo",
        f"--model={hf_parent}",
        f"--curriculum={curriculum}",
        f"--kl-coeff={kl_coeff}",
        f"--group-size={group_size}",
        f"--output-dir={ckpt_dir}",
        f"--run-name={experiment_id}",
        f"--wandb-mode={wandb_mode}",
    ]
    if extra_args:
        grpo_cmd.extend(extra_args)

    print(f"[grpo] Running: {' '.join(grpo_cmd)}")
    subprocess.run(grpo_cmd, check=True)

    # Run eval
    eval_output = f"/results/eval_{experiment_id}.json"
    eval_cmd = [
        "python", "-m", "scripts.eval.run_eval",
        f"--checkpoint={ckpt_dir}/final.pt",
        "--datasets=gsm8k,math500,aime",
        "--mode=full",
        f"--depth={depth}",
        f"--output={eval_output}",
    ]
    print(f"[grpo] Running eval: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, check=True)

    vol_checkpoints.commit()
    vol_results.commit()

    eval_results = {}
    if Path(eval_output).exists():
        eval_results = json.loads(Path(eval_output).read_text())

    return {
        "checkpoint_dir": ckpt_dir,
        "final_checkpoint": f"{ckpt_dir}/final.pt",
        "eval_results": eval_results,
    }
