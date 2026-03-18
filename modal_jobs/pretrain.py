"""Modal job: pretrain a single model.

GPU: H100, Timeout: 8h
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
def run_pretrain(
    depth: int,
    mixture: str,
    token_multiplier: int,
    experiment_id: str,
    wandb_mode: str = "online",
    extra_args: list[str] | None = None,
) -> dict:
    """Run a pretrain job on Modal.

    Returns dict with checkpoint paths and final metrics.
    """
    import json
    import subprocess
    from pathlib import Path

    ckpt_dir = f"/checkpoints/d{depth}/pretrain/{experiment_id}"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "base_train",
        f"--depth={depth}",
        f"--data-source={mixture}",
        f"--token-multiplier={token_multiplier}",
        f"--run-name={experiment_id}",
        f"--checkpoint-dir={ckpt_dir}",
        f"--wandb-mode={wandb_mode}",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"[pretrain] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Run post-training eval
    eval_output = f"/results/eval_{experiment_id}.json"
    eval_cmd = [
        "python", "-m", "scripts.eval.run_eval",
        f"--checkpoint={ckpt_dir}/final.pt",
        "--datasets=gsm8k,math500",
        "--mode=full",
        f"--depth={depth}",
        f"--output={eval_output}",
    ]
    print(f"[pretrain] Running eval: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, check=True)

    # Commit volume changes
    vol_checkpoints.commit()
    vol_results.commit()

    # Read eval results
    eval_results = {}
    if Path(eval_output).exists():
        eval_results = json.loads(Path(eval_output).read_text())

    return {
        "checkpoint_dir": ckpt_dir,
        "final_checkpoint": f"{ckpt_dir}/final.pt",
        "eval_results": eval_results,
    }
