"""Modal job: SFT on a pretrained checkpoint.

GPU: H100, Timeout: 4h
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
    timeout=4 * 3600,
    volumes=VOLUME_MOUNTS,
    secrets=[WANDB_SECRET],
)
def run_sft(
    depth: int,
    parent_checkpoint: str,
    sft_recipe: str,
    experiment_id: str,
    epochs: int = 3,
    lr: float = 2e-5,
    max_seq_len: int = 2048,
    wandb_mode: str = "online",
    extra_args: list[str] | None = None,
) -> dict:
    """Run SFT on Modal.

    Converts nanochat checkpoint to HF format, runs TRL SFT, converts back.
    """
    import json
    import subprocess
    from pathlib import Path

    ckpt_dir = f"/checkpoints/d{depth}/sft/{experiment_id}"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Convert parent checkpoint to HF format for TRL
    hf_parent = f"/checkpoints/d{depth}/sft/{experiment_id}_hf_parent"
    convert_cmd = [
        "python", "-m", "scripts.train.convert_to_hf",
        f"--checkpoint={parent_checkpoint}",
        f"--output={hf_parent}",
        f"--depth={depth}",
        "--direction=nanochat_to_hf",
    ]
    print(f"[sft] Converting checkpoint: {' '.join(convert_cmd)}")
    subprocess.run(convert_cmd, check=True)

    # Run SFT via TRL
    sft_cmd = [
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
    if extra_args:
        sft_cmd.extend(extra_args)

    print(f"[sft] Running: {' '.join(sft_cmd)}")
    subprocess.run(sft_cmd, check=True)

    # Convert best checkpoint back to nanochat format
    best_hf = f"{ckpt_dir}/best_gsm8k"
    best_nanochat = f"{ckpt_dir}/best_gsm8k.pt"
    if Path(best_hf).exists():
        convert_back = [
            "python", "-m", "scripts.train.convert_to_hf",
            f"--checkpoint={best_hf}",
            f"--output={best_nanochat}",
            f"--depth={depth}",
            "--direction=hf_to_nanochat",
        ]
        subprocess.run(convert_back, check=True)

    # Run eval
    eval_output = f"/results/eval_{experiment_id}.json"
    eval_cmd = [
        "python", "-m", "scripts.eval.run_eval",
        f"--checkpoint={ckpt_dir}/final.pt",
        "--datasets=gsm8k,math500",
        "--mode=full",
        f"--depth={depth}",
        f"--output={eval_output}",
    ]
    print(f"[sft] Running eval: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, check=True)

    vol_checkpoints.commit()
    vol_results.commit()

    eval_results = {}
    if Path(eval_output).exists():
        eval_results = json.loads(Path(eval_output).read_text())

    return {
        "checkpoint_dir": ckpt_dir,
        "final_checkpoint": f"{ckpt_dir}/final.pt",
        "best_checkpoint": best_nanochat if Path(best_nanochat).exists() else f"{ckpt_dir}/final.pt",
        "eval_results": eval_results,
    }
