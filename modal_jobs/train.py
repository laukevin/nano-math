"""Modal job: training launcher that calls nanochat directly.

nanochat handles all training (pretrain, SFT, GRPO), model saving,
eval, and W&B logging. We just pass the right flags.

For math SFT, we use our own scripts/math_sft.py instead of nanochat's
chat_sft.py (which NaN's on small models due to data packing issues).
"""

from __future__ import annotations

from modal_jobs.common import (
    HF_SECRET,
    VOLUME_MOUNTS,
    WANDB_SECRET,
    app,
    train_image,
    vol_checkpoints,
    vol_results,
)


@app.function(
    image=train_image,
    gpu="A100",
    timeout=4 * 3600,
    volumes=VOLUME_MOUNTS,
    secrets=[s for s in [WANDB_SECRET, HF_SECRET] if s is not None],
)
def run_pretrain(
    depth: int = 2,
    max_seq_len: int = 512,
    num_iterations: int = -1,
    save_every: int = 100,
    device_batch_size: int = 32,
    run_name: str = "dummy",
) -> dict:
    """Run pretrain via nanochat on Modal."""
    import subprocess
    import time

    cmd = [
        "python", "-m", "scripts.base_train",
        f"--depth={depth}",
        f"--max-seq-len={max_seq_len}",
        "--window-pattern=L",
        "--pos-encoding=nope",
        f"--save-every={save_every}",
        f"--device-batch-size={device_batch_size}",
        f"--core-metric-every=-1",
        f"--run={run_name}",
    ]
    if num_iterations > 0:
        cmd.append(f"--num-iterations={num_iterations}")

    import os
    from modal_jobs.common import vol_data

    base_dir = "/data"
    data_dir = os.path.join(base_dir, "base_data_climbmix")
    tok_dir = os.path.join(base_dir, "tokenizer")

    # Copy tokenizer to data volume if not there yet
    if not os.path.exists(tok_dir):
        import shutil
        shutil.copytree("/root/.cache/nanochat/tokenizer", tok_dir)
        vol_data.commit()

    # Download dataset on first run (persisted in volume)
    if not os.path.exists(data_dir):
        print("[pretrain] Downloading dataset (first time only)...")
        dl_env = {**os.environ, "NANOCHAT_BASE_DIR": base_dir}
        dl_result = subprocess.run(
            ["python", "-m", "nanochat.dataset", "-n", "10"],
            cwd="/root/math-nano/vendor/nanochat",
            capture_output=True, text=True, env=dl_env,
        )
        print(dl_result.stdout[-1000:] if len(dl_result.stdout) > 1000 else dl_result.stdout)
        if dl_result.returncode != 0:
            print("DL STDERR:", dl_result.stderr[-500:])
        vol_data.commit()

    print(f"[pretrain] Running: {' '.join(cmd)}")
    start = time.monotonic()
    env = {**os.environ, "WANDB_MODE": "disabled", "NANOCHAT_BASE_DIR": base_dir}
    result = subprocess.run(cmd, cwd="/root/math-nano/vendor/nanochat", capture_output=True, text=True, env=env)
    elapsed = time.monotonic() - start

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-1000:])

    vol_checkpoints.commit()

    return {
        "stage": "pretrain",
        "depth": depth,
        "exit_code": result.returncode,
        "wall_clock_s": elapsed,
    }


@app.function(
    image=train_image,
    gpu="A100",
    timeout=2 * 3600,
    volumes=VOLUME_MOUNTS,
    secrets=[s for s in [WANDB_SECRET, HF_SECRET] if s is not None],
)
def run_math_sft(
    model_tag: str = "d2",
    num_steps: int = 500,
    batch_size: int = 16,
    lr: float = 1e-4,
    max_seq_len: int = 512,
    save_every: int = 100,
    eval_every: int = 50,
    data: str = "",
) -> dict:
    """Run math SFT using our script on Modal."""
    import os
    import subprocess
    import time

    env = {**os.environ, "WANDB_MODE": "disabled", "NANOCHAT_BASE_DIR": "/data"}
    cmd = [
        "python", "/root/math-nano/scripts/math_sft.py",
        f"--model-tag={model_tag}",
        f"--num-steps={num_steps}",
        f"--batch-size={batch_size}",
        f"--lr={lr}",
        f"--max-seq-len={max_seq_len}",
        f"--save-every={save_every}",
        f"--eval-every={eval_every}",
    ]
    if data:
        cmd.append(f"--data=/root/math-nano/{data}")

    print(f"[math_sft] Running: {' '.join(cmd)}")
    start = time.monotonic()
    result = subprocess.run(cmd, cwd="/root/math-nano/vendor/nanochat", capture_output=True, text=True, env=env)
    elapsed = time.monotonic() - start

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-1000:])

    vol_checkpoints.commit()

    return {
        "stage": "math_sft",
        "model_tag": model_tag,
        "exit_code": result.returncode,
        "wall_clock_s": elapsed,
    }


@app.function(
    image=train_image,
    gpu="A100",
    timeout=1 * 3600,
    volumes=VOLUME_MOUNTS,
    secrets=[s for s in [HF_SECRET] if s is not None],
)
def run_math_eval(
    model_tag: str = "d2",
    phase: str = "base",
    benchmark: str = "gsm8k",
    n_problems: int = 10,
    max_tokens: int = 128,
) -> dict:
    """Run math eval on Modal."""
    import subprocess
    import time

    cmd = [
        "python", "-m", "scripts.eval.run",
        f"--model-tag={model_tag}",
        f"--phase={phase}",
        f"--benchmark={benchmark}",
        f"--n-problems={n_problems}",
        f"--max-tokens={max_tokens}",
        f"--output=/results/eval_{model_tag}_{phase}_{benchmark}.json",
    ]

    import os
    env = {**os.environ, "PYTHONPATH": "/root/math-nano", "NANOCHAT_BASE_DIR": "/data"}

    print(f"[eval] Running: {' '.join(cmd)}")
    start = time.monotonic()
    result = subprocess.run(
        cmd, cwd="/root/math-nano/vendor/nanochat",
        capture_output=True, text=True, env=env,
    )
    elapsed = time.monotonic() - start

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-1000:])

    vol_results.commit()

    return {
        "stage": "eval",
        "model_tag": model_tag,
        "phase": phase,
        "exit_code": result.returncode,
        "wall_clock_s": elapsed,
    }
