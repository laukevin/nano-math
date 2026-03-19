"""Modal training jobs.

Contains both nanochat (from-scratch) and HF (LoRA SFT) pipelines.

Nanochat: pretrain, math SFT, eval, SFT sweep
HF/LoRA: run_sft_lora (Qwen3-0.6B + LoRA SFT + eval + registry)
"""

from __future__ import annotations

from modal_jobs.common import (
    HF_SECRET,
    VOLUME_MOUNTS,
    WANDB_SECRET,
    app,
    train_image,
    vol_checkpoints,
    vol_data,
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


@app.function(
    image=train_image,
    gpu="A100",
    timeout=4 * 3600,
    volumes=VOLUME_MOUNTS,
    secrets=[s for s in [HF_SECRET] if s is not None],
)
def run_sft_sweep(
    model_tag: str = "d8",
    n_problems: int = 50,
    max_tokens: int = 256,
) -> dict:
    """Eval across all SFT checkpoints to find optimal training duration.

    Loads the base model once, then swaps in SFT weights at each checkpoint
    step and runs eval. Also evals the base (pre-SFT) model as step 0.
    """
    import json
    import os
    import sys
    from pathlib import Path

    import torch

    project_root = Path("/root/math-nano")
    nanochat_dir = project_root / "vendor" / "nanochat"
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(nanochat_dir))

    from nanochat.checkpoint_manager import load_model
    from scripts.eval.run import run_eval, make_gsm8k_mini, make_svamp

    base_dir = Path(os.environ.get("NANOCHAT_BASE_DIR", "/data"))
    os.environ["NANOCHAT_BASE_DIR"] = str(base_dir)
    device = torch.device("cuda")

    # Load base model (once)
    print(f"Loading base model (tag={model_tag})...")
    model, tokenizer, _meta = load_model(
        "base", device, phase="eval", model_tag=model_tag, step=None,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Save base weights so we can restore between SFT checkpoints
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Load eval problems (once)
    gsm8k_problems = make_gsm8k_mini(n=n_problems)
    svamp_problems = make_svamp(n=n_problems)

    # Find all SFT checkpoints
    sft_dir = base_dir / "mathsft_checkpoints" / model_tag
    ckpt_files = sorted(sft_dir.glob("model_*.pt"))
    steps = []
    for f in ckpt_files:
        step_str = f.stem.replace("model_", "")
        steps.append(int(step_str))
    steps.sort()
    print(f"\nFound {len(steps)} SFT checkpoints: {steps}")

    all_results = []

    # Eval base model (step 0)
    print("\n" + "=" * 70)
    print("Evaluating BASE model (step 0, no SFT)")
    print("=" * 70)
    model.load_state_dict(base_state)
    gsm8k_summary = run_eval(model, tokenizer, "cuda", gsm8k_problems, max_tokens=max_tokens)
    svamp_summary = run_eval(model, tokenizer, "cuda", svamp_problems, max_tokens=max_tokens)
    all_results.append({
        "step": 0,
        "phase": "base",
        "gsm8k_accuracy": gsm8k_summary["accuracy"],
        "gsm8k_extraction": gsm8k_summary["extraction_rate"],
        "gsm8k_format_boxed": gsm8k_summary["format_boxed_rate"],
        "svamp_accuracy": svamp_summary["accuracy"],
        "svamp_extraction": svamp_summary["extraction_rate"],
        "svamp_format_boxed": svamp_summary["format_boxed_rate"],
    })

    # Eval each SFT checkpoint
    for step in steps:
        print(f"\n{'=' * 70}")
        print(f"Evaluating SFT checkpoint step={step}")
        print("=" * 70)

        sft_path = sft_dir / f"model_{step:06d}.pt"
        state_dict = torch.load(sft_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        gsm8k_summary = run_eval(model, tokenizer, "cuda", gsm8k_problems, max_tokens=max_tokens)
        svamp_summary = run_eval(model, tokenizer, "cuda", svamp_problems, max_tokens=max_tokens)

        # Load SFT meta for training loss
        meta_path = sft_dir / f"meta_{step:06d}.json"
        sft_loss = None
        if meta_path.exists():
            with open(meta_path) as f:
                sft_meta = json.load(f)
            sft_loss = sft_meta.get("loss")

        all_results.append({
            "step": step,
            "phase": "mathsft",
            "sft_loss": sft_loss,
            "gsm8k_accuracy": gsm8k_summary["accuracy"],
            "gsm8k_extraction": gsm8k_summary["extraction_rate"],
            "gsm8k_format_boxed": gsm8k_summary["format_boxed_rate"],
            "svamp_accuracy": svamp_summary["accuracy"],
            "svamp_extraction": svamp_summary["extraction_rate"],
            "svamp_format_boxed": svamp_summary["format_boxed_rate"],
        })

    # Print summary table
    print("\n" + "=" * 70)
    print("SFT SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'Step':>6} | {'Loss':>8} | {'GSM8K':>6} | {'SVAMP':>6} | {'GSM8K ext':>9} | {'SVAMP ext':>9} | {'boxed':>6}")
    print("-" * 70)
    for r in all_results:
        loss_str = f"{r.get('sft_loss', 0):.4f}" if r.get("sft_loss") else "   n/a"
        print(f"{r['step']:>6} | {loss_str:>8} | {r['gsm8k_accuracy']*100:>5.1f}% | {r['svamp_accuracy']*100:>5.1f}% | {r['gsm8k_extraction']*100:>8.1f}% | {r['svamp_extraction']*100:>8.1f}% | {r['gsm8k_format_boxed']*100:>5.1f}%")

    # Save results
    output = {
        "model_tag": model_tag,
        "n_params": n_params,
        "n_problems": n_problems,
        "max_tokens": max_tokens,
        "steps": all_results,
    }
    out_path = f"/results/sft_sweep_{model_tag}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    vol_results.commit()

    return output


# ---------------------------------------------------------------------------
# HF / LoRA SFT pipeline (Qwen3-0.6B + LoRA + eval + registry)
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    gpu="A100",
    timeout=4 * 3600,
    volumes=VOLUME_MOUNTS,
    secrets=[s for s in [WANDB_SECRET, HF_SECRET] if s is not None],
)
def run_sft_lora(
    experiment_id: str = "sft-test",
    data_source: str = "gsm8k",
    data_size: int = -1,
    base_model: str = "Qwen/Qwen3-0.6B-Base",
    lr: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 8,
    max_seq_len: int = 2048,
    lora_rank: int = 16,
    prompt_format: str = "chat_think",
    eval_benchmarks: str = "gsm8k,svamp",
    eval_n_problems: int = 50,
    eval_max_tokens: int = 1024,
) -> dict:
    """Run LoRA SFT on a HuggingFace model, eval, and log to registry.

    Full pipeline:
    1. Prepare data (download from HF if needed, normalize to JSONL)
    2. Run LoRA SFT training
    3. Run eval on benchmarks
    4. Log results to experiment registry
    """
    import json
    import os
    import subprocess
    import time

    os.environ["HF_HOME"] = "/data/hf_cache"
    project = "/root/math-nano"
    start_time = time.time()

    # --- 1. Prepare data ---
    data_dir = f"/data/sft/{data_source}"
    data_path = f"{data_dir}/train.jsonl"

    # Check if we need to (re-)download data
    need_download = not os.path.exists(data_path)
    if not need_download:
        line_count = sum(1 for _ in open(data_path))
        if data_size < 0 and line_count < 1000:
            # Cached file is tiny (likely from a smoke test), re-download full
            print(f"[data] Cached file only has {line_count} samples, re-downloading full dataset")
            need_download = True
        elif data_size > 0 and abs(line_count - data_size) > data_size * 0.1:
            # Cached file size doesn't match requested size
            print(f"[data] Cached={line_count} samples, want={data_size}, re-downloading")
            need_download = True

    if need_download:
        print(f"[data] Preparing {data_source} dataset...")
        os.makedirs(data_dir, exist_ok=True)
        cmd = [
            "python", f"{project}/scripts/data/normalize_dataset.py",
            f"--dataset={data_source}",
            f"--output={data_path}",
        ]
        if data_size > 0:
            cmd.append(f"--max-samples={data_size}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout[-1000:])
        if result.returncode != 0:
            print("STDERR:", result.stderr[-500:])
            return {"error": "data prep failed", "stderr": result.stderr[-500:]}
        vol_data.commit()
    else:
        print(f"[data] Using cached {data_source} at {data_path} ({line_count} samples)")

    # --- 2. Run SFT ---
    output_dir = f"/checkpoints/{experiment_id}"
    print(f"\n[sft] Training {experiment_id}...")
    sft_cmd = [
        "python", f"{project}/scripts/train/sft_lora.py",
        f"--base-model={base_model}",
        f"--data={data_path}",
        f"--output-dir={output_dir}",
        f"--lr={lr}",
        f"--epochs={epochs}",
        f"--batch-size={batch_size}",
        f"--max-seq-len={max_seq_len}",
        f"--lora-rank={lora_rank}",
        f"--prompt-format={prompt_format}",
    ]
    if data_size > 0:
        sft_cmd.append(f"--data-size={data_size}")

    sft_result = subprocess.run(sft_cmd, capture_output=True, text=True)
    print(sft_result.stdout[-2000:])
    if sft_result.returncode != 0:
        print("STDERR:", sft_result.stderr[-1000:])
        return {"error": "sft failed", "stderr": sft_result.stderr[-1000:]}
    vol_checkpoints.commit()

    # Load training metadata
    meta_path = os.path.join(output_dir, "training_meta.json")
    training_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            training_meta = json.load(f)

    # --- 3. Run eval ---
    eval_output = f"/results/eval_{experiment_id}.json"
    print(f"\n[eval] Evaluating {experiment_id}...")
    eval_cmd = [
        "python", f"{project}/scripts/eval/run_hf.py",
        f"--base-model={base_model}",
        f"--adapter={output_dir}",
        f"--benchmarks={eval_benchmarks}",
        f"--n-problems={eval_n_problems}",
        f"--max-tokens={eval_max_tokens}",
        f"--prompt-format={prompt_format}",
        "--eval-batch-size=8",
        f"--output={eval_output}",
    ]
    eval_result = subprocess.run(
        eval_cmd, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": project},
    )
    print(eval_result.stdout[-2000:])
    if eval_result.returncode != 0:
        print("STDERR:", eval_result.stderr[-1000:])

    # Load eval results
    eval_data = {}
    if os.path.exists(eval_output):
        with open(eval_output) as f:
            eval_data = json.load(f)

    elapsed = time.time() - start_time

    # --- 4. Log to registry ---
    # Build eval summary
    eval_summary = {}
    for bench_name, bench_data in eval_data.get("benchmarks", {}).items():
        eval_summary[f"{bench_name}_greedy"] = bench_data.get("accuracy", 0)
        eval_summary[f"{bench_name}_extraction"] = bench_data.get("extraction_rate", 0)

    record = {
        "experiment_id": experiment_id,
        "base_model": base_model,
        "method": "sft-lora",
        "prompt_format": prompt_format,
        "data_source": data_source,
        "data_size": training_meta.get("data_size", data_size),
        "lora_rank": lora_rank,
        "lr": lr,
        "epochs": epochs,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "final_loss": training_meta.get("final_loss"),
        "eval": eval_summary,
        "checkpoint_path": output_dir,
        "wall_clock_min": elapsed / 60,
    }

    # Import and use registry
    import sys
    sys.path.insert(0, project)
    from scripts.registry import append_result
    append_result(record, "/results/experiment_registry.jsonl")

    vol_results.commit()
    vol_checkpoints.commit()

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE: {experiment_id}")
    print(f"  Data: {data_source} ({training_meta.get('data_size', '?')} samples)")
    print(f"  Loss: {training_meta.get('final_loss', '?')}")
    for k, v in eval_summary.items():
        if "greedy" in k:
            print(f"  {k}: {v*100:.1f}%")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"{'='*70}")

    return record
