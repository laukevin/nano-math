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
# Eval-only: re-eval an existing checkpoint on all 4 tiers
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    gpu="A100",
    timeout=1 * 3600,
    volumes=VOLUME_MOUNTS,
    secrets=[s for s in [WANDB_SECRET, HF_SECRET] if s is not None],
)
def run_eval_only(
    experiment_id: str = "sft-gsm8k-full-v1",
    base_model: str = "Qwen/Qwen3-0.6B-Base",
    adapter: str = "",
    prompt_format: str = "chat_think",
    eval_benchmarks: str = "svamp,gsm8k,math,aime_2025",
    eval_n_problems: int = 100,
    eval_max_tokens: int = 1024,
    update_registry: bool = True,
) -> dict:
    """Run eval on an existing checkpoint (no training).

    Use adapter="" for base model eval, or adapter="/checkpoints/sft-xxx" for LoRA.
    Results are saved and optionally update the experiment registry.
    """
    import json
    import os
    import subprocess
    import sys
    import time

    os.environ["HF_HOME"] = "/data/hf_cache"
    project = "/root/math-nano"
    start_time = time.time()

    eval_output = f"/results/eval_{experiment_id}.json"
    print(f"[eval] Evaluating {experiment_id} on {eval_benchmarks}...")

    eval_cmd = [
        "python", "-u", f"{project}/scripts/eval/run_hf.py",
        f"--base-model={base_model}",
        f"--benchmarks={eval_benchmarks}",
        f"--n-problems={eval_n_problems}",
        f"--max-tokens={eval_max_tokens}",
        f"--prompt-format={prompt_format}",
        f"--output={eval_output}",
    ]
    if adapter:
        eval_cmd.append(f"--adapter={adapter}")

    eval_result = subprocess.run(
        eval_cmd, text=True,
        stdout=sys.stdout, stderr=sys.stderr,
        env={**os.environ, "PYTHONPATH": project},
    )
    if eval_result.returncode != 0:
        print("STDERR:", eval_result.stderr[-1000:])
        return {"error": "eval failed", "stderr": eval_result.stderr[-1000:]}

    # Load eval results
    eval_data = {}
    if os.path.exists(eval_output):
        with open(eval_output) as f:
            eval_data = json.load(f)

    elapsed = time.time() - start_time

    # Build eval summary
    eval_summary = {}
    for bench_name, bench_data in eval_data.get("benchmarks", {}).items():
        eval_summary[f"{bench_name}_greedy"] = bench_data.get("accuracy", 0)
        eval_summary[f"{bench_name}_extraction"] = bench_data.get("extraction_rate", 0)

    # Optionally update registry
    if update_registry:
        import sys
        sys.path.insert(0, project)
        from scripts.registry import append_result

        record = {
            "experiment_id": experiment_id,
            "base_model": base_model,
            "method": "eval-only" if not adapter else "sft-lora",
            "prompt_format": prompt_format,
            "adapter": adapter or None,
            "eval": eval_summary,
            "eval_wall_clock_min": elapsed / 60,
        }
        append_result(record, "/results/experiment_registry.jsonl")

    vol_results.commit()

    print(f"\n{'='*70}")
    print(f"EVAL COMPLETE: {experiment_id}")
    for k, v in eval_summary.items():
        if "greedy" in k:
            print(f"  {k}: {v*100:.1f}%")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"{'='*70}")

    return {"experiment_id": experiment_id, "eval": eval_summary, "wall_clock_min": elapsed / 60}


# ---------------------------------------------------------------------------
# Batch eval sweep: load base model once, swap adapters in a loop
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    gpu="A100",
    timeout=6 * 3600,
    volumes=VOLUME_MOUNTS,
    secrets=[s for s in [WANDB_SECRET, HF_SECRET] if s is not None],
)
def run_eval_sweep(
    experiments: str = "",
    base_model: str = "Qwen/Qwen3-0.6B-Base",
    prompt_format: str = "chat_think",
    eval_benchmarks: str = "svamp,gsm8k,math,amc12,aime_2025",
    eval_n_problems: int = 100,
    eval_max_tokens: int = 1024,
) -> dict:
    """Eval all adapters sequentially on one GPU — loads base model once.

    experiments: list of experiment IDs (maps to /checkpoints/<id>).
                 Defaults to all sft-* checkpoints found in /checkpoints/.
    """
    import json
    import os
    import sys
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = "/data/hf_cache"
    project = "/root/math-nano"
    sys.path.insert(0, project)

    from scripts.eval.run_hf import _eos_token_ids, load_benchmark, run_eval
    from scripts.registry import append_result

    # Discover experiments if not specified
    experiments_list = [e.strip() for e in experiments.split(",") if e.strip()] if experiments else []
    if not experiments_list:
        experiments_list = sorted([
            d for d in os.listdir("/checkpoints")
            if d.startswith("sft-") and os.path.isdir(f"/checkpoints/{d}")
        ])
    print(f"[sweep] Evaluating {len(experiments_list)} experiments: {experiments_list}", flush=True)

    # Load base model once
    print(f"\n[sweep] Loading base model: {base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    eos_ids = _eos_token_ids(tokenizer)
    print(f"[sweep] Base model loaded. EOS token IDs: {eos_ids}", flush=True)

    # Load benchmarks once
    benchmarks = eval_benchmarks.split(",")
    print(f"[sweep] Loading benchmarks: {benchmarks}", flush=True)
    benchmark_problems = {}
    for b in benchmarks:
        benchmark_problems[b] = load_benchmark(b, n=eval_n_problems)
        print(f"  {b}: {len(benchmark_problems[b])} problems", flush=True)

    all_results = {}
    sweep_start = time.time()

    for experiment_id in experiments_list:
        # Special case: eval base model with no adapter (matches "base-no-sft" or "base-no-sft-*")
        if experiment_id == "base-no-sft" or experiment_id.startswith("base-no-sft-"):
            print(f"\n[sweep] {experiment_id} — evaluating base model (no adapter)", flush=True)
            model = base
            model.eval()
            load_path = "none"
        else:
            adapter_path = f"/checkpoints/{experiment_id}"
            # Find the best checkpoint: prefer final adapter, fall back to latest checkpoint-N
            if os.path.exists(f"{adapter_path}/adapter_model.safetensors"):
                load_path = adapter_path
            else:
                ckpts = sorted([
                    d for d in os.listdir(adapter_path)
                    if d.startswith("checkpoint-")
                ], key=lambda x: int(x.split("-")[1]))
                if not ckpts:
                    print(f"[sweep] {experiment_id}: no adapter found, skipping", flush=True)
                    continue
                load_path = f"{adapter_path}/{ckpts[-1]}"

            print(f"\n[sweep] {experiment_id} — loading adapter from {load_path}", flush=True)
            t0 = time.time()

            from peft import PeftModel
            model = PeftModel.from_pretrained(base, load_path)
            model.eval()
            print(f"  Adapter loaded in {time.time()-t0:.1f}s", flush=True)

        exp_results = {"base_model": base_model, "adapter": load_path, "benchmarks": {}}
        for bench_name, problems in benchmark_problems.items():
            summary = run_eval(
                model, tokenizer, problems,
                max_tokens=eval_max_tokens,
                prompt_format=prompt_format,
            )
            # keep per_problem for aime (30 problems, critical for analysis); strip for larger benchmarks
            if bench_name.startswith("aime"):
                exp_results["benchmarks"][bench_name] = summary
            else:
                exp_results["benchmarks"][bench_name] = {
                    k: v for k, v in summary.items() if k != "per_problem"
                }

        # Save result to volume
        out_path = f"/results/eval_{experiment_id}_v2.json"
        with open(out_path, "w") as f:
            json.dump(exp_results, f, indent=2)

        # Update registry
        eval_summary = {}
        for b, metrics in exp_results["benchmarks"].items():
            eval_summary[f"{b}_greedy"] = metrics.get("accuracy", 0)
            eval_summary[f"{b}_extraction"] = metrics.get("extraction_rate", 0)
        append_result({
            "experiment_id": experiment_id,
            "base_model": base_model,
            "adapter": load_path,
            "prompt_format": prompt_format,
            "eval": eval_summary,
            "eos_fix": True,
        }, "/results/experiment_registry.jsonl")

        all_results[experiment_id] = eval_summary

        # Unload adapter weights to free VRAM before next adapter
        if experiment_id != "base-no-sft":
            model = model.unload()
            del model

        elapsed = (time.time() - sweep_start) / 60
        print(f"  Done. Cumulative time: {elapsed:.1f} min", flush=True)

    vol_results.commit()

    print(f"\n[sweep] Complete. {len(all_results)} experiments evaluated.", flush=True)
    print(f"\n{'Experiment':<35} {'GSM8K':>7} {'MATH':>7} {'SVAMP':>7} {'AMC12':>7} {'AIME':>7}")
    print("-" * 71)
    for eid, r in all_results.items():
        print(
            f"{eid:<35} "
            f"{r.get('gsm8k_greedy', 0)*100:>6.1f}% "
            f"{r.get('math_greedy', 0)*100:>6.1f}% "
            f"{r.get('svamp_greedy', 0)*100:>6.1f}% "
            f"{r.get('amc12_greedy', 0)*100:>6.1f}% "
            f"{r.get('aime_2025_greedy', 0)*100:>6.1f}%"
        )
    return all_results


# ---------------------------------------------------------------------------
# HF / LoRA SFT pipeline (Qwen3-0.6B + LoRA + eval + registry)
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    gpu="A100",
    timeout=6 * 3600,
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
    batch_size: int = 0,
    gradient_accumulation_steps: int = 1,
    max_seq_len: int = 2048,
    lora_rank: int = 16,
    prompt_format: str = "chat_think",
    packing: bool = False,
    max_tokens_per_batch: int = -1,
    eval_benchmarks: str = "amc12",
    eval_n_problems: int = 30,
    eval_max_tokens: int = 1024,
    init_adapter: str = "",
    min_chars: int = 0,
    max_chars: int = 0,
    save_every: int = 500,
    eval_every: int = -1,
) -> dict:
    """Run LoRA SFT on a HuggingFace model, eval, and log to registry.

    Post-training eval defaults to amc12/30 (fast smoketest ~2 min).
    For full benchmark results, run run_eval_sweep on promising checkpoints.

    Full pipeline:
    1. Estimate memory, print GPU plan
    2. Prepare data (download from HF if needed, normalize to JSONL)
    3. Run LoRA SFT training (with OOM recovery)
    4. Run eval on benchmarks
    5. Log results to experiment registry
    """
    import json
    import os
    import subprocess
    import sys
    import time

    os.environ["HF_HOME"] = "/data/hf_cache"
    project = "/root/math-nano"
    sys.path.insert(0, project)
    start_time = time.time()

    # Print experiment header immediately — visible with `modal app logs <id> | head -20`
    print("=" * 70, flush=True)
    print(f"EXPERIMENT: {experiment_id}", flush=True)
    print(f"  data:       {data_source}  size={data_size}  chars=[{min_chars},{max_chars}]", flush=True)
    print(f"  model:      {base_model}  lora_rank={lora_rank}", flush=True)
    print(f"  training:   seq={max_seq_len}  batch={batch_size}  epochs={epochs}  lr={lr}", flush=True)
    print(f"  eval:       {eval_benchmarks}  n={eval_n_problems}  max_tok={eval_max_tokens}", flush=True)
    print("=" * 70, flush=True)

    from scripts.gpu_config import estimate_training_memory_gb, estimate_eval_memory_gb, recommend_batch_size, GPU_SPECS

    # --- 1. Prepare data ---
    # Use a separate cache dir when char filters are applied so filtered and
    # unfiltered versions don't collide.
    if min_chars > 0 or max_chars > 0:
        char_suffix = f"-chars{min_chars}-{max_chars}"
    else:
        char_suffix = ""
    data_dir = f"/data/sft/{data_source}{char_suffix}"
    data_path = f"{data_dir}/train.jsonl"

    # Check if we need to (re-)download data
    meta_path = data_path + ".meta"
    need_download = not os.path.exists(data_path)
    if not need_download:
        line_count = sum(1 for _ in open(data_path))
        has_meta = os.path.exists(meta_path)
        cached_size = -1
        if has_meta:
            import json as _json
            cached_size = _json.load(open(meta_path)).get("data_size", -1)
        if data_size < 0 and line_count < 1000:
            # Cached file is tiny (likely from a smoke test), re-download full
            print(f"[data] Cached file only has {line_count} samples, re-downloading full dataset")
            need_download = True
        elif data_size < 0 and not has_meta:
            # Cache predates meta system — may have been bounded, re-download to be safe
            print(f"[data] Cached {line_count} samples has no .meta (pre-dates meta system), "
                  f"re-downloading full dataset (data_size=-1)")
            need_download = True
        elif data_size < 0 and cached_size > 0:
            # We want the full dataset but the cache was created with a size cap — re-download
            print(f"[data] Cached {line_count} samples was created with data_size={cached_size}, "
                  f"re-downloading full dataset (data_size=-1)")
            need_download = True
        elif data_size > 0 and abs(line_count - data_size) > data_size * 0.1:
            # Cached file size doesn't match requested size
            print(f"[data] Cached={line_count} samples, want={data_size}, re-downloading")
            need_download = True

    if need_download:
        print(f"[data] Preparing {data_source} dataset...")
        os.makedirs(data_dir, exist_ok=True)
        cmd = [
            "python", "-u", f"{project}/scripts/data/normalize_dataset.py",
            f"--dataset={data_source}",
            f"--output={data_path}",
        ]
        if data_size > 0:
            cmd.append(f"--max-samples={data_size}")
        if min_chars > 0:
            cmd.append(f"--min-chars={min_chars}")
        if max_chars > 0:
            cmd.append(f"--max-chars={max_chars}")
        result = subprocess.run(cmd, text=True, stdout=sys.stdout, stderr=sys.stderr)
        if result.returncode != 0:
            # HuggingFace streaming datasets trigger a PyGILState crash at exit from
            # background worker threads — benign, happens after data is fully written.
            # Treat as success if the output file exists with content.
            if os.path.exists(data_path) and os.path.getsize(data_path) > 0:
                print(f"[data] normalize_dataset exited with code {result.returncode} "
                      f"but data file exists ({os.path.getsize(data_path)//1024}KB) — continuing",
                      flush=True)
            else:
                return {"error": "data prep failed"}
        # Write meta so future runs can detect bounded-vs-unlimited cache mismatches
        import json as _json
        with open(meta_path, "w") as _mf:
            _json.dump({"data_size": data_size, "min_chars": min_chars, "max_chars": max_chars}, _mf)
        vol_data.commit()
    else:
        line_count = sum(1 for _ in open(data_path))
        print(f"[data] Using cached {data_source} at {data_path} ({line_count} samples)")

    # --- 2. GPU memory plan (uses real dataset seq lengths) ---
    # Quick char-count scan to estimate avg tokenized length (chars / 3.5 + template overhead)
    import statistics as _stats
    _sol_lens = []
    _prob_lens = []
    with open(data_path) as _f:
        for _i, _line in enumerate(_f):
            if data_size > 0 and _i >= data_size:
                break
            _row = json.loads(_line)
            _sol_lens.append(len(_row.get("solution", "")))
            _prob_lens.append(len(_row.get("problem", "")))
    avg_sol_chars = _stats.mean(_sol_lens) if _sol_lens else 1000
    avg_prob_chars = _stats.mean(_prob_lens) if _prob_lens else 200
    avg_seq_len = int((avg_sol_chars + avg_prob_chars) / 3.5) + 60  # 60 tok chat template overhead
    p90_seq_len = int(sorted(_sol_lens + _prob_lens)[int(len(_sol_lens) * 0.9)] / 3.5) + 60

    train_est = estimate_training_memory_gb(
        batch_size=batch_size, seq_len=max_seq_len, lora_rank=lora_rank,
        packing=packing, avg_seq_len=avg_seq_len,
    )
    eval_est = estimate_eval_memory_gb(batch_size=64, seq_len=eval_max_tokens, lora_rank=lora_rank)
    gpu_mem = GPU_SPECS.get("A100-40GB", 40.0)
    pack_label = "packing" if packing else "no packing"
    print(f"\n[gpu] Memory plan for {experiment_id}:", flush=True)
    print(f"  Dataset: {len(_sol_lens)} samples, avg_seq~{avg_seq_len} tok, p90~{p90_seq_len} tok", flush=True)
    print(f"  Training: batch={batch_size}, seq={max_seq_len}, {pack_label} "
          f"(eff_seq={train_est['effective_seq_len']}) -> {train_est['total_gb']:.1f}GB "
          f"({train_est['total_gb']/gpu_mem*100:.0f}% of {gpu_mem:.0f}GB)", flush=True)
    print(f"    Breakdown: logits={train_est['logits_gb']:.1f}GB, "
          f"activations={train_est['activations_gb']:.1f}GB, "
          f"fixed={train_est['fixed_gb']:.1f}GB", flush=True)
    print(f"  Eval: batch=64, seq={eval_max_tokens} -> {eval_est['total_gb']:.1f}GB "
          f"({eval_est['total_gb']/gpu_mem*100:.0f}% of {gpu_mem:.0f}GB)", flush=True)
    print(f"  Headroom: {gpu_mem - train_est['total_gb']:.1f}GB training, "
          f"{gpu_mem - eval_est['total_gb']:.1f}GB eval", flush=True)

    # Guard: if min_chars filter is set, check that most data actually fits at max_seq_len.
    # Rule of thumb: max_chars_that_fit ≈ max_seq_len × 3.5 chars/token.
    # If min_chars > max_chars_that_fit, virtually every sample will be dropped.
    if min_chars > 0:
        max_chars_that_fit = int(max_seq_len * 3.5)
        if min_chars > max_chars_that_fit * 0.9:
            print(
                f"\n  WARNING: min_chars={min_chars} > seq_len capacity "
                f"({max_chars_that_fit} chars at seq={max_seq_len}). "
                f"Nearly all samples will be dropped at training time! "
                f"Use --max-seq-len {int(min_chars / 3.5 / 1024 + 1) * 1024} or higher.",
                flush=True,
            )

    if train_est['total_gb'] > gpu_mem * 0.95:
        rec = recommend_batch_size(
            mode="train", seq_len=max_seq_len, lora_rank=lora_rank,
            packing=packing, avg_seq_len=avg_seq_len,
        )
        print(f"  WARNING: estimate ({train_est['total_gb']:.1f}GB) exceeds 95% of GPU! "
              f"Recommended: batch={rec['recommended_batch_size']}", flush=True)
        if max_tokens_per_batch <= 0:
            # seq≥8192: cap at 1× (batch=1) — logits alone at 3× hit ~7.5GB,
            # plus gradient-checkpointing spike pushes >40GB on A100-40GB.
            # seq=4096: 2× is safe (~3.8GB logits), better throughput than 1×.
            if max_seq_len >= 8192:
                max_tokens_per_batch = max_seq_len  # batch=1
            else:
                max_tokens_per_batch = max_seq_len * 2  # batch~2 at seq4096
            print(f"  AUTO-FIX: switching to token-budget batching "
                  f"(max_tokens_per_batch={max_tokens_per_batch})", flush=True)

    # --- 3. Run SFT ---
    output_dir = f"/checkpoints/{experiment_id}"
    print(f"\n[sft] Training {experiment_id}...", flush=True)
    sft_cmd = [
        "python", "-u", f"{project}/scripts/train/sft_lora.py",
        f"--base-model={base_model}",
        f"--data={data_path}",
        f"--output-dir={output_dir}",
        f"--lr={lr}",
        f"--epochs={epochs}",
        f"--batch-size={batch_size}",
        f"--gradient-accumulation-steps={gradient_accumulation_steps}",
        f"--max-seq-len={max_seq_len}",
        f"--lora-rank={lora_rank}",
        f"--prompt-format={prompt_format}",
    ]
    if data_size > 0:
        sft_cmd.append(f"--data-size={data_size}")
    if packing:
        sft_cmd.append("--packing")
    if max_tokens_per_batch > 0:
        sft_cmd.append(f"--max-tokens-per-batch={max_tokens_per_batch}")
    if init_adapter:
        sft_cmd.append(f"--init-adapter={init_adapter}")
    if save_every > 0:
        sft_cmd.append(f"--save-every={save_every}")
    sft_cmd.append(f"--eval-every={eval_every}")  # always pass: -1=auto, 0=disable, N=every N steps
    sft_cmd.append("--registry-path=/results/experiment_registry.jsonl")

    # Run training with periodic checkpoint + results commits (every 10 min) so
    # checkpoints and mid-run registry entries survive a container crash mid-run.
    COMMIT_INTERVAL_S = 600  # 10 minutes
    sft_proc = subprocess.Popen(sft_cmd, stdout=sys.stdout, stderr=sys.stderr)
    last_commit = time.time()
    while sft_proc.poll() is None:
        time.sleep(30)
        if time.time() - last_commit >= COMMIT_INTERVAL_S:
            try:
                vol_checkpoints.commit()
                print("[checkpoint] Committed checkpoints to volume", flush=True)
            except Exception as e:
                print(f"[checkpoint] commit failed (non-fatal): {e}", flush=True)
            try:
                vol_results.commit()
                print("[results] Committed results to volume", flush=True)
            except Exception as e:
                print(f"[results] commit failed (non-fatal): {e}", flush=True)
            last_commit = time.time()

    sft_result = sft_proc

    # Final commit after training completes
    try:
        vol_checkpoints.commit()
    except Exception:
        pass

    if sft_result.returncode != 0:
        print(f"\n[error] SFT failed with exit code {sft_result.returncode}", flush=True)
        # Check if we got partial checkpoints (useful for epoch-2 recovery)
        partial_ckpts = []
        if os.path.exists(output_dir):
            for d in sorted(os.listdir(output_dir)):
                if d.startswith("checkpoint-"):
                    partial_ckpts.append(d)
        if partial_ckpts:
            print(f"  Partial checkpoints saved: {partial_ckpts}", flush=True)
            print(f"  Can eval from: /checkpoints/{experiment_id}/{partial_ckpts[-1]}", flush=True)
        return {
            "error": "sft failed",
            "exit_code": sft_result.returncode,
            "partial_checkpoints": partial_ckpts,
            "experiment_id": experiment_id,
        }

    # Check that adapter was actually saved (training can "succeed" with 0 samples)
    adapter_config = os.path.join(output_dir, "adapter_config.json")
    if not os.path.exists(adapter_config):
        return {"error": "sft produced no checkpoint (likely 0 training samples)"}

    # Load training metadata
    meta_path = os.path.join(output_dir, "training_meta.json")
    training_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            training_meta = json.load(f)

    # --- 4. Run eval ---
    eval_output = f"/results/eval_{experiment_id}.json"
    print(f"\n[eval] Evaluating {experiment_id} (max_tokens={eval_max_tokens})...", flush=True)
    eval_cmd = [
        "python", "-u", f"{project}/scripts/eval/run_hf.py",
        f"--base-model={base_model}",
        f"--adapter={output_dir}",
        f"--benchmarks={eval_benchmarks}",
        f"--n-problems={eval_n_problems}",
        f"--max-tokens={eval_max_tokens}",
        f"--prompt-format={prompt_format}",
        f"--output={eval_output}",
    ]
    eval_result = subprocess.run(
        eval_cmd, text=True, stdout=sys.stdout, stderr=sys.stderr,
        env={**os.environ, "PYTHONPATH": project},
    )
    if eval_result.returncode != 0:
        print(f"\n[error] Eval failed with exit code {eval_result.returncode}", flush=True)
        print("  Training succeeded but eval crashed. Saving partial results...", flush=True)

    # Load eval results (may be empty if eval crashed)
    eval_data = {}
    if os.path.exists(eval_output):
        with open(eval_output) as f:
            eval_data = json.load(f)

    elapsed = time.time() - start_time

    # --- 5. Log to registry ---
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


# ---------------------------------------------------------------------------
# Batch size smoketest: find max safe batch without OOM
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    gpu="A100",
    timeout=30 * 60,
    volumes=VOLUME_MOUNTS,
    secrets=[s for s in [HF_SECRET] if s is not None],
)
def run_batch_smoketest(
    base_model: str = "Qwen/Qwen3-0.6B-Base",
    data_source: str = "acemath",
    batch_sizes: str = "8,16,32,48,64",
    max_seq_len: int = 2048,
    lora_rank: int = 16,
    n_steps: int = 3,
) -> dict:
    """Run a few training steps at each batch size and report actual GPU memory.

    Use this to find the largest safe batch size before committing to a full run.
    Tries each batch size in ascending order, stops at first OOM.

    Example:
        uv run modal run modal_jobs/train.py::run_batch_smoketest \\
            --data-source acemath --batch-sizes '[8, 16, 32, 48, 64]'
    """
    import os
    import sys
    import time

    import torch

    os.environ["HF_HOME"] = "/data/hf_cache"
    project = "/root/math-nano"
    sys.path.insert(0, project)

    batch_sizes_list = [int(x.strip()) for x in batch_sizes.split(",")]

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scripts.gpu_config import estimate_training_memory_gb

    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gpu_name = torch.cuda.get_device_properties(0).name
    print(f"GPU: {gpu_name}, {gpu_mem_total:.1f} GB total", flush=True)

    # Load real samples from data volume for realistic seq lengths
    data_path = f"/data/sft/{data_source}/train.jsonl"
    samples = []
    if os.path.exists(data_path):
        import json as _json
        with open(data_path) as f:
            for line in f:
                samples.append(_json.loads(line))
                if len(samples) >= max(batch_sizes_list) * 2:
                    break
        print(f"Loaded {len(samples)} real samples from {data_source}", flush=True)
    else:
        print(f"No cached data for {data_source}, using synthetic sequences", flush=True)

    # Estimate avg seq len from data
    avg_seq_len = 512  # fallback
    if samples:
        import statistics as _stats
        char_lens = [len(s.get("solution", "")) + len(s.get("problem", "")) for s in samples]
        avg_seq_len = int(_stats.mean(char_lens) / 3.5) + 60
        print(f"Data avg seq len estimate: ~{avg_seq_len} tokens", flush=True)

    print(f"\nLoading model: {base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-5)

    def make_batch(batch_size: int):
        """Build a real or synthetic batch."""
        if samples:
            # Tokenize a few real samples and pad to batch max
            sys.path.insert(0, project)
            from scripts.train.sft_lora import tokenize_chat_think
            tokenized = []
            for s in samples[:batch_size * 3]:
                tok = tokenize_chat_think(s, tokenizer, max_seq_len)
                if tok:
                    tokenized.append(tok)
                if len(tokenized) >= batch_size:
                    break
            if len(tokenized) >= batch_size:
                batch = tokenized[:batch_size]
                pad_id = tokenizer.pad_token_id
                max_len = max(len(t["input_ids"]) for t in batch)
                max_len = min(max_len, max_seq_len)
                input_ids = torch.tensor(
                    [t["input_ids"][:max_len] + [pad_id] * (max_len - len(t["input_ids"][:max_len])) for t in batch],
                    device="cuda",
                )
                labels = torch.tensor(
                    [t["labels"][:max_len] + [-100] * (max_len - len(t["labels"][:max_len])) for t in batch],
                    device="cuda",
                )
                return input_ids, labels, max_len
        # Fallback: synthetic batch at avg_seq_len
        syn_len = min(avg_seq_len, max_seq_len)
        input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, syn_len), device="cuda")
        labels = input_ids.clone()
        return input_ids, labels, syn_len

    results = {}
    print(f"\n{'Batch':>6}  {'Est GB':>7}  {'Real GB':>8}  {'Util%':>6}  {'s/step':>7}  {'Status':>8}")
    print("-" * 55)

    for bs in sorted(batch_sizes_list):
        est = estimate_training_memory_gb(
            batch_size=bs, seq_len=max_seq_len, lora_rank=lora_rank, avg_seq_len=avg_seq_len,
        )
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            input_ids, labels, actual_seq_len = make_batch(bs)

            step_times = []
            for _ in range(n_steps):
                t0 = time.time()
                out = model(input_ids=input_ids, labels=labels)
                out.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                step_times.append(time.time() - t0)

            peak_gb = torch.cuda.max_memory_reserved() / 1024**3
            avg_step = sum(step_times) / len(step_times)
            util_pct = peak_gb / gpu_mem_total * 100
            print(
                f"{bs:>6}  {est['total_gb']:>6.1f}GB  {peak_gb:>7.1f}GB  {util_pct:>5.0f}%  "
                f"{avg_step:>6.2f}s  {'OK':>8}  (seq={actual_seq_len})",
                flush=True,
            )
            results[bs] = {
                "status": "ok",
                "estimated_gb": est["total_gb"],
                "peak_gb": peak_gb,
                "util_pct": util_pct,
                "avg_step_s": avg_step,
                "actual_seq_len": actual_seq_len,
            }
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>6}  {est['total_gb']:>6.1f}GB  {'OOM':>8}  {'---':>6}  {'---':>7}  {'OOM':>8}", flush=True)
            results[bs] = {"status": "oom", "estimated_gb": est["total_gb"]}
            torch.cuda.empty_cache()
            break  # no point trying larger batches

    ok_batches = [bs for bs, r in results.items() if r["status"] == "ok"]
    recommended = max(ok_batches) if ok_batches else None
    print(f"\nRecommended batch size: {recommended}", flush=True)
    if recommended:
        r = results[recommended]
        print(f"  Peak memory: {r['peak_gb']:.1f}GB / {gpu_mem_total:.1f}GB ({r['util_pct']:.0f}%)", flush=True)
        print(f"  Step time: {r['avg_step_s']:.2f}s at batch={recommended}", flush=True)

    return {"gpu": gpu_name, "gpu_mem_gb": gpu_mem_total, "avg_seq_len": avg_seq_len, "results": results, "recommended_batch": recommended}


@app.local_entrypoint()
def run_parallel_sft(config_file: str = "", dry_run: bool = False) -> None:
    """Launch multiple SFT experiments in parallel on Modal.

    Each experiment runs in its own container (one A100 each), all starting
    simultaneously. Much faster than running sequentially when searching over
    data sources, learning rates, or hyperparameters.

    config_file: path to a JSON file containing a list of experiment dicts.
    Each dict is passed as kwargs to run_sft_lora.

    Example experiments.json:
        [
          {"experiment_id": "sft-gsm8k-lr1e5", "data_source": "gsm8k", "lr": 1e-5},
          {"experiment_id": "sft-acemath-lr2e5", "data_source": "acemath", "lr": 2e-5},
          {"experiment_id": "sft-acemath-seq4k", "data_source": "acemath", "max_seq_len": 4096}
        ]

    Usage:
        # Launch and wait for all to complete:
        uv run modal run modal_jobs/train.py::run_parallel_sft --config-file experiments.json

        # Dry run to verify config without launching:
        uv run modal run modal_jobs/train.py::run_parallel_sft --config-file experiments.json --dry-run
    """
    import json

    if not config_file:
        print("ERROR: --config-file is required")
        return

    with open(config_file) as f:
        experiments = json.load(f)

    if not isinstance(experiments, list) or not experiments:
        print(f"ERROR: config_file must contain a non-empty JSON list, got: {type(experiments)}")
        return

    if dry_run:
        print(f"DRY RUN — would launch {len(experiments)} experiments in parallel:")
        for exp in experiments:
            eid = exp.get("experiment_id", "???")
            rest = {k: v for k, v in exp.items() if k != "experiment_id"}
            print(f"  {eid}: {rest}")
        return

    print(f"Launching {len(experiments)} SFT experiments in parallel...")
    handles = []
    for exp in experiments:
        eid = exp.get("experiment_id", "?")
        h = run_sft_lora.spawn(**exp)
        handles.append((eid, h))
        print(f"  Spawned: {eid}  (call_id={h.object_id})", flush=True)

    print(f"\nAll {len(handles)} jobs launched. Collecting results (Ctrl+C to stop watching)...\n")
    results = {}
    for eid, h in handles:
        try:
            result = h.get()
            ev = result.get("eval", {})
            results[eid] = result
            print(
                f"  DONE: {eid:<40} "
                f"GSM8K={ev.get('gsm8k_greedy', 0)*100:.1f}%  "
                f"MATH={ev.get('math_greedy', 0)*100:.1f}%  "
                f"SVAMP={ev.get('svamp_greedy', 0)*100:.1f}%",
                flush=True,
            )
        except Exception as e:
            results[eid] = {"error": str(e)}
            print(f"  FAILED: {eid} -> {e}", flush=True)

    print(f"\n{'='*75}")
    print(f"PARALLEL SFT COMPLETE — {len(results)} experiments")
    print(f"{'Experiment':<42} {'GSM8K':>7} {'MATH':>7} {'SVAMP':>7} {'AMC12':>7}")
    print("-" * 70)
    for eid, r in results.items():
        ev = r.get("eval", {})
        err = r.get("error")
        if err:
            print(f"  {eid:<40} ERROR: {err}")
        else:
            print(
                f"  {eid:<40} "
                f"{ev.get('gsm8k_greedy', 0)*100:>6.1f}%  "
                f"{ev.get('math_greedy', 0)*100:>6.1f}%  "
                f"{ev.get('svamp_greedy', 0)*100:>6.1f}%  "
                f"{ev.get('amc12_greedy', 0)*100:>6.1f}%"
            )


@app.function(
    image=train_image,
    gpu=None,
    timeout=1 * 3600,
    volumes=VOLUME_MOUNTS,
    secrets=[s for s in [WANDB_SECRET, HF_SECRET] if s is not None],
)
def run_script(script: str = "scripts/data/sample_datasets.py") -> None:
    """Run an arbitrary script from the project on Modal (CPU only, HF access).

    Useful for data exploration, dataset sampling, etc.

    Example:
        uv run modal run modal_jobs/train.py::run_script --script scripts/data/sample_datasets.py
    """
    import os
    import subprocess
    import sys

    os.environ["HF_HOME"] = "/data/hf_cache"
    project = "/root/math-nano"
    result = subprocess.run(
        ["python", "-u", f"{project}/{script}"],
        stdout=sys.stdout, stderr=sys.stderr,
        env={**os.environ, "PYTHONPATH": project},
    )
    if result.returncode != 0:
        # PyGIL crash (exit -6) from HF streaming datasets is benign — happens at
        # subprocess cleanup after all output has been produced. Warn, don't raise.
        print(f"[run_script] Script exited with code {result.returncode} (likely benign PyGIL crash)", flush=True)
