"""Eval for HuggingFace models (with optional LoRA adapter).

Runs greedy eval on GSM8K, SVAMP, and/or MATH benchmarks.
Uses the same extraction/scoring as the nanochat eval pipeline.

Usage:
    # Eval base model
    python scripts/eval/run_hf.py --base-model Qwen/Qwen3-0.6B-Base

    # Eval LoRA adapter
    python scripts/eval/run_hf.py --base-model Qwen/Qwen3-0.6B-Base \
        --adapter /checkpoints/sft-001

    # Specific benchmarks
    python scripts/eval/run_hf.py --base-model Qwen/Qwen3-0.6B-Base \
        --benchmarks gsm8k,svamp,math --n-problems 50
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.extraction import extract_answer, normalize_answer


def log_gpu_stats(prefix: str = ""):
    """Log GPU memory usage if CUDA is available."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  {prefix}GPU mem: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total ({allocated/total*100:.0f}% used)", flush=True)


def _detect_memory_gb() -> tuple[float, str]:
    """Return (usable_memory_gb, device_label) for the current device.

    CUDA: full VRAM (dedicated, not shared).
    MPS:  70% of total system RAM — leaves headroom for OS + other processes
          since Apple unified memory is shared.
    CPU:  0.0 (signals caller to skip batching).
    """
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        name = torch.cuda.get_device_properties(0).name
        return mem, f"CUDA {name} ({mem:.0f}GB)"
    if torch.backends.mps.is_available():
        try:
            import subprocess as _sp
            total_ram = int(_sp.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()) / 1e9
        except Exception:
            total_ram = 8.0
        usable = total_ram * 0.70  # leave ~30% for OS, other processes, CPU tensors
        return usable, f"MPS ({usable:.0f}GB usable of {total_ram:.0f}GB RAM)"
    return 0.0, "CPU"


def auto_batch_size(mode: str = "eval", seq_len: int = 1024) -> int:
    """Pick a safe batch size for the current device using the memory estimator.

    mode:    "eval"  — inference only (KV cache, no gradients/optimizer)
             "train" — LoRA SFT (gradients + optimizer states + logits)
    seq_len: generation length for eval; training context length for train.

    Target utilization (fraction of usable memory):
      eval  → 70%  (light; only KV cache scales with batch)
      train → 55%  (heavy; logits + gradient-checkpointing spikes need headroom)

    Hard caps prevent absurdly large values that give no practical speedup:
      eval  → 64   (30-100 problems fit in 1-2 batches at 64; no benefit going higher)
      train → 32   (larger batches don't help throughput much on these small models)
    """
    mem_gb, device_label = _detect_memory_gb()
    if mem_gb == 0.0:
        return 1

    from scripts.gpu_config import recommend_batch_size
    target = 0.70 if mode == "eval" else 0.55
    rec = recommend_batch_size(
        mode=mode,
        seq_len=seq_len,
        target_utilization=target,
        memory_gb=mem_gb,
    )
    bs = rec["recommended_batch_size"]

    cap = 64 if mode == "eval" else 32
    bs = min(bs, cap)

    # Re-estimate memory at the actual (possibly capped) batch size for accurate display
    from scripts.gpu_config import estimate_eval_memory_gb, estimate_training_memory_gb
    actual_est = (estimate_eval_memory_gb if mode == "eval" else estimate_training_memory_gb)(
        batch_size=bs, seq_len=seq_len
    )
    print(
        f"  [auto batch] device={device_label}  mode={mode}  seq_len={seq_len}"
        f"  -> batch={bs}  est={actual_est['total_gb']:.1f}GB"
        f"  ({actual_est['total_gb']/mem_gb*100:.0f}% of {mem_gb:.0f}GB, cap={cap})",
        flush=True,
    )
    return bs


def load_hf_model(base_model: str, adapter_path: str | None = None):
    """Load HF model with optional LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel

        print(f"Loading LoRA adapter: {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log_gpu_stats("After model load: ")
    return model, tokenizer, n_params


FEW_SHOT_PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is \\boxed{6}.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is \\boxed{5}.

Q: """


def make_eval_prompt(problem: str, tokenizer=None, prompt_format: str = "chat_think") -> str:
    """Format eval prompt.

    chat_think: Qwen3 chat template with enable_thinking=True. Model generates
    <think>reasoning</think> then \\boxed{answer}. Best for generalization.

    few_shot: Plain text Q&A with 2 examples. Works on base model without SFT.
    """
    if prompt_format == "chat_think" and tokenizer is not None:
        instruction = (
            "Solve the following math problem step by step. "
            "Put your final answer in \\boxed{}.\n\n" + problem
        )
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    return FEW_SHOT_PROMPT + problem + "\nA:"


def _eos_token_ids(tokenizer) -> list[int]:
    """Return all token IDs that should stop generation.

    Qwen3 chat template ends turns with <|im_end|> (151645), but
    tokenizer.eos_token_id is <|endoftext|> (151643). Without both,
    generation never stops after the assistant turn.
    """
    ids = {tokenizer.eos_token_id}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end != tokenizer.unk_token_id:
        ids.add(im_end)
    return list(ids)


@torch.no_grad()
def generate_hf(model, tokenizer, prompt: str, max_tokens: int = 256) -> str:
    """Generate completion with HF model (greedy), single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        eos_token_id=_eos_token_ids(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def generate_hf_batch(
    model, tokenizer, prompts: list[str], max_tokens: int = 256, batch_size: int = 16
) -> list[str]:
    """Generate completions for multiple prompts in batches."""
    all_outputs = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    t_start = time.time()
    for i in range(0, len(prompts), batch_size):
        batch_num = i // batch_size + 1
        batch_prompts = prompts[i : i + batch_size]
        t_batch = time.time()
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True,
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=_eos_token_ids(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
        batch_elapsed = time.time() - t_batch
        total_elapsed = time.time() - t_start
        eta = (total_elapsed / batch_num) * (n_batches - batch_num)
        print(
            f"  Batch {batch_num}/{n_batches} ({len(batch_prompts)} prompts) "
            f"done in {batch_elapsed:.1f}s  [elapsed: {total_elapsed:.0f}s, ETA: {eta:.0f}s]",
            flush=True,
        )
        for j, output in enumerate(outputs):
            prompt_len = inputs["attention_mask"][j].sum().item()
            generated = output[prompt_len:]
            all_outputs.append(
                tokenizer.decode(generated, skip_special_tokens=True)
            )
    return all_outputs


def load_benchmark(name: str, n: int = 50) -> list[dict]:
    """Load benchmark problems from HuggingFace."""
    from datasets import load_dataset

    print(f"  Loading {name} benchmark...", flush=True)
    t0 = time.time()

    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        problems = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            answer = row["answer"].split("####")[-1].strip()
            problems.append(
                {
                    "id": f"gsm8k_{i:04d}",
                    "problem": row["question"],
                    "answer": answer,
                    "source": "gsm8k",
                }
            )
        print(f"  Loaded {len(problems)} {name} problems in {time.time()-t0:.1f}s", flush=True)
        return problems

    elif name == "svamp":
        ds = load_dataset("ChilleD/SVAMP", split="test")
        problems = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            problems.append(
                {
                    "id": f"svamp_{i:04d}",
                    "problem": row["question_concat"].strip(),
                    "answer": str(int(float(row["Answer"]))),
                    "source": "svamp",
                }
            )
        print(f"  Loaded {len(problems)} {name} problems in {time.time()-t0:.1f}s", flush=True)
        return problems

    elif name == "math":
        # Stratified sample: ~n/7 problems per subject, preferring Level 4-5.
        # Fixes the concatenate_datasets bias where first 100 = all algebra.
        configs = [
            "algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
        ]
        n_per_subject = max(1, n // len(configs))
        problems = []
        for config in configs:
            ds = load_dataset("EleutherAI/hendrycks_math", config, split="test")
            rows = list(ds)
            # Prefer hard problems (Level 4-5) for better model discrimination
            hard = [r for r in rows if r.get("level", "") in ("Level 4", "Level 5")]
            pool = hard if len(hard) >= n_per_subject else rows
            pool = sorted(pool, key=lambda r: r.get("level", ""), reverse=True)
            for row in pool[:n_per_subject]:
                boxed = re.findall(r"\\boxed\{([^}]+)\}", row["solution"])
                answer = boxed[-1] if boxed else ""
                problems.append(
                    {
                        "id": f"math_{config[:4]}_{len(problems):03d}",
                        "problem": row["problem"],
                        "answer": answer,
                        "source": "math",
                        "level": row.get("level", ""),
                    }
                )
        print(
            f"  Loaded {len(problems)} {name} problems "
            f"({n_per_subject}/subject, levels 4-5 preferred) in {time.time()-t0:.1f}s",
            flush=True,
        )
        return problems[:n]

    elif name == "aime_2024":
        ds = load_dataset("MathArena/aime_2024", split="train")
        problems = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            problems.append(
                {
                    "id": f"aime24_{i:04d}",
                    "problem": row["problem"],
                    "answer": str(row["answer"]),
                    "source": "aime_2024",
                }
            )
        print(f"  Loaded {len(problems)} {name} problems in {time.time()-t0:.1f}s", flush=True)
        return problems

    elif name == "aime_2025":
        ds = load_dataset("MathArena/aime_2025", split="train")
        problems = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            problems.append(
                {
                    "id": f"aime25_{i:04d}",
                    "problem": row["problem"],
                    "answer": str(row["answer"]),
                    "source": "aime_2025",
                }
            )
        print(f"  Loaded {len(problems)} {name} problems in {time.time()-t0:.1f}s", flush=True)
        return problems

    elif name == "amc12":
        # AMC 12 2025 only — least likely to appear in any training data.
        # 42 text-only problems (figure-based excluded).
        # Answers are the actual answer value, not the A-E letter.
        try:
            ds = load_dataset("sonthenguyen/amc12-2025-non-figure", split="train")
        except Exception as e:
            raise RuntimeError(f"Could not load amc12-2025: {e}") from e

        problems = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            problems.append({
                "id": f"amc25_{i:04d}",
                "problem": row["question"],
                "answer": str(row["answer"]),
                "source": "amc12_2025",
            })
        print(f"  Loaded {len(problems)} amc12 2025 problems in {time.time()-t0:.1f}s", flush=True)
        return problems

    raise ValueError(f"Unknown benchmark: {name}")


def run_eval(model, tokenizer, problems, max_tokens=256, prompt_format="chat_think",
             eval_batch_size=0):
    """Run eval on problems. Returns summary dict.

    eval_batch_size=0 (default) auto-detects a safe batch size based on device
    memory and the actual prompt lengths in this benchmark. Accounts for both
    prompt tokens and generation budget in the KV cache estimate, so AIME
    (long prompts) gets a smaller batch than SVAMP (short prompts).
    """
    model.eval()  # Ensure eval mode
    if eval_batch_size == 0:
        # Sample prompts to measure actual input length for this benchmark
        sample = problems[:min(8, len(problems))]
        sample_prompts = [make_eval_prompt(p["problem"], tokenizer, prompt_format) for p in sample]
        tok_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in sample_prompts]
        avg_prompt_tokens = int(sum(tok_lens) / len(tok_lens)) if tok_lens else 256
        effective_seq = avg_prompt_tokens + max_tokens  # total KV cache per sequence
        eval_batch_size = auto_batch_size(mode="eval", seq_len=effective_seq)
    n_correct = 0
    n_extracted = 0
    n_boxed = 0
    results = []

    print(f"\nEvaluating {len(problems)} problems (max_tokens={max_tokens}, format={prompt_format}, batch={eval_batch_size})", flush=True)
    print("-" * 70, flush=True)
    log_gpu_stats("Before generation: ")

    prompts = [make_eval_prompt(p["problem"], tokenizer, prompt_format) for p in problems]

    t0 = time.time()
    if eval_batch_size > 1:
        outputs = generate_hf_batch(model, tokenizer, prompts, max_tokens=max_tokens, batch_size=eval_batch_size)
    else:
        outputs = [generate_hf(model, tokenizer, p, max_tokens=max_tokens) for p in prompts]
    total_elapsed = time.time() - t0
    print(f"  Generation done in {total_elapsed:.1f}s ({total_elapsed/len(problems):.1f}s/problem)", flush=True)
    log_gpu_stats("After generation: ")

    for i, (prob, output) in enumerate(zip(problems, outputs)):
        extracted = extract_answer(output)
        gt = normalize_answer(str(prob["answer"]))
        correct = extracted is not None and extracted == gt

        if extracted is not None:
            n_extracted += 1
        if "\\boxed{" in output:
            n_boxed += 1
        if correct:
            n_correct += 1

        is_aime = prob["source"].startswith("aime")
        entry: dict = {
            "id": prob["id"],
            "correct": correct,
            "extracted": extracted,
            "ground_truth": gt,
            "output_preview": output[:500] if is_aime else output[:200],
        }
        if is_aime:
            entry["problem"] = prob["problem"]
        results.append(entry)

        status = "CORRECT" if correct else ("EXTRACTED" if extracted else "NO_ANS")
        print(
            f"  [{i+1:3d}/{len(problems)}] {status:9s} "
            f"gt={gt:>8s} pred={str(extracted):>8s}"
        )
        if is_aime and correct:
            print(f"  *** AIME SOLVED [{prob['id']}] ans={gt} ***", flush=True)

    n = len(problems)
    accuracy = n_correct / n if n else 0
    print(f"\nResults: {n_correct}/{n} correct ({accuracy*100:.1f}%)", flush=True)

    return {
        "n_problems": n,
        "accuracy": accuracy,
        "extraction_rate": n_extracted / n if n else 0,
        "boxed_rate": n_boxed / n if n else 0,
        "per_problem": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Eval HF model on math benchmarks")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--benchmarks", type=str, default="gsm8k,svamp")
    parser.add_argument("--n-problems", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--prompt-format", type=str, default="chat_think",
        choices=["chat_think", "few_shot"],
    )
    parser.add_argument("--eval-batch-size", type=int, default=0,
                        help="Batch size for generation. 0=auto-detect from device/VRAM (default).")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model, tokenizer, n_params = load_hf_model(args.base_model, args.adapter)
    print(f"  Params: {n_params:,}", flush=True)
    print(f"  Prompt format: {args.prompt_format}", flush=True)

    all_results = {}
    total_benchmarks = len(args.benchmarks.split(","))
    for bench_idx, bench_name in enumerate(args.benchmarks.split(","), 1):
        bench_name = bench_name.strip()
        print(f"\n{'='*70}", flush=True)
        print(f"Benchmark [{bench_idx}/{total_benchmarks}]: {bench_name}", flush=True)
        print(f"{'='*70}", flush=True)

        problems = load_benchmark(bench_name, n=args.n_problems)
        summary = run_eval(model, tokenizer, problems, args.max_tokens, args.prompt_format,
                           eval_batch_size=args.eval_batch_size)
        all_results[bench_name] = summary

    # Build output
    output = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "n_params": n_params,
        "benchmarks": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_problem"}
            for k, v in all_results.items()
        },
        "detailed": all_results,
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}", flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    for bench, res in all_results.items():
        print(f"  {bench}: {res['accuracy']*100:.1f}% ({res['n_problems']} problems)", flush=True)

    return output


if __name__ == "__main__":
    main()
