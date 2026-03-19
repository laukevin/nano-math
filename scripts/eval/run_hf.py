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


def load_hf_model(base_model: str, adapter_path: str | None = None):
    """Load HF model with optional LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel

        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, tokenizer, n_params


FEW_SHOT_PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is \\boxed{6}.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is \\boxed{5}.

Q: """


def make_eval_prompt(problem: str) -> str:
    """Format eval prompt using few-shot format.

    Qwen3-0.6B-Base works best with few-shot prompting (not chat template).
    Chat template with enable_thinking=False produces garbage on the base model.
    """
    return FEW_SHOT_PROMPT + problem + "\nA:"


@torch.no_grad()
def generate_hf(model, tokenizer, prompt: str, max_tokens: int = 256) -> str:
    """Generate completion with HF model (greedy)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def load_benchmark(name: str, n: int = 50) -> list[dict]:
    """Load benchmark problems from HuggingFace."""
    from datasets import load_dataset

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
        return problems

    elif name == "math":
        ds = load_dataset("lighteval/MATH", split="test")
        problems = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            boxed = re.findall(r"\\boxed\{([^}]+)\}", row["solution"])
            answer = boxed[-1] if boxed else ""
            problems.append(
                {
                    "id": f"math_{i:04d}",
                    "problem": row["problem"],
                    "answer": answer,
                    "source": "math",
                }
            )
        return problems

    raise ValueError(f"Unknown benchmark: {name}")


def run_eval(model, tokenizer, problems, max_tokens=256):
    """Run eval on problems. Returns summary dict."""
    n_correct = 0
    n_extracted = 0
    n_boxed = 0
    results = []

    print(f"\nEvaluating {len(problems)} problems (max_tokens={max_tokens})")
    print("-" * 70)

    for i, prob in enumerate(problems):
        prompt = make_eval_prompt(prob["problem"])
        t0 = time.time()
        output = generate_hf(model, tokenizer, prompt, max_tokens=max_tokens)
        elapsed = time.time() - t0

        extracted = extract_answer(output)
        gt = normalize_answer(str(prob["answer"]))
        correct = extracted is not None and extracted == gt

        if extracted is not None:
            n_extracted += 1
        if "\\boxed{" in output:
            n_boxed += 1
        if correct:
            n_correct += 1

        results.append(
            {
                "id": prob["id"],
                "correct": correct,
                "extracted": extracted,
                "ground_truth": gt,
                "time_s": elapsed,
                "output_preview": output[:200],
            }
        )

        status = "CORRECT" if correct else ("EXTRACTED" if extracted else "NO_ANS")
        print(
            f"  [{i+1:3d}/{len(problems)}] {status:9s} "
            f"gt={gt:>8s} pred={str(extracted):>8s} | {elapsed:.1f}s"
        )

    n = len(problems)
    accuracy = n_correct / n if n else 0
    print(f"\nResults: {n_correct}/{n} correct ({accuracy*100:.1f}%)")

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
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model, tokenizer, n_params = load_hf_model(args.base_model, args.adapter)
    print(f"  Params: {n_params:,}")

    all_results = {}
    for bench_name in args.benchmarks.split(","):
        bench_name = bench_name.strip()
        print(f"\n{'='*70}")
        print(f"Benchmark: {bench_name}")
        print(f"{'='*70}")

        problems = load_benchmark(bench_name, n=args.n_problems)
        summary = run_eval(model, tokenizer, problems, args.max_tokens)
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
        print(f"\nResults saved to {args.output}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for bench, res in all_results.items():
        print(f"  {bench}: {res['accuracy']*100:.1f}% ({res['n_problems']} problems)")

    return output


if __name__ == "__main__":
    main()
