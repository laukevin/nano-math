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
        from datasets import concatenate_datasets
        configs = [
            "algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
        ]
        parts = [load_dataset("EleutherAI/hendrycks_math", c, split="test") for c in configs]
        ds = concatenate_datasets(parts)
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
        print(f"  Loaded {len(problems)} {name} problems in {time.time()-t0:.1f}s", flush=True)
        return problems

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
        # Combine 2023 + 2024 + 2025 AMC 12 problems.
        # These are post-2022 and unlikely to appear in training data.
        # Answers are stored as the actual answer value (not A-E letter).
        problems = []

        loaders = [
            # (dataset_id, split, problem_field, answer_field, id_prefix)
            ("math-ai/amc23",                       "test",  "question", "answer", "amc23"),
            ("rawsh/2024_AMC12",                    "train", "problem",  "answer", "amc24"),
            ("sonthenguyen/amc12-2025-non-figure",  "train", "question", "answer", "amc25"),
        ]
        for hf_id, split, prob_field, ans_field, prefix in loaders:
            try:
                ds = load_dataset(hf_id, split=split, trust_remote_code=True)
                for i, row in enumerate(ds):
                    problems.append({
                        "id": f"{prefix}_{i:04d}",
                        "problem": row[prob_field],
                        "answer": str(row[ans_field]),
                        "source": prefix,
                    })
            except Exception as e:
                print(f"  Warning: could not load {hf_id}: {e}", flush=True)

        # Deterministic order: sort by source then index so eval is reproducible
        problems.sort(key=lambda p: p["id"])
        problems = problems[:n]
        print(f"  Loaded {len(problems)} amc12 problems ({set(p['source'] for p in problems)}) "
              f"in {time.time()-t0:.1f}s", flush=True)
        return problems

    raise ValueError(f"Unknown benchmark: {name}")


def run_eval(model, tokenizer, problems, max_tokens=256, prompt_format="chat_think",
             eval_batch_size=1):
    """Run eval on problems. Returns summary dict.

    eval_batch_size=1 (default) runs sequentially — safe for local/MPS.
    On GPU (Modal), pass eval_batch_size=8 or 16 for much faster eval.
    """
    model.eval()  # Ensure eval mode
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
    parser.add_argument("--eval-batch-size", type=int, default=1,
                        help="Batch size for generation. 1=sequential (local), 16-64=batched (GPU)")
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
