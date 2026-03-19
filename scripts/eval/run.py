"""Run math eval on a nanochat checkpoint.

Usage:
    # Short eval on pretrained model
    python -m scripts.eval.run --model-tag d2 --suite small

    # With specific step
    python -m scripts.eval.run --model-tag d2 --step 300 --suite small
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Add project root to path so we can import from scripts.eval
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.extraction import extract_answer, normalize_answer
from scripts.eval.data import format_eval_prompt, SUITE_DATASETS


def load_nanochat_model(model_tag: str, step: int | None, device: str, phase: str = "base"):
    """Load a nanochat checkpoint using nanochat's own loader.

    phase: "base" for pretrained, "mathsft" for our math SFT checkpoints.
    """
    # Add nanochat to path
    nanochat_dir = PROJECT_ROOT / "vendor" / "nanochat"
    sys.path.insert(0, str(nanochat_dir))

    from nanochat.checkpoint_manager import load_model
    from nanochat.common import autodetect_device_type

    if device == "auto":
        device = autodetect_device_type()

    device_obj = torch.device(device)

    if phase == "mathsft":
        # Load base model structure, then overlay SFT weights
        model, tokenizer, meta = load_model(
            "base", device_obj, phase="eval",
            model_tag=model_tag, step=None,
        )
        # Load our math SFT weights
        base = Path(os.environ.get("NANOCHAT_BASE_DIR", str(Path.home() / ".cache" / "nanochat")))
        sft_dir = base / "mathsft_checkpoints" / model_tag
        if step is None:
            # Find latest step
            pts = sorted(sft_dir.glob("model_*.pt"))
            if not pts:
                raise FileNotFoundError(f"No mathsft checkpoints in {sft_dir}")
            sft_path = pts[-1]
        else:
            sft_path = sft_dir / f"model_{step:06d}.pt"
        print(f"Loading math SFT weights from {sft_path}")
        state_dict = torch.load(sft_path, map_location=device_obj, weights_only=True)
        model.load_state_dict(state_dict)
        # Load SFT metadata
        meta_path = sft_path.with_name(sft_path.name.replace("model_", "meta_").replace(".pt", ".json"))
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                sft_meta = json.load(f)
            meta.update(sft_meta)
    else:
        model, tokenizer, meta = load_model(
            "base", device_obj, phase="eval",
            model_tag=model_tag, step=step,
        )

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, tokenizer, device, n_params, meta


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, device: str, max_tokens: int = 256):
    """Generate a single completion. Simple, no batching."""
    bos_id = tokenizer.get_bos_token_id()
    input_ids = tokenizer.encode(prompt)
    tokens = torch.tensor([input_ids], device=device)

    for _ in range(max_tokens):
        logits = model(tokens)
        if isinstance(logits, tuple):
            logits = logits[0]
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=-1)

        # Stop on BOS (nanochat's document delimiter)
        if next_token.item() == bos_id:
            break

    output_ids = tokens[0, len(input_ids):].tolist()
    # Truncate at BOS
    if bos_id in output_ids:
        output_ids = output_ids[:output_ids.index(bos_id)]
    return tokenizer.decode(output_ids)


def make_gsm8k_mini(n: int = 50) -> list[dict]:
    """Load GSM8K test split and take first n problems."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        problems = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            # GSM8K answer is after ####
            answer_text = row["answer"].split("####")[-1].strip()
            problems.append({
                "id": f"gsm8k_{i:04d}",
                "problem": row["question"],
                "answer": answer_text,
                "source": "gsm8k",
            })
        return problems
    except ImportError:
        print("WARNING: 'datasets' not installed, using hardcoded mini problems")
        return _hardcoded_mini()


def _hardcoded_mini() -> list[dict]:
    """Tiny hardcoded eval set for testing without HF datasets."""
    return [
        {"id": "mini_0", "problem": "What is 2 + 3?", "answer": "5", "source": "mini"},
        {"id": "mini_1", "problem": "What is 7 * 8?", "answer": "56", "source": "mini"},
        {"id": "mini_2", "problem": "What is 100 - 37?", "answer": "63", "source": "mini"},
        {"id": "mini_3", "problem": "What is 144 / 12?", "answer": "12", "source": "mini"},
        {"id": "mini_4", "problem": "What is 15 + 27?", "answer": "42", "source": "mini"},
        {"id": "mini_5", "problem": "If you have 3 bags with 4 apples each, how many apples total?", "answer": "12", "source": "mini"},
        {"id": "mini_6", "problem": "What is 25% of 80?", "answer": "20", "source": "mini"},
        {"id": "mini_7", "problem": "A train travels 60 miles in 2 hours. What is its speed in mph?", "answer": "30", "source": "mini"},
        {"id": "mini_8", "problem": "What is 9 squared?", "answer": "81", "source": "mini"},
        {"id": "mini_9", "problem": "What is 1000 - 999?", "answer": "1", "source": "mini"},
    ]


def make_svamp(n: int = 50) -> list[dict]:
    """Load SVAMP dataset — single-step arithmetic word problems.

    SVAMP is much simpler than GSM8K: each problem requires exactly one
    arithmetic operation. Good for testing basic math ability.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("ChilleD/SVAMP", split="test")
        problems = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            problems.append({
                "id": f"svamp_{i:04d}",
                "problem": row["question_concat"].strip(),
                "answer": str(int(float(row["Answer"]))),
                "source": "svamp",
            })
        return problems
    except Exception as e:
        print(f"WARNING: Could not load SVAMP dataset: {e}")
        print("Falling back to generated arithmetic problems")
        return _generated_arithmetic(n)


def _generated_arithmetic(n: int = 50) -> list[dict]:
    """Generate simple arithmetic problems as fallback."""
    import random
    random.seed(42)
    problems = []
    for i in range(n):
        op = random.choice(["+", "-"])
        if op == "+":
            a, b = random.randint(1, 100), random.randint(1, 100)
            answer = a + b
        else:
            a = random.randint(10, 200)
            b = random.randint(1, a)
            answer = a - b
        problems.append({
            "id": f"arith_{i:04d}",
            "problem": f"What is {a} {op} {b}?",
            "answer": str(answer),
            "source": "arithmetic",
        })
    return problems


def eval_format_score(output: str) -> dict:
    """Score formatting quality of a single output.

    Returns dict with:
        has_boxed: bool — output contains \boxed{...}
        has_number: bool — extracted answer is a number
        has_steps: bool — output has multiple lines (attempted reasoning)
    """
    import re

    has_boxed = bool(re.search(r"\\boxed\{", output))
    extracted = extract_answer(output)
    has_number = False
    if extracted is not None:
        try:
            float(extracted)
            has_number = True
        except ValueError:
            pass
    has_steps = output.count("\n") >= 2

    return {
        "has_boxed": has_boxed,
        "has_number": has_number,
        "has_steps": has_steps,
    }


def run_eval(model, tokenizer, device, problems, max_tokens=256):
    """Run eval on a list of problems. Returns results dict."""
    results = []
    n_correct = 0
    n_format_boxed = 0
    n_format_number = 0
    n_format_steps = 0
    n_extracted = 0

    print(f"\nRunning eval on {len(problems)} problems (max_tokens={max_tokens})")
    print("-" * 70)

    for i, problem in enumerate(problems):
        prompt = format_eval_prompt(problem["problem"])
        start = time.time()
        output = generate_one(model, tokenizer, prompt, device, max_tokens=max_tokens)
        elapsed = time.time() - start

        # Format scoring
        fmt = eval_format_score(output)
        n_format_boxed += fmt["has_boxed"]
        n_format_number += fmt["has_number"]
        n_format_steps += fmt["has_steps"]

        # Correctness
        extracted = extract_answer(output)
        if extracted is not None:
            n_extracted += 1
        gt = normalize_answer(str(problem["answer"]))
        correct = extracted is not None and extracted == gt

        if correct:
            n_correct += 1

        results.append({
            "id": problem["id"],
            "problem": problem["problem"][:80],
            "ground_truth": gt,
            "extracted": extracted,
            "correct": correct,
            "format": fmt,
            "output_preview": output[:200],
            "time_s": elapsed,
        })

        status = "CORRECT" if correct else ("EXTRACTED" if extracted else "NO_ANSWER")
        print(f"  [{i+1:3d}/{len(problems)}] {status:10s} | gt={gt:>8s} pred={str(extracted):>8s} | {elapsed:.1f}s | {problem['problem'][:50]}")

    n = len(problems)
    print("-" * 70)
    print(f"\n=== RESULTS ({n} problems) ===")
    print(f"  Accuracy:       {n_correct}/{n} ({100*n_correct/n:.1f}%)")
    print(f"  Extracted:      {n_extracted}/{n} ({100*n_extracted/n:.1f}%)")
    print(f"  Format scores:")
    print(f"    has_boxed:    {n_format_boxed}/{n} ({100*n_format_boxed/n:.1f}%)")
    print(f"    has_number:   {n_format_number}/{n} ({100*n_format_number/n:.1f}%)")
    print(f"    has_steps:    {n_format_steps}/{n} ({100*n_format_steps/n:.1f}%)")

    summary = {
        "n_problems": n,
        "accuracy": n_correct / n if n else 0,
        "extraction_rate": n_extracted / n if n else 0,
        "format_boxed_rate": n_format_boxed / n if n else 0,
        "format_number_rate": n_format_number / n if n else 0,
        "format_steps_rate": n_format_steps / n if n else 0,
        "per_problem": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run math eval on nanochat checkpoint")
    parser.add_argument("--model-tag", type=str, default=None, help="nanochat model tag (e.g. d2)")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step (default=latest)")
    parser.add_argument("--phase", type=str, default="base", choices=["base", "mathsft"], help="checkpoint phase (base=pretrain, mathsft=math SFT)")
    parser.add_argument("--device", type=str, default="auto", help="device (auto/cpu/mps/cuda)")
    parser.add_argument("--max-tokens", type=int, default=256, help="max generation tokens")
    parser.add_argument("--n-problems", type=int, default=10, help="number of problems to eval")
    parser.add_argument("--benchmark", type=str, default="gsm8k", choices=["gsm8k", "svamp"], help="benchmark to eval on")
    parser.add_argument("--output", type=str, default=None, help="save results JSON to this path")
    args = parser.parse_args()

    print(f"Loading model (tag={args.model_tag}, step={args.step}, phase={args.phase})...")
    model, tokenizer, device, n_params, meta = load_nanochat_model(
        args.model_tag, args.step, args.device, phase=args.phase
    )
    print(f"  Device: {device}")
    print(f"  Params: {n_params:,}")
    print(f"  Config: {meta.get('model_config', {})}")

    # Load problems
    if args.benchmark == "svamp":
        problems = make_svamp(n=args.n_problems)
    else:
        problems = make_gsm8k_mini(n=args.n_problems)
    if not problems:
        print("No eval problems available")
        return

    # Run eval
    start = time.time()
    summary = run_eval(model, tokenizer, device, problems, max_tokens=args.max_tokens)
    total_time = time.time() - start

    summary["model_tag"] = args.model_tag
    summary["benchmark"] = args.benchmark
    summary["step"] = args.step
    summary["n_params"] = n_params
    summary["device"] = device
    summary["total_time_s"] = total_time

    print(f"\n  Total eval time: {total_time:.1f}s")

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove non-serializable items for JSON
        json_summary = {k: v for k, v in summary.items()}
        with open(out_path, "w") as f:
            json.dump(json_summary, f, indent=2, default=str)
        print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
