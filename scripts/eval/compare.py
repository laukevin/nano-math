"""Compare two checkpoints with paired bootstrap significance testing.

Usage:
    python scripts/eval/compare.py \
      --checkpoint-a results/sft-m/best.pt \
      --checkpoint-b results/grpo-m/best.pt \
      --depth 16 --suite full --samples 16
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from scripts.eval.extraction import extract_answer, normalize_answer
from scripts.eval.metrics import bootstrap_ci, is_significant_improvement
from scripts.eval.run_eval import (
    BOOTSTRAP_SEED,
    MAX_NEW_TOKENS,
    SUITE_DATASETS,
    format_eval_prompt,
    generate_batch,
    load_eval_dataset,
    load_model,
)

logger = logging.getLogger(__name__)


def collect_per_problem_correctness(
    model,
    tokenizer,
    problems: list[dict],
    dataset_name: str = "",  # noqa: ARG001
    device: str = "cpu",
    batch_size: int = 32,
) -> list[bool]:
    """Run greedy generation and return per-problem correctness."""
    prompts = [format_eval_prompt(p["problem"]) for p in problems]
    ground_truths = [normalize_answer(str(p["answer"])) for p in problems]

    outputs, _ = generate_batch(
        model,
        tokenizer,
        prompts,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        batch_size=batch_size,
        device=device,
    )

    correct = []
    for output, gt in zip(outputs, ground_truths):
        predicted = extract_answer(output)
        correct.append(predicted is not None and predicted == gt)

    return correct


def compare_checkpoints(
    model_a,
    tokenizer_a,
    model_b,
    tokenizer_b,
    problems: list[dict],
    dataset_name: str,
    device: str = "cpu",
    batch_size: int = 32,
) -> dict:
    """Compare two models on the same problems.

    Returns comparison dict with metrics and significance test.
    """
    correct_a = collect_per_problem_correctness(
        model_a, tokenizer_a, problems, dataset_name, device, batch_size
    )
    correct_b = collect_per_problem_correctness(
        model_b, tokenizer_b, problems, dataset_name, device, batch_size
    )

    mean_a, ci_low_a, ci_high_a = bootstrap_ci(correct_a, seed=BOOTSTRAP_SEED)
    mean_b, ci_low_b, ci_high_b = bootstrap_ci(correct_b, seed=BOOTSTRAP_SEED)

    significant, p_value = is_significant_improvement(
        correct_a, correct_b, seed=BOOTSTRAP_SEED
    )

    # Count agreement/disagreement
    n = len(correct_a)
    both_correct = sum(a and b for a, b in zip(correct_a, correct_b))
    both_wrong = sum(not a and not b for a, b in zip(correct_a, correct_b))
    a_only = sum(a and not b for a, b in zip(correct_a, correct_b))
    b_only = sum(not a and b for a, b in zip(correct_a, correct_b))

    return {
        "dataset": dataset_name,
        "n_problems": n,
        "model_a": {
            "pass_at_1": mean_a,
            "ci95": [ci_low_a, ci_high_a],
        },
        "model_b": {
            "pass_at_1": mean_b,
            "ci95": [ci_low_b, ci_high_b],
        },
        "delta": mean_b - mean_a,
        "significant": significant,
        "p_value": p_value,
        "agreement": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "a_only": a_only,
            "b_only": b_only,
        },
    }


def format_comparison_table(comparisons: list[dict], ckpt_a: str, ckpt_b: str) -> str:
    """Format comparison results as a readable table."""
    lines = [
        f"Checkpoint A: {ckpt_a}",
        f"Checkpoint B: {ckpt_b}",
        "",
        f"{'Dataset':<15} {'A pass@1':>10} {'B pass@1':>10} {'Delta':>8} {'p-value':>8} {'Sig?':>5}",
        "-" * 60,
    ]

    for c in comparisons:
        sig_mark = "*" if c["significant"] else ""
        lines.append(
            f"{c['dataset']:<15} "
            f"{c['model_a']['pass_at_1']:>10.3f} "
            f"{c['model_b']['pass_at_1']:>10.3f} "
            f"{c['delta']:>+8.3f} "
            f"{c['p_value']:>8.4f} "
            f"{sig_mark:>5}"
        )

    lines.append("")
    lines.append("* = significant at p < 0.05 (paired bootstrap)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Compare two checkpoints")
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument(
        "--suite", choices=["small", "full"], default="small"
    )
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", default="data/eval")
    parser.add_argument(
        "--output", default=None, help="Output JSON path"
    )

    args = parser.parse_args(argv)

    datasets = args.datasets or SUITE_DATASETS[args.suite]
    data_dir = Path(args.data_dir)

    # Load both models
    logger.info("Loading model A: %s", args.checkpoint_a)
    model_a, tok_a, device, _params_a = load_model(
        args.checkpoint_a, args.depth, args.device
    )
    logger.info("Loading model B: %s", args.checkpoint_b)
    model_b, tok_b, _, _params_b = load_model(
        args.checkpoint_b, args.depth, args.device
    )

    comparisons = []
    for ds_name in datasets:
        logger.info("Comparing on %s...", ds_name)
        problems = load_eval_dataset(ds_name, data_dir)
        comp = compare_checkpoints(
            model_a, tok_a, model_b, tok_b,
            problems, ds_name, device, args.batch_size,
        )
        comparisons.append(comp)

    # Print table
    table = format_comparison_table(
        comparisons, args.checkpoint_a, args.checkpoint_b
    )
    print(table)

    # Save JSON if requested
    if args.output:
        output = {
            "checkpoint_a": args.checkpoint_a,
            "checkpoint_b": args.checkpoint_b,
            "depth": args.depth,
            "comparisons": comparisons,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, indent=2))
        logger.info("Comparison saved to %s", args.output)


if __name__ == "__main__":
    main()
