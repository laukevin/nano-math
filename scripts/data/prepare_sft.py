#!/usr/bin/env python3
"""Prepare SFT data: format datasets into chat JSONL for fine-tuning.

Supports all 5 SFT recipes from spec 04:
  - distill-r1: Long CoT from DeepSeek-R1 (100K samples, 4096 max_seq_len)
  - concise-cot: Short solutions from MetaMath (100K samples, 2048 max_seq_len)
  - kitchen-sink: Diverse mix from 4 sources (200K samples, 2048 max_seq_len)
  - quality: High-quality small set (30K samples, 2048 max_seq_len, 10 epochs)
  - progressive: Staged easy-to-hard curriculum (2 stages)

Usage:
    python scripts/data/prepare_sft.py --recipe distill-r1 --output data/sft/distill-r1/
    python scripts/data/prepare_sft.py --recipe quality --output data/sft/quality/
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from math_nano.data.answers import ensure_boxed_answer
from math_nano.data.hf import load_hf_dataset
from math_nano.data.io import write_jsonl
from math_nano.data.tokenizer import get_tokenizer

# SFT recipe definitions
RECIPES = {
    "distill-r1": {
        "recipe_id": "sft-distill-r1",
        "sources": [
            {
                "dataset": "deepseek-r1-distill",
                "hf_id": "deepseek-ai/DeepSeek-R1",
                "samples": 100_000,
                "text_columns": {"problem": "problem", "solution": "generated_solution"},
            },
        ],
        "system_prompt": "You are a helpful math assistant. Think step by step.",
        "max_seq_len": 4096,
        "epochs": 3,
    },
    "concise-cot": {
        "recipe_id": "sft-concise-cot",
        "sources": [
            {
                "dataset": "metamath",
                "hf_id": "meta-math/MetaMathQA",
                "samples": 100_000,
                "text_columns": {"problem": "query", "solution": "response"},
            },
        ],
        "system_prompt": "Solve the problem step by step. Be concise.",
        "max_seq_len": 2048,
        "epochs": 3,
    },
    "kitchen-sink": {
        "recipe_id": "sft-kitchen-sink",
        "sources": [
            {
                "dataset": "openmathreasoning-sft",
                "hf_id": "nvidia/OpenMathReasoning",
                "samples": 50_000,
                "text_columns": {"problem": "problem", "solution": "generated_solution"},
            },
            {
                "dataset": "metamath",
                "hf_id": "meta-math/MetaMathQA",
                "samples": 50_000,
                "text_columns": {"problem": "query", "solution": "response"},
            },
            {
                "dataset": "numinamath-cot",
                "hf_id": "AI-MO/NuminaMath-CoT",
                "samples": 50_000,
                "text_columns": {"problem": "problem", "solution": "solution"},
            },
            {
                "dataset": "orca-math",
                "hf_id": "microsoft/orca-math-word-problems-200k",
                "samples": 50_000,
                "text_columns": {"problem": "question", "solution": "answer"},
            },
        ],
        "system_prompt": "You are a helpful math assistant. Think step by step.",
        "max_seq_len": 2048,
        "epochs": 3,
    },
    "quality": {
        "recipe_id": "sft-quality",
        "sources": [
            {
                "dataset": "math-train",
                "hf_id": "hendrycks/competition_math",
                "samples": None,  # all
                "split": "train",
                "text_columns": {"problem": "problem", "solution": "solution"},
            },
            {
                "dataset": "gsm8k-train",
                "hf_id": "openai/gsm8k",
                "subset": "main",
                "samples": None,  # all
                "split": "train",
                "text_columns": {"problem": "question", "solution": "answer"},
            },
            {
                "dataset": "numinamath-cot",
                "hf_id": "AI-MO/NuminaMath-CoT",
                "samples": 15_000,
                "text_columns": {"problem": "problem", "solution": "solution"},
            },
        ],
        "system_prompt": "You are a helpful math assistant. Think step by step.",
        "max_seq_len": 2048,
        "epochs": 10,
    },
    "progressive": {
        "recipe_id": "sft-progressive",
        "stages": {
            "stage_1": {
                "sources": [
                    {
                        "dataset": "gsm8k-train",
                        "hf_id": "openai/gsm8k",
                        "subset": "main",
                        "samples": None,
                        "split": "train",
                        "text_columns": {"problem": "question", "solution": "answer"},
                    },
                    {
                        "dataset": "metamath",
                        "hf_id": "meta-math/MetaMathQA",
                        "samples": 30_000,
                        "text_columns": {"problem": "query", "solution": "response"},
                        "filter_difficulty": "easy",
                    },
                ],
                "epochs": 3,
            },
            "stage_2": {
                "sources": [
                    {
                        "dataset": "numinamath-cot",
                        "hf_id": "AI-MO/NuminaMath-CoT",
                        "samples": 50_000,
                        "text_columns": {"problem": "problem", "solution": "solution"},
                    },
                    {
                        "dataset": "openmathreasoning-sft",
                        "hf_id": "nvidia/OpenMathReasoning",
                        "samples": 50_000,
                        "text_columns": {"problem": "problem", "solution": "generated_solution"},
                        "filter_difficulty": "hard",
                    },
                ],
                "epochs": 2,
            },
        },
        "system_prompt": "You are a helpful math assistant. Think step by step.",
        "max_seq_len": 2048,
    },
}


def truncate_preserving_answer(
    text: str, max_tokens: int, tokenizer
) -> tuple[str, bool]:
    r"""Truncate text from the LEFT of the chain-of-thought, keeping the answer.

    Per spec: truncate from the left of the CoT, never truncate the answer.

    Returns:
        (truncated_text, was_truncated)
    """
    tokens = tokenizer.encode_ordinary(text)
    if len(tokens) <= max_tokens:
        return text, False

    # Keep the last ~200 tokens always (to preserve the answer)
    answer_reserve = min(200, max_tokens // 4)
    content_budget = max_tokens - answer_reserve

    tail_tokens = tokens[-answer_reserve:]
    head_tokens = tokens[:content_budget]

    truncated = (
        tokenizer.decode(head_tokens)
        + "\n... [truncated] ...\n"
        + tokenizer.decode(tail_tokens)
    )

    return truncated, True


def format_chat_sample(
    problem: str,
    solution: str,
    system_prompt: str,
    max_seq_len: int | None = None,
    tokenizer=None,
) -> tuple[dict, bool]:
    """Format a single (problem, solution) pair into chat format.

    Returns:
        (formatted_sample, was_truncated)
    """
    solution = ensure_boxed_answer(solution)

    was_truncated = False
    if max_seq_len is not None and tokenizer is not None:
        overhead = 50 + len(tokenizer.encode_ordinary(problem))
        solution_budget = max_seq_len - overhead
        if solution_budget > 0:
            solution, was_truncated = truncate_preserving_answer(
                solution, solution_budget, tokenizer
            )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution},
        ]
    }, was_truncated


def estimate_difficulty(solution: str, tokenizer=None) -> int:
    """Estimate problem difficulty (1-5) using solution length as proxy.

    Level 1: <100 tokens, Level 2: 100-300, Level 3: 300-600,
    Level 4: 600-1200, Level 5: 1200+
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    n_tokens = len(tokenizer.encode_ordinary(solution))
    if n_tokens < 100:
        return 1
    elif n_tokens < 300:
        return 2
    elif n_tokens < 600:
        return 3
    elif n_tokens < 1200:
        return 4
    else:
        return 5


def process_source(
    source_config: dict,
    system_prompt: str,
    max_seq_len: int,
    tokenizer,
) -> tuple[list[dict], dict]:
    """Process a single data source into formatted samples.

    Returns:
        (samples, stats)
    """
    ds = load_hf_dataset(
        source_config["hf_id"],
        subset=source_config.get("subset"),
        split=source_config.get("split", "train"),
    )
    cols = source_config["text_columns"]
    max_samples = source_config.get("samples")
    filter_difficulty = source_config.get("filter_difficulty")

    samples = []
    stats = {
        "dataset": source_config["dataset"],
        "total_seen": 0,
        "total_kept": 0,
        "truncated": 0,
        "missing_boxed": 0,
        "token_lengths": [],
    }

    for doc in ds:
        problem = doc.get(cols["problem"], "")
        solution = doc.get(cols["solution"], "")

        if not problem or not solution:
            continue

        stats["total_seen"] += 1

        # Difficulty filtering for progressive recipe
        if filter_difficulty:
            diff = estimate_difficulty(solution, tokenizer)
            if filter_difficulty == "easy" and diff > 2:
                continue
            if filter_difficulty == "hard" and diff < 3:
                continue

        sample, was_truncated = format_chat_sample(
            problem=problem.strip(),
            solution=solution.strip(),
            system_prompt=system_prompt,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
        )

        assistant_content = sample["messages"][-1]["content"]
        if r"\boxed{" not in assistant_content:
            stats["missing_boxed"] += 1

        if was_truncated:
            stats["truncated"] += 1

        n_tokens = len(tokenizer.encode_ordinary(assistant_content))
        stats["token_lengths"].append(n_tokens)

        samples.append(sample)
        stats["total_kept"] += 1

        if max_samples is not None and stats["total_kept"] >= max_samples:
            break

        if stats["total_seen"] % 10000 == 0:
            print(
                f"  [{source_config['dataset']}] "
                f"Seen {stats['total_seen']:,}, kept {stats['total_kept']:,}"
            )

    return samples, stats


def _summarize_lengths(lengths: list[int]) -> dict:
    """Summarize token length distribution."""
    if not lengths:
        return {"count": 0}
    arr = np.array(lengths)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(np.max(arr)),
        "min": int(np.min(arr)),
    }


def prepare_recipe(
    recipe_name: str,
    output_dir: str,
    max_seq_len_override: int | None = None,
):
    """Prepare all data for a given SFT recipe."""
    if recipe_name not in RECIPES:
        raise ValueError(
            f"Unknown recipe: {recipe_name}. Choose from: {list(RECIPES)}"
        )

    recipe = RECIPES[recipe_name]
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = get_tokenizer()
    system_prompt = recipe.get(
        "system_prompt", "You are a helpful math assistant. Think step by step."
    )
    max_seq_len = max_seq_len_override or recipe.get("max_seq_len", 2048)

    print(f"Preparing recipe: {recipe_name}")
    print(f"  Max seq len: {max_seq_len}")
    print(f"  System prompt: {system_prompt[:60]}...")

    # Handle progressive recipe (multiple stages)
    if "stages" in recipe:
        all_stats = {"recipe": recipe_name, "stages": {}}
        for stage_name, stage_config in recipe["stages"].items():
            print(f"\n--- {stage_name} ---")
            stage_dir = os.path.join(output_dir, stage_name)
            os.makedirs(stage_dir, exist_ok=True)

            all_samples = []
            stage_stats = []
            for source_config in stage_config["sources"]:
                print(f"  Processing: {source_config['dataset']}")
                samples, stats = process_source(
                    source_config, system_prompt, max_seq_len, tokenizer
                )
                all_samples.extend(samples)
                stats["token_lengths"] = _summarize_lengths(stats["token_lengths"])
                stage_stats.append(stats)

            rng = np.random.RandomState(42)
            rng.shuffle(all_samples)

            jsonl_path = os.path.join(stage_dir, "train.jsonl")
            write_jsonl(all_samples, jsonl_path)

            all_stats["stages"][stage_name] = {
                "total_samples": len(all_samples),
                "epochs": stage_config["epochs"],
                "sources": stage_stats,
            }
            print(f"  {stage_name}: {len(all_samples):,} samples -> {jsonl_path}")

        stats_path = os.path.join(output_dir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nStats written to {stats_path}")
        return

    # Standard recipe (single stage)
    all_samples = []
    all_stats = {"recipe": recipe_name, "sources": []}

    for source_config in recipe["sources"]:
        print(f"  Processing: {source_config['dataset']}")
        samples, stats = process_source(
            source_config, system_prompt, max_seq_len, tokenizer
        )
        all_samples.extend(samples)
        stats["token_lengths"] = _summarize_lengths(stats["token_lengths"])
        all_stats["sources"].append(stats)

    rng = np.random.RandomState(42)
    rng.shuffle(all_samples)

    jsonl_path = os.path.join(output_dir, "train.jsonl")
    write_jsonl(all_samples, jsonl_path)

    all_stats["total_samples"] = len(all_samples)
    all_stats["max_seq_len"] = max_seq_len
    all_stats["epochs"] = recipe.get("epochs", 3)
    all_stats["system_prompt"] = system_prompt

    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nRecipe: {recipe_name}")
    print(f"  Total samples: {len(all_samples):,}")
    print(f"  Output: {jsonl_path}")
    print(f"  Stats: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data for math-nano")
    parser.add_argument(
        "--recipe",
        required=True,
        choices=list(RECIPES.keys()),
        help="SFT recipe to prepare",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Override max sequence length",
    )

    args = parser.parse_args()

    prepare_recipe(
        recipe_name=args.recipe,
        output_dir=args.output,
        max_seq_len_override=args.max_seq_len,
    )


if __name__ == "__main__":
    main()
