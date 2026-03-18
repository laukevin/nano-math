#!/usr/bin/env python3
"""Download tiny data subsets for local testing.

Downloads 1 shard per pretrain source and 1000 SFT samples for quick
local development and smoke testing.

Usage:
    python scripts/data/download_sample.py
    python scripts/data/download_sample.py --output data/sample/ --num-sft 500
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from math_nano.data.answers import ensure_boxed_answer
from math_nano.data.dataloader import write_shard
from math_nano.data.hf import load_hf_dataset
from math_nano.data.io import write_jsonl
from math_nano.data.tokenizer import EOT_TOKEN, get_tokenizer

# Sources to sample from
PRETRAIN_SOURCES = {
    "fineweb-edu": {
        "hf_id": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "text_column": "text",
    },
    "openwebmath": {
        "hf_id": "open-web-math/open-web-math",
        "subset": None,
        "text_column": "text",
    },
    "openmathreasoning": {
        "hf_id": "nvidia/OpenMathReasoning",
        "subset": None,
        "text_column": "generated_solution",
        "problem_column": "problem",
    },
}

SFT_SOURCES = [
    {
        "name": "gsm8k",
        "hf_id": "openai/gsm8k",
        "subset": "main",
        "split": "train",
        "problem_col": "question",
        "solution_col": "answer",
        "samples": 500,
    },
    {
        "name": "metamath",
        "hf_id": "meta-math/MetaMathQA",
        "subset": None,
        "split": "train",
        "problem_col": "query",
        "solution_col": "response",
        "samples": 500,
    },
]


def download_pretrain_sample(source_name: str, output_dir: str, max_tokens: int):
    """Download a small sample of pretrain data and tokenize into 1 shard."""
    config = PRETRAIN_SOURCES[source_name]
    print(f"  Downloading {source_name} sample...")

    ds = load_hf_dataset(
        config["hf_id"],
        subset=config.get("subset"),
    )

    enc = get_tokenizer()
    tokens = []
    doc_offsets = [0]
    n_docs = 0

    for doc in ds:
        text = doc.get(config["text_column"], "")
        if config.get("problem_column") and config["problem_column"] in doc:
            problem = doc[config["problem_column"]]
            text = f"Problem: {problem}\n\nSolution: {text}"

        if not text or not text.strip():
            continue

        doc_tokens = enc.encode_ordinary(text)
        doc_tokens.append(EOT_TOKEN)
        tokens.extend(doc_tokens)
        doc_offsets.append(len(tokens))
        n_docs += 1

        if len(tokens) >= max_tokens:
            break

    # Write shard
    shard_dir = os.path.join(output_dir, source_name)
    os.makedirs(shard_dir, exist_ok=True)
    token_arr = np.array(tokens[:max_tokens], dtype=np.uint16)
    offset_arr = np.array(doc_offsets, dtype=np.int64)
    shard_path = os.path.join(shard_dir, "shard_000000")
    write_shard(token_arr, offset_arr, shard_path)

    print(f"    {source_name}: {len(token_arr):,} tokens, {n_docs} docs -> {shard_dir}")


def download_sft_sample(output_dir: str, num_samples: int):
    """Download a small sample of SFT data as chat JSONL."""
    samples = []

    for src in SFT_SOURCES:
        target = min(src["samples"], num_samples // len(SFT_SOURCES))
        print(f"  Downloading {src['name']} SFT sample ({target} samples)...")

        ds = load_hf_dataset(
            src["hf_id"],
            subset=src.get("subset"),
            split=src["split"],
        )

        count = 0
        for doc in ds:
            problem = doc.get(src["problem_col"], "")
            solution = doc.get(src["solution_col"], "")
            if not problem or not solution:
                continue

            solution = ensure_boxed_answer(solution)

            sample = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful math assistant. Think step by step.",
                    },
                    {"role": "user", "content": problem.strip()},
                    {"role": "assistant", "content": solution.strip()},
                ]
            }
            samples.append(sample)
            count += 1
            if count >= target:
                break

        print(f"    {src['name']}: {count} samples collected")

    # Shuffle and write
    rng = np.random.RandomState(42)
    rng.shuffle(samples)

    jsonl_path = os.path.join(output_dir, "sft_sample.jsonl")
    write_jsonl(samples, jsonl_path)

    print(f"  SFT sample: {len(samples)} samples -> {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download tiny data subsets for local testing"
    )
    parser.add_argument(
        "--output",
        default="data/sample/",
        help="Output directory (default: data/sample/)",
    )
    parser.add_argument(
        "--shard-tokens",
        type=int,
        default=1_000_000,
        help="Tokens per pretrain sample shard (default: 1M)",
    )
    parser.add_argument(
        "--num-sft",
        type=int,
        default=1000,
        help="Number of SFT samples to download (default: 1000)",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Pretrain sources to download (default: all)",
    )

    args = parser.parse_args()
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Download pretrain samples
    sources = args.sources or list(PRETRAIN_SOURCES.keys())
    print("Downloading pretrain samples:")
    for source in sources:
        if source in PRETRAIN_SOURCES:
            download_pretrain_sample(source, output_dir, args.shard_tokens)

    # Download SFT samples
    print("\nDownloading SFT samples:")
    download_sft_sample(output_dir, args.num_sft)

    print(f"\nDone! Sample data in {output_dir}")


if __name__ == "__main__":
    main()
