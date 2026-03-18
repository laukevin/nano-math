#!/usr/bin/env python3
"""Download and tokenize pretrain data from HuggingFace.

Downloads from HuggingFace (FineWeb-Edu, OpenWebMath, OpenMathReasoning),
tokenizes with GPT-2 BPE (tiktoken), outputs .bin shards of uint16 token IDs.

Usage:
    python scripts/data/download_and_tokenize.py \
        --source openwebmath \
        --output data/tokenized/openwebmath/ \
        --shard-size 100000000

    python scripts/data/download_and_tokenize.py \
        --source fineweb-edu \
        --output data/tokenized/fineweb-edu/ \
        --shard-size 100000000 \
        --max-docs 10000
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from math_nano.data.dataloader import write_shard
from math_nano.data.hf import load_hf_dataset
from math_nano.data.tokenizer import EOT_TOKEN, get_tokenizer

# Source configs: HuggingFace ID, dataset subset, text column
SOURCES = {
    "fineweb-edu": {
        "hf_id": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "text_column": "text",
        "split": "train",
    },
    "openwebmath": {
        "hf_id": "open-web-math/open-web-math",
        "subset": None,
        "text_column": "text",
        "split": "train",
    },
    "openmathreasoning": {
        "hf_id": "nvidia/OpenMathReasoning",
        "subset": None,
        "text_column": "generated_solution",
        "split": "train",
        "extra_columns": ["problem", "expected_answer"],
    },
}


def download_and_tokenize(
    source: str,
    output_dir: str,
    shard_size: int = 100_000_000,
    max_docs: int | None = None,
    val_tokens: int = 10_000_000,
):
    """Download a dataset from HuggingFace and tokenize into shards.

    Args:
        source: Source name (fineweb-edu, openwebmath, openmathreasoning).
        output_dir: Output directory for .bin/.idx shard files.
        shard_size: Tokens per shard (default 100M).
        max_docs: Maximum documents to process (None for all).
        val_tokens: Tokens to hold out for validation (default 10M).
    """
    if source not in SOURCES:
        raise ValueError(f"Unknown source: {source}. Choose from: {list(SOURCES)}")

    config = SOURCES[source]
    os.makedirs(output_dir, exist_ok=True)
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(val_dir, exist_ok=True)

    print(f"Loading dataset: {config['hf_id']}")
    ds = load_hf_dataset(
        config["hf_id"],
        subset=config.get("subset"),
        split=config["split"],
    )

    enc = get_tokenizer()

    # State for current shard
    shard_idx = 0
    token_buf = []
    doc_offsets = [0]
    total_tokens = 0
    total_docs = 0
    val_token_buf = []
    val_doc_offsets = [0]
    val_collected = 0
    collecting_val = True

    def flush_shard(tokens, offsets, directory, idx):
        """Write accumulated tokens to a shard file."""
        if not tokens:
            return
        token_arr = np.array(tokens, dtype=np.uint16)
        offset_arr = np.array(offsets, dtype=np.int64)
        shard_path = os.path.join(directory, f"shard_{idx:06d}")
        write_shard(token_arr, offset_arr, shard_path)
        n = len(token_arr)
        print(f"  Wrote {shard_path}.bin ({n:,} tokens)")

    print(f"Tokenizing with GPT-2 BPE (shard size: {shard_size:,} tokens)")
    print(f"Holding out {val_tokens:,} tokens for validation")

    for doc in ds:
        text = doc.get(config["text_column"], "")

        # For OpenMathReasoning, prepend the problem
        if source == "openmathreasoning" and "problem" in doc:
            problem = doc.get("problem", "")
            text = f"Problem: {problem}\n\nSolution: {text}"

        if not text or not text.strip():
            continue

        tokens = enc.encode_ordinary(text)
        tokens.append(EOT_TOKEN)

        # Collect validation tokens first
        if collecting_val and val_collected < val_tokens:
            val_token_buf.extend(tokens)
            val_doc_offsets.append(len(val_token_buf))
            val_collected += len(tokens)
            if val_collected >= val_tokens:
                collecting_val = False
                flush_shard(val_token_buf, val_doc_offsets, val_dir, 0)
                print(f"  Validation set: {val_collected:,} tokens")
            continue

        # Add to current shard
        token_buf.extend(tokens)
        doc_offsets.append(len(token_buf))
        total_tokens += len(tokens)
        total_docs += 1

        # Flush shard if full
        if len(token_buf) >= shard_size:
            flush_shard(token_buf, doc_offsets, output_dir, shard_idx)
            shard_idx += 1
            token_buf = []
            doc_offsets = [0]

        if total_docs % 10000 == 0:
            print(
                f"  Processed {total_docs:,} docs, {total_tokens:,} tokens, "
                f"{shard_idx} shards"
            )

        if max_docs is not None and total_docs >= max_docs:
            break

    # Flush remaining tokens
    if token_buf:
        flush_shard(token_buf, doc_offsets, output_dir, shard_idx)
        shard_idx += 1

    # Write val shard if not already written
    if collecting_val and val_token_buf:
        flush_shard(val_token_buf, val_doc_offsets, val_dir, 0)

    print(f"\nDone! {source}:")
    print(f"  Total documents: {total_docs:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total shards: {shard_idx}")
    print(f"  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and tokenize pretrain data from HuggingFace"
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=list(SOURCES.keys()),
        help="Data source to download",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for tokenized shards",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100_000_000,
        help="Tokens per shard (default: 100M)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to process (default: all)",
    )
    parser.add_argument(
        "--val-tokens",
        type=int,
        default=10_000_000,
        help="Tokens to hold out for validation (default: 10M)",
    )

    args = parser.parse_args()

    download_and_tokenize(
        source=args.source,
        output_dir=args.output,
        shard_size=args.shard_size,
        max_docs=args.max_docs,
        val_tokens=args.val_tokens,
    )


if __name__ == "__main__":
    main()
