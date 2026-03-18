"""Thin wrapper around nanochat's chat_sft.py for math-nano experiments.

Translates our ExperimentConfig into nanochat's CLI flags.
nanochat handles everything: model loading, training loop, eval, checkpointing.

Usage:
    python -m scripts.train.run_sft \
        --depth 16 \
        --model-tag pt-m-broad \
        --recipe sft-concise-cot \
        --output-tag sft-m-concise
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT via nanochat")
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--model-tag", required=True, help="Pretrained model tag to load from")
    parser.add_argument("--output-tag", required=True, help="Tag for output checkpoints")
    parser.add_argument("--recipe", default="sft-concise-cot")
    parser.add_argument("--num-iterations", type=int, default=-1, help="-1 = full epoch")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--device-batch-size", type=int, default=None)
    parser.add_argument("--device-type", default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--wandb-mode", default="online")
    # Compat flags (ignored, for harness runner)
    parser.add_argument("--parent", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Build nanochat chat_sft command
    cmd = [
        sys.executable, "-m", "scripts.chat_sft",
        f"--run={args.output_tag}" if args.wandb_mode != "disabled" else "--run=dummy",
        f"--model-tag={args.model_tag}",
        f"--max-seq-len={args.max_seq_len}",
        f"--eval-every={args.eval_every}",
    ]

    if args.num_iterations > 0:
        cmd.append(f"--num-iterations={args.num_iterations}")

    if args.device_batch_size:
        cmd.append(f"--device-batch-size={args.device_batch_size}")

    if args.device_type:
        cmd.append(f"--device-type={args.device_type}")

    logger.info("Running nanochat SFT: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
