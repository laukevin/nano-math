"""Thin wrapper around nanochat's chat_rl.py for math-nano experiments.

Translates our ExperimentConfig into nanochat's CLI flags.
nanochat handles everything: model loading, GRPO, generation, eval, checkpointing.

Usage:
    python -m scripts.train.run_grpo \
        --depth 16 \
        --model-tag sft-m-concise \
        --output-tag grpo-m-easy2hard
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO via nanochat")
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--model-tag", required=True, help="SFT model tag to load from")
    parser.add_argument("--output-tag", required=True, help="Tag for output checkpoints")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=16, help="Samples per question for GRPO")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device-batch-size", type=int, default=8)
    parser.add_argument("--device-type", default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--eval-every", type=int, default=60)
    parser.add_argument("--wandb-mode", default="online")
    # Compat flags (ignored, for harness runner)
    parser.add_argument("--parent", default=None)
    parser.add_argument("--curriculum", default="easy-to-hard")
    parser.add_argument("--kl-coeff", type=float, default=0.05)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Build nanochat chat_rl command
    cmd = [
        sys.executable, "-m", "scripts.chat_rl",
        f"--run={args.output_tag}" if args.wandb_mode != "disabled" else "--run=dummy",
        f"--model-tag={args.model_tag}",
        f"--num-epochs={args.num_epochs}",
        f"--num-samples={args.num_samples}",
        f"--max-new-tokens={args.max_new_tokens}",
        f"--device-batch-size={args.device_batch_size}",
        f"--eval-every={args.eval_every}",
    ]

    if args.device_type:
        cmd.append(f"--device-type={args.device_type}")

    logger.info("Running nanochat GRPO: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
