"""SFT training script using HuggingFace TRL.

Loads an HF-format model, fine-tunes on a recipe's data, saves checkpoints.

Usage:
    python -m scripts.train.run_sft \
        --model hf_model/ \
        --recipe sft-concise-cot \
        --epochs 3 \
        --output-dir checkpoints/sft/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# SFT recipe configs — matches spec 04
RECIPE_CONFIGS = {
    "sft-distill-r1": {
        "datasets": [{"id": "deepseek-ai/DeepSeek-R1", "samples": 100_000}],
        "system_prompt": "You are a helpful math assistant. Think step by step.",
        "max_seq_len": 4096,
    },
    "sft-concise-cot": {
        "datasets": [{"id": "meta-math/MetaMathQA", "samples": 100_000}],
        "system_prompt": "Solve the problem step by step. Be concise.",
        "max_seq_len": 2048,
    },
    "sft-kitchen-sink": {
        "datasets": [
            {"id": "nvidia/OpenMathReasoning", "samples": 50_000},
            {"id": "meta-math/MetaMathQA", "samples": 50_000},
            {"id": "AI-MO/NuminaMath-CoT", "samples": 50_000},
            {"id": "microsoft/orca-math-word-problems-200k", "samples": 50_000},
        ],
        "system_prompt": "You are a helpful math assistant. Think step by step.",
        "max_seq_len": 2048,
    },
    "sft-quality": {
        "datasets": [
            {"id": "hendrycks/competition_math", "samples": 7500, "split": "train"},
            {"id": "openai/gsm8k", "samples": 7500, "split": "train"},
            {"id": "AI-MO/NuminaMath-CoT", "samples": 15_000},
        ],
        "system_prompt": "You are a helpful math assistant. Think step by step.",
        "max_seq_len": 2048,
        "epochs_override": 10,
    },
    "sft-progressive": {
        "datasets": [
            {"id": "openai/gsm8k", "samples": 7500, "split": "train"},
            {"id": "meta-math/MetaMathQA", "samples": 30_000},
        ],
        "system_prompt": "You are a helpful math assistant. Think step by step.",
        "max_seq_len": 2048,
    },
}


def load_sft_data(recipe_id: str, max_samples: int | None = None) -> list[dict]:
    """Load and format SFT data for a recipe.

    Returns list of {"messages": [...]} dicts in chat format.
    """
    from datasets import load_dataset

    recipe = RECIPE_CONFIGS[recipe_id]
    all_samples = []

    for ds_config in recipe["datasets"]:
        ds_id = ds_config["id"]
        n_samples = ds_config.get("samples", 10_000)
        split = ds_config.get("split", "train")

        logger.info("Loading %s (split=%s, n=%d)", ds_id, split, n_samples)
        ds = load_dataset(ds_id, split=split, streaming=True)

        count = 0
        for row in ds:
            if count >= n_samples:
                break

            # Extract problem and solution from various dataset formats
            problem = row.get("problem") or row.get("question") or row.get("input", "")
            solution = row.get("solution") or row.get("answer") or row.get("output", "")

            if not problem or not solution:
                continue

            # Ensure answer has \boxed{} format
            if "\\boxed{" not in solution:
                # Try to extract final answer and wrap it
                from scripts.eval.extraction import extract_answer
                ans = extract_answer(solution)
                if ans:
                    solution = solution.rstrip() + f"\n\nThe answer is \\boxed{{{ans}}}"

            sample = {
                "messages": [
                    {"role": "system", "content": recipe["system_prompt"]},
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ]
            }
            all_samples.append(sample)
            count += 1

    if max_samples:
        all_samples = all_samples[:max_samples]

    logger.info("Loaded %d SFT samples for recipe %s", len(all_samples), recipe_id)
    return all_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training with TRL")
    parser.add_argument("--model", required=True, help="Path to HF model directory")
    parser.add_argument("--recipe", required=True, choices=list(RECIPE_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="sft-run")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--parent", help="Parent checkpoint (unused, for CLI compat)")
    parser.add_argument("--depth", type=int, help="Model depth (unused, for CLI compat)")
    parser.add_argument("--max-samples", type=int, help="Limit training samples")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    import os
    os.environ["WANDB_MODE"] = args.wandb_mode

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    recipe = RECIPE_CONFIGS[args.recipe]
    epochs = recipe.get("epochs_override", args.epochs)

    logger.info("Loading model from %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading SFT data for recipe %s", args.recipe)
    train_data = load_sft_data(args.recipe, max_samples=args.max_samples)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        max_seq_length=min(args.max_seq_len, recipe["max_seq_len"]),
        save_strategy="epoch",
        logging_steps=10,
        report_to="wandb" if args.wandb_mode == "online" else "none",
        run_name=args.run_name,
        bf16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training: %d epochs, lr=%s", epochs, args.lr)
    trainer.train()

    # Save final checkpoint in nanochat format
    trainer.save_model(str(output_dir / "hf_final"))
    logger.info("SFT complete. Model saved to %s", output_dir)

    # Convert back to nanochat .pt format
    if args.depth:
        from scripts.train.convert_to_hf import hf_to_nanochat
        hf_to_nanochat(str(output_dir / "hf_final"), str(output_dir / "final.pt"), args.depth)
        logger.info("Converted to nanochat format: %s", output_dir / "final.pt")


if __name__ == "__main__":
    main()
