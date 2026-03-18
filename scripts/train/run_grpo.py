"""GRPO training script using HuggingFace TRL.

Loads an HF-format model (post-SFT), runs Group Relative Policy Optimization
with curriculum-based problem sets.

Usage:
    python -m scripts.train.run_grpo \
        --model hf_model/ \
        --curriculum easy-to-hard \
        --output-dir checkpoints/grpo/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# Curriculum strategies — matches spec 06
CURRICULUM_CONFIGS = {
    "easy-to-hard": {
        "stages": [
            {"dataset": "openai/gsm8k", "split": "train", "advance_at": 0.40, "max_steps": 500},
            {"dataset": "AI-MO/NuminaMath-CoT", "filter": "amc", "advance_at": 0.15, "max_steps": 500},
            {"dataset": "AI-MO/aimo-validation-aime", "advance_at": None, "max_steps": 1000},
        ],
    },
    "hard-only": {
        "stages": [
            {"dataset": "hendrycks/competition_math", "split": "test", "advance_at": None, "max_steps": 2000},
        ],
    },
    "interleaved": {
        "stages": [
            {
                "mixed": [
                    {"dataset": "openai/gsm8k", "split": "train", "weight": 0.6},
                    {"dataset": "AI-MO/NuminaMath-CoT", "filter": "amc", "weight": 0.3},
                    {"dataset": "AI-MO/aimo-validation-aime", "weight": 0.1},
                ],
                "advance_at": None,
                "max_steps": 2000,
            },
        ],
    },
    "reverse": {
        "stages": [
            {"dataset": "AI-MO/aimo-validation-aime", "advance_at": None, "max_steps": 500},
            {"dataset": "openai/gsm8k", "split": "train", "advance_at": 0.40, "max_steps": 500},
            {"dataset": "AI-MO/aimo-validation-aime", "advance_at": None, "max_steps": 1000},
        ],
    },
}


def load_rl_problems(dataset_id: str, split: str = "train", max_problems: int = 5000) -> list[dict]:
    """Load problems for RL training. Returns list of {"prompt": str, "answer": str}."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split, streaming=True)
    problems = []
    for row in ds:
        if len(problems) >= max_problems:
            break
        problem = row.get("problem") or row.get("question") or row.get("input", "")
        answer = row.get("answer") or row.get("solution") or row.get("output", "")
        if problem and answer:
            # Normalize answer for reward computation
            from scripts.eval.extraction import extract_answer
            normalized = extract_answer(answer) or answer.strip()
            problems.append({"prompt": problem, "answer": normalized})
    return problems


def make_reward_fn(problems_by_prompt: dict[str, str]):
    """Create a reward function for GRPO that checks answer correctness."""
    from scripts.eval.extraction import extract_answer, normalize_answer

    def reward_fn(completions: list[str], prompts: list[str]) -> list[float]:
        rewards = []
        for completion, prompt in zip(completions, prompts):
            ground_truth = problems_by_prompt.get(prompt, "")
            predicted = extract_answer(completion)
            if predicted is None:
                rewards.append(0.0)
            elif predicted == normalize_answer(ground_truth):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    return reward_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training with TRL")
    parser.add_argument("--model", required=True, help="Path to HF model directory")
    parser.add_argument("--curriculum", default="easy-to-hard", choices=list(CURRICULUM_CONFIGS.keys()))
    parser.add_argument("--kl-coeff", type=float, default=0.05)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="grpo-run")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--parent", help="Parent checkpoint (unused, for CLI compat)")
    parser.add_argument("--depth", type=int, help="Model depth (unused, for CLI compat)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    import os
    os.environ["WANDB_MODE"] = args.wandb_mode

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    logger.info("Loading model from %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    curriculum = CURRICULUM_CONFIGS[args.curriculum]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 0

    for stage_idx, stage_config in enumerate(curriculum["stages"]):
        logger.info("=== Curriculum stage %d ===", stage_idx + 1)
        max_steps = stage_config.get("max_steps", 500)

        # Load problems for this stage
        if "mixed" in stage_config:
            problems = []
            for src in stage_config["mixed"]:
                subset = load_rl_problems(src["dataset"], src.get("split", "train"))
                problems.extend(subset)
        else:
            problems = load_rl_problems(
                stage_config["dataset"],
                stage_config.get("split", "train"),
            )

        if not problems:
            logger.warning("No problems loaded for stage %d, skipping", stage_idx + 1)
            continue

        # Build prompt→answer mapping for reward
        prompts_answers = {p["prompt"]: p["answer"] for p in problems}
        reward_fn = make_reward_fn(prompts_answers)

        # Format prompts for GRPO
        prompts = [
            f"Solve the following math problem step by step. "
            f"Put your final answer in \\boxed{{}}.\n\n"
            f"Problem: {p['prompt']}\n\nSolution:"
            for p in problems
        ]

        grpo_config = GRPOConfig(
            output_dir=str(output_dir / f"stage_{stage_idx}"),
            max_steps=max_steps,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            num_generations=args.group_size,
            max_completion_length=args.max_new_tokens,
            logging_steps=10,
            save_steps=100,
            report_to="wandb" if args.wandb_mode == "online" else "none",
            run_name=f"{args.run_name}-stage{stage_idx}",
            bf16=torch.cuda.is_available(),
            beta=args.kl_coeff,
        )

        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            reward_funcs=reward_fn,
            processing_class=tokenizer,
        )

        logger.info("Starting GRPO stage %d: %d problems, max %d steps", stage_idx + 1, len(problems), max_steps)
        trainer.train()
        total_steps += max_steps

        # Save stage checkpoint
        trainer.save_model(str(output_dir / f"stage_{stage_idx}_final"))

    # Save final model
    logger.info("GRPO complete (%d total steps). Saving final model.", total_steps)
    if 'trainer' in locals():
        trainer.save_model(str(output_dir / "hf_final"))

    # Convert back to nanochat format
    if args.depth:
        from scripts.train.convert_to_hf import hf_to_nanochat
        hf_to_nanochat(str(output_dir / "hf_final"), str(output_dir / "final.pt"), args.depth)
        logger.info("Converted to nanochat format: %s", output_dir / "final.pt")


if __name__ == "__main__":
    main()
