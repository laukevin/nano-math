"""Reward functions for GRPO training."""

from __future__ import annotations

from scripts.eval.extraction import extract_answer


def compute_reward(model_output: str, ground_truth: str) -> float:
    """Binary reward: 1.0 if correct, 0.0 if wrong.

    Extracts the answer from model output and compares to ground truth.
    """
    predicted = extract_answer(model_output)
    if predicted is None:
        return 0.0
    expected = extract_answer(f"\\boxed{{{ground_truth}}}") or ground_truth.strip()
    return 1.0 if predicted == expected else 0.0
