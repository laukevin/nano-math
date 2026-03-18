"""Core evaluation logic: score completions, build result JSON."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np

from scripts.eval.data import (
    BOOTSTRAP_SEED,
    EVAL_VERSION,
    MAX_NEW_TOKENS,
    format_eval_prompt,
)
from scripts.eval.extraction import extract_answer, normalize_answer
from scripts.eval.inference import generate_batch
from scripts.eval.metrics import bootstrap_ci, compute_pass_at_k, pass_at_k


def evaluate_completions(
    outputs: list[str] | list[list[str]],
    ground_truths: list[str],
    problem_ids: list[str],
    n_samples: int,
) -> dict:
    """Evaluate model outputs against ground truths.

    Args:
        outputs: If n_samples==1, list[str] (one per problem).
                 If n_samples>1, list[list[str]] (n_samples per problem).
        ground_truths: Normalized answer strings, one per problem.
        problem_ids: Unique ID per problem.
        n_samples: Number of samples per problem.

    Returns:
        Result dict matching spec 07 format.
    """
    n_problems = len(ground_truths)
    extraction_failures = 0

    if n_samples == 1:
        per_problem = []
        greedy_correct: list[bool] = []

        for pid, output, gt in zip(problem_ids, outputs, ground_truths):
            predicted = extract_answer(output)  # type: ignore[arg-type]
            if predicted is None:
                extraction_failures += 1
                correct = False
            else:
                correct = predicted == gt
            greedy_correct.append(correct)
            per_problem.append(
                {"id": pid, "correct_samples": int(correct), "total_samples": 1}
            )

        mean, ci_low, ci_high = bootstrap_ci(
            greedy_correct, seed=BOOTSTRAP_SEED
        )

        return {
            "n_problems": n_problems,
            "pass_at_1_greedy": mean,
            "pass_at_1_greedy_ci95": [ci_low, ci_high],
            "extraction_failures": extraction_failures,
            "extraction_failure_rate": (
                extraction_failures / n_problems if n_problems else 0.0
            ),
            "per_problem": per_problem,
        }

    # Sampled mode
    per_problem = []
    pass_k_inputs: list[dict] = []

    for idx, (pid, gt) in enumerate(zip(problem_ids, ground_truths)):
        sample_list = outputs[idx]  # type: ignore[index]
        n_correct = 0
        for output in sample_list:
            predicted = extract_answer(output)
            if predicted is None:
                extraction_failures += 1
            elif predicted == gt:
                n_correct += 1

        per_problem.append(
            {
                "id": pid,
                "correct_samples": n_correct,
                "total_samples": n_samples,
            }
        )
        pass_k_inputs.append(
            {
                "problem_id": pid,
                "n_samples": n_samples,
                "n_correct": n_correct,
            }
        )

    k_values = [k for k in [1, 4, 8] if k <= n_samples]
    pass_k_metrics = compute_pass_at_k(pass_k_inputs, k_values)

    result: dict = {
        "n_problems": n_problems,
        "n_samples_per_problem": n_samples,
        "extraction_failures": extraction_failures,
        "extraction_failure_rate": (
            extraction_failures / (n_problems * n_samples)
            if n_problems
            else 0.0
        ),
        "per_problem": per_problem,
    }

    for k in k_values:
        per_problem_scores = [
            pass_at_k(r["n_samples"], r["n_correct"], k)
            for r in pass_k_inputs
        ]
        result[f"pass_at_{k}_sampled"] = pass_k_metrics[f"pass@{k}"]

        _, ci_l, ci_h = bootstrap_ci(
            per_problem_scores, seed=BOOTSTRAP_SEED  # type: ignore[arg-type]
        )
        result[f"pass_at_{k}_sampled_ci95"] = [ci_l, ci_h]

    return result


def run_dataset_eval(
    model,
    tokenizer,
    problems: list[dict],
    dataset_name: str,
    n_samples: int = 1,
    temperature: float = 0.0,
    device: str = "cpu",
    batch_size: int = 32,
) -> dict:
    """Run eval on a single dataset: generate + evaluate."""
    prompts = [format_eval_prompt(p["problem"]) for p in problems]
    ground_truths = [normalize_answer(str(p["answer"])) for p in problems]
    problem_ids = [
        p.get("id", f"{dataset_name}_{i:04d}") for i, p in enumerate(problems)
    ]

    start_time = time.time()

    if n_samples == 1:
        outputs, gen_stats = generate_batch(
            model, tokenizer, prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            batch_size=batch_size,
            device=device,
        )
    else:
        expanded = [p for p in prompts for _ in range(n_samples)]
        flat_outputs, gen_stats = generate_batch(
            model, tokenizer, expanded,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            batch_size=batch_size,
            device=device,
        )
        outputs = [
            flat_outputs[i * n_samples : (i + 1) * n_samples]
            for i in range(len(prompts))
        ]

    elapsed = time.time() - start_time
    n_total_gens = len(prompts) * n_samples

    result = evaluate_completions(outputs, ground_truths, problem_ids, n_samples)

    result["avg_output_tokens"] = (
        gen_stats["total_tokens"] / n_total_gens if n_total_gens else 0
    )
    result["avg_inference_ms"] = (
        (elapsed * 1000) / n_total_gens if n_total_gens else 0
    )

    return result


def build_output_json(
    checkpoint: str,
    depth: int,
    model_params: int,
    suite: str,
    n_samples: int,
    temperature: float,
    dataset_results: dict[str, dict],
    manifest_sha: str,
    experiment_id: str = "",
    stage: str = "",
) -> dict:
    """Build the full eval output JSON per spec 07."""
    greedy_scores = []
    sampled_scores = []
    for ds_result in dataset_results.values():
        if "pass_at_1_greedy" in ds_result:
            greedy_scores.append(ds_result["pass_at_1_greedy"])
        if "pass_at_1_sampled" in ds_result:
            sampled_scores.append(ds_result["pass_at_1_sampled"])

    aggregate: dict = {}
    if greedy_scores:
        aggregate["avg_pass_at_1_greedy"] = float(np.mean(greedy_scores))
    if sampled_scores:
        aggregate["avg_pass_at_1_sampled"] = float(np.mean(sampled_scores))

    return {
        "eval_version": EVAL_VERSION,
        "checkpoint": checkpoint,
        "model_depth": depth,
        "model_params": model_params,
        "stage": stage,
        "experiment_id": experiment_id,
        "eval_suite": suite,
        "n_samples_per_problem": n_samples,
        "temperature": temperature,
        "max_new_tokens": MAX_NEW_TOKENS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_data_manifest_sha": manifest_sha,
        "results": dataset_results,
        "aggregate": aggregate,
    }
