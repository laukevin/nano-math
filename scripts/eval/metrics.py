"""Eval metrics: pass@k and bootstrap confidence intervals."""

from __future__ import annotations

from math import comb

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k.

    Args:
        n: total samples generated
        c: number of correct samples
        k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def compute_pass_at_k(
    results: list[dict], k_values: list[int] | None = None
) -> dict[str, float]:
    """Compute pass@k for multiple problems.

    Args:
        results: list of {"problem_id": str, "n_samples": int, "n_correct": int}
        k_values: which k values to compute (default: [1, 4, 8])

    Returns:
        {"pass@1": float, "pass@4": float, ...}
    """
    if k_values is None:
        k_values = [1, 4, 8]

    metrics = {}
    for k in k_values:
        per_problem = [pass_at_k(r["n_samples"], r["n_correct"], k) for r in results]
        metrics[f"pass@{k}"] = float(np.mean(per_problem))
    return metrics


def bootstrap_ci(
    per_problem_correct: list[bool],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for accuracy.

    Args:
        per_problem_correct: list of True/False for each problem
        n_bootstrap: number of bootstrap samples
        ci: confidence level
        seed: random seed for reproducibility

    Returns:
        (mean, ci_low, ci_high)
    """
    rng = np.random.RandomState(seed)
    n = len(per_problem_correct)
    scores = np.array(per_problem_correct, dtype=float)
    mean = float(scores.mean())

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        bootstrap_means.append(sample.mean())

    alpha = (1 - ci) / 2
    ci_low = float(np.percentile(bootstrap_means, 100 * alpha))
    ci_high = float(np.percentile(bootstrap_means, 100 * (1 - alpha)))

    return mean, ci_low, ci_high


def is_significant_improvement(
    results_a: list[bool],
    results_b: list[bool],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> tuple[bool, float]:
    """Paired bootstrap test: is B significantly better than A?

    Returns:
        (is_significant, p_value)
    """
    assert len(results_a) == len(results_b)
    rng = np.random.RandomState(seed)
    n = len(results_a)
    a = np.array(results_a, dtype=float)
    b = np.array(results_b, dtype=float)

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diffs.append(b[idx].mean() - a[idx].mean())

    p_value = float(np.mean([d <= 0 for d in diffs]))
    return p_value < 0.05, p_value
