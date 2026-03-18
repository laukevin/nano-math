"""Tests for bootstrap confidence intervals and significance testing."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.eval.metrics import bootstrap_ci, is_significant_improvement


class TestBootstrapCI:
    """Tests for bootstrap_ci function."""

    def test_all_correct(self):
        """All-correct results should give mean=1.0, tight CI."""
        results = [True] * 100
        mean, ci_low, ci_high = bootstrap_ci(results)
        assert mean == 1.0
        assert ci_low == 1.0
        assert ci_high == 1.0

    def test_all_wrong(self):
        """All-wrong results should give mean=0.0, tight CI."""
        results = [False] * 100
        mean, ci_low, ci_high = bootstrap_ci(results)
        assert mean == 0.0
        assert ci_low == 0.0
        assert ci_high == 0.0

    def test_mixed_results(self):
        """Mixed results should give mean between 0 and 1, CI contains mean."""
        results = [True] * 30 + [False] * 70
        mean, ci_low, ci_high = bootstrap_ci(results)
        assert mean == pytest.approx(0.3)
        assert ci_low <= mean <= ci_high
        # CI should be reasonable for 100 samples at 30%
        assert ci_low > 0.15
        assert ci_high < 0.45

    def test_ci_contains_mean(self):
        """The CI should always contain the sample mean."""
        for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            n = 200
            results = [True] * int(n * frac) + [False] * (n - int(n * frac))
            mean, ci_low, ci_high = bootstrap_ci(results)
            assert ci_low <= mean <= ci_high

    def test_seed_reproducibility(self):
        """Same seed should give same results."""
        results = [True] * 50 + [False] * 50
        r1 = bootstrap_ci(results, seed=42)
        r2 = bootstrap_ci(results, seed=42)
        assert r1 == r2

    def test_different_seeds(self):
        """Different seeds may give different CIs."""
        results = [True] * 50 + [False] * 50
        r1 = bootstrap_ci(results, seed=42)
        r2 = bootstrap_ci(results, seed=123)
        # Means should be identical (data is the same)
        assert r1[0] == r2[0]
        # CIs may differ (different bootstrap samples)
        # Not guaranteed to differ but very likely with 50/50 split

    def test_ci_width_decreases_with_more_problems(self):
        """Larger sample sizes should give tighter CIs."""
        small = [True] * 15 + [False] * 35  # 50 problems, 30%
        large = [True] * 150 + [False] * 350  # 500 problems, 30%

        _, lo_s, hi_s = bootstrap_ci(small, seed=42)
        _, lo_l, hi_l = bootstrap_ci(large, seed=42)

        width_small = hi_s - lo_s
        width_large = hi_l - lo_l
        assert width_large < width_small

    def test_95_confidence_level(self):
        """Default CI should be 95%."""
        np.random.seed(0)
        true_p = 0.4
        n = 200
        n_trials = 200
        covers = 0

        for _ in range(n_trials):
            results = list(np.random.random(n) < true_p)
            _, ci_low, ci_high = bootstrap_ci(results, n_bootstrap=2000)
            if ci_low <= true_p <= ci_high:
                covers += 1

        # Should cover ~95% of the time (allow some slack)
        coverage = covers / n_trials
        assert coverage > 0.85, f"Coverage {coverage:.2f} too low"

    def test_custom_confidence_level(self):
        """Wider CI for higher confidence."""
        results = [True] * 40 + [False] * 60
        _, lo_95, hi_95 = bootstrap_ci(results, ci=0.95, seed=42)
        _, lo_99, hi_99 = bootstrap_ci(results, ci=0.99, seed=42)

        width_95 = hi_95 - lo_95
        width_99 = hi_99 - lo_99
        assert width_99 >= width_95

    def test_small_sample(self):
        """Should work with very small samples (though CI will be wide)."""
        results = [True, False, True]
        mean, ci_low, ci_high = bootstrap_ci(results)
        assert mean == pytest.approx(2 / 3, abs=1e-10)
        assert 0.0 <= ci_low <= ci_high <= 1.0


class TestIsSignificantImprovement:
    """Tests for paired bootstrap significance testing."""

    def test_identical_results(self):
        """Identical results should not be significant."""
        results = [True] * 30 + [False] * 70
        significant, p_value = is_significant_improvement(results, results)
        assert not significant
        assert p_value > 0.05

    def test_clearly_better(self):
        """Clearly better model B should be significant."""
        results_a = [True] * 20 + [False] * 80
        results_b = [True] * 60 + [False] * 40
        significant, p_value = is_significant_improvement(results_a, results_b)
        assert significant
        assert p_value < 0.05

    def test_clearly_worse(self):
        """Clearly worse model B should not be significant improvement."""
        results_a = [True] * 60 + [False] * 40
        results_b = [True] * 20 + [False] * 80
        significant, p_value = is_significant_improvement(results_a, results_b)
        assert not significant
        assert p_value > 0.95  # B is worse, so p should be high

    def test_seed_reproducibility(self):
        """Same seed should give same p-value."""
        a = [True] * 30 + [False] * 70
        b = [True] * 40 + [False] * 60
        r1 = is_significant_improvement(a, b, seed=42)
        r2 = is_significant_improvement(a, b, seed=42)
        assert r1 == r2

    def test_requires_same_length(self):
        """Should raise if inputs have different lengths."""
        with pytest.raises(AssertionError):
            is_significant_improvement([True, False], [True])

    def test_marginal_improvement(self):
        """Small improvement should not be significant with few problems."""
        # 50 problems, 1 more correct — unlikely significant
        a = [True] * 24 + [False] * 26
        b = [True] * 25 + [False] * 25
        significant, p_value = is_significant_improvement(a, b, seed=42)
        assert not significant

    def test_p_value_bounds(self):
        """p-value should be between 0 and 1."""
        a = [True] * 30 + [False] * 70
        b = [True] * 35 + [False] * 65
        _, p_value = is_significant_improvement(a, b)
        assert 0.0 <= p_value <= 1.0
