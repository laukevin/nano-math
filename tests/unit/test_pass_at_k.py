"""Tests for pass@k estimator."""

import pytest

from scripts.eval.metrics import pass_at_k, compute_pass_at_k


class TestPassAtK:
    def test_all_correct(self):
        assert pass_at_k(n=8, c=8, k=1) == 1.0

    def test_none_correct(self):
        assert pass_at_k(n=8, c=0, k=1) == 0.0

    def test_half_correct_pass1(self):
        assert pass_at_k(n=8, c=4, k=1) == 0.5

    def test_pass_at_k_monotonic(self):
        for c in range(1, 8):
            scores = [pass_at_k(n=8, c=c, k=k) for k in [1, 2, 4, 8]]
            assert scores == sorted(scores), f"Not monotonic for c={c}: {scores}"

    def test_pass_at_1_equals_c_over_n(self):
        for n in [8, 16, 32]:
            for c in range(n + 1):
                assert abs(pass_at_k(n, c, 1) - c / n) < 1e-10

    def test_pass_at_n_is_0_or_1(self):
        assert pass_at_k(n=8, c=1, k=8) == 1.0
        assert pass_at_k(n=8, c=0, k=8) == 0.0

    def test_known_value(self):
        # pass@2 with n=8, c=3: 1 - C(5,2)/C(8,2) = 1 - 10/28 = 18/28
        expected = 1.0 - 10.0 / 28.0
        assert abs(pass_at_k(n=8, c=3, k=2) - expected) < 1e-10


class TestComputePassAtK:
    def test_single_problem(self):
        results = [{"problem_id": "p1", "n_samples": 8, "n_correct": 4}]
        metrics = compute_pass_at_k(results, k_values=[1, 4, 8])
        assert metrics["pass@1"] == 0.5

    def test_multiple_problems(self):
        results = [
            {"problem_id": "p1", "n_samples": 8, "n_correct": 8},
            {"problem_id": "p2", "n_samples": 8, "n_correct": 0},
        ]
        metrics = compute_pass_at_k(results, k_values=[1])
        assert metrics["pass@1"] == 0.5
