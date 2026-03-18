"""Tests for GRPO reward function."""

import pytest

from scripts.eval.reward import compute_reward


class TestReward:
    def test_correct_boxed(self):
        assert compute_reward("\\boxed{42}", "42") == 1.0

    def test_wrong_boxed(self):
        assert compute_reward("\\boxed{43}", "42") == 0.0

    def test_correct_hash(self):
        assert compute_reward("#### 42", "42") == 1.0

    def test_correct_answer_is(self):
        assert compute_reward("The answer is 42.", "42") == 1.0

    def test_no_answer(self):
        assert compute_reward("I don't know", "42") == 0.0

    def test_empty_output(self):
        assert compute_reward("", "42") == 0.0

    def test_numeric_equivalence(self):
        assert compute_reward("\\boxed{42.0}", "42") == 1.0

    def test_with_commas(self):
        assert compute_reward("\\boxed{1,234}", "1234") == 1.0

    def test_negative_number(self):
        assert compute_reward("\\boxed{-7}", "-7") == 1.0

    def test_correct_but_verbose(self):
        text = "Step 1: blah blah\n" * 100 + "\\boxed{42}"
        assert compute_reward(text, "42") == 1.0


class TestRewardEdgeCases:
    def test_multiple_boxed_takes_last(self):
        assert compute_reward("\\boxed{5} ... \\boxed{42}", "42") == 1.0
