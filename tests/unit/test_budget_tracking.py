"""Tests for budget tracking and cost estimation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.config import ExperimentConfig, PHASE_BUDGETS
from harness.experiment_state import ExperimentState
from harness.runner import estimate_cost, ConfigError, ExperimentRunner


class TestEstimateCost:
    """Test cost estimation logic."""

    def test_pretrain_small(self):
        config = ExperimentConfig(
            experiment_id="test", stage="pretrain", phase="1a",
            depth=12, mixture="mix-math-broad",
        )
        cost = estimate_cost(config)
        assert cost > 0
        # S model pretrain ~4h * $3.50/hr = ~$14
        assert 10 <= cost <= 20

    def test_pretrain_xl(self):
        config = ExperimentConfig(
            experiment_id="test", stage="pretrain", phase="1a",
            depth=24, mixture="mix-math-broad",
        )
        cost = estimate_cost(config)
        # XL pretrain ~20h * $3.50/hr = ~$70
        assert 50 <= cost <= 100

    def test_pretrain_scales_with_token_multiplier(self):
        config_50x = ExperimentConfig(
            experiment_id="test", stage="pretrain", phase="1a",
            depth=12, mixture="mix-math-broad", token_multiplier=50,
        )
        config_100x = ExperimentConfig(
            experiment_id="test", stage="pretrain", phase="1a",
            depth=12, mixture="mix-math-broad", token_multiplier=100,
        )
        cost_50 = estimate_cost(config_50x)
        cost_100 = estimate_cost(config_100x)
        assert cost_100 == pytest.approx(cost_50 * 2, rel=0.01)

    def test_sft_cost(self):
        config = ExperimentConfig(
            experiment_id="test", stage="sft", phase="2a",
            depth=16, sft_recipe="sft-concise-cot",
            parent_checkpoint="pt-m-broad-final",
        )
        cost = estimate_cost(config)
        assert cost > 0
        # SFT ~2.5h * $3.50/hr = ~$8.75
        assert 5 <= cost <= 15

    def test_sft_cost_scales_with_epochs(self):
        config_3 = ExperimentConfig(
            experiment_id="test", stage="sft", phase="2a",
            depth=16, sft_recipe="sft-concise-cot",
            parent_checkpoint="x", sft_epochs=3,
        )
        config_6 = ExperimentConfig(
            experiment_id="test", stage="sft", phase="2a",
            depth=16, sft_recipe="sft-concise-cot",
            parent_checkpoint="x", sft_epochs=6,
        )
        assert estimate_cost(config_6) > estimate_cost(config_3)

    def test_grpo_cost(self):
        config = ExperimentConfig(
            experiment_id="test", stage="grpo", phase="3a",
            depth=16, parent_checkpoint="sft-m-concise-best",
        )
        cost = estimate_cost(config)
        assert cost > 0
        assert 5 <= cost <= 20

    def test_cost_increases_with_depth(self):
        costs = []
        for depth in [10, 12, 16, 20, 24]:
            config = ExperimentConfig(
                experiment_id="test", stage="pretrain", phase="1a",
                depth=depth, mixture="mix-math-broad",
            )
            costs.append(estimate_cost(config))
        # Costs should be monotonically increasing
        for i in range(len(costs) - 1):
            assert costs[i] < costs[i + 1], f"depth cost not increasing: {costs}"


class TestExperimentStateBudget:
    """Test budget tracking in experiment state."""

    def test_initial_budget(self, tmp_path: Path):
        state = ExperimentState(path=tmp_path / "state.json")
        assert state.total_spend_usd == 0.0

    def test_mark_completed_adds_cost(self, tmp_path: Path):
        state = ExperimentState(path=tmp_path / "state.json")
        state.mark_completed("exp-1", cost_usd=14.70)
        assert state.total_spend_usd == 14.70

    def test_cumulative_spend(self, tmp_path: Path):
        state = ExperimentState(path=tmp_path / "state.json")
        state.mark_completed("exp-1", cost_usd=14.70)
        state.mark_completed("exp-2", cost_usd=21.00)
        assert state.total_spend_usd == pytest.approx(35.70)

    def test_spend_persists(self, tmp_path: Path):
        path = tmp_path / "state.json"
        state = ExperimentState(path=path)
        state.mark_completed("exp-1", cost_usd=14.70)
        state.save()

        reloaded = ExperimentState.load(path)
        assert reloaded.total_spend_usd == 14.70

    def test_phase_budgets_exist(self):
        assert "pretrain" in PHASE_BUDGETS
        assert "sft" in PHASE_BUDGETS
        assert "grpo" in PHASE_BUDGETS
        assert all(v > 0 for v in PHASE_BUDGETS.values())

    def test_runner_budget_check(self, tmp_path: Path):
        """Runner should reject experiments that exceed budget."""
        # Create a state file showing most budget spent
        state_path = tmp_path / "results" / "experiment_state.json"
        state_path.parent.mkdir(parents=True)
        state_data = {
            "current_phase": 1,
            "current_wave": 1,
            "completed_experiments": [],
            "running_experiments": [],
            "pending_experiments": [],
            "failed_experiments": [],
            "total_spend_usd": 295.0,  # Almost all of pretrain budget
            "phase_budget_remaining_usd": 5.0,
            "last_gate_status": None,
            "decisions_log": [],
        }
        state_path.write_text(json.dumps(state_data))

        # estimate_cost for a pretrain should be > $5
        config = ExperimentConfig(
            experiment_id="test", stage="pretrain", phase="1a",
            depth=12, mixture="mix-math-broad",
        )
        cost = estimate_cost(config)
        assert cost > 5.0  # Should exceed remaining budget


class TestExperimentStateTracking:
    """Test experiment state transitions."""

    def test_mark_running(self, tmp_path: Path):
        state = ExperimentState(path=tmp_path / "state.json")
        state.add_pending(["exp-1", "exp-2"])
        state.mark_running("exp-1")
        assert "exp-1" in state.running_experiments
        assert "exp-1" not in state.pending_experiments

    def test_mark_failed(self, tmp_path: Path):
        state = ExperimentState(path=tmp_path / "state.json")
        state.mark_running("exp-1")
        state.mark_failed("exp-1")
        assert "exp-1" not in state.running_experiments
        assert "exp-1" in state._data["failed_experiments"]

    def test_advance_wave(self, tmp_path: Path):
        state = ExperimentState(path=tmp_path / "state.json")
        assert state.current_wave == 1
        state.advance_wave()
        assert state.current_wave == 2

    def test_advance_phase_resets_wave(self, tmp_path: Path):
        state = ExperimentState(path=tmp_path / "state.json")
        state.advance_wave()
        state.advance_wave()
        assert state.current_wave == 3
        state.advance_phase()
        assert state.current_phase == 2
        assert state.current_wave == 1

    def test_summary(self, tmp_path: Path):
        state = ExperimentState(path=tmp_path / "state.json")
        state.mark_completed("exp-1", cost_usd=10.0)
        state.mark_running("exp-2")
        summary = state.summary()
        assert "Phase: 1" in summary
        assert "Completed: 1" in summary
        assert "Running: 1" in summary
        assert "$10.00" in summary
