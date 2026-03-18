"""Tests for experiment config validation."""

import pytest

from harness.config import ExperimentConfig, validate_config


class TestConfigValidation:
    def test_valid_pretrain_config(self):
        config = ExperimentConfig(
            experiment_id="test-pt",
            stage="pretrain",
            phase="1a",
            depth=10,
            mixture="mix-math-broad",
        )
        errors = validate_config(config)
        assert errors == []

    def test_pretrain_missing_mixture(self):
        config = ExperimentConfig(
            experiment_id="test-pt",
            stage="pretrain",
            phase="1a",
            depth=10,
        )
        errors = validate_config(config)
        assert any("mixture" in e.lower() for e in errors)

    def test_sft_missing_parent(self):
        config = ExperimentConfig(
            experiment_id="test-sft",
            stage="sft",
            phase="2a",
            depth=16,
            sft_recipe="sft-concise-cot",
        )
        errors = validate_config(config)
        assert any("parent" in e.lower() for e in errors)

    def test_sft_missing_recipe(self):
        config = ExperimentConfig(
            experiment_id="test-sft",
            stage="sft",
            phase="2a",
            depth=16,
            parent_checkpoint="some-checkpoint",
        )
        errors = validate_config(config)
        assert any("recipe" in e.lower() for e in errors)

    def test_invalid_depth(self):
        config = ExperimentConfig(
            experiment_id="test-pt",
            stage="pretrain",
            phase="1a",
            depth=7,
            mixture="mix-math-broad",
        )
        errors = validate_config(config)
        assert any("depth" in e.lower() for e in errors)

    def test_valid_sft_config(self):
        config = ExperimentConfig(
            experiment_id="test-sft",
            stage="sft",
            phase="2a",
            depth=16,
            sft_recipe="sft-concise-cot",
            parent_checkpoint="pt-m-broad-final",
        )
        errors = validate_config(config)
        assert errors == []

    def test_grpo_missing_parent(self):
        config = ExperimentConfig(
            experiment_id="test-grpo",
            stage="grpo",
            phase="3a",
            depth=16,
        )
        errors = validate_config(config)
        assert any("parent" in e.lower() for e in errors)
