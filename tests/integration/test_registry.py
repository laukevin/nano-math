"""Integration tests for data & model registries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.bookkeeper import DataRegistry, ModelRegistry
from harness.config import ExperimentConfig


class TestDataRegistry:
    """Test data registry operations."""

    def test_create_empty_registry(self, tmp_path: Path):
        reg = DataRegistry(path=tmp_path / "registry.json")
        assert reg.datasets == {}
        assert reg.mixtures == {}

    def test_register_dataset(self, tmp_path: Path):
        reg = DataRegistry(path=tmp_path / "registry.json")
        reg.register_dataset(
            name="fineweb-edu-v1",
            dataset_type="pretrain",
            source="HuggingFaceFW/fineweb-edu",
            tokenized_path="data/tokenized/fineweb-edu/",
            total_tokens=5_200_000_000,
            content_hash="sha256:abc123",
        )
        assert "fineweb-edu-v1" in reg.datasets
        ds = reg.get_dataset("fineweb-edu-v1")
        assert ds["total_tokens"] == 5_200_000_000
        assert ds["content_hash"] == "sha256:abc123"

    def test_immutable_datasets(self, tmp_path: Path):
        reg = DataRegistry(path=tmp_path / "registry.json")
        reg.register_dataset(
            name="test-v1",
            dataset_type="pretrain",
            source="test",
            tokenized_path="data/test/",
            total_tokens=100,
        )
        with pytest.raises(ValueError, match="already registered"):
            reg.register_dataset(
                name="test-v1",
                dataset_type="pretrain",
                source="test",
                tokenized_path="data/test/",
                total_tokens=200,
            )

    def test_register_mixture(self, tmp_path: Path):
        reg = DataRegistry(path=tmp_path / "registry.json")
        reg.register_dataset(
            name="ds-a", dataset_type="pretrain", source="a",
            tokenized_path="a/", total_tokens=100,
        )
        reg.register_dataset(
            name="ds-b", dataset_type="pretrain", source="b",
            tokenized_path="b/", total_tokens=200,
        )
        reg.register_mixture(
            "mix-ab",
            components={"ds-a": 0.6, "ds-b": 0.4},
            description="Test mixture",
        )
        mix = reg.get_mixture("mix-ab")
        assert mix is not None
        assert mix["components"]["ds-a"] == 0.6

    def test_get_mixture_datasets(self, tmp_path: Path):
        reg = DataRegistry(path=tmp_path / "registry.json")
        reg.register_dataset(
            name="ds-a", dataset_type="pretrain", source="a",
            tokenized_path="a/", total_tokens=100,
        )
        reg.register_mixture("mix-a", components={"ds-a": 1.0})
        assert reg.get_mixture_datasets("mix-a") == ["ds-a"]

    def test_persistence(self, tmp_path: Path):
        path = tmp_path / "registry.json"
        reg = DataRegistry(path=path)
        reg.register_dataset(
            name="test-v1", dataset_type="pretrain", source="test",
            tokenized_path="data/", total_tokens=100,
        )

        # Reload
        reg2 = DataRegistry(path=path)
        assert "test-v1" in reg2.datasets

    def test_missing_dataset_returns_none(self, tmp_path: Path):
        reg = DataRegistry(path=tmp_path / "registry.json")
        assert reg.get_dataset("nonexistent") is None


class TestModelRegistry:
    """Test model registry and lineage tracking."""

    def _make_config(self, **kwargs) -> ExperimentConfig:
        defaults = {
            "experiment_id": "test",
            "stage": "pretrain",
            "phase": "1a",
            "depth": 12,
            "mixture": "mix-math-broad",
        }
        defaults.update(kwargs)
        return ExperimentConfig(**defaults)

    def test_create_empty_registry(self, tmp_path: Path):
        reg = ModelRegistry(path=tmp_path / "model_registry.json")
        assert reg.models == {}

    def test_register_model(self, tmp_path: Path):
        reg = ModelRegistry(path=tmp_path / "model_registry.json")
        config = self._make_config()
        reg.register(
            model_id="pt-s-broad-best",
            experiment_id="pt-s-broad",
            stage="pretrain",
            depth=12,
            checkpoint_path="/ckpt/final.pt",
            parent_model=None,
            config=config,
            training_info={"wall_clock_hours": 4.2, "cost_usd": 14.70},
            eval_results={"gsm8k_pass1_greedy": 0.02},
        )
        model = reg.get("pt-s-broad-best")
        assert model is not None
        assert model["depth"] == 12
        assert model["eval_results"]["gsm8k_pass1_greedy"] == 0.02

    def test_lineage_chain(self, tmp_path: Path):
        reg = ModelRegistry(path=tmp_path / "model_registry.json")

        # Register pretrain
        reg.register(
            model_id="pt-m-best",
            experiment_id="pt-m",
            stage="pretrain",
            depth=16,
            checkpoint_path="/ckpt/pt.pt",
            parent_model=None,
            config=self._make_config(depth=16),
            training_info={},
            eval_results={"gsm8k_pass1_greedy": 0.02},
        )

        # Register SFT with pretrain as parent
        reg.register(
            model_id="sft-m-best",
            experiment_id="sft-m",
            stage="sft",
            depth=16,
            checkpoint_path="/ckpt/sft.pt",
            parent_model="pt-m-best",
            config=self._make_config(
                stage="sft", depth=16,
                sft_recipe="sft-concise-cot",
                parent_checkpoint="pt-m-best",
            ),
            training_info={},
            eval_results={"gsm8k_pass1_greedy": 0.34},
        )

        # Register GRPO with SFT as parent
        reg.register(
            model_id="grpo-m-best",
            experiment_id="grpo-m",
            stage="grpo",
            depth=16,
            checkpoint_path="/ckpt/grpo.pt",
            parent_model="sft-m-best",
            config=self._make_config(
                stage="grpo", depth=16,
                parent_checkpoint="sft-m-best",
            ),
            training_info={},
            eval_results={"gsm8k_pass1_greedy": 0.42},
        )

        # Check lineage
        lineage = reg.get_lineage("grpo-m-best")
        assert len(lineage) == 3
        assert lineage[0]["model_id"] == "grpo-m-best"
        assert lineage[1]["model_id"] == "sft-m-best"
        assert lineage[2]["model_id"] == "pt-m-best"

    def test_data_lineage(self, tmp_path: Path):
        reg = ModelRegistry(path=tmp_path / "model_registry.json")

        reg.register(
            model_id="pt-m-best",
            experiment_id="pt-m",
            stage="pretrain",
            depth=16,
            checkpoint_path="/ckpt/pt.pt",
            parent_model=None,
            config=self._make_config(depth=16),
            training_info={},
            eval_results={},
        )
        reg.register(
            model_id="sft-m-best",
            experiment_id="sft-m",
            stage="sft",
            depth=16,
            checkpoint_path="/ckpt/sft.pt",
            parent_model="pt-m-best",
            config=self._make_config(
                stage="sft", depth=16,
                sft_recipe="sft-concise-cot",
                parent_checkpoint="pt-m-best",
            ),
            training_info={},
            eval_results={},
        )

        data_lineage = reg.get_data_lineage("sft-m-best")
        assert "pretrain_data" in data_lineage
        assert "sft_data" in data_lineage

    def test_compare_models(self, tmp_path: Path):
        reg = ModelRegistry(path=tmp_path / "model_registry.json")
        config = self._make_config()

        reg.register(
            model_id="model-a", experiment_id="a", stage="pretrain",
            depth=12, checkpoint_path="/a.pt", parent_model=None,
            config=config, training_info={},
            eval_results={"gsm8k_pass1_greedy": 0.10},
        )
        reg.register(
            model_id="model-b", experiment_id="b", stage="pretrain",
            depth=12, checkpoint_path="/b.pt", parent_model=None,
            config=config, training_info={},
            eval_results={"gsm8k_pass1_greedy": 0.20},
        )

        comparison = reg.compare("model-a", "model-b")
        assert "model_a" in comparison
        assert "model_b" in comparison
        assert comparison["model_a"]["eval_results"]["gsm8k_pass1_greedy"] == 0.10

    def test_compare_missing_model(self, tmp_path: Path):
        reg = ModelRegistry(path=tmp_path / "model_registry.json")
        comparison = reg.compare("missing-a", "missing-b")
        assert "error" in comparison

    def test_check_invalidations_missing_parent(self, tmp_path: Path):
        reg = ModelRegistry(path=tmp_path / "model_registry.json")
        # Manually insert a model with a non-existent parent
        reg._data["models"]["orphan"] = {
            "experiment_id": "orphan",
            "stage": "sft",
            "parent_model": "nonexistent-parent",
            "data": {},
        }
        warnings = reg.check_invalidations()
        assert any("nonexistent-parent" in w for w in warnings)

    def test_list_models_by_stage(self, tmp_path: Path):
        reg = ModelRegistry(path=tmp_path / "model_registry.json")
        config = self._make_config()

        reg.register(
            model_id="pt-1", experiment_id="pt-1", stage="pretrain",
            depth=12, checkpoint_path="/a.pt", parent_model=None,
            config=config, training_info={}, eval_results={},
        )
        reg.register(
            model_id="sft-1", experiment_id="sft-1", stage="sft",
            depth=12, checkpoint_path="/b.pt", parent_model="pt-1",
            config=self._make_config(
                stage="sft", sft_recipe="sft-concise-cot",
                parent_checkpoint="pt-1",
            ),
            training_info={}, eval_results={},
        )

        assert reg.list_models(stage="pretrain") == ["pt-1"]
        assert reg.list_models(stage="sft") == ["sft-1"]
        assert len(reg.list_models()) == 2

    def test_persistence(self, tmp_path: Path):
        path = tmp_path / "model_registry.json"
        reg = ModelRegistry(path=path)
        config = self._make_config()
        reg.register(
            model_id="test-model", experiment_id="test", stage="pretrain",
            depth=12, checkpoint_path="/x.pt", parent_model=None,
            config=config, training_info={}, eval_results={},
        )

        reg2 = ModelRegistry(path=path)
        assert "test-model" in reg2.models
