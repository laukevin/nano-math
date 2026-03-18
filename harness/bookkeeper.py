"""Data & model registry with lineage tracking.

Data registry: data/registry.json
Model registry: results/model_registry.json
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.runner import get_git_hash

logger = logging.getLogger(__name__)

DATA_REGISTRY_PATH = Path("data/registry.json")
MODEL_REGISTRY_PATH = Path("results/model_registry.json")


def _hash_file(path: str | Path) -> str:
    """SHA256 of a file."""
    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


class DataRegistry:
    """Manages data/registry.json — tracks datasets, eval sets, and mixtures."""

    def __init__(self, path: Path | None = None):
        self.path = path or DATA_REGISTRY_PATH
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"version": "1.0", "datasets": {}, "eval_sets": {}, "mixtures": {}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2) + "\n")

    @property
    def datasets(self) -> dict:
        return self._data.get("datasets", {})

    @property
    def eval_sets(self) -> dict:
        return self._data.get("eval_sets", {})

    @property
    def mixtures(self) -> dict:
        return self._data.get("mixtures", {})

    def register_dataset(
        self,
        name: str,
        dataset_type: str,
        source: str,
        tokenized_path: str,
        total_tokens: int,
        content_hash: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Register a new dataset version. Immutable once created."""
        if name in self.datasets:
            raise ValueError(
                f"Dataset '{name}' already registered. Create a new version instead."
            )
        entry = {
            "type": dataset_type,
            "source": source,
            "tokenized_path": tokenized_path,
            "total_tokens": total_tokens,
            "content_hash": content_hash or "pending",
            "registered_at": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self._data["datasets"][name] = entry
        self.save()
        logger.info("Registered dataset: %s", name)

    def register_mixture(
        self, name: str, components: dict[str, float], description: str = ""
    ) -> None:
        """Register a data mixture."""
        # Validate components reference known datasets
        for dataset_name in components:
            if dataset_name not in self.datasets:
                logger.warning("Mixture '%s' references unknown dataset '%s'", name, dataset_name)
        self._data["mixtures"][name] = {
            "components": components,
            "description": description,
        }
        self.save()

    def get_dataset(self, name: str) -> dict | None:
        return self.datasets.get(name)

    def get_mixture(self, name: str) -> dict | None:
        return self.mixtures.get(name)

    def get_mixture_datasets(self, mixture_name: str) -> list[str]:
        """Get dataset names used in a mixture."""
        mixture = self.get_mixture(mixture_name)
        if not mixture:
            return []
        return list(mixture["components"].keys())

    def verify_checksums(self) -> list[str]:
        """Check that all registered datasets still match their content hashes."""
        warnings = []
        for name, ds in self.datasets.items():
            if ds.get("content_hash", "pending") == "pending":
                continue
            path = ds.get("tokenized_path", "")
            if path and Path(path).exists():
                current = _hash_file(path)
                if current != ds["content_hash"]:
                    warnings.append(
                        f"Dataset '{name}' content hash mismatch: "
                        f"registered={ds['content_hash']}, current={current}"
                    )
        return warnings


class ModelRegistry:
    """Manages results/model_registry.json — tracks all trained models with lineage."""

    def __init__(self, path: Path | None = None):
        self.path = path or MODEL_REGISTRY_PATH
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"version": "1.0", "models": {}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2) + "\n")

    @property
    def models(self) -> dict:
        return self._data.get("models", {})

    def get(self, model_id: str) -> dict | None:
        return self.models.get(model_id)

    def register(
        self,
        model_id: str,
        experiment_id: str,
        stage: str,
        depth: int,
        checkpoint_path: str,
        parent_model: str | None,
        config: Any,
        training_info: dict[str, Any],
        eval_results: dict[str, Any],
    ) -> None:
        """Register a trained model with full provenance."""
        from dataclasses import asdict

        from harness.config import ExperimentConfig

        # Build data info based on stage
        data_info: dict[str, Any] = {}
        if isinstance(config, ExperimentConfig):
            if stage == "pretrain":
                data_info = {
                    "mixture": config.mixture,
                    "token_multiplier": config.token_multiplier,
                }
            elif stage == "sft":
                data_info = {
                    "recipe": config.sft_recipe,
                    "epochs": config.sft_epochs,
                }
            elif stage == "grpo":
                data_info = {
                    "curriculum": config.rl_curriculum,
                    "kl_coeff": config.rl_kl_coeff,
                    "group_size": config.rl_group_size,
                }

        # Build hyperparams
        hyperparams: dict[str, Any] = {}
        if isinstance(config, ExperimentConfig):
            cfg_dict = asdict(config)
            # Extract stage-relevant hyperparams
            if stage == "sft":
                hyperparams = {
                    "lr": config.sft_lr,
                    "max_seq_len": config.sft_max_seq_len,
                    "epochs": config.sft_epochs,
                }
            elif stage == "grpo":
                hyperparams = {
                    "kl_coeff": config.rl_kl_coeff,
                    "group_size": config.rl_group_size,
                }

        entry = {
            "experiment_id": experiment_id,
            "stage": stage,
            "depth": depth,
            "checkpoint_path": checkpoint_path,
            "parent_model": parent_model,
            "data": data_info,
            "hyperparams": hyperparams,
            "training": training_info,
            "eval_results": eval_results,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_hash": get_git_hash(),
        }

        self._data["models"][model_id] = entry
        self.save()
        logger.info("Registered model: %s", model_id)

    def get_lineage(self, model_id: str) -> list[dict]:
        """Walk up the parent chain to get full lineage."""
        chain = []
        current_id = model_id
        visited = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            model = self.get(current_id)
            if not model:
                break
            chain.append({"model_id": current_id, **model})
            current_id = model.get("parent_model")
        return chain

    def get_data_lineage(self, model_id: str) -> dict[str, Any]:
        """Get the full data lineage for a model (what data went in at each stage)."""
        model = self.get(model_id)
        if not model:
            return {}

        if model["stage"] == "pretrain":
            return {"pretrain_data": model.get("data", {})}
        else:
            result = {}
            parent = model.get("parent_model")
            if parent:
                result = self.get_data_lineage(parent)
            result[f"{model['stage']}_data"] = model.get("data", {})
            return result

    def compare(self, model_a: str, model_b: str) -> dict[str, Any]:
        """Compare two models side by side."""
        a = self.get(model_a)
        b = self.get(model_b)
        if not a or not b:
            missing = []
            if not a:
                missing.append(model_a)
            if not b:
                missing.append(model_b)
            return {"error": f"Models not found: {missing}"}

        return {
            "model_a": {"id": model_a, **a},
            "model_b": {"id": model_b, **b},
        }

    def check_invalidations(self, data_registry: DataRegistry | None = None) -> list[str]:
        """Check if any registered results are invalidated by changes."""
        warnings = []
        for model_id, model in self.models.items():
            # Check parent still exists
            parent = model.get("parent_model")
            if parent and parent not in self.models:
                warnings.append(f"Parent '{parent}' of '{model_id}' not in registry")

        if data_registry:
            for model_id, model in self.models.items():
                data = model.get("data", {})
                # Check mixture datasets still match
                for dv in data.get("data_versions", []):
                    ds = data_registry.get_dataset(dv)
                    if not ds:
                        warnings.append(
                            f"Model '{model_id}' references unknown dataset '{dv}'"
                        )

        return warnings

    def list_models(self, stage: str | None = None) -> list[str]:
        """List all model IDs, optionally filtered by stage."""
        if stage is None:
            return list(self.models.keys())
        return [
            mid for mid, m in self.models.items() if m.get("stage") == stage
        ]
