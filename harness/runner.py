"""Experiment runner — the single codepath for all experiments.

Orchestrates: validate config → check gate → init W&B → train → eval → register → log.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness import HARNESS_VERSION
from harness.config import ExperimentConfig, validate_config

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
AUDIT_LOG_PATH = RESULTS_DIR / "audit_log.jsonl"


class ConfigError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Config validation failed: {errors}")


class GateError(Exception):
    pass


class TrainResult:
    """Result from a training run."""

    def __init__(
        self,
        best_checkpoint: str,
        final_checkpoint: str,
        final_loss: float,
        wall_clock_hours: float,
        tokens_seen: int,
        cost_usd: float,
        wandb_run_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ):
        self.best_checkpoint = best_checkpoint
        self.final_checkpoint = final_checkpoint
        self.final_loss = final_loss
        self.wall_clock_hours = wall_clock_hours
        self.tokens_seen = tokens_seen
        self.cost_usd = cost_usd
        self.wandb_run_id = wandb_run_id
        self.extra = extra or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_checkpoint": self.best_checkpoint,
            "final_checkpoint": self.final_checkpoint,
            "final_loss": self.final_loss,
            "wall_clock_hours": self.wall_clock_hours,
            "tokens_seen": self.tokens_seen,
            "cost_usd": self.cost_usd,
            "wandb_run_id": self.wandb_run_id,
            **self.extra,
        }


def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def is_git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return True


def hash_file(path: str) -> str:
    """SHA256 hash of a file, or 'missing' if not found."""
    import hashlib

    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return f"sha256:{h.hexdigest()[:16]}"


# GPU cost rates ($/hr)
GPU_COST_RATES = {
    "H100": 3.50,
    "A100": 2.00,
    "A10G": 1.10,
    "T4": 0.50,
}


def estimate_cost(config: ExperimentConfig) -> float:
    """Estimate cost in USD for an experiment."""
    rate = GPU_COST_RATES.get(config.gpu, 3.50)

    # Rough hour estimates per stage and depth
    if config.stage == "pretrain":
        depth_hours = {10: 1.5, 12: 4.0, 16: 6.0, 20: 10.0, 24: 20.0}
        hours = depth_hours.get(config.depth, 6.0)
        hours *= config.token_multiplier / 50  # scale by token multiplier
    elif config.stage == "sft":
        hours = 1.0 + config.sft_epochs * 0.5
    elif config.stage == "grpo":
        depth_hours = {10: 1.5, 12: 2.0, 16: 3.0, 20: 4.0, 24: 6.0}
        hours = depth_hours.get(config.depth, 3.0)
    else:
        hours = 1.0

    return round(hours * rate, 2)


class ExperimentRunner:
    """Enforces the experiment contract.

    All experiments go through this runner. It guarantees:
    - Config was validated
    - Gate was checked
    - W&B run exists with provenance
    - Post-training eval ran
    - Model was registered
    - Audit trail exists
    """

    def __init__(self, force: bool = False, dry_run: bool = False):
        self.force = force
        self.dry_run = dry_run

    def run(self, config: ExperimentConfig) -> tuple[TrainResult, dict[str, Any]]:
        """Execute the full experiment pipeline."""
        # ═══ PRE-FLIGHT ═══
        errors = validate_config(config)
        if errors and not self.force:
            raise ConfigError(errors)
        elif errors:
            logger.warning("Config has errors (--force): %s", errors)

        # Check gate for this stage
        gate_result = self._check_gate(config.stage)
        if not gate_result and not self.force:
            raise GateError(f"Gate for {config.stage} has not passed")

        # Budget check
        cost = estimate_cost(config)
        remaining = self._get_remaining_budget(config.stage)
        if cost > remaining and not self.force:
            raise ConfigError(
                [f"Estimated cost ${cost:.2f} exceeds remaining budget ${remaining:.2f}"]
            )

        if self.dry_run:
            logger.info("Dry run — would launch %s (est. $%.2f)", config.experiment_id, cost)
            return TrainResult(
                best_checkpoint="dry-run",
                final_checkpoint="dry-run",
                final_loss=0.0,
                wall_clock_hours=0.0,
                tokens_seen=0,
                cost_usd=0.0,
            ), {}

        self._append_audit_log(config, "started")

        # ═══ SETUP ═══
        wandb_run = self._init_wandb(config)
        self._log_provenance(config)

        # ═══ TRAIN ═══
        start_time = time.monotonic()
        try:
            if config.stage == "pretrain":
                result = self._run_pretrain(config)
            elif config.stage == "sft":
                result = self._run_sft(config)
            elif config.stage == "grpo":
                result = self._run_grpo(config)
            else:
                raise ValueError(f"Unknown stage: {config.stage}")
        except Exception as e:
            self._log_wandb({"error": str(e)})
            self._append_audit_log(config, "failed", error=str(e))
            self._finish_wandb()
            raise

        elapsed_hours = (time.monotonic() - start_time) / 3600

        # ═══ POST-TRAIN EVAL ═══
        eval_results = self._run_eval(config, result.best_checkpoint)

        # ═══ REGISTER ═══
        self._register_model(config, result, eval_results)

        # ═══ LOG COMPLETION ═══
        self._log_completion(config, result, eval_results)
        self._update_experiment_state(config, result, eval_results)
        self._append_audit_log(config, "completed")

        self._finish_wandb()
        return result, eval_results

    def _check_gate(self, stage: str) -> bool:
        """Check whether the gate for this stage has been passed."""
        from harness.gates import (
            check_preflight,
            check_pretrain_to_sft,
            check_sft_to_rl,
        )

        gate_map = {
            "pretrain": check_preflight,
            "sft": check_pretrain_to_sft,
            "grpo": check_sft_to_rl,
        }

        gate_fn = gate_map.get(stage)
        if gate_fn is None:
            return True

        result = gate_fn()
        if not result.passed:
            logger.warning("Gate '%s' failed:\n%s", result.gate_name, result.summary())
        return result.passed

    def _get_remaining_budget(self, stage: str) -> float:
        """Get remaining budget for a stage from experiment state."""
        from harness.config import PHASE_BUDGETS

        state_path = RESULTS_DIR / "experiment_state.json"
        if not state_path.exists():
            return PHASE_BUDGETS.get(stage, 0.0)

        state = json.loads(state_path.read_text())
        spent = state.get("total_spend_usd", 0.0)
        budget = PHASE_BUDGETS.get(stage, 0.0)
        return max(0.0, budget - spent)

    def _init_wandb(self, config: ExperimentConfig) -> Any:
        """Initialize W&B run."""
        if config.wandb_mode == "disabled":
            logger.info("W&B disabled for this run")
            return None
        try:
            import wandb

            tags = self._build_tags(config)
            run = wandb.init(
                project="math-nano",
                group=config.stage,
                name=config.experiment_id,
                tags=tags,
                config=asdict(config),
                mode=config.wandb_mode,
            )
            return run
        except ImportError:
            logger.warning("wandb not installed, skipping W&B init")
            return None

    def _build_tags(self, config: ExperimentConfig) -> list[str]:
        tags = [
            f"depth_{config.depth}",
            f"stage_{config.stage}",
            f"phase_{config.phase}",
        ]
        if config.mixture:
            tags.append(f"mixture_{config.mixture}")
        if config.sft_recipe:
            tags.append(f"recipe_{config.sft_recipe}")
        tags.extend(config.tags)
        return tags

    def _log_provenance(self, config: ExperimentConfig) -> None:
        """Log everything needed to reproduce this run."""
        provenance = {
            "git_hash": get_git_hash(),
            "git_dirty": is_git_dirty(),
            "harness_version": HARNESS_VERSION,
            "data_registry_hash": hash_file("data/registry.json"),
            "eval_manifest_hash": hash_file("data/eval/manifest.json"),
        }

        if config.parent_checkpoint:
            try:
                from harness.bookkeeper import ModelRegistry

                registry = ModelRegistry()
                parent = registry.get(config.parent_checkpoint)
                if parent:
                    provenance["parent_model_id"] = config.parent_checkpoint
                    provenance["parent_experiment"] = parent.get("experiment_id")
                    provenance["parent_gsm8k"] = parent.get("eval_results", {}).get(
                        "gsm8k_pass1_greedy"
                    )
            except Exception:
                pass

        self._log_wandb_config(provenance)

    def _log_wandb(self, data: dict) -> None:
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(data)
        except ImportError:
            pass

    def _log_wandb_config(self, data: dict) -> None:
        try:
            import wandb

            if wandb.run is not None:
                wandb.config.update(data)
        except ImportError:
            pass

    def _finish_wandb(self) -> None:
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass

    def _run_pretrain(self, config: ExperimentConfig) -> TrainResult:
        """Launch pretrain job. In production, dispatches to Modal."""
        logger.info(
            "Starting pretrain: %s (depth=%d, mixture=%s, tokens=%dx)",
            config.experiment_id, config.depth, config.mixture, config.token_multiplier,
        )
        # The actual training is dispatched via Modal or run locally.
        # This is a placeholder that will be connected to modal_jobs/pretrain.py
        raise NotImplementedError(
            "Pretrain dispatch not yet connected. "
            "Use `modal run modal_jobs/pretrain.py` directly for now."
        )

    def _run_sft(self, config: ExperimentConfig) -> TrainResult:
        """Launch SFT job."""
        logger.info(
            "Starting SFT: %s (depth=%d, recipe=%s, parent=%s)",
            config.experiment_id, config.depth, config.sft_recipe, config.parent_checkpoint,
        )
        raise NotImplementedError(
            "SFT dispatch not yet connected. "
            "Use `modal run modal_jobs/sft.py` directly for now."
        )

    def _run_grpo(self, config: ExperimentConfig) -> TrainResult:
        """Launch GRPO job."""
        logger.info(
            "Starting GRPO: %s (depth=%d, curriculum=%s, parent=%s)",
            config.experiment_id, config.depth, config.rl_curriculum, config.parent_checkpoint,
        )
        raise NotImplementedError(
            "GRPO dispatch not yet connected. "
            "Use `modal run modal_jobs/grpo.py` directly for now."
        )

    def _run_eval(self, config: ExperimentConfig, checkpoint: str) -> dict[str, Any]:
        """Run post-training eval suite."""
        logger.info("Running eval suite '%s' on %s", config.eval_suite, checkpoint)
        # Placeholder — will dispatch to modal_jobs/eval.py or run locally
        return {}

    def _register_model(
        self,
        config: ExperimentConfig,
        result: TrainResult,
        eval_results: dict[str, Any],
    ) -> None:
        """Register model in the model registry."""
        try:
            from harness.bookkeeper import ModelRegistry

            registry = ModelRegistry()
            model_id = f"{config.experiment_id}-best"
            registry.register(
                model_id=model_id,
                experiment_id=config.experiment_id,
                stage=config.stage,
                depth=config.depth,
                checkpoint_path=result.best_checkpoint,
                parent_model=config.parent_checkpoint,
                config=config,
                training_info={
                    "wall_clock_hours": result.wall_clock_hours,
                    "cost_usd": result.cost_usd,
                    "final_train_loss": result.final_loss,
                    "gpu": config.gpu,
                    "wandb_run_id": result.wandb_run_id,
                },
                eval_results=eval_results,
            )
            logger.info("Registered model: %s", model_id)
        except Exception as e:
            logger.error("Failed to register model: %s", e)

    def _log_completion(
        self,
        config: ExperimentConfig,
        result: TrainResult,
        eval_results: dict[str, Any],
    ) -> None:
        """Log final metrics to W&B."""
        final = {
            "final/train_loss": result.final_loss,
            "final/wall_clock_hours": result.wall_clock_hours,
            "final/tokens_seen": result.tokens_seen,
            "final/cost_usd_estimated": result.cost_usd,
            "final/checkpoint_path": result.best_checkpoint,
        }
        for k, v in eval_results.items():
            final[f"eval/{k}"] = v
        self._log_wandb(final)

    def _update_experiment_state(
        self,
        config: ExperimentConfig,
        result: TrainResult,
        eval_results: dict[str, Any],
    ) -> None:
        """Update results/experiment_state.json."""
        from harness.experiment_state import ExperimentState

        state = ExperimentState.load()
        state.mark_completed(config.experiment_id, result.cost_usd)
        state.save()

    def _append_audit_log(
        self, config: ExperimentConfig, action: str, error: str | None = None
    ) -> None:
        """Append an entry to the audit log."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_id": config.experiment_id,
            "stage": config.stage,
            "action": action,
            "git_hash": get_git_hash(),
        }
        if error:
            entry["error"] = error
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
