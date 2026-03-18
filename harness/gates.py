"""Validation gates between experiment phases."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
REGISTRY_PATH = RESULTS_DIR / "model_registry.json"
EVAL_DIR = RESULTS_DIR / "eval"


@dataclass
class GateResult:
    """Result of a gate check."""

    gate_name: str
    passed: bool
    checks: dict[str, bool]
    notes: list[str]

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Gate '{self.gate_name}': {status}"]
        for check, ok in self.checks.items():
            mark = "+" if ok else "x"
            lines.append(f"  [{mark}] {check}")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        return "\n".join(lines)


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def _get_completed_experiments(stage: str) -> list[dict]:
    """Get registry entries for completed experiments in a stage."""
    registry = _load_registry()
    return [
        m for m in registry.values()
        if isinstance(m, dict) and m.get("stage") == stage
    ]


def check_preflight() -> GateResult:
    """Gate 0: Pre-flight checks before any training."""
    checks = {}
    notes = []

    # Check eval data exists
    eval_dir = Path("data/eval")
    manifest = eval_dir / "manifest.json"
    checks["eval_manifest_exists"] = manifest.exists()
    if not manifest.exists():
        notes.append("Run scripts/data/download_eval.py to create eval datasets")

    # Check at least one eval dataset exists
    if manifest.exists():
        datasets = json.loads(manifest.read_text()).get("datasets", {})
        has_data = any((eval_dir / d["file"]).exists() for d in datasets.values())
        checks["eval_datasets_present"] = has_data
    else:
        checks["eval_datasets_present"] = False

    # Check pyproject.toml exists (project is set up)
    checks["project_setup"] = Path("pyproject.toml").exists()

    # Check model can initialize (import test)
    try:
        from harness.config import VALID_DEPTHS
        checks["config_loadable"] = True
    except Exception:
        checks["config_loadable"] = False

    return GateResult(
        gate_name="preflight",
        passed=all(checks.values()),
        checks=checks,
        notes=notes,
    )


def check_pretrain_to_sft() -> GateResult:
    """Gate 1: Can we proceed from pretrain to SFT?"""
    checks = {}
    notes = []

    pretrain_models = _get_completed_experiments("pretrain")
    checks["has_pretrain_models"] = len(pretrain_models) >= 1
    if not pretrain_models:
        notes.append("No pretrain models registered yet")
        return GateResult(
            gate_name="pretrain_to_sft", passed=False, checks=checks, notes=notes,
        )

    notes.append(f"Found {len(pretrain_models)} pretrain model(s)")

    # Check that at least one model has checkpoints
    has_checkpoint = False
    for m in pretrain_models:
        ckpt = m.get("checkpoint_path", "")
        if ckpt and Path(ckpt).exists():
            has_checkpoint = True
            break
    checks["checkpoint_exists"] = has_checkpoint

    # Check that eval was run on at least one model
    has_eval = any(m.get("eval_results") for m in pretrain_models)
    checks["eval_completed"] = has_eval

    # Check scaling: if multiple depths, verify loss decreases with depth
    depths_and_loss = []
    for m in pretrain_models:
        depth = m.get("depth")
        training = m.get("training", {})
        loss = training.get("final_train_loss")
        if depth and loss:
            depths_and_loss.append((depth, loss))

    if len(depths_and_loss) >= 2:
        depths_and_loss.sort()
        monotonic = all(
            depths_and_loss[i][1] >= depths_and_loss[i + 1][1]
            for i in range(len(depths_and_loss) - 1)
        )
        checks["scaling_monotonic"] = monotonic
        if not monotonic:
            notes.append("Warning: loss does not decrease monotonically with depth")
    else:
        notes.append("Not enough models to check scaling monotonicity")

    return GateResult(
        gate_name="pretrain_to_sft",
        passed=all(checks.values()),
        checks=checks,
        notes=notes,
    )


def check_sft_to_rl() -> GateResult:
    """Gate 2: Can we proceed from SFT to RL?"""
    checks = {}
    notes = []

    sft_models = _get_completed_experiments("sft")
    checks["has_sft_models"] = len(sft_models) >= 1
    if not sft_models:
        notes.append("No SFT models registered yet")
        return GateResult(
            gate_name="sft_to_rl", passed=False, checks=checks, notes=notes,
        )

    notes.append(f"Found {len(sft_models)} SFT model(s)")

    # Check checkpoint exists
    has_checkpoint = any(
        m.get("checkpoint_path") and Path(m["checkpoint_path"]).exists()
        for m in sft_models
    )
    checks["checkpoint_exists"] = has_checkpoint

    # Check SFT improved over pretrain (compare gsm8k scores)
    pretrain_models = _get_completed_experiments("pretrain")
    if pretrain_models and sft_models:
        best_pretrain_gsm8k = max(
            (m.get("eval_results", {}).get("gsm8k_pass1_greedy", 0.0) for m in pretrain_models),
            default=0.0,
        )
        best_sft_gsm8k = max(
            (m.get("eval_results", {}).get("gsm8k_pass1_greedy", 0.0) for m in sft_models),
            default=0.0,
        )
        improved = best_sft_gsm8k > best_pretrain_gsm8k
        checks["sft_improves_over_pretrain"] = improved
        notes.append(f"Best pretrain GSM8K: {best_pretrain_gsm8k:.3f}, best SFT: {best_sft_gsm8k:.3f}")
    else:
        notes.append("Cannot compare SFT vs pretrain (missing data)")

    # Check format compliance if available
    has_format = False
    for m in sft_models:
        compliance = m.get("eval_results", {}).get("format_compliance")
        if compliance is not None:
            has_format = True
            checks["format_compliance_gt_80"] = compliance > 0.80
            notes.append(f"Format compliance: {compliance:.1%}")
            break
    if not has_format:
        notes.append("Format compliance data not available")

    return GateResult(
        gate_name="sft_to_rl",
        passed=all(checks.values()),
        checks=checks,
        notes=notes,
    )
