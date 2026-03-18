"""Validation gates between experiment phases."""

from __future__ import annotations

from dataclasses import dataclass


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


def check_preflight() -> GateResult:
    """Gate 0: Pre-flight checks before any training."""
    checks = {}
    notes = []

    # TODO: implement checks
    # - Data shards exist and pass validation
    # - Model initializes at all depths
    # - Eval harness runs on sample data
    # - W&B connection works

    return GateResult(
        gate_name="preflight",
        passed=all(checks.values()) if checks else False,
        checks=checks,
        notes=notes or ["Not yet implemented"],
    )


def check_pretrain_to_sft() -> GateResult:
    """Gate 1: Can we proceed from pretrain to SFT?"""
    checks = {}
    notes = []

    # TODO: implement checks
    # - Loss decreased during training
    # - Scaling is monotonic (bigger model = lower loss)
    # - All checkpoints saved and loadable
    # - W&B logging complete

    return GateResult(
        gate_name="pretrain_to_sft",
        passed=all(checks.values()) if checks else False,
        checks=checks,
        notes=notes or ["Not yet implemented"],
    )


def check_sft_to_rl() -> GateResult:
    """Gate 2: Can we proceed from SFT to RL?"""
    checks = {}
    notes = []

    # TODO: implement checks
    # - SFT improved over pretrain baseline
    # - Format compliance > 80% (\boxed{} usage)
    # - No obvious overfitting (train loss vs eval gap)

    return GateResult(
        gate_name="sft_to_rl",
        passed=all(checks.values()) if checks else False,
        checks=checks,
        notes=notes or ["Not yet implemented"],
    )
