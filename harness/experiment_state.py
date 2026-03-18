"""Experiment state tracking — results/experiment_state.json.

Tracks current phase, wave, completed/running experiments, and spend.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATE_PATH = Path("results/experiment_state.json")


class ExperimentState:
    """Manages the experiment state file."""

    def __init__(self, data: dict[str, Any] | None = None, path: Path | None = None):
        self.path = path or STATE_PATH
        self._data = data or self._default()

    @staticmethod
    def _default() -> dict[str, Any]:
        return {
            "current_phase": 1,
            "current_wave": 1,
            "completed_experiments": [],
            "running_experiments": [],
            "pending_experiments": [],
            "failed_experiments": [],
            "total_spend_usd": 0.0,
            "phase_budget_remaining_usd": 300.0,
            "last_gate_status": None,
            "decisions_log": [],
        }

    @classmethod
    def load(cls, path: Path | None = None) -> ExperimentState:
        p = path or STATE_PATH
        if p.exists():
            data = json.loads(p.read_text())
            return cls(data=data, path=p)
        return cls(path=p)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2) + "\n")

    @property
    def current_phase(self) -> int:
        return self._data["current_phase"]

    @property
    def current_wave(self) -> int:
        return self._data["current_wave"]

    @property
    def completed_experiments(self) -> list[str]:
        return self._data["completed_experiments"]

    @property
    def running_experiments(self) -> list[str]:
        return self._data["running_experiments"]

    @property
    def pending_experiments(self) -> list[str]:
        return self._data["pending_experiments"]

    @property
    def total_spend_usd(self) -> float:
        return self._data["total_spend_usd"]

    def mark_running(self, experiment_id: str) -> None:
        if experiment_id not in self._data["running_experiments"]:
            self._data["running_experiments"].append(experiment_id)
        if experiment_id in self._data["pending_experiments"]:
            self._data["pending_experiments"].remove(experiment_id)

    def mark_completed(self, experiment_id: str, cost_usd: float = 0.0) -> None:
        if experiment_id in self._data["running_experiments"]:
            self._data["running_experiments"].remove(experiment_id)
        if experiment_id not in self._data["completed_experiments"]:
            self._data["completed_experiments"].append(experiment_id)
        self._data["total_spend_usd"] = round(
            self._data["total_spend_usd"] + cost_usd, 2
        )

    def mark_failed(self, experiment_id: str) -> None:
        if experiment_id in self._data["running_experiments"]:
            self._data["running_experiments"].remove(experiment_id)
        if experiment_id not in self._data.get("failed_experiments", []):
            self._data.setdefault("failed_experiments", []).append(experiment_id)

    def add_pending(self, experiment_ids: list[str]) -> None:
        for eid in experiment_ids:
            if eid not in self._data["pending_experiments"]:
                self._data["pending_experiments"].append(eid)

    def advance_wave(self) -> None:
        self._data["current_wave"] += 1

    def advance_phase(self) -> None:
        self._data["current_phase"] += 1
        self._data["current_wave"] = 1

    def set_gate_status(self, gate: str, passed: bool) -> None:
        self._data["last_gate_status"] = {
            "gate": gate,
            "passed": passed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def log_decision(self, decision: str, decided_by: str = "human") -> None:
        self._data["decisions_log"].append({
            "wave": self.current_wave,
            "decision": decision,
            "decided_by": decided_by,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def summary(self) -> str:
        """Human-readable status summary."""
        lines = [
            f"Phase: {self.current_phase}",
            f"Wave: {self.current_wave}",
            f"Completed: {len(self.completed_experiments)}",
            f"Running: {len(self.running_experiments)}",
            f"Pending: {len(self.pending_experiments)}",
            f"Budget: ${self.total_spend_usd:.2f} spent",
        ]
        gate = self._data.get("last_gate_status")
        if gate:
            status = "passed" if gate["passed"] else "FAILED"
            lines.append(f"Last gate: {gate['gate']} ({status})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)
