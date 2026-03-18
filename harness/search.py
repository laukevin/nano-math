"""Smart search strategy logic for experiment selection.

Implements:
- Phased elimination: pilot → rank → sweep
- Binary search for capability thresholds
- Cost-aware experiment prioritization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from harness.config import VALID_DEPTHS

logger = logging.getLogger(__name__)

# Map depth to approximate param count (millions)
DEPTH_TO_PARAMS = {
    10: 50,
    12: 85,
    16: 130,
    20: 200,
    24: 320,
}


@dataclass
class ExperimentProposal:
    """A proposed experiment with cost estimate."""

    experiment_id: str
    stage: str
    depth: int
    config_overrides: dict[str, Any] = field(default_factory=dict)
    estimated_cost: float = 0.0
    rationale: str = ""


@dataclass
class SearchResult:
    """Result of a search phase."""

    phase: str
    proposals: list[ExperimentProposal]
    total_estimated_cost: float
    rationale: str


class PhasedElimination:
    """Three-phase search: pilot → rank → sweep.

    Phase 1 (Pilot): Run a small subset of configs on one model size to
    find promising candidates.

    Phase 2 (Rank): Run the top candidates on 2-3 sizes to confirm the
    ranking holds across scale.

    Phase 3 (Sweep): Run the winner(s) on all sizes to get full scaling
    curves.
    """

    def __init__(
        self,
        candidates: list[str],
        depths: list[int] | None = None,
        pilot_depth: int = 12,
        rank_depths: list[int] | None = None,
    ):
        self.candidates = candidates
        self.depths = depths or VALID_DEPTHS
        self.pilot_depth = pilot_depth
        self.rank_depths = rank_depths or [10, 16]

    def pilot(self, stage: str, phase: str) -> SearchResult:
        """Phase 1: Run all candidates at pilot depth."""
        proposals = []
        for candidate in self.candidates:
            eid = f"{stage[:2]}-{_depth_label(self.pilot_depth)}-{candidate}"
            proposals.append(ExperimentProposal(
                experiment_id=eid,
                stage=stage,
                depth=self.pilot_depth,
                config_overrides={"candidate": candidate},
                rationale=f"Pilot: test {candidate} at depth {self.pilot_depth}",
            ))
        total = sum(p.estimated_cost for p in proposals)
        return SearchResult(
            phase="pilot",
            proposals=proposals,
            total_estimated_cost=total,
            rationale=(
                f"Testing {len(self.candidates)} candidates at depth "
                f"{self.pilot_depth} ({_depth_label(self.pilot_depth)} model)"
            ),
        )

    def rank(
        self, stage: str, phase: str, winners: list[str]
    ) -> SearchResult:
        """Phase 2: Run winners at ranking depths to confirm ordering."""
        proposals = []
        for candidate in winners:
            for depth in self.rank_depths:
                eid = f"{stage[:2]}-{_depth_label(depth)}-{candidate}"
                proposals.append(ExperimentProposal(
                    experiment_id=eid,
                    stage=stage,
                    depth=depth,
                    config_overrides={"candidate": candidate},
                    rationale=f"Rank: confirm {candidate} at depth {depth}",
                ))
        total = sum(p.estimated_cost for p in proposals)
        return SearchResult(
            phase="rank",
            proposals=proposals,
            total_estimated_cost=total,
            rationale=(
                f"Ranking {len(winners)} winners across depths "
                f"{self.rank_depths}"
            ),
        )

    def sweep(self, stage: str, phase: str, winner: str) -> SearchResult:
        """Phase 3: Run the winner at all depths for full scaling curve."""
        proposals = []
        for depth in self.depths:
            eid = f"{stage[:2]}-{_depth_label(depth)}-{winner}"
            proposals.append(ExperimentProposal(
                experiment_id=eid,
                stage=stage,
                depth=depth,
                config_overrides={"candidate": winner},
                rationale=f"Sweep: {winner} at depth {depth}",
            ))
        total = sum(p.estimated_cost for p in proposals)
        return SearchResult(
            phase="sweep",
            proposals=proposals,
            total_estimated_cost=total,
            rationale=(
                f"Full sweep of '{winner}' across all {len(self.depths)} depths"
            ),
        )


class BinarySearchThreshold:
    """Binary search to find the minimum model size that achieves a target metric.

    Example: "What's the smallest model that gets >30% on GSM8K after SFT?"
    """

    def __init__(
        self,
        metric_name: str,
        target: float,
        depths: list[int] | None = None,
    ):
        self.metric_name = metric_name
        self.target = target
        self.depths = sorted(depths or VALID_DEPTHS)
        self._results: dict[int, float] = {}

    def add_result(self, depth: int, value: float) -> None:
        self._results[depth] = value

    def next_depth_to_try(self) -> int | None:
        """Return the next depth to evaluate, or None if search is complete."""
        untried = [d for d in self.depths if d not in self._results]
        if not untried:
            return None

        # Find the boundary between pass and fail
        passing = [d for d, v in self._results.items() if v >= self.target]
        failing = [d for d, v in self._results.items() if v < self.target]

        if not passing and not failing:
            # No results yet — start from the middle
            mid = len(untried) // 2
            return untried[mid]
        elif not passing:
            # Nothing passes yet — try the largest untried
            return max(untried)
        elif not failing:
            # Everything passes — try the smallest untried
            return min(untried)
        else:
            # Binary search between smallest passing and largest failing
            smallest_pass = min(passing)
            largest_fail = max(failing)
            candidates = [d for d in untried if largest_fail < d < smallest_pass]
            if candidates:
                mid = len(candidates) // 2
                return candidates[mid]
            # No candidates in between — search complete
            return None

    def threshold(self) -> int | None:
        """Return the minimum depth that achieves the target, or None."""
        passing = sorted([d for d, v in self._results.items() if v >= self.target])
        return passing[0] if passing else None

    def is_complete(self) -> bool:
        return self.next_depth_to_try() is None

    def summary(self) -> str:
        lines = [f"Binary search for {self.metric_name} >= {self.target}"]
        for depth in self.depths:
            if depth in self._results:
                v = self._results[depth]
                mark = "+" if v >= self.target else "x"
                lines.append(f"  [{mark}] depth {depth}: {v:.4f}")
            else:
                lines.append(f"  [ ] depth {depth}: not tested")
        t = self.threshold()
        if t is not None:
            lines.append(f"  → Threshold: depth {t} ({DEPTH_TO_PARAMS.get(t, '?')}M params)")
        return "\n".join(lines)


def _depth_label(depth: int) -> str:
    """Map depth to size label: xs, s, m, l, xl."""
    labels = {10: "xs", 12: "s", 16: "m", 20: "l", 24: "xl"}
    return labels.get(depth, f"d{depth}")


def suggest_next_experiments(
    stage: str,
    completed: list[dict[str, Any]],
    budget_remaining: float,
) -> list[ExperimentProposal]:
    """Given completed experiments and budget, suggest what to run next.

    This is a heuristic — the agent/human reviews and approves.
    """
    proposals = []

    completed_ids = {e.get("experiment_id", "") for e in completed}
    completed_depths = {e.get("depth") for e in completed if e.get("stage") == stage}

    # Suggest filling in missing depths
    for depth in VALID_DEPTHS:
        if depth not in completed_depths:
            proposals.append(ExperimentProposal(
                experiment_id=f"{stage[:2]}-{_depth_label(depth)}-next",
                stage=stage,
                depth=depth,
                rationale=f"No {stage} experiments at depth {depth} yet",
            ))

    return proposals
