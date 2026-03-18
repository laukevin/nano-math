"""Tests for search strategy logic."""

from __future__ import annotations

import pytest

from harness.search import (
    BinarySearchThreshold,
    PhasedElimination,
    _depth_label,
    suggest_next_experiments,
)


class TestPhasedElimination:
    """Test pilot → rank → sweep search."""

    def test_pilot_runs_all_candidates_at_pilot_depth(self):
        pe = PhasedElimination(
            candidates=["broad", "heavy", "pure"],
            pilot_depth=12,
        )
        result = pe.pilot("pretrain", "1a")
        assert result.phase == "pilot"
        assert len(result.proposals) == 3
        # All at pilot depth
        assert all(p.depth == 12 for p in result.proposals)

    def test_pilot_experiment_ids_contain_candidate(self):
        pe = PhasedElimination(candidates=["broad", "heavy"])
        result = pe.pilot("pretrain", "1a")
        ids = [p.experiment_id for p in result.proposals]
        assert any("broad" in eid for eid in ids)
        assert any("heavy" in eid for eid in ids)

    def test_rank_runs_winners_at_rank_depths(self):
        pe = PhasedElimination(
            candidates=["broad", "heavy", "pure"],
            rank_depths=[10, 16],
        )
        result = pe.rank("pretrain", "1a", winners=["broad", "heavy"])
        # 2 winners * 2 rank depths = 4 proposals
        assert len(result.proposals) == 4
        depths = {p.depth for p in result.proposals}
        assert depths == {10, 16}

    def test_sweep_runs_winner_at_all_depths(self):
        pe = PhasedElimination(
            candidates=["broad"],
            depths=[10, 12, 16, 20, 24],
        )
        result = pe.sweep("pretrain", "1a", winner="broad")
        assert len(result.proposals) == 5
        depths = sorted(p.depth for p in result.proposals)
        assert depths == [10, 12, 16, 20, 24]

    def test_full_pipeline(self):
        """Test the full pilot → rank → sweep flow."""
        candidates = ["mix-a", "mix-b", "mix-c"]
        pe = PhasedElimination(candidates=candidates, pilot_depth=12)

        # Pilot
        pilot = pe.pilot("pretrain", "1a")
        assert len(pilot.proposals) == 3

        # Rank top 2
        rank = pe.rank("pretrain", "1a", winners=["mix-a", "mix-b"])
        assert len(rank.proposals) == 4  # 2 * 2 depths

        # Sweep winner
        sweep = pe.sweep("pretrain", "1a", winner="mix-a")
        assert len(sweep.proposals) == 5


class TestBinarySearchThreshold:
    """Test binary search for capability thresholds."""

    def test_initial_state_suggests_middle(self):
        bs = BinarySearchThreshold(
            metric_name="gsm8k_pass1",
            target=0.30,
            depths=[10, 12, 16, 20, 24],
        )
        next_depth = bs.next_depth_to_try()
        assert next_depth is not None
        assert next_depth in [10, 12, 16, 20, 24]
        # Should start near the middle
        assert next_depth == 16

    def test_all_fail_tries_largest(self):
        bs = BinarySearchThreshold(
            metric_name="gsm8k_pass1",
            target=0.30,
            depths=[10, 12, 16, 20, 24],
        )
        bs.add_result(16, 0.10)  # fail
        next_depth = bs.next_depth_to_try()
        # Should try larger since 16 failed
        assert next_depth == 24

    def test_all_pass_tries_smallest(self):
        bs = BinarySearchThreshold(
            metric_name="gsm8k_pass1",
            target=0.30,
            depths=[10, 12, 16, 20, 24],
        )
        bs.add_result(16, 0.50)  # pass
        next_depth = bs.next_depth_to_try()
        # Should try smaller since 16 passes
        assert next_depth == 10

    def test_binary_narrows(self):
        bs = BinarySearchThreshold(
            metric_name="gsm8k_pass1",
            target=0.30,
            depths=[10, 12, 16, 20, 24],
        )
        bs.add_result(10, 0.05)   # fail
        bs.add_result(24, 0.50)   # pass
        next_depth = bs.next_depth_to_try()
        # Should try between 10 and 24
        assert next_depth in [12, 16, 20]

    def test_threshold_found(self):
        bs = BinarySearchThreshold(
            metric_name="gsm8k_pass1",
            target=0.30,
            depths=[10, 12, 16, 20, 24],
        )
        bs.add_result(10, 0.05)
        bs.add_result(12, 0.15)
        bs.add_result(16, 0.32)
        bs.add_result(20, 0.45)
        bs.add_result(24, 0.50)
        assert bs.threshold() == 16

    def test_no_threshold_if_all_fail(self):
        bs = BinarySearchThreshold(
            metric_name="gsm8k_pass1",
            target=0.90,
            depths=[10, 12, 16],
        )
        bs.add_result(10, 0.05)
        bs.add_result(12, 0.15)
        bs.add_result(16, 0.32)
        assert bs.threshold() is None

    def test_complete_when_all_tested(self):
        bs = BinarySearchThreshold(
            metric_name="test",
            target=0.50,
            depths=[10, 12],
        )
        bs.add_result(10, 0.30)
        bs.add_result(12, 0.60)
        assert bs.is_complete()

    def test_summary(self):
        bs = BinarySearchThreshold(
            metric_name="gsm8k_pass1",
            target=0.30,
            depths=[10, 12, 16],
        )
        bs.add_result(10, 0.05)
        bs.add_result(16, 0.40)
        s = bs.summary()
        assert "gsm8k_pass1" in s
        assert "[x]" in s  # depth 10 failed
        assert "[+]" in s  # depth 16 passed
        assert "[ ]" in s  # depth 12 not tested


class TestDepthLabel:
    def test_known_depths(self):
        assert _depth_label(10) == "xs"
        assert _depth_label(12) == "s"
        assert _depth_label(16) == "m"
        assert _depth_label(20) == "l"
        assert _depth_label(24) == "xl"

    def test_unknown_depth(self):
        assert _depth_label(8) == "d8"


class TestSuggestNext:
    def test_suggests_missing_depths(self):
        completed = [
            {"experiment_id": "pt-s-broad", "stage": "pretrain", "depth": 12},
        ]
        proposals = suggest_next_experiments("pretrain", completed, budget_remaining=200.0)
        suggested_depths = {p.depth for p in proposals}
        # Should suggest depths we haven't run yet
        assert 10 in suggested_depths
        assert 16 in suggested_depths
        assert 12 not in suggested_depths

    def test_no_suggestions_when_all_done(self):
        completed = [
            {"experiment_id": f"pt-{d}", "stage": "pretrain", "depth": d}
            for d in [10, 12, 16, 20, 24]
        ]
        proposals = suggest_next_experiments("pretrain", completed, budget_remaining=200.0)
        assert len(proposals) == 0
