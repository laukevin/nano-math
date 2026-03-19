"""Experiment registry: append results, read, and compare.

Registry is a JSONL file — each line is a complete experiment record.

Usage:
    # Print leaderboard
    python scripts/registry.py --registry /results/experiment_registry.jsonl
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone


def append_result(
    record: dict,
    registry_path: str = "/results/experiment_registry.jsonl",
):
    """Append an experiment result to the registry."""
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    os.makedirs(os.path.dirname(registry_path) or ".", exist_ok=True)
    with open(registry_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    print(
        f"[registry] Logged {record.get('experiment_id', '?')} to {registry_path}"
    )


def read_registry(
    registry_path: str = "/results/experiment_registry.jsonl",
) -> list[dict]:
    """Read all experiment records."""
    if not os.path.exists(registry_path):
        return []
    records = []
    with open(registry_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_leaderboard(
    records: list[dict] | None = None,
    registry_path: str = "/results/experiment_registry.jsonl",
):
    """Print a leaderboard sorted by GSM8K accuracy."""
    if records is None:
        records = read_registry(registry_path)

    if not records:
        print("No experiments in registry.")
        return

    def sort_key(r):
        ev = r.get("eval", {})
        return ev.get("gsm8k_greedy", 0)

    records.sort(key=sort_key, reverse=True)

    header = (
        f"{'ID':<20} {'Data':<15} {'Size':>6} "
        f"{'GSM8K':>7} {'SVAMP':>7} {'MATH':>7} {'Loss':>7} {'Time':>6}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for r in records:
        ev = r.get("eval", {})
        gsm = f"{ev['gsm8k_greedy']*100:.1f}%" if "gsm8k_greedy" in ev else "  n/a"
        sva = f"{ev['svamp_greedy']*100:.1f}%" if "svamp_greedy" in ev else "  n/a"
        mat = f"{ev['math_greedy']*100:.1f}%" if "math_greedy" in ev else "  n/a"
        loss = f"{r['final_loss']:.4f}" if r.get("final_loss") else "  n/a"
        wall = f"{r['wall_clock_min']:.0f}m" if r.get("wall_clock_min") else " n/a"

        print(
            f"{r.get('experiment_id', '?'):<20} "
            f"{r.get('data_source', '?'):<15} "
            f"{r.get('data_size', '?'):>6} "
            f"{gsm:>7} {sva:>7} {mat:>7} {loss:>7} {wall:>6}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Show experiment leaderboard")
    parser.add_argument(
        "--registry", default="/results/experiment_registry.jsonl"
    )
    args = parser.parse_args()
    print_leaderboard(registry_path=args.registry)
