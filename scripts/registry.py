"""Experiment registry: append results, read, and compare.

Supports two on-disk formats:
  - Old: experiment_registry.jsonl, each line has an "eval" dict with *_greedy keys
  - New: individual eval_*.json files with a "benchmarks" dict

Usage:
    # Pull from Modal volume and print leaderboard (sorted by AIME)
    uv run python scripts/registry.py

    # Sort by a different metric
    uv run python scripts/registry.py --sort gsm8k

    # Only show experiments with AIME > 0%
    uv run python scripts/registry.py --min-aime 0.01

    # Use a local dir instead of downloading
    uv run python scripts/registry.py --local-dir /tmp/my-results
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


BENCHMARKS = ["aime_2025", "gsm8k", "svamp", "math"]


# ---------------------------------------------------------------------------
# Write helpers (used by Modal eval jobs)
# ---------------------------------------------------------------------------

def append_result(
    record: dict,
    registry_path: str = "/results/experiment_registry.jsonl",
):
    """Append an experiment result to the registry."""
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    os.makedirs(os.path.dirname(registry_path) or ".", exist_ok=True)
    with open(registry_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    print(f"[registry] Logged {record.get('experiment_id', '?')} to {registry_path}")


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def read_registry(
    registry_path: str = "/results/experiment_registry.jsonl",
) -> list[dict]:
    """Read all experiment records from the old JSONL registry."""
    if not os.path.exists(registry_path):
        return []
    records = []
    with open(registry_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _parse_new_eval(path: str, filename: str) -> dict | None:
    """Parse a new-format eval_*.json file into a normalised record."""
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:
        print(f"  skip {filename}: {e}", file=sys.stderr)
        return None

    benches = d.get("benchmarks", {})
    if not benches:
        return None

    adapter = d.get("adapter", "")
    if adapter:
        exp_id = adapter.lstrip("/").replace("checkpoints/", "").replace("sft-", "")
    else:
        exp_id = filename.replace("eval_sft-", "").replace("eval_", "").replace(".json", "")

    record = {
        "experiment_id": exp_id,
        "base_model": d.get("base_model", ""),
        "adapter": adapter,
        "timestamp": d.get("timestamp", ""),
        "source": "eval_json",
    }
    for bench in BENCHMARKS:
        b = benches.get(bench, {})
        record[bench] = b.get("accuracy")
    return record


def _parse_registry_record(d: dict) -> dict | None:
    """Normalise a single old-format registry record."""
    ev = d.get("eval", {})
    if not ev:
        return None
    record = {
        "experiment_id": d.get("experiment_id", "?"),
        "base_model": d.get("base_model", ""),
        "adapter": d.get("adapter", ""),
        "timestamp": d.get("timestamp", ""),
        "source": "registry",
    }
    record["aime_2025"] = ev.get("aime_2025_greedy") or ev.get("aime_2024_greedy")
    record["gsm8k"] = ev.get("gsm8k_greedy")
    record["svamp"] = ev.get("svamp_greedy")
    record["math"] = ev.get("math_greedy")
    return record


def _normalize_id(exp_id: str) -> str:
    """Strip leading 'sft-' for deduplication (old registry keeps it, new files don't)."""
    return exp_id.removeprefix("sft-")


def load_all_results(results_dir: str) -> list[dict]:
    """Load and deduplicate results from a directory.

    eval_*.json files take priority over registry.jsonl entries for the same
    experiment_id (newer format is more reliable).
    """
    records_by_id: dict[str, dict] = {}

    # 1. Old registry (lower priority)
    registry_path = os.path.join(results_dir, "experiment_registry.jsonl")
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = _parse_registry_record(json.loads(line))
                except Exception:
                    continue
                if r:
                    records_by_id[_normalize_id(r["experiment_id"])] = r

    # 2. New eval_*.json files (higher priority — overwrite registry entries)
    for filename in sorted(os.listdir(results_dir)):
        if not (filename.endswith(".json") and filename.startswith("eval_")):
            continue
        r = _parse_new_eval(os.path.join(results_dir, filename), filename)
        if r:
            records_by_id[_normalize_id(r["experiment_id"])] = r

    return list(records_by_id.values())


def download_volume(volume: str, dest_dir: str) -> None:
    result = subprocess.run(
        ["uv", "run", "modal", "volume", "get", volume, "/", dest_dir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Warning: volume download failed:\n{result.stderr.strip()}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _sort_key(r: dict, primary: str) -> tuple:
    def g(k):
        v = r.get(k)
        return v if v is not None else -1.0
    return (g(primary), g("gsm8k"), g("svamp"), g("math"))


def _pct(v) -> str:
    if v is None:
        return "  n/a"
    return f"{v*100:5.1f}%"


def print_leaderboard(
    records: list[dict] | None = None,
    registry_path: str = "/results/experiment_registry.jsonl",
    sort_by: str = "aime_2025",
    min_aime: float = -1.0,
):
    """Print a sorted leaderboard."""
    if records is None:
        records = read_registry(registry_path)
        records = [r for r in (_parse_registry_record(r) for r in records) if r]

    if min_aime >= 0:
        records = [r for r in records if (r.get("aime_2025") or 0) >= min_aime]

    records.sort(key=lambda r: _sort_key(r, sort_by), reverse=True)

    if not records:
        print("No experiments found.")
        return

    col = 44
    header = f"{'EXPERIMENT':<{col}}  {'AIME':>7}  {'GSM8K':>7}  {'SVAMP':>7}  {'MATH':>7}"
    print()
    print(header)
    print("-" * len(header))
    for r in records:
        exp = r["experiment_id"]
        if len(exp) > col:
            exp = exp[:col - 1] + "…"
        print(
            f"{exp:<{col}}  "
            f"{_pct(r.get('aime_2025')):>7}  "
            f"{_pct(r.get('gsm8k')):>7}  "
            f"{_pct(r.get('svamp')):>7}  "
            f"{_pct(r.get('math')):>7}"
        )
    print()
    print(f"  {len(records)} experiments  |  sorted by {sort_by}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Show experiment leaderboard")
    parser.add_argument("--sort", default="aime_2025",
                        choices=["aime_2025", "gsm8k", "svamp", "math"],
                        help="Primary sort metric (default: aime_2025)")
    parser.add_argument("--min-aime", type=float, default=-1.0,
                        help="Only show rows with AIME >= this (e.g. 0.01 = >0%%)")
    parser.add_argument("--volume", default="math-nano-results",
                        help="Modal results volume name")
    parser.add_argument("--local-dir", default=None,
                        help="Use local directory instead of downloading")
    args = parser.parse_args()

    if args.local_dir:
        results_dir = args.local_dir
        print(f"Using local dir: {results_dir}")
    else:
        tmp = tempfile.mkdtemp(prefix="nano-results-")
        print(f"Downloading from '{args.volume}'...", end=" ", flush=True)
        download_volume(args.volume, tmp)
        print("done")
        results_dir = tmp

    all_records = load_all_results(results_dir)
    print(f"Loaded {len(all_records)} experiments")
    print_leaderboard(all_records, sort_by=args.sort, min_aime=args.min_aime)
