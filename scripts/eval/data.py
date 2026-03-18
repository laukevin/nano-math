"""Eval data loading, prompt formatting, and constants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EVAL_VERSION = "1.0"
MAX_NEW_TOKENS = 1024
GREEDY_TEMPERATURE = 0.0
SAMPLED_TEMPERATURE = 0.7
BOOTSTRAP_SEED = 42
EOS_TOKEN_ID = 50256  # GPT-2 EOS

SUITE_DATASETS = {
    "small": ["gsm8k_mini", "math_mini"],
    "full": ["gsm8k", "math500", "amc", "aime", "minerva"],
}


def format_eval_prompt(problem: str) -> str:
    """Format a math problem for evaluation."""
    return (
        "Solve the following math problem step by step. "
        "Put your final answer in \\boxed{}.\n\n"
        f"Problem: {problem}\n\n"
        "Solution:"
    )


def load_eval_dataset(dataset_name: str, data_dir: Path) -> list[dict]:
    """Load an eval dataset from JSONL, verifying checksum against manifest."""
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        datasets_info = manifest.get("datasets", {})
        if dataset_name in datasets_info:
            entry = datasets_info[dataset_name]
            file_path = data_dir / entry["file"]
            if not file_path.exists():
                raise FileNotFoundError(f"Eval dataset not found: {file_path}")

            content = file_path.read_bytes()
            actual_sha = hashlib.sha256(content).hexdigest()
            expected_sha = entry.get("sha256", "")
            if expected_sha and actual_sha != expected_sha:
                raise ValueError(
                    f"Checksum mismatch for {dataset_name}: "
                    f"expected {expected_sha[:16]}..., got {actual_sha[:16]}..."
                )

            problems = [
                json.loads(line)
                for line in content.decode().splitlines()
                if line.strip()
            ]
            expected_n = entry.get("n")
            if expected_n and len(problems) != expected_n:
                raise ValueError(
                    f"Count mismatch for {dataset_name}: "
                    f"expected {expected_n}, got {len(problems)}"
                )
            return problems

    # Fallback: try direct file load
    for pattern in [f"{dataset_name}.jsonl", f"{dataset_name}_test.jsonl"]:
        file_path = data_dir / pattern
        if file_path.exists():
            return [
                json.loads(line)
                for line in file_path.read_text().splitlines()
                if line.strip()
            ]

    raise FileNotFoundError(
        f"No eval data found for '{dataset_name}' in {data_dir}"
    )


def get_manifest_sha(data_dir: Path) -> str:
    """Return SHA256 of the manifest file itself."""
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        return hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return ""
