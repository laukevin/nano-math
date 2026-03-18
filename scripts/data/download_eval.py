#!/usr/bin/env python3
"""Download and freeze blessed eval datasets.

Downloads GSM8K, MATH500, AMC, AIME, and Minerva eval sets into
data/eval/ as JSONL files + creates manifest.json with SHA256 checksums.

These files are version-controlled and NEVER modified after initial creation.

Usage:
    python scripts/data/download_eval.py
    python scripts/data/download_eval.py --output data/eval/ --seed 42
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from math_nano.data.answers import (
    extract_answer_gsm8k,
    extract_boxed,
    normalize_answer_for_eval,
)
from math_nano.data.hf import load_hf_dataset
from math_nano.data.io import sha256_file, write_jsonl


def create_manifest(output_dir: str, datasets_info: dict) -> str:
    """Create manifest.json with SHA256 checksums for all eval files."""
    manifest = {
        "version": "1.0",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datasets": {},
    }

    for name, info in datasets_info.items():
        filepath = info["path"]
        if os.path.exists(filepath):
            manifest["datasets"][name] = {
                "file": os.path.basename(filepath),
                "sha256": sha256_file(filepath),
                "n": info["count"],
            }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def verify_manifest(output_dir: str) -> bool:
    """Verify all eval files match their manifest checksums."""
    manifest_path = os.path.join(output_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print("No manifest.json found!")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    all_ok = True
    for name, info in manifest["datasets"].items():
        filepath = os.path.join(output_dir, info["file"])
        if not os.path.exists(filepath):
            print(f"  MISSING: {info['file']}")
            all_ok = False
            continue

        actual_hash = sha256_file(filepath)
        if actual_hash != info["sha256"]:
            print(
                f"  MISMATCH: {info['file']} "
                f"(expected {info['sha256'][:16]}..., got {actual_hash[:16]}...)"
            )
            all_ok = False
        else:
            print(f"  OK: {info['file']} ({info['n']} problems)")

    return all_ok


# ── Dataset downloaders ──


def download_gsm8k(output_dir: str, seed: int) -> tuple[str, str, int, int]:
    """Download GSM8K test set and create mini subset."""
    print("  Loading GSM8K test set...")
    ds = load_hf_dataset("openai/gsm8k", subset="main", split="test", streaming=False)

    samples = []
    for item in ds:
        answer = extract_answer_gsm8k(item["answer"]) or ""
        answer = normalize_answer_for_eval(answer)

        samples.append(
            {
                "problem_id": f"gsm8k_{len(samples):04d}",
                "dataset": "gsm8k",
                "problem": item["question"],
                "answer": answer,
                "solution": item["answer"],
                "difficulty": None,
            }
        )

    full_path = os.path.join(output_dir, "gsm8k_test.jsonl")
    write_jsonl(samples, full_path)

    # Mini subset (200 problems, seeded)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(samples), size=min(200, len(samples)), replace=False)
    mini_samples = [samples[i] for i in sorted(indices)]

    mini_path = os.path.join(output_dir, "gsm8k_mini.jsonl")
    write_jsonl(mini_samples, mini_path)

    print(f"    GSM8K: {len(samples)} test, {len(mini_samples)} mini")
    return full_path, mini_path, len(samples), len(mini_samples)


def download_math500(output_dir: str, seed: int) -> tuple[str, str, int, int]:
    """Download MATH500 test subset and create mini subset."""
    print("  Loading MATH test set...")
    try:
        ds = load_hf_dataset("lighteval/MATH", split="test", streaming=False)
    except Exception:
        ds = load_hf_dataset("hendrycks/competition_math", split="test", streaming=False)

    samples = []
    for item in ds:
        solution = item.get("solution", "")
        answer = extract_boxed(solution) or ""
        answer = normalize_answer_for_eval(answer)

        level = item.get("level", None)
        if isinstance(level, str) and level.startswith("Level "):
            level = int(level.split()[-1])

        samples.append(
            {
                "problem_id": f"math_{len(samples):04d}",
                "dataset": "math500",
                "problem": item.get("problem", ""),
                "answer": answer,
                "solution": solution,
                "difficulty": level,
                "subject": item.get("type", None),
            }
        )

    # Cap at 500
    if len(samples) > 500:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(samples), size=500, replace=False)
        samples = [samples[i] for i in sorted(indices)]

    full_path = os.path.join(output_dir, "math500.jsonl")
    write_jsonl(samples, full_path)

    # Mini subset: 100 problems
    rng = np.random.RandomState(seed)
    n_mini = min(100, len(samples))
    indices = rng.choice(len(samples), size=n_mini, replace=False)
    mini_samples = [samples[i] for i in sorted(indices)]

    mini_path = os.path.join(output_dir, "math_mini.jsonl")
    write_jsonl(mini_samples, mini_path)

    print(f"    MATH500: {len(samples)} test, {len(mini_samples)} mini")
    return full_path, mini_path, len(samples), len(mini_samples)


def download_amc(output_dir: str) -> tuple[str, int]:
    """Download AMC problems from NuminaMath-CoT."""
    print("  Loading AMC problems from NuminaMath-CoT...")
    ds = load_hf_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)

    samples = []
    for item in ds:
        source = item.get("source", "")
        if "amc" not in source.lower():
            continue

        solution = item.get("solution", "")
        answer = extract_boxed(solution) or ""
        answer = normalize_answer_for_eval(answer)

        samples.append(
            {
                "problem_id": f"amc_{len(samples):04d}",
                "dataset": "amc",
                "problem": item.get("problem", ""),
                "answer": answer,
                "solution": solution,
                "difficulty": None,
                "source": source,
            }
        )

        if len(samples) >= 300:
            break

    path = os.path.join(output_dir, "amc.jsonl")
    write_jsonl(samples, path)
    print(f"    AMC: {len(samples)} problems")
    return path, len(samples)


def download_aime(output_dir: str) -> tuple[str, int]:
    """Download AIME problems."""
    print("  Loading AIME problems...")
    try:
        ds = load_hf_dataset(
            "AI-MO/aimo-validation-aime", split="train", streaming=False
        )
    except Exception:
        # Fallback: extract from NuminaMath
        print("    Falling back to NuminaMath-CoT for AIME problems...")
        ds = load_hf_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
        samples = []
        for item in ds:
            source = item.get("source", "")
            if "aime" not in source.lower():
                continue

            solution = item.get("solution", "")
            answer = extract_boxed(solution) or ""
            answer = normalize_answer_for_eval(answer)

            samples.append(
                {
                    "problem_id": f"aime_{len(samples):04d}",
                    "dataset": "aime",
                    "problem": item.get("problem", ""),
                    "answer": answer,
                    "solution": solution,
                    "difficulty": None,
                    "source": source,
                }
            )
            if len(samples) >= 150:
                break

        path = os.path.join(output_dir, "aime.jsonl")
        write_jsonl(samples, path)
        print(f"    AIME: {len(samples)} problems")
        return path, len(samples)

    samples = []
    for item in ds:
        problem = item.get("problem", item.get("question", ""))
        answer = str(item.get("answer", ""))
        answer = normalize_answer_for_eval(answer)

        samples.append(
            {
                "problem_id": f"aime_{len(samples):04d}",
                "dataset": "aime",
                "problem": problem,
                "answer": answer,
                "solution": item.get("solution", ""),
                "difficulty": None,
            }
        )

    path = os.path.join(output_dir, "aime.jsonl")
    write_jsonl(samples, path)
    print(f"    AIME: {len(samples)} problems")
    return path, len(samples)


def download_minerva(output_dir: str) -> tuple[str, int]:
    """Download Minerva math problems."""
    print("  Loading Minerva math problems...")
    try:
        ds = load_hf_dataset("google/minerva_math", split="test", streaming=False)
    except Exception:
        print("    Minerva dataset not directly available. Creating placeholder.")
        path = os.path.join(output_dir, "minerva.jsonl")
        write_jsonl([], path)
        return path, 0

    samples = []
    for item in ds:
        answer = str(item.get("answer", item.get("solution", "")))
        boxed = extract_boxed(answer)
        if boxed:
            answer = boxed
        answer = normalize_answer_for_eval(answer)

        samples.append(
            {
                "problem_id": f"minerva_{len(samples):04d}",
                "dataset": "minerva",
                "problem": item.get("problem", item.get("question", "")),
                "answer": answer,
                "solution": item.get("solution", ""),
                "difficulty": None,
                "subject": item.get("subject", item.get("type", None)),
            }
        )

    path = os.path.join(output_dir, "minerva.jsonl")
    write_jsonl(samples, path)
    print(f"    Minerva: {len(samples)} problems")
    return path, len(samples)


def main():
    parser = argparse.ArgumentParser(
        description="Download and freeze blessed eval datasets"
    )
    parser.add_argument(
        "--output",
        default="data/eval/",
        help="Output directory (default: data/eval/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subset selection (default: 42)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing manifest checksums",
    )

    args = parser.parse_args()
    output_dir = args.output

    if args.verify_only:
        print("Verifying eval dataset checksums:")
        ok = verify_manifest(output_dir)
        sys.exit(0 if ok else 1)

    os.makedirs(output_dir, exist_ok=True)

    print("Downloading blessed eval datasets:")
    datasets_info = {}

    gsm8k_path, gsm8k_mini_path, gsm8k_n, gsm8k_mini_n = download_gsm8k(
        output_dir, args.seed
    )
    datasets_info["gsm8k"] = {"path": gsm8k_path, "count": gsm8k_n}
    datasets_info["gsm8k_mini"] = {"path": gsm8k_mini_path, "count": gsm8k_mini_n}

    math_path, math_mini_path, math_n, math_mini_n = download_math500(
        output_dir, args.seed
    )
    datasets_info["math500"] = {"path": math_path, "count": math_n}
    datasets_info["math_mini"] = {"path": math_mini_path, "count": math_mini_n}

    amc_path, amc_n = download_amc(output_dir)
    datasets_info["amc"] = {"path": amc_path, "count": amc_n}

    aime_path, aime_n = download_aime(output_dir)
    datasets_info["aime"] = {"path": aime_path, "count": aime_n}

    minerva_path, minerva_n = download_minerva(output_dir)
    datasets_info["minerva"] = {"path": minerva_path, "count": minerva_n}

    manifest_path = create_manifest(output_dir, datasets_info)

    total = sum(info["count"] for info in datasets_info.values())
    print(f"\nDone! {total} total eval problems in {output_dir}")
    print(f"Manifest: {manifest_path}")

    print("\nVerifying checksums:")
    verify_manifest(output_dir)


if __name__ == "__main__":
    main()
