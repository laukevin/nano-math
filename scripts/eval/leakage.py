"""Check for eval-vs-training data leakage via exact-match deduplication."""

from __future__ import annotations

import json
from pathlib import Path


def normalize_for_dedup(text: str) -> str:
    """Normalize text for exact-match comparison.

    Strips whitespace, lowercases, removes extra spaces.
    """
    return " ".join(text.lower().split())


def load_eval_problems(eval_dir: Path) -> dict[str, list[dict]]:
    """Load all eval problems from JSONL files.

    Returns:
        {dataset_name: [{"problem": str, "id": str, ...}, ...]}
    """
    problems: dict[str, list[dict]] = {}
    for jsonl_path in sorted(eval_dir.glob("*.jsonl")):
        ds_name = jsonl_path.stem
        entries = []
        for i, line in enumerate(jsonl_path.read_text().splitlines()):
            if not line.strip():
                continue
            entry = json.loads(line)
            if "id" not in entry:
                entry["id"] = f"{ds_name}_{i:04d}"
            entries.append(entry)
        if entries:
            problems[ds_name] = entries
    return problems


def load_train_texts(train_path: Path) -> set[str]:
    """Load training texts for deduplication.

    Supports:
    - Directory of JSONL files (reads "text" or "problem" field)
    - Single JSONL file
    - Directory of .txt files
    """
    texts: set[str] = set()

    if train_path.is_file():
        paths = [train_path]
    else:
        paths = list(train_path.glob("**/*.jsonl")) + list(
            train_path.glob("**/*.txt")
        )

    for p in paths:
        if p.suffix == ".jsonl":
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                text = entry.get("text") or entry.get("problem") or ""
                if text:
                    texts.add(normalize_for_dedup(text))
        elif p.suffix == ".txt":
            content = p.read_text()
            if content.strip():
                texts.add(normalize_for_dedup(content))

    return texts


def check_leakage(
    eval_problems: dict[str, list[dict]],
    train_texts: set[str],
) -> dict:
    """Check each eval problem against training corpus.

    Returns:
        Report dict with matches per dataset.
    """
    report: dict = {
        "n_train_texts": len(train_texts),
        "datasets": {},
        "total_eval_problems": 0,
        "total_matches": 0,
    }

    for ds_name, problems in eval_problems.items():
        matches = []
        for p in problems:
            problem_text = p.get("problem", "")
            normalized = normalize_for_dedup(problem_text)
            if normalized in train_texts:
                matches.append(
                    {
                        "id": p.get("id", "unknown"),
                        "problem_preview": problem_text[:100],
                    }
                )

        report["datasets"][ds_name] = {
            "n_problems": len(problems),
            "n_matches": len(matches),
            "match_rate": len(matches) / len(problems) if problems else 0.0,
            "matches": matches,
        }
        report["total_eval_problems"] += len(problems)
        report["total_matches"] += len(matches)

    return report
