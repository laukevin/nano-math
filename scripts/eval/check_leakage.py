"""Check for eval-vs-training data leakage via exact-match deduplication.

Usage:
    python scripts/eval/check_leakage.py \
      --eval-dir data/eval \
      --train-dir data/tokenized \
      --output results/leakage_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Check eval-vs-training data leakage"
    )
    parser.add_argument(
        "--eval-dir",
        default="data/eval",
        help="Directory containing eval JSONL files",
    )
    parser.add_argument(
        "--train-dir",
        required=True,
        help="Directory or file containing training data",
    )
    parser.add_argument(
        "--output",
        default="results/leakage_report.json",
        help="Output report path",
    )

    args = parser.parse_args(argv)
    eval_dir = Path(args.eval_dir)
    train_path = Path(args.train_dir)

    if not eval_dir.exists():
        logger.error("Eval directory not found: %s", eval_dir)
        return

    if not train_path.exists():
        logger.error("Training data not found: %s", train_path)
        return

    # Load data
    logger.info("Loading eval problems from %s", eval_dir)
    eval_problems = load_eval_problems(eval_dir)
    for ds_name, probs in eval_problems.items():
        logger.info("  %s: %d problems", ds_name, len(probs))

    logger.info("Loading training texts from %s", train_path)
    train_texts = load_train_texts(train_path)
    logger.info("  %d unique training texts loaded", len(train_texts))

    # Check leakage
    report = check_leakage(eval_problems, train_texts)

    # Print summary
    print(f"\n{'Dataset':<20} {'Problems':>10} {'Matches':>10} {'Rate':>8}")
    print("-" * 50)
    for ds_name, ds_report in report["datasets"].items():
        print(
            f"{ds_name:<20} "
            f"{ds_report['n_problems']:>10} "
            f"{ds_report['n_matches']:>10} "
            f"{ds_report['match_rate']:>8.2%}"
        )
    print("-" * 50)
    print(
        f"{'TOTAL':<20} "
        f"{report['total_eval_problems']:>10} "
        f"{report['total_matches']:>10}"
    )

    if report["total_matches"] > 0:
        print(
            f"\nWARNING: {report['total_matches']} eval problems found "
            f"in training data!"
        )
    else:
        print("\nNo leakage detected.")

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    logger.info("Report saved to %s", output_path)


if __name__ == "__main__":
    main()
