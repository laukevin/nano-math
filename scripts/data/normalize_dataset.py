"""Download and normalize math datasets from HuggingFace to JSONL.

Output format: one JSON object per line:
    {"problem": "...", "solution": "...", "source": "dataset_name"}

Usage:
    python scripts/data/normalize_dataset.py --dataset gsm8k --output /data/sft/gsm8k/train.jsonl
    python scripts/data/normalize_dataset.py --dataset gsm8k --output /data/sft/gsm8k/train.jsonl --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import os
import re

DATASETS = {
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "subset": "main",
        "split": "train",
    },
    "metamath": {
        "hf_id": "meta-math/MetaMathQA",
        "split": "train",
    },
    "numinamath": {
        "hf_id": "AI-MO/NuminaMath-CoT",
        "split": "train",
    },
    "math": {
        "hf_id": "lighteval/MATH",
        "split": "train",
    },
    "openmathinstruct2": {
        "hf_id": "nvidia/OpenMathInstruct-2",
        "split": "train_1M",
    },
    "openthoughts3": {
        "hf_id": "open-thoughts/OpenThoughts-114k",
        "split": "train",
    },
    "stratos": {
        "hf_id": "bespokelabs/Bespoke-Stratos-17k",
        "split": "train",
    },
    "dartmath": {
        "hf_id": "hkust-nlp/dart-math-hard",
        "split": "train",
    },
    "mathinstruct": {
        "hf_id": "TIGER-Lab/MathInstruct",
        "split": "train",
    },
    "numinamath15": {
        "hf_id": "AI-MO/NuminaMath-1.5",
        "split": "train",
    },
    "acemath": {
        "hf_id": "nvidia/AceMath-Instruct-Training-Data",
        "subset": "math_sft",
        "split": "train",
    },
}


def ensure_boxed(text: str) -> str:
    """Ensure solution text has \\boxed{} format for the final answer."""
    if "\\boxed{" in text:
        return text
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return text + f"\n\nThe answer is \\boxed{{{numbers[-1]}}}."
    return text


def normalize_gsm8k(row: dict) -> dict | None:
    question = row["question"]
    answer_text = row["answer"]
    answer_text = re.sub(r"<<[^>]+>>", "", answer_text)
    parts = answer_text.split("####")
    reasoning = parts[0].strip()
    final_answer = parts[-1].strip() if len(parts) > 1 else ""
    if not final_answer:
        return None
    solution = f"{reasoning}\n\nThe answer is \\boxed{{{final_answer}}}."
    return {"problem": question, "solution": solution, "source": "gsm8k"}


def normalize_metamath(row: dict) -> dict | None:
    return {
        "problem": row["query"],
        "solution": ensure_boxed(row["response"]),
        "source": "metamath",
    }


def normalize_numinamath(row: dict) -> dict | None:
    return {
        "problem": row["problem"],
        "solution": ensure_boxed(row["solution"]),
        "source": "numinamath",
    }


def normalize_math(row: dict) -> dict | None:
    return {
        "problem": row["problem"],
        "solution": ensure_boxed(row["solution"]),
        "source": "math",
    }


def normalize_openmathinstruct2(row: dict) -> dict | None:
    return {
        "problem": row["problem"],
        "solution": ensure_boxed(row["generated_solution"]),
        "source": "openmathinstruct2",
    }


def normalize_openthoughts3(row: dict) -> dict | None:
    # OpenThoughts format: has 'problem', 'solution', and 'conversations'
    problem = row.get("problem") or ""
    solution = row.get("solution") or ""
    if not problem or not solution:
        return None
    return {
        "problem": problem,
        "solution": ensure_boxed(solution),
        "source": "openthoughts3",
    }


def normalize_stratos(row: dict) -> dict | None:
    # Bespoke-Stratos: conversations format
    conversations = row.get("conversations", [])
    problem = solution = ""
    for turn in conversations:
        if turn.get("role") == "user":
            problem = turn.get("content", "")
        elif turn.get("role") == "assistant":
            solution = turn.get("content", "")
    if not problem or not solution:
        return None
    return {
        "problem": problem,
        "solution": ensure_boxed(solution),
        "source": "stratos",
    }


def normalize_dartmath(row: dict) -> dict | None:
    problem = row.get("query") or row.get("problem") or ""
    solution = row.get("response") or row.get("solution") or ""
    if not problem or not solution:
        return None
    return {
        "problem": problem,
        "solution": ensure_boxed(solution),
        "source": "dartmath",
    }


def normalize_mathinstruct(row: dict) -> dict | None:
    problem = row.get("instruction") or ""
    solution = row.get("output") or ""
    if not problem or not solution:
        return None
    return {
        "problem": problem,
        "solution": ensure_boxed(solution),
        "source": "mathinstruct",
    }


def normalize_numinamath15(row: dict) -> dict | None:
    problem = row.get("problem") or ""
    solution = row.get("solution") or ""
    if not problem or not solution:
        return None
    return {
        "problem": problem,
        "solution": ensure_boxed(solution),
        "source": "numinamath15",
    }


def normalize_acemath(row: dict) -> dict | None:
    # AceMath uses messages format
    messages = row.get("messages", [])
    problem = solution = ""
    for msg in messages:
        if msg.get("role") == "user":
            problem = msg.get("content", "")
        elif msg.get("role") == "assistant":
            solution = msg.get("content", "")
    if not problem or not solution:
        return None
    return {
        "problem": problem,
        "solution": ensure_boxed(solution),
        "source": "acemath",
    }


NORMALIZERS = {
    "gsm8k": normalize_gsm8k,
    "metamath": normalize_metamath,
    "numinamath": normalize_numinamath,
    "math": normalize_math,
    "openmathinstruct2": normalize_openmathinstruct2,
    "openthoughts3": normalize_openthoughts3,
    "stratos": normalize_stratos,
    "dartmath": normalize_dartmath,
    "mathinstruct": normalize_mathinstruct,
    "numinamath15": normalize_numinamath15,
    "acemath": normalize_acemath,
}


def main():
    parser = argparse.ArgumentParser(description="Download and normalize math datasets")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-samples", type=int, default=-1)
    args = parser.parse_args()

    from datasets import load_dataset

    config = DATASETS[args.dataset]
    print(f"Loading {args.dataset} from {config['hf_id']}...")
    ds = load_dataset(
        config["hf_id"],
        name=config.get("subset"),
        split=config["split"],
        trust_remote_code=True,
    )

    normalize = NORMALIZERS[args.dataset]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    count = 0
    skipped = 0
    with open(args.output, "w") as f:
        for row in ds:
            if 0 < args.max_samples <= count:
                break
            try:
                normalized = normalize(row)
                if normalized and normalized["problem"].strip() and normalized["solution"].strip():
                    f.write(json.dumps(normalized) + "\n")
                    count += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

    print(f"Saved {count} samples to {args.output} ({skipped} skipped)")

    # Stats file
    stats_dir = os.path.dirname(args.output)
    stats_path = os.path.join(stats_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "hf_id": config["hf_id"],
                "samples": count,
                "skipped": skipped,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
