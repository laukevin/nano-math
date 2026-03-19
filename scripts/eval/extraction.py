"""Answer extraction from model outputs."""

from __future__ import annotations

import math
import re
from typing import Optional


def extract_answer(text: str) -> Optional[str]:
    """Extract the final numerical answer from model output.

    Priority order:
    1. \\boxed{...}
    2. #### pattern (GSM8K style)
    3. "The answer is ..."
    4. Last number in output (aggressive fallback)
    """
    if not text or not text.strip():
        return None

    # 1. \boxed{...} — take the last one
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return normalize_answer(boxed[-1])

    # 2. #### pattern (GSM8K style)
    hash_match = re.search(r"####\s*(.+)", text)
    if hash_match:
        return normalize_answer(hash_match.group(1))

    # 3. "The answer is ..."
    answer_match = re.search(r"[Tt]he (?:final )?answer is[:\s]*(.+?)[\.\n]", text)
    if answer_match:
        return normalize_answer(answer_match.group(1))

    # 4. Last number in output (fallback)
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return normalize_answer(numbers[-1])

    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    answer = answer.strip()
    # Remove $, \, spaces, commas
    answer = re.sub(r"[\$\\,\s]", "", answer)
    # Try to evaluate as number
    try:
        val = float(answer)
        if not math.isfinite(val):
            return answer.lower()
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        return answer.lower()
