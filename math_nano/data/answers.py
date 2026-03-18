r"""Answer extraction and normalization for math problems.

Handles \boxed{}, ####, "The answer is", and last-number fallback.
"""

import re


def extract_boxed(text: str) -> str | None:
    r"""Extract the last \boxed{...} content from text.

    Returns None if no \boxed{} found.
    """
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1] if matches else None


def extract_answer_gsm8k(text: str) -> str | None:
    """Extract the answer after #### (GSM8K format).

    Returns None if no #### found.
    """
    match = re.search(r"####\s*(.+?)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def extract_last_number(text: str) -> str | None:
    """Extract the last number in text (aggressive fallback)."""
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else None


def ensure_boxed_answer(solution: str) -> str:
    r"""Ensure the solution contains a \boxed{} answer.

    If already present, return as-is. Otherwise try to extract the answer
    via ####, "The answer is", or last-number fallback and append \boxed{}.
    """
    if r"\boxed{" in solution:
        return solution

    # Try #### format (GSM8K)
    answer = extract_answer_gsm8k(solution)
    if answer:
        return solution + f"\n\nThe answer is \\boxed{{{answer}}}"

    # Try "The answer is X"
    match = re.search(r"[Tt]he (?:final )?answer is[:\s]*(.+?)[\.\n]", solution)
    if match:
        answer = match.group(1).strip()
        return solution + f"\n\nThe answer is \\boxed{{{answer}}}"

    # Try last number
    answer = extract_last_number(solution)
    if answer:
        return solution + f"\n\nThe answer is \\boxed{{{answer}}}"

    return solution


def normalize_answer_for_eval(answer: str) -> str:
    """Normalize an answer string for consistent eval comparison.

    Strips whitespace, removes commas and trailing periods.
    """
    answer = answer.strip()
    answer = answer.replace(",", "")
    answer = answer.rstrip(".")
    return answer
