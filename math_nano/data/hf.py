"""HuggingFace dataset loading utilities."""

from collections.abc import Iterator
from typing import Any


def load_hf_dataset(
    hf_id: str,
    subset: str | None = None,
    split: str = "train",
    streaming: bool = True,
):
    """Load a HuggingFace dataset.

    Thin wrapper that standardizes the load_dataset call across all scripts.

    Args:
        hf_id: HuggingFace dataset ID (e.g. "openai/gsm8k").
        subset: Dataset subset/config name (e.g. "main").
        split: Dataset split (default "train").
        streaming: Whether to stream (default True).
    """
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": hf_id,
        "split": split,
        "streaming": streaming,
    }
    if subset is not None:
        kwargs["name"] = subset

    return load_dataset(**kwargs)


def iter_texts(
    dataset,
    text_column: str,
    max_docs: int | None = None,
) -> Iterator[str]:
    """Iterate non-empty text values from a streaming HF dataset.

    Args:
        dataset: A HuggingFace (streaming) dataset.
        text_column: Column name to extract text from.
        max_docs: Stop after this many non-empty documents.

    Yields:
        Non-empty text strings.
    """
    count = 0
    for doc in dataset:
        text = doc.get(text_column, "")
        if not text or not text.strip():
            continue
        yield text
        count += 1
        if max_docs is not None and count >= max_docs:
            break
