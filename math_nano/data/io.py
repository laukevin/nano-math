"""Shared I/O utilities: JSONL and file hashing."""

import hashlib
import json


def write_jsonl(samples: list[dict], path: str) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def sha256_file(path: str) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
