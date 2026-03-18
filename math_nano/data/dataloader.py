"""Multi-source dataloader for pretrain data mixtures.

Implements Option B from spec 02: accepts multiple shard directories with
weights for flexible mixture sweeps without re-tokenizing.

Shard format:
  - .bin: flat array of uint16 token IDs
  - .idx: flat array of int64 document-start offsets (byte positions into .bin)
"""

import glob
import os
import struct
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset


DTYPE = np.uint16
HEADER_MAGIC = b"MNDS"  # math-nano data shard
HEADER_VERSION = 1


def write_shard(tokens: np.ndarray, doc_offsets: np.ndarray, path: str) -> None:
    """Write a tokenized shard to disk.

    Args:
        tokens: uint16 array of token IDs.
        doc_offsets: int64 array of document start indices into tokens.
        path: output path (without extension). Writes .bin and .idx files.
    """
    bin_path = path if path.endswith(".bin") else path + ".bin"
    idx_path = bin_path.replace(".bin", ".idx")

    tokens = np.asarray(tokens, dtype=np.uint16)
    doc_offsets = np.asarray(doc_offsets, dtype=np.int64)

    tokens.tofile(bin_path)
    doc_offsets.tofile(idx_path)


def read_shard(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a tokenized shard from disk.

    Args:
        path: path to .bin file.

    Returns:
        (tokens, doc_offsets) where doc_offsets may be None if no .idx file.
    """
    bin_path = path if path.endswith(".bin") else path + ".bin"
    idx_path = bin_path.replace(".bin", ".idx")

    tokens = np.fromfile(bin_path, dtype=np.uint16)
    doc_offsets = None
    if os.path.exists(idx_path):
        doc_offsets = np.fromfile(idx_path, dtype=np.int64)

    return tokens, doc_offsets


def list_shards(shard_dir: str) -> list[str]:
    """List all .bin shard files in a directory, sorted."""
    pattern = os.path.join(shard_dir, "*.bin")
    return sorted(glob.glob(pattern))


@dataclass
class DataSourceConfig:
    """Configuration for a single data source in a mixture."""

    path: str
    weight: float

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative, got {self.weight}")


class ShardedDataSource:
    """Loads tokenized shards from a single directory."""

    def __init__(self, shard_dir: str, seed: int = 42):
        self.shard_dir = shard_dir
        self.shard_files = list_shards(shard_dir)
        if not self.shard_files:
            raise FileNotFoundError(f"No .bin shards found in {shard_dir}")
        self.rng = np.random.RandomState(seed)
        self._current_shard_idx = 0
        self._current_tokens: np.ndarray | None = None
        self._current_pos = 0
        self._shuffle_shards()

    def _shuffle_shards(self):
        """Shuffle shard order for this epoch."""
        self.rng.shuffle(self.shard_files)
        self._current_shard_idx = 0

    def _load_next_shard(self):
        """Load the next shard into memory."""
        if self._current_shard_idx >= len(self.shard_files):
            self._shuffle_shards()
        path = self.shard_files[self._current_shard_idx]
        self._current_tokens, _ = read_shard(path)
        self._current_pos = 0
        self._current_shard_idx += 1

    def get_tokens(self, count: int) -> np.ndarray:
        """Get the next `count` tokens from this source.

        Automatically advances through shards and wraps around.
        """
        result = []
        remaining = count

        while remaining > 0:
            if self._current_tokens is None or self._current_pos >= len(
                self._current_tokens
            ):
                self._load_next_shard()

            available = len(self._current_tokens) - self._current_pos
            take = min(available, remaining)
            result.append(
                self._current_tokens[self._current_pos : self._current_pos + take]
            )
            self._current_pos += take
            remaining -= take

        return np.concatenate(result)

    @property
    def total_tokens(self) -> int:
        """Total tokens across all shards (computed on first call)."""
        total = 0
        for path in list_shards(self.shard_dir):
            total += os.path.getsize(path) // 2  # uint16 = 2 bytes
        return total


class MultiSourceDataLoader(IterableDataset):
    """Dataloader that samples from multiple shard directories with weights.

    Implements Option B from spec 02: interleave tokens from multiple sources
    according to configurable mixture weights.

    Usage:
        sources = [
            DataSourceConfig(path="data/fineweb-edu/", weight=0.5),
            DataSourceConfig(path="data/openwebmath/", weight=0.4),
            DataSourceConfig(path="data/openmathreasoning/", weight=0.1),
        ]
        loader = MultiSourceDataLoader(
            sources=sources,
            seq_len=1024,
            batch_size=8,
            seed=42,
        )
        for batch in loader:
            # batch shape: (batch_size, seq_len)
            ...
    """

    def __init__(
        self,
        sources: list[DataSourceConfig],
        seq_len: int = 1024,
        batch_size: int = 8,
        seed: int = 42,
        max_steps: int | None = None,
    ):
        if not sources:
            raise ValueError("At least one data source is required")

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.seed = seed

        # Normalize weights
        total_weight = sum(s.weight for s in sources)
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")
        self.weights = [s.weight / total_weight for s in sources]

        # Create data sources
        self.sources = [
            ShardedDataSource(s.path, seed=seed + i) for i, s in enumerate(sources)
        ]
        self.source_names = [os.path.basename(s.path.rstrip("/")) for s in sources]
        self.rng = np.random.RandomState(seed)

    def _get_batch(self) -> np.ndarray:
        """Get one batch of tokens by sampling from sources by weight.

        Each sequence in the batch is drawn entirely from one source,
        selected according to the mixture weights.
        """
        batch = np.zeros((self.batch_size, self.seq_len), dtype=np.uint16)
        source_indices = self.rng.choice(
            len(self.sources), size=self.batch_size, p=self.weights
        )

        for i, src_idx in enumerate(source_indices):
            batch[i] = self.sources[src_idx].get_tokens(self.seq_len)

        return batch

    def __iter__(self):
        step = 0
        while self.max_steps is None or step < self.max_steps:
            batch = self._get_batch()
            yield torch.from_numpy(batch.astype(np.int64))
            step += 1

    def get_mixture_info(self) -> dict:
        """Return info about the configured mixture."""
        return {
            "sources": [
                {
                    "name": name,
                    "path": src.shard_dir,
                    "weight": w,
                    "n_shards": len(src.shard_files),
                    "total_tokens": src.total_tokens,
                }
                for name, src, w in zip(
                    self.source_names, self.sources, self.weights
                )
            ],
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
        }


def create_dataloader_from_config(
    config: dict | list[dict],
    seq_len: int = 1024,
    batch_size: int = 8,
    seed: int = 42,
    max_steps: int | None = None,
) -> MultiSourceDataLoader:
    """Create a MultiSourceDataLoader from a config dict or list.

    Config format (list):
        [
            {"path": "data/fineweb-edu/", "weight": 0.5},
            {"path": "data/openwebmath/", "weight": 0.4},
        ]

    Config format (dict with data_sources key):
        {"data_sources": [{"path": "...", "weight": ...}, ...]}
    """
    if isinstance(config, dict):
        if "data_sources" not in config:
            raise ValueError("Config dict must have 'data_sources' key")
        config = config["data_sources"]

    sources = [DataSourceConfig(path=c["path"], weight=c["weight"]) for c in config]

    return MultiSourceDataLoader(
        sources=sources,
        seq_len=seq_len,
        batch_size=batch_size,
        seed=seed,
        max_steps=max_steps,
    )


# Predefined mixture configs from spec 02
MIXTURES = {
    "mix-general": [
        {"path": "data/tokenized/fineweb-edu/", "weight": 1.0},
    ],
    "mix-math-broad": [
        {"path": "data/tokenized/fineweb-edu/", "weight": 0.5},
        {"path": "data/tokenized/openwebmath/", "weight": 0.4},
        {"path": "data/tokenized/openmathreasoning/", "weight": 0.1},
    ],
    "mix-math-heavy": [
        {"path": "data/tokenized/fineweb-edu/", "weight": 0.2},
        {"path": "data/tokenized/openwebmath/", "weight": 0.5},
        {"path": "data/tokenized/openmathreasoning/", "weight": 0.3},
    ],
    "mix-math-pure": [
        {"path": "data/tokenized/openwebmath/", "weight": 0.6},
        {"path": "data/tokenized/openmathreasoning/", "weight": 0.4},
    ],
    "mix-reasoning": [
        {"path": "data/tokenized/fineweb-edu/", "weight": 0.3},
        {"path": "data/tokenized/openwebmath/", "weight": 0.2},
        {"path": "data/tokenized/openmathreasoning/", "weight": 0.5},
    ],
}
