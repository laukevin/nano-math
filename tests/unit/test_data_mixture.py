"""Unit tests for data mixture weight math and dataloader logic.

Tests the mixture weight normalization, weighted sampling distribution,
shard reading/writing, and config validation — all without I/O to HuggingFace.
"""

import os

import numpy as np
import pytest

from math_nano.data.dataloader import (
    DataSourceConfig,
    MultiSourceDataLoader,
    ShardedDataSource,
    create_dataloader_from_config,
    list_shards,
    read_shard,
    write_shard,
    MIXTURES,
)


# ── Shard I/O ──


class TestShardIO:
    def test_write_read_roundtrip(self, tmp_path):
        """Write a shard, read it back, verify identical."""
        tokens = np.array([100, 200, 300, 50256, 400, 500, 50256], dtype=np.uint16)
        offsets = np.array([0, 4, 7], dtype=np.int64)

        shard_path = str(tmp_path / "shard_000000")
        write_shard(tokens, offsets, shard_path)

        read_tokens, read_offsets = read_shard(shard_path + ".bin")
        np.testing.assert_array_equal(read_tokens, tokens)
        np.testing.assert_array_equal(read_offsets, offsets)

    def test_write_creates_bin_and_idx(self, tmp_path):
        """write_shard creates both .bin and .idx files."""
        tokens = np.array([1, 2, 3], dtype=np.uint16)
        offsets = np.array([0, 3], dtype=np.int64)

        shard_path = str(tmp_path / "test")
        write_shard(tokens, offsets, shard_path)

        assert os.path.exists(shard_path + ".bin")
        assert os.path.exists(shard_path + ".idx")

    def test_bin_is_uint16(self, tmp_path):
        """Verify .bin file size matches uint16 encoding."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
        offsets = np.array([0], dtype=np.int64)

        shard_path = str(tmp_path / "test")
        write_shard(tokens, offsets, shard_path)

        # uint16 = 2 bytes per token
        assert os.path.getsize(shard_path + ".bin") == 10

    def test_read_without_idx(self, tmp_path):
        """Reading a shard without .idx returns None for offsets."""
        tokens = np.array([1, 2, 3], dtype=np.uint16)
        bin_path = tmp_path / "shard.bin"
        tokens.tofile(str(bin_path))

        read_tokens, read_offsets = read_shard(str(bin_path))
        np.testing.assert_array_equal(read_tokens, tokens)
        assert read_offsets is None

    def test_list_shards(self, tmp_path):
        """list_shards finds all .bin files sorted."""
        for i in [2, 0, 1]:
            (tmp_path / f"shard_{i:06d}.bin").write_bytes(b"\x00\x00")
        shards = list_shards(str(tmp_path))
        assert len(shards) == 3
        assert "shard_000000.bin" in shards[0]
        assert "shard_000002.bin" in shards[2]

    def test_list_shards_empty_dir(self, tmp_path):
        """list_shards returns empty list for dir with no .bin files."""
        assert list_shards(str(tmp_path)) == []

    def test_tokens_fit_uint16(self):
        """GPT-2 max token ID (50256) fits in uint16 (max 65535)."""
        assert 50256 < np.iinfo(np.uint16).max


# ── DataSourceConfig ──


class TestDataSourceConfig:
    def test_valid_config(self):
        cfg = DataSourceConfig(path="/some/path", weight=0.5)
        assert cfg.weight == 0.5

    def test_zero_weight(self):
        cfg = DataSourceConfig(path="/some/path", weight=0.0)
        assert cfg.weight == 0.0

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            DataSourceConfig(path="/some/path", weight=-0.1)


# ── Weight Normalization ──


class TestWeightNormalization:
    def _make_sources(self, tmp_path, weights):
        """Helper: create shard dirs with dummy data and return DataSourceConfigs."""
        configs = []
        for i, w in enumerate(weights):
            d = tmp_path / f"source_{i}"
            d.mkdir()
            tokens = np.arange(1000, dtype=np.uint16)
            offsets = np.array([0, 1000], dtype=np.int64)
            write_shard(tokens, offsets, str(d / "shard_000000"))
            configs.append(DataSourceConfig(path=str(d), weight=w))
        return configs

    def test_weights_sum_to_one(self, tmp_path):
        """Weights are normalized to sum to 1.0."""
        configs = self._make_sources(tmp_path, [1.0, 3.0, 6.0])
        loader = MultiSourceDataLoader(configs, seq_len=10, batch_size=4, seed=42)
        assert abs(sum(loader.weights) - 1.0) < 1e-10

    def test_equal_weights(self, tmp_path):
        """Equal input weights produce equal normalized weights."""
        configs = self._make_sources(tmp_path, [1.0, 1.0, 1.0])
        loader = MultiSourceDataLoader(configs, seq_len=10, batch_size=4, seed=42)
        for w in loader.weights:
            assert abs(w - 1 / 3) < 1e-10

    def test_single_source_weight_one(self, tmp_path):
        """Single source gets weight 1.0."""
        configs = self._make_sources(tmp_path, [5.0])
        loader = MultiSourceDataLoader(configs, seq_len=10, batch_size=4, seed=42)
        assert loader.weights == [1.0]

    def test_large_ratio(self, tmp_path):
        """90/10 split normalizes correctly."""
        configs = self._make_sources(tmp_path, [9.0, 1.0])
        loader = MultiSourceDataLoader(configs, seq_len=10, batch_size=4, seed=42)
        assert abs(loader.weights[0] - 0.9) < 1e-10
        assert abs(loader.weights[1] - 0.1) < 1e-10


# ── Weighted Sampling ──


class TestWeightedSampling:
    def _make_loader(self, tmp_path, weights, batch_size=1000, seq_len=10, seed=42):
        """Create a loader with distinguishable sources."""
        configs = []
        for i, w in enumerate(weights):
            d = tmp_path / f"source_{i}"
            d.mkdir()
            # Each source has a unique token value so we can identify it
            token_val = (i + 1) * 100  # 100, 200, 300...
            tokens = np.full(100_000, token_val, dtype=np.uint16)
            offsets = np.array([0, 100_000], dtype=np.int64)
            write_shard(tokens, offsets, str(d / "shard_000000"))
            configs.append(DataSourceConfig(path=str(d), weight=w))
        return MultiSourceDataLoader(
            configs, seq_len=seq_len, batch_size=batch_size, seed=seed, max_steps=1
        )

    def test_sampling_respects_weights(self, tmp_path):
        """With 90/10 weights, ~90% of sequences come from source 0."""
        loader = self._make_loader(tmp_path, [9.0, 1.0], batch_size=10000, seq_len=10)
        batch = next(iter(loader))
        # Check first token of each sequence to identify source
        first_tokens = batch[:, 0].numpy()
        from_source_0 = np.sum(first_tokens == 100)
        ratio = from_source_0 / len(first_tokens)
        # Should be ~0.9, allow tolerance
        assert 0.85 < ratio < 0.95, f"Expected ~0.9, got {ratio}"

    def test_all_sources_sampled(self, tmp_path):
        """With 3 sources, all get sampled over many batches."""
        loader = self._make_loader(
            tmp_path, [0.5, 0.3, 0.2], batch_size=1000, seq_len=10
        )
        batch = next(iter(loader))
        first_tokens = set(batch[:, 0].numpy().tolist())
        assert 100 in first_tokens  # source 0
        assert 200 in first_tokens  # source 1
        assert 300 in first_tokens  # source 2

    def test_zero_weight_source_not_sampled(self, tmp_path):
        """A source with weight 0 should never be sampled."""
        configs = []
        for i, w in enumerate([1.0, 0.0]):
            d = tmp_path / f"source_{i}"
            d.mkdir()
            token_val = (i + 1) * 100
            tokens = np.full(100_000, token_val, dtype=np.uint16)
            offsets = np.array([0, 100_000], dtype=np.int64)
            write_shard(tokens, offsets, str(d / "shard_000000"))
            configs.append(DataSourceConfig(path=str(d), weight=w))

        # With one zero weight, only 1 source contributes
        # The loader normalizes [1.0, 0.0] but np.random.choice can't handle p=0
        # in the array. Actually it can — p=[1.0, 0.0] is valid.
        loader = MultiSourceDataLoader(
            configs, seq_len=10, batch_size=100, seed=42, max_steps=1
        )
        batch = next(iter(loader))
        first_tokens = set(batch[:, 0].numpy().tolist())
        assert 100 in first_tokens
        assert 200 not in first_tokens


# ── Batch Shape and Types ──


class TestBatchFormat:
    def _make_loader(self, tmp_path, seq_len=128, batch_size=4):
        d = tmp_path / "source"
        d.mkdir()
        tokens = np.arange(100_000, dtype=np.uint16) % 50257
        offsets = np.array([0, 100_000], dtype=np.int64)
        write_shard(tokens, offsets, str(d / "shard_000000"))
        return MultiSourceDataLoader(
            [DataSourceConfig(path=str(d), weight=1.0)],
            seq_len=seq_len,
            batch_size=batch_size,
            seed=42,
            max_steps=3,
        )

    def test_batch_shape(self, tmp_path):
        loader = self._make_loader(tmp_path, seq_len=128, batch_size=4)
        batch = next(iter(loader))
        assert batch.shape == (4, 128)

    def test_batch_dtype_int64(self, tmp_path):
        """Batches are int64 for PyTorch compatibility."""
        import torch

        loader = self._make_loader(tmp_path, seq_len=128, batch_size=4)
        batch = next(iter(loader))
        assert batch.dtype == torch.int64

    def test_token_values_valid(self, tmp_path):
        """All token IDs are in valid GPT-2 range [0, 50257)."""
        loader = self._make_loader(tmp_path, seq_len=128, batch_size=4)
        batch = next(iter(loader))
        assert (batch >= 0).all()
        assert (batch < 50257).all()

    def test_max_steps_respected(self, tmp_path):
        loader = self._make_loader(tmp_path, seq_len=32, batch_size=2)
        batches = list(loader)
        assert len(batches) == 3


# ── ShardedDataSource ──


class TestShardedDataSource:
    def test_get_tokens(self, tmp_path):
        """get_tokens returns the requested number of tokens."""
        tokens = np.arange(1000, dtype=np.uint16)
        offsets = np.array([0, 1000], dtype=np.int64)
        write_shard(tokens, offsets, str(tmp_path / "shard_000000"))

        src = ShardedDataSource(str(tmp_path), seed=42)
        result = src.get_tokens(100)
        assert len(result) == 100

    def test_wraps_around_shard(self, tmp_path):
        """get_tokens wraps to next shard when current is exhausted."""
        for i in range(2):
            tokens = np.full(500, i + 1, dtype=np.uint16)
            offsets = np.array([0, 500], dtype=np.int64)
            write_shard(tokens, offsets, str(tmp_path / f"shard_{i:06d}"))

        src = ShardedDataSource(str(tmp_path), seed=42)
        # Request more than one shard holds
        result = src.get_tokens(800)
        assert len(result) == 800

    def test_no_shards_raises(self, tmp_path):
        """Raises FileNotFoundError if directory has no .bin files."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            ShardedDataSource(str(empty))

    def test_total_tokens(self, tmp_path):
        """total_tokens sums across all shards."""
        for i in range(3):
            tokens = np.zeros(1000, dtype=np.uint16)
            offsets = np.array([0], dtype=np.int64)
            write_shard(tokens, offsets, str(tmp_path / f"shard_{i:06d}"))

        src = ShardedDataSource(str(tmp_path), seed=42)
        assert src.total_tokens == 3000


# ── Config Parsing ──


class TestCreateDataloaderFromConfig:
    def _make_shard_dir(self, tmp_path, name="source"):
        d = tmp_path / name
        d.mkdir()
        tokens = np.arange(10000, dtype=np.uint16)
        offsets = np.array([0, 10000], dtype=np.int64)
        write_shard(tokens, offsets, str(d / "shard_000000"))
        return str(d)

    def test_list_config(self, tmp_path):
        """Create loader from list of dicts."""
        path = self._make_shard_dir(tmp_path)
        config = [{"path": path, "weight": 1.0}]
        loader = create_dataloader_from_config(config, seq_len=32, batch_size=2)
        batch = next(iter(loader))
        assert batch.shape == (2, 32)

    def test_dict_config(self, tmp_path):
        """Create loader from dict with data_sources key."""
        path = self._make_shard_dir(tmp_path)
        config = {"data_sources": [{"path": path, "weight": 1.0}]}
        loader = create_dataloader_from_config(config, seq_len=32, batch_size=2)
        batch = next(iter(loader))
        assert batch.shape == (2, 32)

    def test_dict_missing_key_raises(self, tmp_path):
        with pytest.raises(ValueError, match="data_sources"):
            create_dataloader_from_config({"wrong_key": []})

    def test_empty_sources_raises(self, tmp_path):
        with pytest.raises(ValueError, match="At least one"):
            create_dataloader_from_config([])

    def test_multi_source_config(self, tmp_path):
        """Multiple sources in config."""
        p1 = self._make_shard_dir(tmp_path, "src1")
        p2 = self._make_shard_dir(tmp_path, "src2")
        config = [
            {"path": p1, "weight": 0.7},
            {"path": p2, "weight": 0.3},
        ]
        loader = create_dataloader_from_config(
            config, seq_len=32, batch_size=4, max_steps=1
        )
        batch = next(iter(loader))
        assert batch.shape == (4, 32)


# ── Mixture Info ──


class TestMixtureInfo:
    def test_get_mixture_info(self, tmp_path):
        d = tmp_path / "source"
        d.mkdir()
        tokens = np.arange(5000, dtype=np.uint16)
        offsets = np.array([0, 5000], dtype=np.int64)
        write_shard(tokens, offsets, str(d / "shard_000000"))

        loader = MultiSourceDataLoader(
            [DataSourceConfig(path=str(d), weight=1.0)],
            seq_len=64,
            batch_size=4,
        )
        info = loader.get_mixture_info()
        assert len(info["sources"]) == 1
        assert info["sources"][0]["weight"] == 1.0
        assert info["sources"][0]["n_shards"] == 1
        assert info["sources"][0]["total_tokens"] == 5000
        assert info["seq_len"] == 64
        assert info["batch_size"] == 4


# ── Predefined Mixtures ──


class TestPredefinedMixtures:
    def test_all_mixtures_defined(self):
        """All 5 spec mixtures are defined."""
        expected = {"mix-general", "mix-math-broad", "mix-math-heavy",
                    "mix-math-pure", "mix-reasoning"}
        assert set(MIXTURES.keys()) == expected

    def test_weights_sum_correctly(self):
        """Each predefined mixture's weights sum to 1.0."""
        for name, sources in MIXTURES.items():
            total = sum(s["weight"] for s in sources)
            assert abs(total - 1.0) < 1e-10, f"{name}: weights sum to {total}"

    def test_mix_general_is_fineweb_only(self):
        """mix-general uses only FineWeb-Edu at 100%."""
        sources = MIXTURES["mix-general"]
        assert len(sources) == 1
        assert "fineweb" in sources[0]["path"]
        assert sources[0]["weight"] == 1.0

    def test_mix_math_pure_no_fineweb(self):
        """mix-math-pure has no FineWeb-Edu."""
        sources = MIXTURES["mix-math-pure"]
        for s in sources:
            assert "fineweb" not in s["path"]

    def test_all_sources_have_path_and_weight(self):
        """Every source in every mixture has path and weight keys."""
        for name, sources in MIXTURES.items():
            for s in sources:
                assert "path" in s, f"{name}: source missing 'path'"
                assert "weight" in s, f"{name}: source missing 'weight'"
                assert s["weight"] >= 0, f"{name}: negative weight"
