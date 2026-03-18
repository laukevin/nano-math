"""Integration tests for the data loading pipeline.

Tests shard creation, loading, tokenization roundtrip, multi-source
dataloader with real (tiny) data, and SFT formatting. All tests run
on CPU with no network access required (uses synthetic data).
"""

import json
import os
import tempfile

import numpy as np
import pytest
import tiktoken

from math_nano.data.dataloader import (
    DataSourceConfig,
    MultiSourceDataLoader,
    ShardedDataSource,
    read_shard,
    write_shard,
)


# ── Fixtures ──


@pytest.fixture(scope="module")
def tokenizer():
    """GPT-2 BPE tokenizer."""
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def sample_texts():
    """A few sample math-like texts for tokenization tests."""
    return [
        "What is 2 + 2? The answer is 4.",
        "Solve for x: 3x + 5 = 20. Subtract 5: 3x = 15. Divide by 3: x = 5.",
        "The area of a circle with radius r is A = pi * r^2.",
        "If f(x) = x^2 + 3x + 2, then f(0) = 2 and f(1) = 6.",
        "The sum of the first n natural numbers is n*(n+1)/2.",
    ]


@pytest.fixture
def shard_dir_with_data(tmp_path, tokenizer, sample_texts):
    """Create a shard directory with tokenized sample data."""
    all_tokens = []
    doc_offsets = [0]

    for text in sample_texts:
        tokens = tokenizer.encode_ordinary(text)
        tokens.append(50256)  # EOT
        all_tokens.extend(tokens)
        doc_offsets.append(len(all_tokens))

    token_arr = np.array(all_tokens, dtype=np.uint16)
    offset_arr = np.array(doc_offsets, dtype=np.int64)

    shard_path = str(tmp_path / "shard_000000")
    write_shard(token_arr, offset_arr, shard_path)

    return tmp_path, token_arr, offset_arr


@pytest.fixture
def multi_source_dirs(tmp_path, tokenizer):
    """Create multiple shard directories simulating different data sources."""
    sources = {}
    source_texts = {
        "fineweb": [
            "The history of mathematics spans thousands of years.",
            "Education is the foundation of a productive society.",
            "Scientific research advances human knowledge.",
        ],
        "openwebmath": [
            "Let x be a real number such that x^2 = 4. Then x = 2 or x = -2.",
            "The derivative of sin(x) is cos(x).",
            "The integral of 1/x dx is ln|x| + C.",
        ],
        "openmathreasoning": [
            "Problem: Find 15% of 80.\nStep 1: 15% = 0.15\nStep 2: 0.15 * 80 = 12\nAnswer: 12",
            "Problem: Solve 2x + 4 = 10.\nStep 1: 2x = 6\nStep 2: x = 3\nAnswer: 3",
        ],
    }

    for name, texts in source_texts.items():
        d = tmp_path / name
        d.mkdir()

        all_tokens = []
        doc_offsets = [0]
        for text in texts:
            tokens = tokenizer.encode_ordinary(text)
            tokens.append(50256)
            all_tokens.extend(tokens)
            doc_offsets.append(len(all_tokens))

        token_arr = np.array(all_tokens, dtype=np.uint16)
        offset_arr = np.array(doc_offsets, dtype=np.int64)
        write_shard(token_arr, offset_arr, str(d / "shard_000000"))

        sources[name] = str(d)

    return sources


# ── Tokenization Roundtrip ──


class TestTokenizationRoundtrip:
    def test_encode_decode_roundtrip(self, tokenizer, sample_texts):
        """Tokenize and detokenize produces original text."""
        for text in sample_texts:
            tokens = tokenizer.encode_ordinary(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text

    def test_tokens_fit_uint16(self, tokenizer, sample_texts):
        """All token IDs fit in uint16 range."""
        for text in sample_texts:
            tokens = tokenizer.encode_ordinary(text)
            for t in tokens:
                assert 0 <= t <= 65535, f"Token {t} out of uint16 range"
            # Also check EOT token
            assert 50256 <= 65535

    def test_eot_token_value(self, tokenizer):
        """EOT token ID is 50256 for GPT-2."""
        # The EOT token for GPT-2 is well-known to be 50256
        assert tokenizer.n_vocab == 50257
        # Token 50256 is the last valid token (EOT)

    def test_shard_roundtrip(self, shard_dir_with_data, tokenizer):
        """Write tokens to shard, read back, detokenize to original text."""
        shard_dir, original_tokens, original_offsets = shard_dir_with_data

        # Read back
        read_tokens, read_offsets = read_shard(
            str(shard_dir / "shard_000000.bin")
        )

        np.testing.assert_array_equal(read_tokens, original_tokens)
        np.testing.assert_array_equal(read_offsets, original_offsets)

    def test_document_boundaries(self, shard_dir_with_data, tokenizer, sample_texts):
        """Document boundaries in .idx allow extracting individual docs."""
        shard_dir, tokens, offsets = shard_dir_with_data

        for i in range(len(sample_texts)):
            start = offsets[i]
            end = offsets[i + 1]
            doc_tokens = tokens[start:end]

            # Last token should be EOT
            assert doc_tokens[-1] == 50256

            # Decode without EOT should match original
            decoded = tokenizer.decode(doc_tokens[:-1].tolist())
            assert decoded == sample_texts[i]


# ── Shard Loading ──


class TestShardLoading:
    def test_load_single_shard(self, shard_dir_with_data):
        """ShardedDataSource loads a single shard correctly."""
        shard_dir, tokens, _ = shard_dir_with_data
        src = ShardedDataSource(str(shard_dir), seed=42)
        assert len(src.shard_files) == 1

    def test_get_exact_tokens(self, shard_dir_with_data):
        """get_tokens returns exactly the requested count."""
        shard_dir, _, _ = shard_dir_with_data
        src = ShardedDataSource(str(shard_dir), seed=42)
        for count in [1, 10, 50, 100]:
            result = src.get_tokens(count)
            assert len(result) == count

    def test_multiple_shards(self, tmp_path):
        """Source with multiple shards reads through all of them."""
        for i in range(3):
            tokens = np.full(100, i + 1, dtype=np.uint16)
            offsets = np.array([0, 100], dtype=np.int64)
            write_shard(tokens, offsets, str(tmp_path / f"shard_{i:06d}"))

        src = ShardedDataSource(str(tmp_path), seed=0)
        # Read enough to span all shards
        result = src.get_tokens(300)
        assert len(result) == 300
        # Should contain values from different shards
        unique = set(result.tolist())
        assert len(unique) == 3


# ── Multi-Source DataLoader ──


class TestMultiSourceDataLoader:
    def test_creates_from_multiple_sources(self, multi_source_dirs):
        """Loader initializes with multiple source directories."""
        sources = [
            DataSourceConfig(path=multi_source_dirs["fineweb"], weight=0.5),
            DataSourceConfig(path=multi_source_dirs["openwebmath"], weight=0.4),
            DataSourceConfig(path=multi_source_dirs["openmathreasoning"], weight=0.1),
        ]
        loader = MultiSourceDataLoader(
            sources, seq_len=32, batch_size=4, seed=42, max_steps=1
        )
        batch = next(iter(loader))
        assert batch.shape == (4, 32)

    def test_different_seq_lens(self, multi_source_dirs):
        """Loader works with different sequence lengths."""
        sources = [
            DataSourceConfig(path=multi_source_dirs["fineweb"], weight=1.0),
        ]
        for seq_len in [16, 32, 64, 128]:
            loader = MultiSourceDataLoader(
                sources, seq_len=seq_len, batch_size=2, seed=42, max_steps=1
            )
            batch = next(iter(loader))
            assert batch.shape == (2, seq_len)

    def test_reproducible_with_same_seed(self, multi_source_dirs):
        """Same seed produces same batches."""
        sources = [
            DataSourceConfig(path=multi_source_dirs["fineweb"], weight=0.5),
            DataSourceConfig(path=multi_source_dirs["openwebmath"], weight=0.5),
        ]

        loader1 = MultiSourceDataLoader(
            sources, seq_len=32, batch_size=4, seed=123, max_steps=3
        )
        loader2 = MultiSourceDataLoader(
            sources, seq_len=32, batch_size=4, seed=123, max_steps=3
        )

        for b1, b2 in zip(loader1, loader2):
            assert (b1 == b2).all()

    def test_different_seeds_differ(self, multi_source_dirs):
        """Different seeds produce different batches."""
        sources = [
            DataSourceConfig(path=multi_source_dirs["fineweb"], weight=0.5),
            DataSourceConfig(path=multi_source_dirs["openwebmath"], weight=0.5),
        ]

        loader1 = MultiSourceDataLoader(
            sources, seq_len=32, batch_size=8, seed=42, max_steps=1
        )
        loader2 = MultiSourceDataLoader(
            sources, seq_len=32, batch_size=8, seed=99, max_steps=1
        )

        b1 = next(iter(loader1))
        b2 = next(iter(loader2))
        # Very unlikely to be identical with different seeds
        assert not (b1 == b2).all()

    def test_mixture_weight_distribution(self, tmp_path):
        """Verify sampling distribution matches weights over many samples."""
        # Create two sources with distinct tokens
        for i, name in enumerate(["src_a", "src_b"]):
            d = tmp_path / name
            d.mkdir()
            val = (i + 1) * 111  # 111 vs 222
            tokens = np.full(100_000, val, dtype=np.uint16)
            offsets = np.array([0, 100_000], dtype=np.int64)
            write_shard(tokens, offsets, str(d / "shard_000000"))

        sources = [
            DataSourceConfig(path=str(tmp_path / "src_a"), weight=0.7),
            DataSourceConfig(path=str(tmp_path / "src_b"), weight=0.3),
        ]
        loader = MultiSourceDataLoader(
            sources, seq_len=10, batch_size=5000, seed=42, max_steps=1
        )
        batch = next(iter(loader))
        first_tokens = batch[:, 0].numpy()
        ratio_a = np.mean(first_tokens == 111)
        # 0.7 ± reasonable tolerance
        assert 0.65 < ratio_a < 0.75, f"Expected ~0.7, got {ratio_a}"

    def test_batch_values_are_valid_tokens(self, multi_source_dirs):
        """All values in batch are valid GPT-2 token IDs."""
        sources = [
            DataSourceConfig(path=multi_source_dirs["fineweb"], weight=1.0),
        ]
        loader = MultiSourceDataLoader(
            sources, seq_len=32, batch_size=4, seed=42, max_steps=5
        )
        for batch in loader:
            assert (batch >= 0).all()
            assert (batch < 50257).all()

    def test_max_steps_stops_iteration(self, multi_source_dirs):
        """Loader stops after max_steps batches."""
        sources = [
            DataSourceConfig(path=multi_source_dirs["fineweb"], weight=1.0),
        ]
        loader = MultiSourceDataLoader(
            sources, seq_len=16, batch_size=2, seed=42, max_steps=5
        )
        batches = list(loader)
        assert len(batches) == 5


# ── SFT Data Format ──


class TestSFTDataFormat:
    def _make_sft_sample(self, problem="What is 2+2?", solution="2+2=4. \\boxed{4}"):
        """Create a minimal SFT sample."""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful math assistant. Think step by step.",
                },
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ]
        }

    def test_chat_format_structure(self):
        """SFT sample has correct 3-message structure."""
        sample = self._make_sft_sample()
        assert len(sample["messages"]) == 3
        assert sample["messages"][0]["role"] == "system"
        assert sample["messages"][1]["role"] == "user"
        assert sample["messages"][2]["role"] == "assistant"

    def test_boxed_answer_present(self):
        """Assistant response contains \\boxed{} answer."""
        sample = self._make_sft_sample()
        assert "\\boxed{" in sample["messages"][2]["content"]

    def test_jsonl_serialization(self, tmp_path):
        """SFT samples serialize to valid JSONL."""
        samples = [
            self._make_sft_sample("What is 1+1?", "1+1=2. \\boxed{2}"),
            self._make_sft_sample("What is 3*4?", "3*4=12. \\boxed{12}"),
        ]

        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        # Read back
        loaded = []
        with open(path) as f:
            for line in f:
                loaded.append(json.loads(line))

        assert len(loaded) == 2
        assert loaded[0]["messages"][1]["content"] == "What is 1+1?"
        assert loaded[1]["messages"][2]["content"] == "3*4=12. \\boxed{12}"


# ── SFT Formatting Functions ──


class TestSFTFormatting:
    def test_ensure_boxed_answer_already_boxed(self):
        """Text with \\boxed{} is returned unchanged."""
        from math_nano.data.answers import ensure_boxed_answer

        text = "The answer is \\boxed{42}"
        assert ensure_boxed_answer(text) == text

    def test_ensure_boxed_answer_from_hash(self):
        """GSM8K-style #### answer gets \\boxed{} appended."""
        from math_nano.data.answers import ensure_boxed_answer

        text = "Step 1: do math\n#### 42"
        result = ensure_boxed_answer(text)
        assert "\\boxed{42}" in result

    def test_ensure_boxed_answer_from_last_number(self):
        """Fallback: last number in text gets \\boxed{}."""
        from math_nano.data.answers import ensure_boxed_answer

        text = "So 3 + 4 = 7"
        result = ensure_boxed_answer(text)
        assert "\\boxed{7}" in result

    def test_ensure_boxed_answer_no_number(self):
        """Text with no numbers returns unchanged."""
        from math_nano.data.answers import ensure_boxed_answer

        text = "I cannot solve this problem."
        result = ensure_boxed_answer(text)
        assert result == text

    def test_format_chat_sample(self):
        """format_chat_sample produces correct structure."""
        from scripts.data.prepare_sft import format_chat_sample

        sample, truncated = format_chat_sample(
            problem="What is 2+2?",
            solution="2+2=4. \\boxed{4}",
            system_prompt="You are a math assistant.",
        )
        assert not truncated
        assert len(sample["messages"]) == 3
        assert sample["messages"][0]["content"] == "You are a math assistant."
        assert sample["messages"][1]["content"] == "What is 2+2?"
        assert "\\boxed{4}" in sample["messages"][2]["content"]

    def test_truncation_preserves_answer(self):
        """Long solutions are truncated but answer is preserved."""
        from scripts.data.prepare_sft import truncate_preserving_answer

        enc = tiktoken.get_encoding("gpt2")
        long_text = "Step: do math. " * 500 + "\\boxed{42}"
        result, was_truncated = truncate_preserving_answer(long_text, 200, enc)
        assert was_truncated
        # The truncated text should still contain the answer
        assert "42" in result

    def test_short_text_not_truncated(self):
        """Short text passes through without truncation."""
        from scripts.data.prepare_sft import truncate_preserving_answer

        enc = tiktoken.get_encoding("gpt2")
        text = "2+2=4. \\boxed{4}"
        result, was_truncated = truncate_preserving_answer(text, 2000, enc)
        assert not was_truncated
        assert result == text

    def test_difficulty_estimation(self):
        """Difficulty scoring produces values 1-5."""
        from scripts.data.prepare_sft import estimate_difficulty

        enc = tiktoken.get_encoding("gpt2")

        # Short solution -> low difficulty
        short = "3 + 5 = 8"
        assert estimate_difficulty(short, enc) <= 2

        # Very long solution -> high difficulty
        long = "Step " + ": detailed computation. " * 200
        assert estimate_difficulty(long, enc) >= 4


# ── Eval Data Format ──


class TestEvalDataFormat:
    def test_eval_sample_structure(self):
        """Eval samples have required fields."""
        sample = {
            "problem_id": "gsm8k_0001",
            "dataset": "gsm8k",
            "problem": "What is 2+2?",
            "answer": "4",
            "solution": "2+2=4. #### 4",
            "difficulty": None,
        }
        required = ["problem_id", "dataset", "problem", "answer"]
        for key in required:
            assert key in sample

    def test_normalize_answer(self):
        """Answer normalization."""
        from math_nano.data.answers import normalize_answer_for_eval

        assert normalize_answer_for_eval("42") == "42"
        assert normalize_answer_for_eval("1,234") == "1234"
        assert normalize_answer_for_eval("  42  ") == "42"
        assert normalize_answer_for_eval("42.") == "42"

    def test_sha256_file(self, tmp_path):
        """SHA256 checksum is computed correctly."""
        from math_nano.data.io import sha256_file

        path = tmp_path / "test.txt"
        path.write_text("hello world\n")

        h = sha256_file(str(path))
        assert len(h) == 64  # SHA256 hex digest is 64 chars
        assert all(c in "0123456789abcdef" for c in h)

    def test_sha256_deterministic(self, tmp_path):
        """Same content produces same hash."""
        from math_nano.data.io import sha256_file

        for name in ["a.txt", "b.txt"]:
            (tmp_path / name).write_text("same content")

        h1 = sha256_file(str(tmp_path / "a.txt"))
        h2 = sha256_file(str(tmp_path / "b.txt"))
        assert h1 == h2

    def test_manifest_format(self, tmp_path):
        """Manifest JSON has correct structure."""
        from scripts.data.download_eval import create_manifest

        # Create a dummy eval file
        eval_path = tmp_path / "test.jsonl"
        eval_path.write_text('{"problem": "test"}\n')

        info = {"test": {"path": str(eval_path), "count": 1}}
        manifest_path = create_manifest(str(tmp_path), info)

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "version" in manifest
        assert "created" in manifest
        assert "datasets" in manifest
        assert "test" in manifest["datasets"]
        assert "sha256" in manifest["datasets"]["test"]
        assert manifest["datasets"]["test"]["n"] == 1

    def test_verify_manifest(self, tmp_path):
        """verify_manifest detects valid files."""
        from scripts.data.download_eval import create_manifest, verify_manifest

        eval_path = tmp_path / "test.jsonl"
        eval_path.write_text('{"problem": "test"}\n')

        info = {"test": {"path": str(eval_path), "count": 1}}
        create_manifest(str(tmp_path), info)

        assert verify_manifest(str(tmp_path)) is True

    def test_verify_manifest_detects_tampering(self, tmp_path):
        """verify_manifest detects modified files."""
        from scripts.data.download_eval import create_manifest, verify_manifest

        eval_path = tmp_path / "test.jsonl"
        eval_path.write_text('{"problem": "original"}\n')

        info = {"test": {"path": str(eval_path), "count": 1}}
        create_manifest(str(tmp_path), info)

        # Tamper with the file
        eval_path.write_text('{"problem": "tampered"}\n')

        assert verify_manifest(str(tmp_path)) is False
