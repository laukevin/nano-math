# Spec 19: Test Suite

## Principle

Tests are the structural backbone. As we iterate on data pipelines,
training code, eval logic, and harness features, tests ensure nothing
silently breaks. Every component has tests. Tests run on CPU, fast,
no GPU required.

**Rule:** No PR merges without tests passing. No experiment launches
without tests passing.

---

## Test Organization

```
tests/
├── unit/                          # Pure logic, no I/O, instant
│   ├── test_answer_extraction.py  # Extract answers from model outputs
│   ├── test_answer_normalization.py
│   ├── test_pass_at_k.py         # pass@k estimator math
│   ├── test_bootstrap_ci.py      # Confidence interval computation
│   ├── test_reward_function.py   # GRPO reward logic
│   ├── test_config_validation.py # Experiment config validation
│   ├── test_budget_tracking.py   # Cost estimation & budget limits
│   ├── test_data_mixture.py      # Mixture weight math
│   └── test_search_strategy.py   # Elimination & binary search logic
│
├── integration/                   # Needs model/data, still fast (CPU)
│   ├── test_model_init.py        # Create models at all depths
│   ├── test_forward_backward.py  # Forward/backward pass works
│   ├── test_checkpoint_roundtrip.py  # Save → load → same state
│   ├── test_data_loading.py      # Load shards, verify format
│   ├── test_sft_formatting.py    # Chat format, loss masking
│   ├── test_eval_pipeline.py     # Eval harness end-to-end (tiny)
│   ├── test_hf_conversion.py     # nanochat ↔ HuggingFace format
│   ├── test_registry.py          # Model & data registry operations
│   └── test_metrics_contract.py  # Verify metric logging shape
│
├── smoke/                         # Quick sanity (CPU, <2 min total)
│   ├── test_pretrain_smoke.py    # 10 pretrain steps
│   ├── test_sft_smoke.py         # 10 SFT steps
│   ├── test_grpo_smoke.py        # 5 GRPO steps
│   └── test_eval_smoke.py        # Eval on 5 problems
│
├── fixtures/                      # Shared test data
│   ├── tiny_model.pt             # Pre-saved depth=10 checkpoint
│   ├── sample_sft.jsonl          # 20 SFT samples
│   ├── sample_gsm8k.jsonl        # 10 GSM8K test problems
│   ├── sample_shard.bin          # 1 tokenized shard (tiny)
│   └── conftest.py               # Pytest fixtures
│
└── conftest.py                    # Root conftest
```

---

## Unit Tests

### test_answer_extraction.py

The most critical test file. If answer extraction is wrong, ALL eval
numbers are wrong.

```python
import pytest
from eval.extraction import extract_answer, normalize_answer

class TestExtractAnswer:
    """Test answer extraction from model outputs."""

    # ── \boxed{} format ──
    def test_boxed_integer(self):
        assert extract_answer("The answer is \\boxed{42}") == "42"

    def test_boxed_negative(self):
        assert extract_answer("\\boxed{-7}") == "-7"

    def test_boxed_decimal(self):
        assert extract_answer("\\boxed{3.14}") == "3.14"

    def test_boxed_fraction(self):
        assert extract_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_boxed_multiple_takes_last(self):
        assert extract_answer("First \\boxed{5}, then \\boxed{10}") == "10"

    def test_boxed_with_surrounding_text(self):
        text = "After solving step by step, we get \\boxed{123}. Done."
        assert extract_answer(text) == "123"

    # ── #### format (GSM8K) ──
    def test_hash_format(self):
        assert extract_answer("#### 42") == "42"

    def test_hash_with_comma(self):
        assert extract_answer("#### 1,234") == "1234"

    # ── "The answer is" format ──
    def test_answer_is(self):
        assert extract_answer("The answer is 42.") == "42"

    def test_final_answer_is(self):
        assert extract_answer("The final answer is 42.") == "42"

    # ── Fallback: last number ──
    def test_last_number_fallback(self):
        assert extract_answer("So we get 3 + 4 = 7") == "7"

    # ── Edge cases ──
    def test_empty_string(self):
        assert extract_answer("") is None

    def test_no_numbers(self):
        assert extract_answer("I don't know the answer.") is None

    def test_only_whitespace(self):
        assert extract_answer("   \n\n  ") is None

    def test_boxed_empty(self):
        # \boxed{} with nothing inside — should fall through
        result = extract_answer("\\boxed{}")
        # Acceptable: None or ""

    def test_very_long_output(self):
        """Model outputs 5000 tokens of reasoning then an answer."""
        text = "Step 1: ...\n" * 500 + "The answer is \\boxed{42}"
        assert extract_answer(text) == "42"


class TestNormalizeAnswer:
    def test_strip_whitespace(self):
        assert normalize_answer("  42  ") == "42"

    def test_strip_dollar_signs(self):
        assert normalize_answer("$42$") == "42"

    def test_strip_commas(self):
        assert normalize_answer("1,234") == "1234"

    def test_float_to_int(self):
        assert normalize_answer("42.0") == "42"

    def test_preserve_decimal(self):
        assert normalize_answer("3.14") == "3.14"

    def test_negative(self):
        assert normalize_answer("-7") == "-7"

    # ── Numeric equivalence ──
    def test_equivalent_representations(self):
        """These should all normalize to the same thing."""
        representations = ["42", "42.0", "42.00", " 42 ", "$42$"]
        normalized = set(normalize_answer(r) for r in representations)
        assert len(normalized) == 1, f"Not all equivalent: {normalized}"
```

### test_pass_at_k.py

```python
import pytest
from eval.metrics import pass_at_k, compute_pass_at_k

class TestPassAtK:
    def test_all_correct(self):
        assert pass_at_k(n=8, c=8, k=1) == 1.0

    def test_none_correct(self):
        assert pass_at_k(n=8, c=0, k=1) == 0.0

    def test_half_correct_pass1(self):
        # 4 correct out of 8 samples → pass@1 = 4/8 = 0.5
        assert pass_at_k(n=8, c=4, k=1) == 0.5

    def test_pass_at_k_monotonic(self):
        """pass@k should increase with k."""
        for c in range(1, 8):
            scores = [pass_at_k(n=8, c=c, k=k) for k in [1, 2, 4, 8]]
            assert scores == sorted(scores), f"Not monotonic for c={c}: {scores}"

    def test_pass_at_1_equals_c_over_n(self):
        """pass@1 with unbiased estimator = c/n."""
        for n in [8, 16, 32]:
            for c in range(n + 1):
                assert abs(pass_at_k(n, c, 1) - c / n) < 1e-10

    def test_pass_at_n_is_0_or_1(self):
        """pass@n: if any correct → 1.0, else 0.0."""
        assert pass_at_k(n=8, c=1, k=8) == 1.0
        assert pass_at_k(n=8, c=0, k=8) == 0.0

    def test_known_value(self):
        # pass@2 with n=8, c=3: 1 - C(5,2)/C(8,2) = 1 - 10/28 = 18/28
        expected = 1.0 - 10.0 / 28.0
        assert abs(pass_at_k(n=8, c=3, k=2) - expected) < 1e-10


class TestComputePassAtK:
    def test_single_problem(self):
        results = [{"problem_id": "p1", "n_samples": 8, "n_correct": 4}]
        metrics = compute_pass_at_k(results, k_values=[1, 4, 8])
        assert metrics["pass@1"] == 0.5

    def test_multiple_problems(self):
        results = [
            {"problem_id": "p1", "n_samples": 8, "n_correct": 8},  # easy
            {"problem_id": "p2", "n_samples": 8, "n_correct": 0},  # impossible
        ]
        metrics = compute_pass_at_k(results, k_values=[1])
        assert metrics["pass@1"] == 0.5  # average of 1.0 and 0.0
```

### test_reward_function.py

```python
import pytest
from grpo.reward import compute_reward

class TestReward:
    def test_correct_boxed(self):
        assert compute_reward("\\boxed{42}", "42") == 1.0

    def test_wrong_boxed(self):
        assert compute_reward("\\boxed{43}", "42") == 0.0

    def test_correct_hash(self):
        assert compute_reward("#### 42", "42") == 1.0

    def test_correct_answer_is(self):
        assert compute_reward("The answer is 42.", "42") == 1.0

    def test_no_answer(self):
        assert compute_reward("I don't know", "42") == 0.0

    def test_empty_output(self):
        assert compute_reward("", "42") == 0.0

    def test_numeric_equivalence(self):
        assert compute_reward("\\boxed{42.0}", "42") == 1.0

    def test_with_commas(self):
        assert compute_reward("\\boxed{1,234}", "1234") == 1.0

    def test_negative_number(self):
        assert compute_reward("\\boxed{-7}", "-7") == 1.0

    def test_correct_but_verbose(self):
        text = ("Step 1: blah blah\n" * 100 + "\\boxed{42}")
        assert compute_reward(text, "42") == 1.0


class TestRewardEdgeCases:
    """These catch potential reward hacking vectors."""

    def test_boxed_with_garbage_inside(self):
        # Model outputs \boxed{...} but with non-answer content
        assert compute_reward("\\boxed{the answer}", "42") == 0.0

    def test_multiple_boxed_takes_last(self):
        assert compute_reward("\\boxed{5} ... \\boxed{42}", "42") == 1.0

    def test_boxed_with_latex(self):
        assert compute_reward("\\boxed{\\frac{1}{2}}", "0.5") == 1.0
        # This test may need refinement — depends on how we handle fractions
```

### test_config_validation.py

```python
import pytest
from harness.config import ExperimentConfig, validate_config

class TestConfigValidation:
    def test_valid_pretrain_config(self):
        config = ExperimentConfig(
            experiment_id="test-pt",
            stage="pretrain",
            phase="1a",
            depth=10,
            mixture="mix-math-broad",
        )
        errors = validate_config(config)
        assert errors == []

    def test_pretrain_missing_mixture(self):
        config = ExperimentConfig(
            experiment_id="test-pt",
            stage="pretrain",
            phase="1a",
            depth=10,
        )
        errors = validate_config(config)
        assert any("mixture" in e.lower() for e in errors)

    def test_sft_missing_parent(self):
        config = ExperimentConfig(
            experiment_id="test-sft",
            stage="sft",
            phase="2a",
            depth=16,
            sft_recipe="sft-concise-cot",
            # no parent_checkpoint!
        )
        errors = validate_config(config)
        assert any("parent" in e.lower() for e in errors)

    def test_sft_missing_recipe(self):
        config = ExperimentConfig(
            experiment_id="test-sft",
            stage="sft",
            phase="2a",
            depth=16,
            parent_checkpoint="some-checkpoint",
            # no recipe!
        )
        errors = validate_config(config)
        assert any("recipe" in e.lower() for e in errors)

    def test_invalid_depth(self):
        config = ExperimentConfig(
            experiment_id="test-pt",
            stage="pretrain",
            phase="1a",
            depth=7,  # not in our model grid
            mixture="mix-math-broad",
        )
        errors = validate_config(config)
        assert any("depth" in e.lower() for e in errors)

    def test_duplicate_experiment_id(self, mock_registry):
        mock_registry.add("existing-experiment")
        config = ExperimentConfig(
            experiment_id="existing-experiment",
            stage="pretrain",
            phase="1a",
            depth=10,
            mixture="mix-math-broad",
        )
        errors = validate_config(config)
        assert any("already exists" in e.lower() for e in errors)
```

---

## Integration Tests

### test_model_init.py

```python
import pytest
import torch

VALID_DEPTHS = [10, 12, 16, 20, 24]

class TestModelInit:
    @pytest.mark.parametrize("depth", VALID_DEPTHS)
    def test_model_creates(self, depth):
        """Model initializes without error at every valid depth."""
        from model import GPT, GPTConfig
        config = GPTConfig(depth=depth)
        model = GPT(config)
        assert model is not None

    @pytest.mark.parametrize("depth", VALID_DEPTHS)
    def test_param_count_reasonable(self, depth):
        """Param count is in expected range."""
        from model import GPT, GPTConfig
        config = GPTConfig(depth=depth)
        model = GPT(config)
        params = sum(p.numel() for p in model.parameters())

        expected_ranges = {
            10: (30_000_000, 80_000_000),
            12: (60_000_000, 120_000_000),
            16: (90_000_000, 180_000_000),
            20: (150_000_000, 280_000_000),
            24: (230_000_000, 450_000_000),
        }
        lo, hi = expected_ranges[depth]
        assert lo < params < hi, f"depth={depth}: {params:,} params outside [{lo:,}, {hi:,}]"


class TestForwardBackward:
    @pytest.mark.parametrize("depth", [10, 12])  # only smallest for speed
    def test_forward_pass(self, depth):
        from model import GPT, GPTConfig
        config = GPTConfig(depth=depth)
        model = GPT(config)
        batch = torch.randint(0, 50257, (2, 128))  # tiny batch
        output = model(batch)
        assert torch.isfinite(output.loss)

    @pytest.mark.parametrize("depth", [10, 12])
    def test_backward_pass(self, depth):
        from model import GPT, GPTConfig
        config = GPTConfig(depth=depth)
        model = GPT(config)
        batch = torch.randint(0, 50257, (2, 128))
        output = model(batch)
        output.loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(p.grad).all(), f"NaN/inf gradient in {name}"
```

### test_checkpoint_roundtrip.py

```python
import pytest
import torch
import tempfile

class TestCheckpoint:
    def test_save_load_roundtrip(self, tiny_model):
        """Save a model, load it back, verify identical."""
        model = tiny_model  # depth=10 fixture

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            # Save
            torch.save({
                "model_state_dict": model.state_dict(),
                "step": 100,
            }, f.name)

            # Load
            checkpoint = torch.load(f.name, weights_only=False)
            model2 = create_model(depth=10)
            model2.load_state_dict(checkpoint["model_state_dict"])

            # Verify
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), model2.named_parameters()
            ):
                assert n1 == n2
                assert torch.equal(p1, p2), f"Mismatch in {n1}"

    def test_checkpoint_contains_required_fields(self, tiny_model):
        """Checkpoint must have all fields the harness expects."""
        # Simulate what the harness saves
        checkpoint = save_checkpoint_harness(tiny_model, step=100, config={}, metrics={})

        required_keys = ["model_state_dict", "step", "config", "experiment_id", "metrics"]
        for key in required_keys:
            assert key in checkpoint, f"Missing key: {key}"
```

### test_sft_formatting.py

```python
class TestSFTFormatting:
    def test_chat_format_structure(self):
        """SFT sample has correct message structure."""
        sample = format_sft_sample(
            problem="What is 2+2?",
            solution="2+2=4. \\boxed{4}",
            system="You are a math assistant."
        )
        assert len(sample["messages"]) == 3
        assert sample["messages"][0]["role"] == "system"
        assert sample["messages"][1]["role"] == "user"
        assert sample["messages"][2]["role"] == "assistant"

    def test_answer_in_boxed_format(self):
        """All SFT samples must end with \boxed{} answer."""
        sample = format_sft_sample(
            problem="What is 2+2?",
            solution="The answer is 4."
        )
        assert "\\boxed{" in sample["messages"][-1]["content"]

    def test_loss_mask_correctness(self):
        """Loss should only be computed on assistant tokens."""
        tokens, loss_mask = tokenize_sft_sample({
            "messages": [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4."},
            ]
        })
        # Find where assistant content starts
        # loss_mask should be 0 for system+user, 1 for assistant
        assert loss_mask[:10].sum() == 0  # system tokens (at least some zeros)
        assert loss_mask[-3:].sum() > 0   # assistant tokens (at least some ones)

    def test_truncation_preserves_answer(self):
        """When truncating long samples, the answer must be preserved."""
        long_cot = "Step: blah. " * 1000 + "\\boxed{42}"
        sample = format_sft_sample(
            problem="Hard problem.",
            solution=long_cot,
            max_seq_len=512,
        )
        assert "\\boxed{42}" in sample["messages"][-1]["content"]
```

### test_eval_pipeline.py

```python
class TestEvalPipeline:
    def test_eval_runs_on_cpu(self, tiny_model, sample_gsm8k):
        """Full eval pipeline works on CPU with tiny model."""
        results = run_eval(
            model=tiny_model,
            dataset=sample_gsm8k,  # 10 problems
            mode="greedy",
            device="cpu",
        )
        assert "pass_at_1_greedy" in results
        assert 0.0 <= results["pass_at_1_greedy"] <= 1.0
        assert results["n_problems"] == 10

    def test_eval_output_format(self, tiny_model, sample_gsm8k):
        """Eval output JSON has all required fields."""
        results = run_eval(tiny_model, sample_gsm8k, mode="greedy", device="cpu")
        required_fields = [
            "pass_at_1_greedy", "n_problems", "extraction_failures",
            "avg_output_tokens",
        ]
        for field in required_fields:
            assert field in results, f"Missing field: {field}"

    def test_eval_deterministic_greedy(self, tiny_model, sample_gsm8k):
        """Greedy eval produces identical results on two runs."""
        r1 = run_eval(tiny_model, sample_gsm8k, mode="greedy", device="cpu")
        r2 = run_eval(tiny_model, sample_gsm8k, mode="greedy", device="cpu")
        assert r1["pass_at_1_greedy"] == r2["pass_at_1_greedy"]

    def test_sampled_eval_has_ci(self, tiny_model, sample_gsm8k):
        """Sampled eval (k>1) produces confidence intervals."""
        results = run_eval(tiny_model, sample_gsm8k, mode="sampled",
                          n_samples=8, device="cpu")
        assert "pass_at_1_sampled_ci95" in results
        ci = results["pass_at_1_sampled_ci95"]
        assert len(ci) == 2
        assert ci[0] <= results["pass_at_1_sampled"] <= ci[1]
```

---

## Smoke Tests

These are "does training actually work?" tests. Slower than unit tests
but catch integration issues.

### test_pretrain_smoke.py

```python
class TestPretrainSmoke:
    @pytest.mark.slow
    def test_pretrain_10_steps(self):
        """Train for 10 steps, loss should decrease."""
        losses = run_training(
            depth=10, max_steps=10, device="cpu",
            data_source="sample", wandb_mode="disabled",
        )
        assert len(losses) == 10
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)
        # Loss should decrease (or at least not explode)
        assert losses[-1] < losses[0] * 1.5  # allow some noise
```

---

## Test Fixtures (conftest.py)

```python
import pytest
import torch

@pytest.fixture(scope="session")
def tiny_model():
    """A depth=10 model for testing. Created once per session."""
    from model import GPT, GPTConfig
    config = GPTConfig(depth=10)
    model = GPT(config)
    model.eval()
    return model

@pytest.fixture
def sample_gsm8k():
    """10 GSM8K test problems for eval testing."""
    return load_jsonl("tests/fixtures/sample_gsm8k.jsonl")

@pytest.fixture
def sample_sft_data():
    """20 SFT samples for formatting tests."""
    return load_jsonl("tests/fixtures/sample_sft.jsonl")

@pytest.fixture
def sample_shard(tmp_path):
    """A tiny tokenized shard for data loading tests."""
    import shutil
    shutil.copy("tests/fixtures/sample_shard.bin", tmp_path / "shard_000.bin")
    return tmp_path
```

---

## Running Tests

```bash
# All tests (fast, CPU only)
uv run pytest tests/ -v

# Just unit tests (instant)
uv run pytest tests/unit/ -v

# Just integration tests
uv run pytest tests/integration/ -v

# Just smoke tests (slower)
uv run pytest tests/smoke/ -v -m slow

# Specific test file
uv run pytest tests/unit/test_answer_extraction.py -v

# With coverage
uv run pytest tests/ --cov=harness --cov=scripts --cov-report=term-missing

# Before launching any experiments (full validation)
uv run pytest tests/ -v && echo "ALL TESTS PASS — safe to launch"
```

### pytest.ini / pyproject.toml markers

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (smoke tests, training runs)",
]
testpaths = ["tests"]
```

---

## Test Coverage Targets

| Module | Target | Critical? |
|--------|--------|-----------|
| `eval/extraction.py` | 100% | YES — wrong extraction = wrong numbers |
| `eval/metrics.py` | 100% | YES — pass@k math must be correct |
| `grpo/reward.py` | 100% | YES — wrong reward = bad RL |
| `harness/config.py` | 90% | YES — catch invalid configs early |
| `harness/bookkeeper.py` | 80% | Important for provenance |
| `harness/runner.py` | 70% | Integration coverage via smoke tests |
| Training loop | 50% | Covered by smoke tests, not unit tests |

---

## When to Add Tests

| Situation | Action |
|-----------|--------|
| New answer format discovered | Add extraction test case |
| Bug found in eval | Add regression test FIRST, then fix |
| New SFT recipe | Add formatting test |
| New reward variant | Add reward function tests |
| New config field | Add validation test |
| Edge case in production | Add test that reproduces it |

**Test-first for bugs:** When a bug is found, write a failing test that
reproduces it before fixing. This prevents regressions.
