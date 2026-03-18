# Spec 07: Eval Harness

## Goal

A single, consistent eval system used across ALL stages (pretrain, SFT, RL).
Results must be directly comparable across experiments.
Shared across every model, every stage, every experiment.

---

## Blessed Eval Suite

Two tiers: **Eval-Small** (fast, during training) and **Eval-Full** (thorough, for reporting).

### Eval-Small (Quick Feedback Loop)

Used during training every N steps. Fast enough to not block training.

| Dataset | Source | Problems | Difficulty | Notes |
|---------|--------|----------|-----------|-------|
| GSM8K-mini | openai/gsm8k test, fixed 200 subset | 200 | Elementary | Randomly sampled once, frozen, seeded |
| MATH-mini | hendrycks/competition_math test, fixed 100 subset | 100 | Mixed competition | Stratified by difficulty level |

**Total: 300 problems.** Greedy decoding. ~5 min on GPU, ~30 min on CPU.

Subset selection: sample once with seed=42, save as `data/eval/gsm8k_mini.jsonl`
and `data/eval/math_mini.jsonl`. These files are version-controlled. Never change them.

### Eval-Full (Blessed Reporting Suite)

Used at end of every stage and for final results. This is what goes in papers/writeups.

| Dataset | Source | Problems | Difficulty | Answer Type |
|---------|--------|----------|-----------|-------------|
| GSM8K | openai/gsm8k | 1319 (full test) | Elementary | Integer |
| MATH500 | lighteval/MATH (500 test subset) | 500 | Mixed competition | Numeric/expression |
| AMC | NuminaMath-CoT, AMC problems | ~200 | High school competition | Integer |
| AIME | AI-MO/aimo-validation-aime | ~90 | Olympiad qualifier | Integer (0-999) |
| Minerva | google/minerva_math | 272 | STEM undergrad | Numeric |

**Total: ~2381 problems.** With k=8 samples: ~19K generations. ~30 min on H100.

### Version Pinning

All eval datasets are:
1. Downloaded once
2. Saved to `data/eval/` as JSONL files
3. Checksummed (SHA256)
4. Version-controlled (or checksum logged to W&B)

```json
// data/eval/manifest.json
{
  "version": "1.0",
  "created": "2026-03-18",
  "datasets": {
    "gsm8k": {"file": "gsm8k_test.jsonl", "sha256": "abc...", "n": 1319},
    "gsm8k_mini": {"file": "gsm8k_mini.jsonl", "sha256": "def...", "n": 200},
    "math500": {"file": "math500.jsonl", "sha256": "ghi...", "n": 500},
    "math_mini": {"file": "math_mini.jsonl", "sha256": "jkl...", "n": 100},
    "amc": {"file": "amc.jsonl", "sha256": "mno...", "n": 200},
    "aime": {"file": "aime.jsonl", "sha256": "pqr...", "n": 90},
    "minerva": {"file": "minerva.jsonl", "sha256": "stu...", "n": 272}
  }
}
```

If you ever update an eval dataset, bump the version and re-run ALL prior
checkpoints against the new version. Otherwise results aren't comparable.

---

## pass@k Protocol

### Definition

**pass@k**: probability that at least 1 of k independent samples is correct.

Given n total samples per problem with c correct:

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

Where C(a,b) is "a choose b". This is the unbiased estimator from
[Chen et al., "Evaluating Large Language Models Trained on Code", 2021].

### k Values We Report

| Metric | k | Samples needed (n) | Purpose |
|--------|---|-------------------|---------|
| pass@1 | 1 | n >= 8 | Standard benchmark, comparability |
| pass@4 | 4 | n >= 8 | "Can the model get it with a few tries?" |
| pass@8 | 8 | n >= 8 | "Is the knowledge in there at all?" |
| pass@16 | 16 | n >= 20 | Optional, for deep analysis only |

Default: generate **n=16 samples** per problem. This lets us compute
pass@1 through pass@16 with reasonable statistical properties.

For Eval-Small (during training): n=1 (greedy only), report just greedy pass@1.
For Eval-Full (reporting): n=16, report pass@1, pass@4, pass@8.

### Implementation

```python
import numpy as np
from math import comb

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k.
    n: total samples generated
    c: number of correct samples
    k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

def compute_pass_at_k(results: list[dict], k_values: list[int] = [1, 4, 8]) -> dict:
    """
    results: list of {"problem_id": str, "n_samples": int, "n_correct": int}
    Returns: {"pass@1": float, "pass@4": float, "pass@8": float}
    """
    metrics = {}
    for k in k_values:
        per_problem = [pass_at_k(r["n_samples"], r["n_correct"], k) for r in results]
        metrics[f"pass@{k}"] = np.mean(per_problem)
    return metrics
```

### Greedy vs Sampled pass@1

We report BOTH:
- **pass@1 (greedy)**: 1 completion, temperature=0, deterministic
- **pass@1 (sampled)**: estimated from n=16 samples at temperature=0.7

These will differ. Greedy is the "deployment" number. Sampled is more robust
statistically and shows what the model "knows" vs what it "says first."

---

## Variance and Confidence Intervals

### Why This Matters

With 200 problems and pass@1=30%, the 95% confidence interval is ±6.4%.
A 5-point "improvement" could be noise. We need to quantify this.

### How We Compute It

**Bootstrap confidence intervals** (non-parametric, doesn't assume distribution):

```python
def bootstrap_ci(per_problem_correct: list[bool], n_bootstrap: int = 10000,
                 ci: float = 0.95) -> tuple[float, float, float]:
    """
    per_problem_correct: list of True/False for each problem
    Returns: (mean, ci_low, ci_high)
    """
    n = len(per_problem_correct)
    scores = np.array(per_problem_correct, dtype=float)
    mean = scores.mean()

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrap_means.append(sample.mean())

    alpha = (1 - ci) / 2
    ci_low = np.percentile(bootstrap_means, 100 * alpha)
    ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha))

    return mean, ci_low, ci_high
```

### What We Report

For every eval result:

```json
{
  "gsm8k": {
    "pass_at_1_greedy": 0.350,
    "pass_at_1_greedy_ci95": [0.324, 0.376],
    "pass_at_1_sampled": 0.380,
    "pass_at_1_sampled_ci95": [0.354, 0.406],
    "pass_at_4": 0.520,
    "pass_at_4_ci95": [0.493, 0.547],
    "pass_at_8": 0.610,
    "pass_at_8_ci95": [0.584, 0.636],
    "n_problems": 1319,
    "n_samples_per_problem": 16
  }
}
```

### When Is an Improvement "Real"?

An improvement from model A to model B is considered real when:
- The 95% CIs don't overlap, OR
- A paired bootstrap test gives p < 0.05

```python
def is_significant_improvement(results_a: list[bool], results_b: list[bool],
                                n_bootstrap: int = 10000) -> tuple[bool, float]:
    """Paired bootstrap test: is B significantly better than A?"""
    assert len(results_a) == len(results_b)
    n = len(results_a)
    a = np.array(results_a, dtype=float)
    b = np.array(results_b, dtype=float)

    observed_diff = b.mean() - a.mean()

    # Bootstrap the difference
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        diffs.append(b[idx].mean() - a[idx].mean())

    p_value = np.mean([d <= 0 for d in diffs])  # fraction where B is not better
    significant = p_value < 0.05

    return significant, p_value
```

---

## Eval Modes Summary

| Mode | Datasets | k (samples) | Decoding | Duration (GPU) | When |
|------|----------|-------------|----------|---------------|------|
| `quick-subset` | GSM8K-mini (200) | 1 (greedy) | temp=0 | ~2 min | Pretrain, every 5K steps |
| `quick` | GSM8K-mini + MATH-mini (300) | 1 (greedy) | temp=0 | ~5 min | SFT/RL, every 200 steps |
| `full` | All 5 datasets (~2381) | 16 per problem | temp=0.7 | ~30 min | End of each stage |
| `full-greedy` | All 5 datasets | 1 (greedy) | temp=0 | ~10 min | Quick full-suite check |

---

## Answer Extraction

```python
def extract_answer(text: str) -> Optional[str]:
    """Extract the final numerical answer from model output."""
    # Priority order:
    # 1. \boxed{...}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return normalize_answer(boxed[-1])  # last boxed expression

    # 2. #### pattern (GSM8K style)
    hash_match = re.search(r'####\s*(.+)', text)
    if hash_match:
        return normalize_answer(hash_match.group(1))

    # 3. "The answer is ..."
    answer_match = re.search(r'[Tt]he (?:final )?answer is[:\s]*(.+?)[\.\n]', text)
    if answer_match:
        return normalize_answer(answer_match.group(1))

    # 4. Last number in output (aggressive fallback)
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return normalize_answer(numbers[-1])

    return None

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.strip()
    # Remove $, \, spaces, commas
    answer = re.sub(r'[\$\\,\s]', '', answer)
    # Try to evaluate as number
    try:
        val = float(answer)
        # Return as int if it is one (42.0 → "42")
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return answer.lower()
```

### Extraction Validation

Before use, validate on ground truth:
- Run extraction on all GSM8K test answers → should match >99%
- Run extraction on all MATH500 answers → should match >95% (harder format)
- Log extraction failure rate per dataset as a metric

---

## Eval Script Interface

```bash
# During training: quick check
python scripts/eval/run_eval.py \
  --checkpoint $CKPT --depth 16 \
  --suite small --mode greedy

# End of stage: full blessed eval
python scripts/eval/run_eval.py \
  --checkpoint $CKPT --depth 16 \
  --suite full --samples 16

# Single dataset, custom k
python scripts/eval/run_eval.py \
  --checkpoint $CKPT --depth 16 \
  --datasets gsm8k --samples 32

# Local CPU test
python scripts/eval/run_eval.py \
  --checkpoint $CKPT --depth 10 \
  --suite small --mode greedy --device cpu

# Compare two checkpoints
python scripts/eval/compare.py \
  --checkpoint-a results/sft-m/best.pt \
  --checkpoint-b results/grpo-m/best.pt \
  --depth 16 --suite full --samples 16
```

## Output Format

```json
{
  "eval_version": "1.0",
  "checkpoint": "results/sft-m-best/best_gsm8k.pt",
  "model_depth": 16,
  "model_params": 130000000,
  "stage": "sft",
  "experiment_id": "sft-m-distill-r1",
  "eval_suite": "full",
  "n_samples_per_problem": 16,
  "temperature": 0.7,
  "max_new_tokens": 1024,
  "timestamp": "2026-03-17T12:00:00Z",
  "eval_data_manifest_sha": "abc123...",
  "results": {
    "gsm8k": {
      "n_problems": 1319,
      "pass_at_1_greedy": 0.350,
      "pass_at_1_greedy_ci95": [0.324, 0.376],
      "pass_at_1_sampled": 0.380,
      "pass_at_1_sampled_ci95": [0.354, 0.406],
      "pass_at_4": 0.520,
      "pass_at_4_ci95": [0.493, 0.547],
      "pass_at_8": 0.610,
      "pass_at_8_ci95": [0.584, 0.636],
      "extraction_failures": 12,
      "extraction_failure_rate": 0.009,
      "avg_output_tokens": 245,
      "avg_inference_ms": 340,
      "per_problem": [
        {"id": "gsm8k_0001", "correct_samples": 6, "total_samples": 16},
        ...
      ]
    },
    "math500": { ... },
    "amc": { ... },
    "aime": { ... },
    "minerva": { ... }
  },
  "aggregate": {
    "avg_pass_at_1_greedy": 0.182,
    "avg_pass_at_1_sampled": 0.201,
    "weighted_pass_at_1": 0.195
  }
}
```

### Per-Problem Results

We save per-problem correctness because:
1. Enables paired statistical tests (compare same problems across models)
2. Enables error analysis (which problems are hard for small models?)
3. Enables difficulty stratification (performance by problem type/difficulty)

## Difficulty Breakdown

For MATH500 (which has difficulty labels 1-5), also report:

```json
{
  "math500_by_level": {
    "level_1": {"n": 100, "pass_at_1": 0.45},
    "level_2": {"n": 100, "pass_at_1": 0.25},
    "level_3": {"n": 100, "pass_at_1": 0.10},
    "level_4": {"n": 100, "pass_at_1": 0.03},
    "level_5": {"n": 100, "pass_at_1": 0.01}
  }
}
```

This shows WHERE the model's capability boundary is.

## W&B Integration

Every eval run logs to W&B:
- As a separate run with type="eval"
- Tagged with: model_size, stage, experiment_id, eval_mode
- Summary metrics: pass@1, pass@4, pass@8 for each dataset
- Artifact: the full JSON output file
- Table: per-problem results (enables W&B's built-in analysis tools)

## CPU Compatibility

The eval harness MUST work on CPU for local testing:
- `--device cpu` flag (default: auto-detect)
- On CPU, auto-suggest `--suite small` if full is requested
- Warn but don't prevent full eval on CPU (will be slow but may be needed)

## Eval Consistency Guarantees

To ensure results are comparable across ALL experiments:
1. **Fixed eval sets** — version-pinned, checksummed, never modified
2. **Fixed generation params** — same max_new_tokens, same temperature
3. **Fixed extraction** — same answer extraction code, versioned
4. **Fixed prompt template** — same system prompt, same problem formatting
5. **Deterministic greedy** — verified that greedy is bit-identical across runs
6. **Bootstrap seed** — fixed seed=42 for CI computation (same CIs for same data)

**If ANY eval code changes → re-run evals on ALL prior checkpoints.**
This is non-negotiable. Otherwise the scaling curves are invalid.
