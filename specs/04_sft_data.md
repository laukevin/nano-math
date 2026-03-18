# Spec 04: SFT Data

## Goal

Fine-tune pretrained models on math reasoning traces. The key questions:
1. What type of reasoning traces work best at small scale?
2. Does trace quality matter more than trace quantity?
3. Does the source model of distillation matter (R1 vs GPT-4 vs etc.)?

## Data Sources

### Distillation Traces (from large models)

| Dataset | HF ID | Size | Source Model | Trace Style |
|---------|-------|------|-------------|-------------|
| DeepSeek-R1 Distill | deepseek-ai/DeepSeek-R1 | ~800K samples | DeepSeek-R1 | Long CoT with <think> tags |
| OpenMathReasoning-SFT | nvidia/OpenMathReasoning | ~1.5M samples | Various | Step-by-step solutions |
| MetaMath | meta-math/MetaMathQA | ~395K samples | GPT-3.5 | Rephrased GSM8K/MATH solutions |
| MathInstruct | TIGER-Lab/MathInstruct | ~260K samples | GPT-4 / human | Mixed CoT + PoT |
| Orca-Math | microsoft/orca-math-word-problems-200k | 200K | GPT-4 | Word problems with solutions |
| NuminaMath-CoT | AI-MO/NuminaMath-CoT | ~860K | Various | Competition math with CoT |

### Curated / High-Quality

| Dataset | HF ID | Size | Content |
|---------|-------|------|---------|
| MATH train | hendrycks/competition_math | ~7.5K | Human-written solutions |
| GSM8K train | openai/gsm8k | ~7.5K | Step-by-step arithmetic |
| PRM800K | openai/prm800k | ~800K steps | Process reward labels (can derive traces) |

## SFT Mixture Recipes

We define distinct SFT recipes to compare:

### Recipe A: "Distill-R1" (Long CoT Focus)
```yaml
recipe_id: sft-distill-r1
sources:
  - dataset: deepseek-r1-distill
    samples: 100K  # subsample — full 800K is too much for small models
    format: chat
    system_prompt: "You are a helpful math assistant. Think step by step."
    max_seq_len: 4096
```
**Hypothesis:** R1-style long chains teach reasoning structure, but may be
too verbose for small models to learn from effectively.

### Recipe B: "Concise-CoT" (Short, Direct Solutions)
```yaml
recipe_id: sft-concise-cot
sources:
  - dataset: metamath
    samples: 100K
    format: chat
    system_prompt: "Solve the problem step by step. Be concise."
    max_seq_len: 2048
```
**Hypothesis:** Shorter traces are easier for small models to fit.
Less capacity wasted on filler tokens in long chains.

### Recipe C: "Kitchen Sink" (Large Diverse Mix)
```yaml
recipe_id: sft-kitchen-sink
sources:
  - dataset: openmathreasoning-sft
    samples: 50K
  - dataset: metamath
    samples: 50K
  - dataset: numinamath-cot
    samples: 50K
  - dataset: orca-math
    samples: 50K
total: 200K samples
max_seq_len: 2048
```
**Hypothesis:** Diversity helps small models generalize.

### Recipe D: "Quality Over Quantity" (Small, High-Quality)
```yaml
recipe_id: sft-quality
sources:
  - dataset: math-train  # hendrycks competition math
    samples: 7.5K (all)
  - dataset: gsm8k-train
    samples: 7.5K (all)
  - dataset: numinamath-cot
    samples: 15K  # top-quality subset
total: 30K samples
max_seq_len: 2048
epochs: 10  # more epochs since dataset is small
```
**Hypothesis:** At small scale, a small set of perfect examples beats
a large set of noisy ones.

### Recipe E: "Progressive" (Staged SFT)
```yaml
recipe_id: sft-progressive
stage_1:  # easy problems first
  sources:
    - dataset: gsm8k-train
      samples: 7.5K
    - dataset: metamath
      samples: 30K
      filter: difficulty < 3  # need to define difficulty score
  epochs: 3
stage_2:  # harder problems
  sources:
    - dataset: numinamath-cot
      samples: 50K
    - dataset: openmathreasoning-sft
      samples: 50K
      filter: difficulty >= 3
  epochs: 2
```
**Hypothesis:** Curriculum (easy-to-hard) during SFT helps small models
build capabilities incrementally rather than being overwhelmed.

## Data Formatting

All SFT data is converted to a standard chat format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful math assistant. Think step by step."},
    {"role": "user", "content": "What is 15% of 80?"},
    {"role": "assistant", "content": "To find 15% of 80:\n\n15% = 15/100 = 0.15\n\n0.15 × 80 = 12\n\nThe answer is \\boxed{12}"}
  ]
}
```

### Answer Formatting Convention
All traces must end with the answer in `\boxed{}` format.
This standardizes answer extraction for eval and RL reward.

### Preprocessing Script
```
scripts/data/prepare_sft.py \
  --recipe sft-distill-r1 \
  --output data/sft/distill-r1/ \
  --max-seq-len 4096 \
  --tokenizer gpt2
```

Outputs:
- `train.jsonl` — formatted chat samples
- `train_tokenized.bin` — tokenized for nanochat's SFT dataloader
- `stats.json` — sample count, avg tokens, max tokens, truncation rate

## Key Design Decisions

### Sequence Length
- R1 traces can be 8K+ tokens. Our small models have limited context.
- Default max_seq_len: 2048 (except R1 recipe which uses 4096)
- Truncation strategy: truncate from the LEFT of the chain-of-thought,
  keeping the final answer intact. Never truncate the answer.
- Log truncation rate per recipe — if >20% of samples are truncated,
  that recipe may not be suitable.

### Sample Count
- Small models can overfit SFT data quickly
- 100K samples is likely more than enough for <500M models
- The "quality" recipe tests whether 30K is sufficient
- Track train loss — if it hits near-zero, we're overfitting

### Epochs
- Default: 3 epochs (validated by Liquid AI for this scale)
- Quality recipe: 10 epochs (small dataset needs more passes)
- Track eval metrics per epoch — stop early if GSM8K degrades

### Difficulty Scoring
For recipes that filter by difficulty, use a simple heuristic:
- Level 1: Single-step arithmetic (e.g., "what is 3+5?")
- Level 2: Multi-step arithmetic (GSM8K-like)
- Level 3: Algebra, basic proofs
- Level 4: Competition math (AMC level)
- Level 5: Olympiad (AIME level)

Can approximate with: solution token length as proxy, or use
NuminaMath's built-in difficulty labels where available.

## Data Integrity

- [ ] No eval problems (GSM8K test, MATH500, AIME) in SFT training data
- [ ] Answer format is consistent (\boxed{} present in >95% of samples)
- [ ] Token length distribution logged per recipe
- [ ] 10 random samples per recipe manually inspected and confirmed readable
