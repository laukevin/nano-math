# Spec 06: Post-Training (RL / GRPO)

## Goal

Apply reinforcement learning via GRPO to push math performance beyond what
SFT alone achieves. Study whether RL provides meaningful gains at small scale,
and whether curriculum design matters.

## Framework

**Primary:** HuggingFace TRL `GRPOTrainer`
- Mature, well-tested, good W&B integration
- Handles the GRPO algorithm (Group Relative Policy Optimization)

**Alternative:** Extend nanochat's existing GRPO stub
- Simpler, stays in nanochat's codebase
- May lack features (KL penalty, reward normalization, etc.)

**Decision:** Start with TRL. If it's too slow or doesn't work well with
nanochat's model format, fall back to extending the stub.

### Model Format Conversion

nanochat saves models in its own format. TRL expects HuggingFace format.
Need a conversion script:

```
scripts/train/convert_to_hf.py \
  --checkpoint results/sft-m-best/best_gsm8k.pt \
  --output models/hf/sft-m-best/ \
  --depth 16
```

This creates a HuggingFace-compatible model directory that TRL can load.
Also need the reverse conversion for eval with nanochat's eval harness.

## Reward Design

### Primary: Exact Match Binary

```python
def compute_reward(model_output: str, ground_truth: str) -> float:
    """
    Extract answer from model output, compare to ground truth.
    Returns 1.0 if correct, 0.0 if wrong.
    """
    predicted = extract_answer(model_output)
    if predicted is None:
        return 0.0  # couldn't parse an answer
    return 1.0 if predicted == ground_truth else 0.0
```

### Answer Extraction

Try these patterns in order:
1. `\boxed{...}` — standard math formatting
2. `The answer is ...` — natural language
3. `#### ...` — GSM8K format
4. Last number in the response — fallback

**Critical:** Log extraction failure rate. If >10% of outputs can't be
parsed, the model hasn't learned the answer format from SFT. This is
a signal that SFT was insufficient, not an RL problem.

### Reward Variants (for experimentation)

| Reward ID | Design | Hypothesis |
|-----------|--------|------------|
| `reward-binary` | 1.0 correct, 0.0 wrong | Baseline, cleanest signal |
| `reward-format` | +0.1 if \boxed{} present, +0.9 if correct | Encourages format compliance |
| `reward-partial` | 0.5 if correct approach but wrong answer, 1.0 if correct | May help small models that get close |
| `reward-length-penalty` | 1.0 - 0.001 * len(output) if correct, 0.0 if wrong | Discourages verbose outputs |

**Start with `reward-binary`.** Only try variants if binary reward
leads to degenerate behavior (reward hacking, empty outputs, etc.)

## Curriculum Strategies

### Strategy A: "Easy to Hard" (Baseline)
```
Stage 1: GSM8K problems → advance when pass@1 > 0.40
Stage 2: AMC problems → advance when pass@1 > 0.15
Stage 3: AIME problems → run to convergence
```
Standard curriculum learning. Most natural.

### Strategy B: "Hard Only"
```
Single stage: MATH500 problems (mixed difficulty)
Run to convergence.
```
**Hypothesis:** Small models might benefit from seeing hard problems early,
because they'll never solve them by rote — they must learn reasoning.

### Strategy C: "Interleaved"
```
Single stage: mix of GSM8K (60%) + AMC (30%) + AIME (10%)
Fixed mixture, no advancement gates.
```
**Hypothesis:** Mixing difficulties is better than staged curriculum.
The easy problems provide reward signal even when the model can't solve
hard ones yet.

### Strategy D: "Reverse Curriculum"
```
Stage 1: AIME/AMC problems → 500 steps (mostly failing, but learning format)
Stage 2: GSM8K problems → advance when pass@1 > 0.40
Stage 3: Back to AIME → run to convergence
```
**Hypothesis:** Exposure to hard problems first creates a "stretch" signal.
Then easy problems let the model consolidate. Then hard problems again.
Inspired by desirable difficulties in learning science.

## RL Datasets

| Dataset | HF ID | Size | Difficulty |
|---------|-------|------|-----------|
| GSM8K | openai/gsm8k (train split) | ~7.5K | Elementary school |
| AMC | From AIME-Preview / NuminaMath | ~500 | High school competition |
| AIME | AI-MO/aimo-validation-aime | ~90 | Olympiad qualifier |
| MATH500 | hendrycks/competition_math | ~500 | Mixed competition |

**Important:** GSM8K train is used for RL. GSM8K test is used for eval.
No leakage between RL training and eval sets.

## GRPO Hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| Group size | 8 | Number of completions per prompt |
| Max new tokens | 1024 | Cap output length |
| KL penalty coeff | 0.05 | Prevents policy from diverging too far |
| Learning rate | 1e-6 | Much lower than SFT |
| Batch size | 16 prompts | = 128 completions per batch |
| Temperature | 0.7 | For sampling completions |
| Steps per curriculum stage | 500 max | Hard cap per stage |
| Total RL steps | 2000 max | Budget constraint |

### Hyperparameter Sweep (optional)

If baseline GRPO works, test:
- KL penalty: {0.01, 0.05, 0.1}
- Group size: {4, 8, 16}
- LR: {5e-7, 1e-6, 5e-6}

## Metrics Logged Per Run

| Metric | Frequency | Source |
|--------|----------|--------|
| `rl/reward_mean` | Every step | GRPO trainer |
| `rl/reward_std` | Every step | GRPO trainer |
| `rl/kl_divergence` | Every step | GRPO trainer |
| `rl/policy_loss` | Every step | GRPO trainer |
| `rl/format_compliance` | Every step | % of outputs with \boxed{} |
| `rl/avg_output_length` | Every step | Tokens per completion |
| `rl/curriculum_stage` | On change | Current dataset |
| `eval/gsm8k_pass1` | Every 100 steps | Eval harness |
| `eval/math500_pass1` | Every 200 steps | Eval harness |
| `eval/amc_pass1` | Every 200 steps | Eval harness |
| `eval/aime_pass1` | Every 200 steps | Eval harness |
| `meta/wall_clock_hours` | Every 100 steps | Timer |

## Checkpointing

- Save when AIME pass@1 improves
- Save when GSM8K pass@1 improves (separate "best_gsm8k" checkpoint)
- Save at curriculum stage transitions
- Save final

## Reward Hacking Detection

See `specs/11_guardrails.md` for full details. Summary:

Watch for these degenerate behaviors:
1. **Empty outputs** → reward mean drops to 0, avg_output_length → 0
2. **Repetitive outputs** → high reward but low diversity (check unique answers)
3. **Format gaming** → outputs that match \boxed{} pattern but contain garbage
4. **Reward inflation** → reward goes up but actual eval metrics don't improve
5. **KL explosion** → policy diverged too far from SFT base

Auto-stop conditions:
- If `rl/reward_mean` > 0.8 but `eval/gsm8k_pass1` hasn't improved → flag
- If `rl/avg_output_length` < 10 for 50 consecutive steps → stop
- If `rl/kl_divergence` > 10 → stop

## Success Criteria for RL Phase

- [ ] At least one model shows >5 point GSM8K improvement from SFT → RL
- [ ] AIME pass@1 > 0 for at least one model (even 1 problem)
- [ ] No reward hacking detected (or hacking detected and mitigated)
- [ ] Curriculum comparison completed: clear ranking of strategies
- [ ] Scaling curve updated with post-RL performance

## Analysis Outputs

1. **RL gain plot**: GSM8K pass@1 before/after RL, per model size
2. **Curriculum comparison**: final eval scores per curriculum strategy
3. **Reward dynamics plot**: reward mean + eval score over RL steps
4. **KL vs performance tradeoff**: scatter plot of KL penalty vs eval gain
5. **Full pipeline scaling curve**: pretrain → SFT → RL at each model size
