# Experiment Hypotheses — math-nano

Tracks the hypothesis, expected outcomes, and downstream impact for each experiment.
Updated as results come in. Read alongside `next_steps.md` for full context.

---

## Background & Current Baselines

| Model | SVAMP | GSM8K | MATH | AIME | Notes |
|-------|-------|-------|------|------|-------|
| base-no-sft | 24% | 40% | 59% | 0% | Strong MATH from pretraining |
| sft-acemath-10k | 83% | 66% | 66% | 0% | Best MATH; acemath is step-by-step CoT |
| sft-dartmath-50k | 82% | 68% | 46% | 7% | Best AIME; hard problems + volume |
| sft-mot-base-phase1 | 61% | 46% | 44% | 0% | Tiny MoT run (2874 samples, seq=2048) |
| sft-stratos-10k | 33% | 19% | 11% | 0% | BROKEN — seq=2048 truncated all traces |

**Core problem**: AIME is 0% for all competent models. dartmath-50k gets 7% (2/30) but at a −13pp MATH cost.
**Diagnosis**: R1-style data (stratos, MoT) has traces of 2K-8K tokens, truncated to garbage by seq_len=2048.
**Theory**: Fix seq_len → exploratory reasoning signal → AIME > 0%.

---

## DAG of Experiments

```
             [E1] stratos-seq4096          [E2] mot7k14k-10k-seq4096
              base → stratos                 base → MoT 7K-14K
              (fix broken run)               (10x the tiny phase1 run)
                     │                              │
                     └──────────┬───────────────────┘
                                │
                   GATE: Does R1 at seq4096 produce AIME > 0%?
                         YES ──────────────────── NO
                          │                        │
              ┌───────────┼──────────────┐    Investigate model
              │           │              │    capacity / eval tokens
       [E4] mot14k   [E5] mot14k    [E6] acemath-stratos
        seq4096       seq8192        curriculum (seq4096)
         (drop ~50%)  (keep ~70%)    (ALREADY RUNNING IN PARALLEL)
              │           │
              └─────┬─────┘
                    │
             seq_len ablation:
             does keeping longer
             traces help AIME?
                    │
             [E7] acemath-mot14k-curriculum
              (if stratos curriculum works)
```

`[E3]` runs in parallel with E1+E2 — doesn't need their results first.

---

## Active Experiments

### E1: `sft-stratos-seq4096`

**What**: Stratos full dataset (~17K samples), seq_len=4096, from base, eval_max_tokens=2048.

**Why this is #1 priority**: The only broken dimension is seq_len. Stratos is the cleanest R1 test —
quality=4.96, exp=8, 100% consistent, 100% is_math. p50=9.8K chars ≈ 2450 tokens, p75=14.7K chars ≈
3675 tokens. Roughly 75% of traces fit at 4096. This directly re-runs sft-stratos-10k with the fix.

**Hypothesis**: Stratos at seq4096 produces **AIME ≥ 7%** (at least matching dartmath-50k's best result)
because:
- Exploratory reasoning (exp=8) teaches search strategies, not just answer patterns
- Competition-level difficulty (diff=7) exposes the model to AMC/AIME-class problems
- 17K samples is enough signal (dartmath-50k got 7% AIME with 50K short CoT samples)

**Expected MATH**: 40-55% (regression from base 59%). Style shift (step-by-step → exploratory) will hurt
MATH, but this is the tradeoff. AIME signal matters more here.

**Expected cost**: ~$1-2 on A100.

**What this unlocks**:
- YES → validates R1 hypothesis, green-lights E4/E5/E7
- NO (AIME=0%) → seq_len was not the only problem. Check if eval_max_tokens is too short, inspect
  generation samples for length, consider model capacity as bottleneck.

---

### E2: `sft-mot7k14k-10k-seq4096`

**What**: MoT 7K-14K chars bucket, 10K samples, seq_len=4096, from base, eval_max_tokens=2048.

**Why this is #2 priority**: We have `sft-mot-base-phase1` (2874 samples, seq=2048, MATH=44%, AIME=0%).
That experiment used too few samples AND the upper end of the 7K-14K bucket (traces ~7K chars ≈ ~1750 tok)
may have been marginally truncated at 2048. This 10K run at proper seq4096 tests both levers at once.

The 7K-14K bucket scores quality=5.0 (perfect in phase2 consistency check), exp=8, diff=7.
All traces fit cleanly within 4096 tokens (7K chars ≈ 1750 tok, 14K chars ≈ 3500 tok). Zero drops expected.

**Hypothesis**: 10K MoT at seq4096 produces **AIME ≥ 3%** and **MATH ≥ 40%** (better than the 2874-sample run).
The tiny phase1 run showed 44% MATH — more data should narrow the regression. AIME gain requires the
exploratory reasoning signal to transfer.

**Comparison**: vs E1 (stratos), this tests whether MoT's lower avg difficulty (diff=7 vs stratos diff=7)
and the specific bucket selection matter. Also tests whether the "phase1" bucket is the right starting point
vs the full 14K+ bucket (E4/E5).

**Expected cost**: ~$1-2 on A100.

**What this unlocks**:
- With E1: establishes "R1 from base" baseline for both datasets
- Informs whether MoT 14K+ bucket (harder, longer) is worth the seq_len overhead (E4 vs E2)
- If MATH regression is worse than E1 → suggests MoT < 7K-14K is suboptimal starting point

---

### E3: `sft-acemath-stratos-curriculum`

**What**: Start from `sft-acemath-10k` adapter, continue training on stratos full dataset (~17K),
seq_len=4096, eval_max_tokens=2048. Uses `init_adapter=/checkpoints/sft-acemath-10k`.

**Why this is #3 priority**: This is the "jackpot" experiment. If it works, we get both:
- acemath's +7pp MATH advantage (66% vs base 59%)
- stratos's exploratory reasoning → AIME signal

Runs in parallel with E1/E2 — acemath-10k adapter already exists on Modal.

**Why stratos (not MoT 14K+) for curriculum**:
- Stratos median ≈ 2450 tokens — less style shock than MoT 14K+ (median ≈ 7000 tokens)
- The acemath→openthoughts curriculum showed catastrophic forgetting (66→36% MATH). The hypothesis is
  that the short-medium traces in stratos are less disruptive than the very long MoT traces.
- Stratos is also being tested from base in E1 — we'll have a direct comparison (base→stratos vs acemath→stratos).

**Hypothesis**: MATH stays at **60-66%** (acemath gains partially preserved) and **AIME ≥ 3%**. The
curriculum avoids catastrophic forgetting because stratos's traces are ~2-4x shorter than MoT 14K+ and
the acemath model's step-by-step foundation can "absorb" the exploratory format more gradually.

**Risk**: acemath's structured CoT (exp=1, step_by_step=9) is maximally different from stratos's exploratory
style (exp=8). Even if traces are medium-length, the output distribution shift is severe. We may still see
forgetting (MATH < 50%). The prior case was openthoughts which had format incompatibility AND wrong domain.
Stratos is math-only, so format is the only issue.

**Expected cost**: ~$1-2 on A100.

**What this unlocks**:
- If MATH ≥ 60% AND AIME > 0% → this is our best model. Try acemath→MoT curriculum next.
- If MATH drops to ~40-50% → curriculum partially works but style mismatch hurts. Try shorter-trace MoT
  (E2's 7K-14K bucket) as phase 2 instead.
- If MATH < 40% → catastrophic forgetting again. Need to investigate learning rate, epochs, or mix.

---

## Pending Experiments (gated on E1/E2/E3 results)

### E4: `sft-mot14k-5k-seq4096`
MoT 14K+ chars (hardest bucket: diff=8, exp=9), 5K samples, seq_len=4096. ~40-50% of traces drop (too long).
**Gate**: run after E1 or E2 confirms AIME > 0% from R1 training.
**Hypothesis**: harder problems + more exploratory traces → higher AIME than E2, at further MATH cost.

### E5: `sft-mot14k-5k-seq8192`
Same as E4 but seq_len=8192. ~70% of traces kept. Tests whether the longer traces (the ones E4 drops)
carry additional AIME signal. Slower and more memory-intensive.
**Gate**: run alongside E4, compare: AIME(E5) vs AIME(E4). If similar → seq4096 is sufficient.

### E6: `sft-acemath-mot7k-curriculum`
acemath-10k → MoT 7K-14K bucket, seq_len=4096.
**Gate**: run if E3 (acemath-stratos) shows catastrophic forgetting (MATH < 50%).
The shorter MoT traces may be less disruptive than stratos's long traces.
**Alternative**: if E3 succeeds, skip this and try acemath→MoT14k directly.

### E7: `sft-acemath-mot14k-curriculum`
acemath-10k → MoT 14K+ bucket, seq_len=8192.
**Gate**: run only if E3 (acemath-stratos) succeeds (MATH ≥ 60%, AIME > 0%).
**Hypothesis**: hardest problems as phase 2 from strong acemath foundation → best AIME yet.

---

## Key Empirical Priors (from dataset survey)

### Why exploratory score matters for AIME
- All datasets with exp ≤ 2 (acemath, dartmath, metamath, openmathinstruct2): AIME = 0%
- dartmath-50k (exp=1, but 50K hard problems): AIME = 7% — volume of hard problems creates signal
- MoT and stratos (exp=8-9): untested at proper seq_len — the primary hypothesis we're testing

### Why terse data hurts (don't use numinamath15 or MATH train set)
- `sft-math` (human terse proofs, median 162 chars): MATH = 19% (−40pp from base)
- `sft-mathinstruct-10k` (CoT+PoT hybrid): MATH = 14% — wrong format destroys pretrained knowledge
- numinamath15 (proof_style=9, median 1.6K chars): high risk of same failure mode. Deprioritized.
- Principle: 600M model can't learn from terse proofs — needs every step explicit

### Why acemath scaling is saturated
- 10K/15K/20K acemath: MATH flat at 65-66%, AIME flat at 0-3.3%
- More acemath data is not the answer. Next levers: R1-style data or RL/GRPO.

### numinamath v1 mystery
- sft-numinamath-10k: MATH = 12%, SVAMP = 12% — collapsed to near-zero
- Needs investigation before running numinamath15 (may have same format bug)

---

## Decision Rules

| Condition | Next action |
|-----------|-------------|
| E1 AIME ≥ 7% | Launch E4+E5 in parallel |
| E1 AIME = 3% | Launch E4 only (E5 if budget allows) |
| E1 AIME = 0% | Inspect generation samples — is model generating long chains or short answers? |
| E3 MATH ≥ 60% AND AIME > 0% | This is the winner; launch E7 |
| E3 MATH 50-60% | Partial success; try E6 (shorter traces) |
| E3 MATH < 50% | Catastrophic forgetting again; try lower LR or fewer epochs for phase 2 |
| E2 MATH >> E1 MATH | MoT 7K-14K is better foundation than stratos; run MoT curriculum (E6) |
