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

## Results (2026-03-21)

### E1: `sft-stratos-seq4096` — COMPLETE
| SVAMP | GSM8K | MATH | AIME |
|-------|-------|------|------|
| 57% | 45% | 42% | **3.3%** |

**Verdict: R1 hypothesis validated.** Seq_len fix (2048→4096) turned a broken experiment (11% MATH, 0% AIME) into a working one. MATH regressed from base 59% (style shift penalty, expected). AIME=3.3% is the first signal from stratos and confirms exploratory reasoning traces produce AIME signal.

### E3: `sft-acemath-stratos-curriculum` — COMPLETE
| SVAMP | GSM8K | MATH | AIME |
|-------|-------|------|------|
| 12% | 33% | 33% | 0% |

**Verdict: Curriculum hurt worse than starting cold.** acemath→stratos collapsed MATH from acemath's 66% to 33% — even below base (59%). E3 is *worse* than E1 (base→stratos: 42% MATH, 3.3% AIME) on every metric. The acemath LoRA weights actively interfere with stratos learning rather than providing a useful foundation. Incompatible adapter initialization is worse than random init.

**Key insight**: curriculum learning only works when phase 1 and phase 2 data are compatible in style. acemath (step-by-step, exp=1) and stratos (exploratory, exp=8) are maximally incompatible — the gradients conflict rather than build.

---

### E2: `sft-mot7k14k-10k-seq4096` — COMPLETE
| SVAMP | GSM8K | MATH | AIME |
|-------|-------|------|------|
| 43% | 30% | 27% | 0% |

**Verdict: MoT 7K-14K underperforms stratos.** Only 6218 of 10K samples survived the seq4096 drop filter — the lower end of the bucket (shorter/easier traces). We inadvertently trained on the least-hard slice of MoT. Stratos (7175 samples, more consistent quality) dominates on every metric: +15pp MATH, +3.3pp AIME.

**Revised view of MoT buckets**: the 7K-14K char bucket, after seq4096 filtering, collapses to mostly medium-difficulty traces. The AIME signal (if any) lives in the 14K+ bucket (diff=8, exp=9) which needs seq8192 to keep most traces.

---

## Wave 2 Summary

| Experiment | MATH | AIME | Verdict |
|-----------|------|------|---------|
| base-no-sft | 59% | 0% | Baseline |
| acemath-10k | 66% | 0% | Best MATH, no AIME |
| dartmath-50k | 46% | 7% | Best AIME overall |
| E1 stratos seq4096 | 42% | **3.3%** | ✅ R1 works at proper seq_len |
| E2 MoT 7k-14k | 27% | 0% | ❌ Wrong bucket, too easy |
| E3 acemath→stratos | 33% | 0% | ❌ Curriculum actively hurts |

**Confirmed findings:**
- Seq_len fix is real: stratos at 2048 (broken) → 4096 (3.3% AIME)
- Curriculum on incompatible data is worse than cold start from base
- MoT 7K-14K is not the right bucket — need 14K+ for hard AIME signal
- Stratos is the best R1 dataset tested so far

**Dead ends:**
- acemath→R1 curriculum: makes things worse, don't retry
- MoT 7K-14K: too easy after seq4096 filtering

### Dartmath-50k AIME source — deep-dive (2026-03-21)

**Finding**: dartmath AIME signal is not generalization — it's memorization depth on 1 problem.

| Dataset | Unique problems | Total rows | Ratio | Max sols/prob |
|---------|----------------|------------|-------|---------------|
| dartmath-50k slice | 1067 | 50K | 46.9× | 270× |
| stratos | 9651 | 10K | 1.04× | 8× |
| acemath | 9993 | 10K | 1.0× | 2× |
| MoT 14K+ | 9965 | 10K | 1.0× | 3× |

Competition-keyword problems in dartmath-50k: **2 unique problems, 190 rows**.
- 1 asy-geometry problem: **188 solutions** (the entire AIME signal source)
- 1 generic "field day competition" problem: 2 solutions

The model achieves 7% AIME (2/30 problems) because it saw one competition-geometry problem solved 188 ways. This is depth-of-memorization, not R1-style exploratory generalization. The signal doesn't extrapolate to new AIME problem types.

**Implication for next experiments**: stratos/MoT are the right AIME path (exploratory reasoning, diverse problems). Dartmath's 7% is a ceiling artifact, not a training signal to build on.

---

## Completed Experiments

### E1: `sft-stratos-seq4096` — COMPLETE
| SVAMP | GSM8K | MATH | AIME |
|-------|-------|------|------|
| 57% | 45% | 42% | **3.3%** |

**What**: Stratos ~7,175 samples (seq4096 survival of full 17K), from base.
**Verdict**: R1 hypothesis partially validated. Seq_len fix (2048→4096) gave first AIME signal (3.3%). But expected ≥7%; hit 3.3% ceiling.
**Key finding**: Dataset analysis shows stratos has 70% survival at seq4096. E1 kept the shortest 70% of stratos. The hardest 30% (longest traces) have never been trained on.

---

### E2: `sft-mot7k14k-10k-seq4096` — COMPLETE
| SVAMP | GSM8K | MATH | AIME |
|-------|-------|------|------|
| 43% | 30% | 27% | 0% |

**What**: MoT 7K-14K bucket, 10K samples, seq_len=4096, from base.
**Verdict**: Underperforms stratos on every metric. Only 6218 of 10K survived — the lower end of the bucket. Stratos dominates (+15pp MATH, +3.3pp AIME). MoT 7K-14K is not the right bucket.

---

### E3: `sft-acemath-stratos-curriculum` — COMPLETE
| SVAMP | GSM8K | MATH | AIME |
|-------|-------|------|------|
| 12% | 33% | 33% | 0% |

**What**: Start from acemath-10k adapter, continue on stratos seq4096.
**Verdict**: Curriculum actively hurt. MATH collapsed from acemath's 66% to 33% — worse than E1 (42%) and even below base (59%). acemath (exp=1) and stratos (exp=8) are maximally incompatible styles. Adapter init is worse than cold start.

---

### E4: `sft-mot14k-5k-seq4096` — FAILED (data-seq_len mismatch)
| SVAMP | GSM8K | MATH | AIME |
|-------|-------|------|------|
| 47% | 59% | 59% | 0% |

**Verdict: Invalid — only 23/5000 samples survived.** MoT 14K+ has `min_chars=14000`; seq4096 handles ~14,336 chars max. Nearly 0% survival. Model is unchanged from base (MATH=59%). Lesson: never run `bucket_min_chars > seq_len × 3.5`.

---

### E8: `sft-stratos-full-17k-seq4096` — COMPLETE
| SVAMP | GSM8K | MATH | AIME |
|-------|-------|------|------|
| 47% | 37% | 39% | **3.3%** |

**What**: Full stratos, all 9,430 seq4096-surviving samples (vs E1's 7,175), from base.
**Hypothesis was**: more stratos data → AIME > 3.3%.
**Verdict: FAILED.** AIME flat at 3.3%. All other metrics *worse* than E1 (−10pp SVAMP, −8pp GSM8K, −3pp MATH).
**Why it failed**: The extra ~2,255 samples vs E1 are not harder traces — they're just more samples from the same distribution. The 30% of stratos dropped at seq4096 (the longest/hardest traces) were dropped in both E1 and E8. More seq4096 samples can't add signal that seq4096 structurally excludes.
**Conclusion**: More R1 traces at the same seq_len does not improve AIME. The ceiling is not about sample count.

---

### E5: `sft-mot14k-5k-seq8192` — TRAINING COMPLETE, EVAL PENDING
**What**: MoT 14K+ bucket, ~5K samples, seq_len=8192, from base. Checkpoint: `sft-mot14k-5k-seq8192`.
**Tests**: do the long hard MoT traces (64% survival at seq8192 vs 0% at seq4096) carry AIME signal?

---

### E9: `sft-mot-phase1-then-14k-seq8192` — TRAINING COMPLETE, EVAL PENDING
**What**: Start from `sft-mot-base-phase1` (MoT 0-7K phase), continue on MoT 14K+ seq8192.
**Tests**: does curriculum on MoT (easy→hard within same dataset style) beat cold start (E5)?
Checkpoint: `sft-mot-phase1-then-14k-seq8192`.

---

## Wave 3 Summary (2026-03-21)

| Experiment | SVAMP | GSM8K | MATH | AIME | Verdict |
|---|---|---|---|---|---|
| E1: stratos-7175-seq4096 | 57% | 45% | 42% | **3.3%** | ✅ First AIME signal |
| E8: stratos-9430-seq4096 | 47% | 37% | 39% | **3.3%** | ❌ More data doesn't scale AIME |
| E4: MoT14k-seq4096 | 47% | 59% | 59% | 0% | ❌ Invalid — 23/5000 samples survived |
| E2: MoT7k-14k-seq4096 | 43% | 30% | 27% | 0% | ❌ Wrong bucket, too easy |
| E3: acemath→stratos | 12% | 33% | 33% | 0% | ❌ Catastrophic forgetting |
| E5: MoT14k-seq8192 | — | — | — | ? | ⏳ Eval pending |
| E9: phase1→MoT14k-seq8192 | — | — | — | ? | ⏳ Eval pending |

**Unexpected finding**: `acemath-think-10k-v2` (not in Wave 3 plan) also hits **3.3% AIME** with dramatically better easy benchmarks (85% SVAMP, 70% GSM8K). Think traces without R1 style match stratos on AIME.

| acemath-think-10k-v2 | 85% | 70% | 62% | **3.3%** | ✅ Ties stratos, much better easy bench |

**Confirmed dead ends:**
- More stratos at seq4096 (E8): AIME plateau at 3.3%, no scaling
- acemath→R1 curriculum (E3): catastrophic forgetting, avoid
- MoT 7K-14K bucket (E2): not enough difficulty after seq4096 filtering
- MoT 14K+ at seq4096 (E4): structurally incompatible, 0% survival

**Open questions:**
1. Does MoT 14K+ at seq8192 produce AIME signal? (E5/E9 pending)
2. Does stratos at seq8192 (89% survival vs 70% at seq4096) break the 3.3% ceiling? (untested)
3. Why does acemath-think tie stratos on AIME despite different style? Is think-tag format sufficient?

---

### E6: `sft-acemath-mot7k-curriculum` — DEAD END (skip)
Gate: E3 MATH < 50%. E3 scored 33% — gate triggered. Incompatibility is fundamental (acemath exp=1 vs MoT exp=8). Skipped.

### E7: `sft-acemath-mot14k-curriculum` — BLOCKED
Gate: E3 MATH ≥ 60% AND AIME > 0%. E3 failed (33% MATH, 0% AIME). E7 is blocked until we find a
curriculum approach that doesn't cause catastrophic forgetting. Not pursuing yet.

---

## Key Empirical Priors (from dataset survey)

### Why exploratory score matters for AIME
- All datasets with exp ≤ 2 (acemath, dartmath, metamath, openmathinstruct2): AIME = 0% (except dartmath-50k = memorization artifact)
- MoT and stratos (exp=8-9) at proper seq_len: both produce 3.3% AIME — confirmed
- **acemath-think** (exp=1 but has `<think>` tags): also 3.3% AIME — think-tag format may matter independently of R1 style
- **AIME ceiling at 3.3%** across all tested approaches. Not broken yet.

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

## Decision Rules (updated 2026-03-21)

**AIME ceiling is 3.3% across all approaches. Next levers to try:**

| Condition | Next action |
|-----------|-------------|
| E5/E9 AIME > 3.3% | MoT 14K+ seq8192 breaks ceiling — scale it up |
| E5/E9 AIME = 3.3% | Ceiling is not about trace length; try stratos seq8192 |
| E5/E9 AIME = 0% | MoT doesn't generalise to AIME even at seq8192; stratos is the only R1 path |
| Any model AIME > 3.3% | Use it as GRPO init — RL reward likely needed to push past 10% |
| All seq8192 experiments flat at 3.3% | Model capacity bottleneck; try rank 32 or 1.5B base model |
