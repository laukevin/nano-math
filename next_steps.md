# Next Steps — math-nano (Seoul)

## Where we are (as of 2026-03-19)

### Phase 1: nanochat from-scratch (complete)
- **Model**: depth-8 (125M params), pretrained on FineWeb-edu (~524M tokens)
- **Best result**: SVAMP 6%, GSM8K 0-2% — ceiling hit at this model size
- **Lesson**: sub-200M models trained from scratch can't chain multi-step math reasoning

### Phase 2: Qwen3-0.6B + LoRA SFT (pipeline working, baseline in progress)
- **Base model**: Qwen/Qwen3-0.6B-Base (606M params, 36T tokens pretraining)
- **Method**: LoRA SFT (rank 16, 1.66% trainable params = 10M params)
- **Pipeline**: data prep → SFT → eval (all 4 tiers) → registry, automated on Modal
- **Prompt format**: `chat_think` (Qwen3 thinking mode)
- **Baseline run**: full GSM8K (7473), 3 epochs, A100 — in progress on Modal

## Infrastructure

### Eval tiers (easy → hard)
Every experiment evals on all 4 tiers so we can see what moves at each difficulty level:

| Tier | Benchmark | Size | Level |
|------|-----------|------|-------|
| Easy | SVAMP | 100 | Grade-school arithmetic |
| Medium | GSM8K | 100 | Grade-school multi-step |
| Hard | MATH | 100 | AMC/competition level |
| Hardest | AIME 2025 | 30 | Olympiad (uncontaminated) |

**AIME contamination note**: virtually all competition-math datasets (MATH, NuminaMath, MetaMath, OpenThoughts, DART-Math) contain historical AIME problems. We eval on AIME 2025 only — post-training-data cutoff, no contamination risk.

### Running an experiment
```bash
# Single experiment — evals on all 4 tiers automatically
uv run modal run --detach modal_jobs/train.py::run_sft_lora \
  --experiment-id sft-gsm8k \
  --data-source gsm8k \
  --data-size -1 \
  --epochs 3

# Check registry
uv run modal volume get math-nano-results experiment_registry.jsonl
```

### Key files
- `scripts/train/sft_lora.py` — LoRA SFT training (transformers + peft)
- `scripts/data/normalize_dataset.py` — Download + normalize HF datasets
- `scripts/eval/run_hf.py` — Eval HF models on SVAMP/GSM8K/MATH/AIME (batched on GPU)
- `scripts/registry.py` — Experiment registry (append, read, leaderboard)
- `modal_jobs/train.py::run_sft_lora()` — Orchestrates: data → train → eval → registry

### Available datasets
| ID | HF Path | Size | Style | Notes |
|----|---------|------|-------|-------|
| `gsm8k` | `openai/gsm8k` | 7.5K | Short CoT | Grade-school, clean baseline |
| `math` | `lighteval/MATH` | 7.5K | Short CoT | Competition math (AMC/AIME level) |
| `metamath` | `meta-math/MetaMathQA` | 395K | Short CoT | Augmented/rephrased GSM8K+MATH |
| `numinamath` | `AI-MO/NuminaMath-CoT` | 860K | CoT | Competition math, broad |
| `numinamath15` | `AI-MO/NuminaMath-1.5` | 900K | CoT | Upgraded v1.5, better decontamination |
| `openmathinstruct2` | `nvidia/OpenMathInstruct-2` | 14M | CoT | Nemotron-generated solutions |
| `openthoughts3` | `open-thoughts/OpenThoughts-114k` | 114K | R1-style long CoT | ⚠️ **Mostly coding problems, not math** — do not use for math SFT |
| `stratos` | `bespokelabs/Bespoke-Stratos-17k` | 17K | R1-style long CoT | High-quality R1 distillation |
| `mixture_of_thoughts` | `open-r1/Mixture-of-Thoughts` (math subset) | 93K | R1-style long CoT | ✅ **Best R1 math dataset** — `<think>` tags, `\boxed{}`, verified, mean 17K chars |
| `dartmath` | `hkust-nlp/dart-math-hard` | 585K | CoT | Difficulty-aware rejection sampling (NeurIPS'24) |
| `mathinstruct` | `TIGER-Lab/MathInstruct` | 260K | CoT + PoT | Hybrid natural language + Python (ICLR'24) |
| `acemath` | `nvidia/AceMath-Instruct-Training-Data` | varies | CoT | Two-stage SFT data (ACL'25) |

### Fixed hyperparams (don't sweep these yet)
- LoRA: rank 16, alpha 32, dropout 0.05
- LR: 2e-5, cosine schedule, warmup 0.03
- Batch size: **use `--max-tokens-per-batch 8192`** (token-budget, not fixed batch size), weight decay: 0.01
- Max seq len: 2048
- Eval: 100 problems per tier (30 for AIME), 1024 max tokens, batch=8 on GPU

### Batching strategy (validated 2026-03-20)
Fixed `batch_size=8` is correct for acemath but severely underutilizes A100 for short-sequence data.
Use `--max-tokens-per-batch 8192` via `run_sft_lora --max-tokens-per-batch 8192`:
- Smoketested on A100-40GB: batch=8 at seq=1167 → 26GB (66%). batch=16 OOMs with random batches.
- With token-budget + group_by_length: short seqs (p50=565 tok) → ~14 samples/batch; long seqs (p99=2182 tok) → ~3 samples/batch
- Memory stays bounded because token count per batch is fixed at ~8192: 3×2182 tok ≈ 8×800 tok ≈ same memory
- Long batches (p99, ~3 samples) are handled automatically — no special code needed, memory is fine (~10GB)
- Gradient scale varies across batches (3 vs 14 samples) but SFT loss is per-token averaged so impact is small
- **Do NOT use `--packing`** with token-budget (redundant, both sort by length)

## Empirical findings (evidence-backed, 2026-03-20)

### Base model behavior
- **Qwen3-0.6B-Base is already strong at MATH (59%)** despite weak SVAMP (24%) / GSM8K (40%).
  *Why*: competition math is massively over-represented in 36T pretraining data (AoPS, arXiv, LaTeX). Grade-school word problems are not.
- **Extraction rate is 100% for base on all benchmarks** (confirmed from eval JSON). Format is not the bottleneck — the base model correctly produces `\boxed{}`. It just gets simple arithmetic wrong.
- Implication: the "few-shot vs chat_think" prompt-format hypothesis is ruled out.

### SFT effects by dataset type
- **Grade-school / mixed-level SFT hurts MATH** (catastrophic forgetting):
  - dartmath-10k: MATH 37% vs base 59% (−22pp)
  - dartmath-50k: MATH 40% (more data barely helps)
  - mathinstruct-10k: **catastrophically bad** — MATH 14%, SVAMP 20%, worse than base everywhere. CoT+PoT hybrid format is toxic for this model.
- **Competition-math SFT is the only thing that beats base on MATH**:
  - acemath-10k: MATH 66% (+7pp over base), SVAMP 83%, GSM8K 66%
  - openmathinstruct-10k: MATH 56% (−3pp vs base, roughly neutral)
- **SFT always helps on easy tiers** (SVAMP/GSM8K) — teaches word-problem reasoning format.

### Data scaling (acemath)
- Scaling effective samples 4,695 → 9,696 (acemath-10k → acemath-10k-v2) showed **no improvement** (83/66/66 vs 81/65/65, within noise at n=100).
- **acemath scaling is saturated — confirmed across 10K/15K/20K**:

| Size | SVAMP | GSM8K | MATH | AIME |
|------|-------|-------|------|------|
| 10K | 83% | 66% | 66% | 0% |
| 15K | 81% | 68% | 65% | 3.3% |
| 20K | 86% | 68% | 65% | 0% |

  MATH is completely flat at 65-66%. GSM8K improves 2pp (66→68%) then stalls. AIME is noise (0/3.3/0% — 1 problem, not real signal).
- **More acemath data is not the answer.** The model has extracted all it can from this distribution at LoRA rank 16, 3 epochs. Next levers: R1-style data, GRPO/RL, or harder problem distributions.

### Thinking format: marginal impact at 10K samples (2026-03-20)
Direct test of whether putting reasoning inside `<think>` blocks improves performance:

| Model | SVAMP | GSM8K | MATH | AIME | Notes |
|-------|-------|-------|------|------|-------|
| sft-acemath-10k | 83% | 66% | 66% | 0% | baseline (no-think format) |
| sft-acemath-think-10k (broken) | 83% | 71% | 64% | 0% | empty `<think></think>`, solution outside |
| sft-acemath-think-10k-v2 (fixed) | 84% | 69% | 63% | 3.3% | reasoning inside `<think>`, `\boxed{}` after `</think>` |

**Finding**: The thinking format fix made no meaningful difference on MATH (63-64% vs 66% baseline). GSM8K is +3-5pp but within statistical noise at n=100. The single AIME solve (3.3%) for v2 is the first non-zero AIME result — suggestive but not conclusive (1/30 problems).

- Loss is slightly higher for v2 (0.209 vs 0.195) — model is learning a harder structural task (reasoning trace → clean boxed answer) but eval numbers don't reflect it at 10K samples.
- Possible explanations: (a) 10K samples insufficient to learn the think→answer pattern, (b) acemath solutions are CoT not exploratory reasoning, (c) Qwen3-0.6B is too small to genuinely benefit from thinking traces via SFT.
- **For real thinking gains, use R1-distilled data (openthoughts, stratos)** where `<think>` contains actual exploration and backtracking. Acemath CoT just reorganizes the same content.

### OpenThoughts3 is a coding dataset, not math (2026-03-21)
Manual inspection of samples revealed that `open-thoughts/OpenThoughts-114k` is primarily **competitive programming problems** ("Generate an executable Python function..."), not math. Solutions use DeepSeek-R1 tags (`<|begin_of_thought|>` / `<|end_of_solution|>`), and `ensure_boxed()` appends garbage answers like `\boxed{000}` and `\boxed{3.}`. All openthoughts experiments are invalid as R1-math experiments.

### R1-style data (openthoughts, stratos) — broken due to seq len truncation (2026-03-20)
We ran `sft-stratos-10k` and `sft-openthoughts-10k` but both are catastrophically worse than base:

| Model | SVAMP | GSM8K | MATH | AIME |
|-------|-------|-------|------|------|
| base-no-sft | 24% | 40% | 59% | 0% |
| sft-openthoughts-10k | 29% | 36% | 6% | 3.3% |
| sft-stratos-10k | 33% | 19% | 11% | 0% |

**Root cause**: `max_seq_len=2048` truncates R1 traces mid-reasoning. OpenThoughts/Stratos `<think>` blocks are 4K-8K tokens long. The model trains on sequences that end mid-thought with no `</think>` and no `\boxed{}`, learning to generate long confused output that never reaches an answer.

**Fix needed before these experiments are meaningful**:
- Rerun with `--max-seq-len 4096` (drops traces > 4096 tokens, ~20-30% of openthoughts)
- Eval with `--max-tokens 4096` so model can finish its reasoning chain
- A100 required (4x longer sequences = 4x more memory per sample)

**Planned experiments (not yet run)**:
1. `sft-openthoughts-5k-seq4096` — 5K R1 traces from base, seq_len=4096
2. `sft-stratos-5k-seq4096` — 5K Stratos traces from base, seq_len=4096
3. `sft-acemath+openthoughts-5k-seq4096` — 5K R1 traces starting from acemath-10k adapter (curriculum: CoT foundation → exploratory reasoning)

Experiment 3 tests whether acemath SFT (66% MATH) is a better starting point for R1-style training than raw base (59% MATH).

### Solution verbosity matters as much as difficulty level
`sft-math` trains on the same competition-math distribution as the MATH benchmark yet scores only 19% — far worse than base (59%). `sft-acemath-10k` on the same distribution scores 66%.

The difference is solution style:
- **MATH train set**: human-written, terse, mathematician-style. 50-150 tokens. Skips steps, assumes background. e.g. *"For the piecewise function to be continuous, the cases must 'meet' at 2 and −2. This implies a(2)+3=2−5..."*
- **acemath math_sft**: LLM-generated, verbose, student-friendly. 200-600 tokens. Numbered steps, explicit LaTeX for every calculation.

A 600M model can't learn from terse proofs — it can't reconstruct the missing steps, so it gets no useful gradient signal and corrupts its pretrained capability instead. Verbose model-generated solutions are better training data for small models because every step is explicit and easy to imitate. This is the same reason R1-distillation works: student models need to see the full reasoning, not the compressed expert version.

Token length distributions (Qwen3 tokenizer, n=7,500 each):

| | median | mean | p10 | p90 |
|---|---|---|---|---|
| MATH train (human-written) | 162 | 228 | 63 | 481 |
| acemath math_sft (LLM-generated) | 513 | 661 | 263 | 1194 |

60% of MATH solutions are under 200 tokens. 60% of acemath solutions are over 400 tokens. Distributions barely overlap — effectively different content types, not just a style difference.

**Implication**: when evaluating datasets, solution length and explicitness matter as much as problem difficulty. Prefer LLM-generated verbose solutions over human-written terse ones for sub-1B SFT.

### Contamination
- **Direct test-set contamination is negligible**: 6 exact MATH test problems appear in first 10K acemath samples. Expected ~0.1 contaminated problems in our 100-problem eval. Not meaningful.
- **Distribution overlap is real**: acemath `math_sft` and MATH benchmark are the same genre (AMC/competition). MATH scores reflect "can do competition math" not just memorization.
- **AIME 2025 is the only clean generalization signal** — post-training-cutoff, provably uncontaminated, 0% for all models so far.

### Qwen3 chat template + thinking mode (discovered 2026-03-20)

The `enable_thinking` flag behaves **opposite to intuition**:

| Call | Output ends with |
|------|-----------------|
| `apply_chat_template(generation_prompt=True, enable_thinking=True)` | `assistant\n` — **no `<think>`** |
| `apply_chat_template(generation_prompt=True, enable_thinking=False)` | `assistant\n<think>\n\n</think>\n\n` — forces empty think, model answers after |
| `apply_chat_template(full_conv, either setting)` | always `<think>\n\n</think>\n\n{solution}` — identical either way |

**What this means for training:**
The prefix (generation prompt) always ends at `assistant\n` — `<think>` is never provided by the prompt. The model must learn to generate `<think>` as its own first output token. It is therefore always in the loss.

**What the original code actually trained:**
```
loss tokens: <think>\n\n</think>\n\n{full solution}
```
Model learned: generate `<think></think>` immediately (empty think), then produce the full CoT solution outside the think block. Thinking mode was structurally present but semantically empty — the model always skipped reasoning.

**The fix in `sft_lora.py` (tokenize_chat_think):**
We now build the sequence manually instead of using `apply_chat_template` on the full conversation:
```
prefix (masked):  ...assistant\n
loss:             <think>\n{reasoning}\n</think>\n\boxed{answer}<|im_end|>
```
The reasoning is now inside `<think>`, and only the final boxed answer comes after `</think>`. The structural change: the model must learn to reason inside the block, then produce a clean answer.

**Result:** The structural change made marginal difference at 10K samples on acemath. See "Thinking format" section under empirical findings for full comparison. The key insight: acemath solutions are verbose CoT, not exploratory reasoning — reformatting them into `<think>` blocks doesn't change what the model learns, only where the tokens appear.

**For R1-style data (openthoughts, stratos):** The training solutions already contain real `<think>...</think>` traces with exploration and backtracking. These will naturally be inside the think block — no restructuring needed. They're the only datasets that teach *genuine* exploratory reasoning vs fixed CoT traces.

## Key research insights

### Data scaling for sub-1B LoRA SFT
- **Sweet spot: 10K-100K high-quality samples** (power-law diminishing returns beyond that for LoRA)
- **Multi-epoch on smaller data > single-epoch on larger data** — "Data Repetition Beats Scaling" (Feb 2026): 128 epochs on 400 samples outperformed 1 epoch on 51K on AIME
- **Quality > quantity** — s1 paper: 1K curated samples matched o1-preview on 32B model. But sub-1B models need more data because less latent reasoning from pretraining
- **Difficulty-aware sampling** (DART-Math, NeurIPS'24): oversampling hard problems beats uniform sampling
- **Two-stage curriculum validated** (AceMath, ACL'25): basic math competence first, then competition-level

### Implication for our experiments
We need to:
1. **Log loss curves + save checkpoints** to detect saturation (not just final loss)
2. **Test multi-epoch** on smaller high-quality data (e.g., 10K × 10 epochs vs 100K × 1 epoch)
3. **Eval all tiers every time** to see which data helps easy vs hard problems

## Experiment plan

### Wave 0: Baseline (complete)

| Exp ID | SVAMP | GSM8K | MATH | AIME | Notes |
|--------|-------|-------|------|------|-------|
| `base-no-sft` | 24% | 40% | 59% | 0% | Strong MATH from pretraining; 100% extraction |
| `sft-gsm8k-full-v1` | — | — | — | — | v2 eval pending |

### Wave 1 results (v2 eval, EOS fix applied, 100 problems each, sorted by MATH)

> **Updated 2026-03-21**: includes thinking-format variants and curriculum experiments.

| Exp ID | SVAMP | GSM8K | MATH | AIME | Notes |
|--------|-------|-------|------|------|-------|
| `sft-acemath-10k` | **83%** | **66%** | **66%** | 0% | Best overall |
| `sft-acemath-15k` | 81% | 68% | 65% | 3.3% | Scaling saturated |
| `sft-acemath-20k` | 86% | 68% | 65% | 0% | Scaling saturated |
| `base-no-sft` | 24% | 40% | 59% | 0% | Pretraining ceiling |
| `sft-dartmath-10k` | 76% | 68% | 52% | 0% | Strong #2 — difficulty-aware sampling |
| `sft-metamath-10k` | 75% | 63% | 50% | 0% | Solid — augmented GSM8K+MATH |
| `sft-openmathinstruct-10k` | 61% | 40% | 46% | 0% | OK — Nemotron-generated |
| `sft-dartmath-50k` | 82% | 68% | 46% | **7%** | Best AIME — 5x data hurts MATH but helps AIME |
| `sft-gsm8k-full-v1` | 51% | 51% | 33% | 0% | Grade-school only |
| `sft-metamath-50k` | 63% | 41% | 31% | 0% | 50k worse than 10k (−19pp) |
| `sft-openmathinstruct-50k` | 59% | 37% | 30% | 0% | 50k worse than 10k (−16pp) |
| `sft-acemath-think-10k-v2` | 84% | 69% | 63% | 3.3% | Fixed thinking, no improvement |
| `sft-math` | 38% | 26% | 19% | 0% | Human terse proofs hurt small models |
| `sft-mathinstruct-10k` | 20% | 20% | 14% | 0% | CoT+PoT hybrid is toxic |
| `sft-stratos-10k` | 26% | 19% | 14% | 0% | **Broken** — seq_len too short |
| `sft-openthoughts-10k` | 29% | 36% | 6% | 3.3% | **Broken** — seq_len too short |
| `sft-numinamath-10k` | 12% | 2% | 2% | 0% | **Broken** — unknown cause, needs investigation |

**Key findings:**
- acemath-think-10k is the new #1 at 69% MATH — thinking format fix helped ~3pp
- acemath-10k is strong; scaling beyond 10K is flat (10K/15K/20K all at 65-66% MATH)
- dartmath-10k is a strong second (52% MATH, 68% GSM8K) — difficulty-aware sampling helps
- dartmath-50k has the **best AIME (6.7% = 2/30)** — exposure to hard competition problems despite lower MATH
- **base-no-sft scores 59% MATH** — Qwen3-0.6B pretraining is already strong; SFT delta is modest
- **50K consistently worse than 10K** on MATH (dartmath −6pp, metamath −19pp, openmathinstruct −16pp)
- R1-style data (openthoughts, stratos) completely broken — seq_len truncation AND openthoughts is coding not math
- numinamath-10k mysteriously broken (2% MATH) — needs investigation

### R1 curriculum on openthoughts3 — failed (2026-03-21)
We ran a 2-phase curriculum: phase1 (short traces < 7K chars, seq2048), phase2 (medium 7K-14K chars, seq4096), starting from both base model and acemath-10k. Dataset turned out to be primarily coding problems, so results are not representative of math R1 training.

| Experiment | SVAMP | GSM8K | MATH | AIME | Notes |
|---|---|---|---|---|---|
| base-phase1 | 52% | 52% | 40% | 0% | Short coding traces from base |
| acemath-phase1 | 47% | 33% | 36% | 3.3% | Short traces hurt acemath (66→36%) |
| base-phase2 | 44% | 33% | 28% | 0% | Medium traces hurt base further |
| acemath-phase2 | 58% | 55% | 48% | 0% | Medium traces partially recover acemath |

**Findings:**
- Curriculum on wrong dataset — openthoughts3 is coding, not math. All results invalid as R1-math tests.
- acemath-phase1 dropped from 66% to 36% MATH — catastrophic forgetting from format mismatch
- acemath-phase2 recovered to 48% — medium traces partially undo phase1 damage, suggesting **skip phase1 for acemath**
- base-phase2 degraded from phase1 (40→28%) — medium traces too hard without math foundations
- AIME signal in acemath-phase1 (3.3%) likely noise given the dataset quality issues

### Mixture-of-Thoughts: the right R1-style math dataset (2026-03-21)
After surveying 5 candidate R1-style datasets (`open-r1/OpenR1-Math-220k`, `open-r1/Mixture-of-Thoughts`, `nvidia/OpenMathReasoning`, `amphora/QwQ-LongCoT-130K`, `a-m-team/AM-DeepSeek-R1-Distilled-1.4M`), **`open-r1/Mixture-of-Thoughts` (math subset)** is the clear choice:

| Dataset | Think tag | `\boxed{}` | Math-only | Notes |
|---|---|---|---|---|
| openr1_math (OpenR1-Math-220k) | ❌ none | ~33% | ✅ | Plain CoT, no thinking tags |
| **mixture_of_thoughts** | ✅ `<think>` | ✅ 100% | ✅ | **Best — Qwen3-compatible, pure math** |
| openmath_reasoning (NVIDIA) | ❌ none | ❌ | ✅ | Solution field empty in streaming |
| qwq_longcot | ❌ none | ✅ | ✅ | Long narrative, no think tags |
| am_deepseek_r1 | ✅ `<think>` | ❌ | mixed | `</answer>` not `\boxed{}`, has physics |

**Mixture-of-Thoughts details:**
- 93K math samples with `<think>reasoning</think>\nfinal_answer_with_boxed` format
- Distilled from DeepSeek-R1, verified correct answers
- Char length distribution (n=2000): mean=16.9K, median=13.4K, p90=34K
- Curriculum splits: phase1 < 7K chars (20%), phase2 7K-14K (32%), phase3 14K-28K (34%)
- Added to `normalize_dataset.py` as `mixture_of_thoughts`; normalizer strips `<think>` tags so `tokenize_chat_think` re-wraps in Qwen3 format. Validated: 100/100 samples have `\boxed{}`, 0 think tag leaks.

### Dataset analysis tooling + full survey (2026-03-21)

Built `scripts/data/analyze_dataset.py` — automated dataset quality analysis pipeline:
- **Phase 1 (5 samples)**: non-LLM regex stats + Gemini 3 Flash rubric extraction
- **Phase 2 (25 samples)**: Gemini 3.1 Flash Lite consistency check against the phase 1 rubric
- Non-LLM stats: boxed rate, think tag rate, coding contamination, R1 opener rate, bold headers, step markers, length distribution (mean/median/p10–p99, bucket breakdown)
- LLM rubric dimensions: style (1–10), verbosity (1–10), step_by_step_score, proof_style_score, exploratory_score (key for AIME), latex_density, difficulty_score + label, reasoning_type, content types, answer format, coding contamination
- Results stored in `logs/dataset_research/<dataset>.json`
- API key in `.env` (gitignored), models: Gemini 3 Flash (phase 1) + Gemini 3.1 Flash Lite (phase 2)

**Full survey results** across all 12 datasets (5 samples phase 1, 25 samples phase 2):

| dataset | style | stp | prf | exp | ltx | diff | reasoning | Q | coding | is_math | styl_match | avgQ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| acemath | structured | 9 | 4 | 1 | 8 | 6 | step_by_step | 9 | none | 100% | 100% | 4.48 |
| dartmath | semi-structured | 6 | 4 | 1 | 8 | 6 | step_by_step | 8 | none | 100% | 92% | 4.68 |
| gsm8k | structured | 3 | 1 | 1 | 2 | 1 | step_by_step | 9 | none | 100% | 88% | 4.80 |
| math | semi-structured | 3 | 2 | 2 | 8 | 5 | step_by_step | 10 | none | 100% | 92% | 4.76 |
| mathinstruct | semi-structured | 3 | 2 | 2 | 4 | 3 | mixed | 4 | some | 96% | 36% | 3.08 |
| metamath | structured | 4 | 2 | 1 | 7 | 4 | step_by_step | 9 | none | 100% | 100% | 4.72 |
| mot [0–7k] | r1_exploratory | 5 | 3 | 9 | 8 | 6 | exploratory_cot | 9 | none | 100% | 100% | 4.92 |
| mot [7k–14k] | r1_exploratory | 4 | 3 | 8 | 8 | 7 | exploratory_cot | 9 | none | 100% | 100% | 5.00 |
| mot [14k+] | r1_exploratory | 6 | 4 | 9 | 9 | 8 | exploratory_cot | 9 | none | 100% | 100% | 4.76 |
| numinamath | structured | 5 | 3 | 1 | 8 | 5 | step_by_step | 9 | none | 100% | 100% | 4.04 |
| numinamath15 | structured | 3 | 9 | 2 | 9 | 9 | proof_style | 9 | none | 100% | 100% | 3.80 |
| openmathinstruct2 | semi-structured | 7 | 3 | 1 | 8 | 6 | step_by_step | 7 | none | 100% | 12% | 3.48 |
| openthoughts3 | r1_exploratory | 3 | 2 | 8 | 3 | 6 | exploratory_cot | 7 | **heavy** | 0% | — | — |
| stratos | r1_exploratory | 6 | 2 | 8 | 8 | 7 | exploratory_cot | 9 | none | 100% | 100% | 4.96 |

*Columns: stp=step_by_step_score, prf=proof_style_score, exp=exploratory_score, ltx=latex_density, diff=difficulty_score (1–10), Q=overall_quality. All scores 1–10.*

**Key findings:**
- **Only MoT and stratos have high exploratory scores (8–9)** — the signal we need for AIME. Every other dataset is exp ≤ 2.
- **numinamath15 is a unique signal** — proof_style=9, olympiad difficulty, but exp=2. Rigorous formal proofs, not search-based reasoning. Potentially complementary.
- **openmathinstruct2 is inconsistent at scale** — 12% style match in phase 2 despite OK phase 1. Avoid or sample carefully.
- **mathinstruct has 40% coding contamination** — confirmed trash for math SFT.
- **openthoughts3 confirmed**: 100% coding in is_math check (is_math=0%), exploratory style but wrong domain entirely.
- **stratos quality=9, 100% consistent, exp=8, competition difficulty** — confirmed as high-quality R1 alternative to MoT. Smaller (17K vs 93K) but may be worth mixing in.
- **MoT 14k+ is the hardest bucket** — difficulty=8 (AMC/AIME range), exploratory=9. The long traces are the primary AIME-targeting signal.
- **acemath step_by_step=9** — the most rigidly structured dataset; exp=1. Complete style opposite to MoT. This is why acemath → MoT training causes catastrophic forgetting: it's not just difficulty mismatch, it's a full output-style reset.

**Style gap between acemath and MoT (confirmed by output inspection):**
- acemath: "**Step 1: Calculate...**" → bold headers, numbered steps, answers the question before writing
- MoT: "Okay, let me think about this... hmm, wait that's wrong... let me try again..." → simulates live reasoning, backtracks, explores dead ends
- Training acemath → MoT phase 1 forces the model to completely change *how it talks* (not just what it knows), which overwrites its existing capabilities. The base → MoT chain avoids this because the base model never had a rigid output style to forget.

### Wave 1: Data source ablation
**Goal**: Which data source helps at each difficulty tier? Fixed size ~10K, 3 epochs.
Every experiment evals on SVAMP (easy), GSM8K (medium), MATH (hard), AIME 2025 (hardest).

**Easy/medium sources** (grade-school level):
| Exp ID | Data | Size | Why |
|--------|------|------|-----|
| `sft-gsm8k` | gsm8k | 7.5K | Clean baseline, all available data |
| `sft-metamath-10k` | metamath | 10K | Augmented GSM8K — does rephrasing help? |

**Hard sources** (competition level):
| Exp ID | Data | Size | Why |
|--------|------|------|-----|
| `sft-math` | math | 7.5K | Pure competition math |
| `sft-numinamath-10k` | numinamath15 | 10K | Broad competition math with better decontam |
| `sft-dartmath-10k` | dartmath | 10K | Difficulty-aware sampling (NeurIPS'24) |

**R1-style reasoning traces** (long CoT / thinking):
| Exp ID | Data | Size | Why |
|--------|------|------|-----|
| `sft-openthoughts-10k` | openthoughts3 | 10K | R1-distilled, long reasoning chains |
| `sft-stratos` | stratos | 10K | High-quality R1 distillation |
| `sft-acemath-10k` | acemath | 10K | Two-stage curriculum data |

**Hybrid**:
| Exp ID | Data | Size | Why |
|--------|------|------|-----|
| `sft-mathinstruct-10k` | mathinstruct | 10K | CoT + program-of-thought hybrid |
| `sft-openmathinstruct-10k` | openmathinstruct2 | 10K | Large-scale Nemotron-generated |

### Wave 2: Data scaling + saturation curves
**Goal**: How much data do we actually need? When do returns diminish?
Run the top 2-3 sources from Wave 1 at multiple sizes. Save checkpoints to plot loss vs accuracy.

| Exp ID | Data | Size | Epochs | Notes |
|--------|------|------|--------|-------|
| `sft-{best}-1k` | winner | 1K | 3 | Minimal data |
| `sft-{best}-1k-10ep` | winner | 1K | 10 | Test multi-epoch hypothesis |
| `sft-{best}-5k` | winner | 5K | 3 | |
| `sft-{best}-10k` | winner | 10K | 3 | |
| `sft-{best}-10k-10ep` | winner | 10K | 10 | Multi-epoch on quality data |
| `sft-{best}-30k` | winner | 30K | 3 | |
| `sft-{best}-100k` | winner | 100K | 1 | Single epoch, more data |

### Wave 3: Data mixtures
**Goal**: Do blends beat single sources? Strategy: blend easy, blend medium, keep hard.
Based on Wave 1 + 2 findings — fill in best sources per tier.

| Exp ID | Mix | Size | Ratio | Logic |
|--------|-----|------|-------|-------|
| `sft-mix-easy-hard` | gsm8k + {hard_winner} | 15K | 50/50 | Easy foundation + hard problems |
| `sft-mix-r1-comp` | {r1_winner} + {comp_winner} | 15K | 50/50 | R1 reasoning + competition math |
| `sft-mix-triple` | gsm8k + {comp_winner} + {r1_winner} | 15K | 33/33/33 | Balanced |
| `sft-mix-heavy-hard` | gsm8k + {hard_winner} | 15K | 20/80 | DART-Math insight: oversample hard |
| `sft-curriculum-easy-hard` | gsm8k then {hard_winner} | 15K | staged | Train easy first, then hard |

### Wave 4: Training recipe ablation (best data from Wave 3)
**Goal**: Does format, epochs, rank, or seq length matter once data is fixed?

| Exp ID | Variable | Value |
|--------|----------|-------|
| `sft-{best}-fewshot` | prompt_format | few_shot (vs chat_think) |
| `sft-{best}-1ep` | epochs | 1 |
| `sft-{best}-5ep` | epochs | 5 |
| `sft-{best}-seq4096` | max_seq_len | 4096 (for long R1 traces) |
| `sft-{best}-rank32` | lora_rank | 32 |
| `sft-{best}-rank8` | lora_rank | 8 |

### Wave 5: GRPO/RL (after best SFT recipe found)
- Use best SFT model as starting point for GRPO
- Outcome reward: correct answer = +1, wrong = 0
- This is where we expect the biggest gains on MATH/AIME

## Training fixes (v2 experiments)

### What changed in v2 (2026-03-20)
Experiments with `-v2` suffix were retrained with three fixes applied to `scripts/train/sft_lora.py`:

1. **Drop-not-truncate**: sequences longer than `max_seq_len` are dropped instead of truncated.
   Truncated solutions teach broken reasoning (answer cut off mid-computation). Old code silently kept
   them; new code returns `None` and skips them.

2. **Token-budget batching** (`--max-tokens-per-batch 8192`): instead of a fixed batch size,
   batches are filled greedily to a token budget. Short sequences → large batches (~14 for acemath);
   long sequences → small batches (~3). Memory stays bounded, GPU utilization improves.

3. **More training data**: the old drop logic (truncate then filter `n_loss_tokens < 5`) silently
   discarded many samples that were truncated to near-zero assistant content. With drop-not-truncate,
   2-3% are dropped (too long) instead of ~50% (truncated + filtered).

**Example**: `sft-acemath-10k` trained on 4695/10000 samples. `sft-acemath-10k-v2` trains on 9696/10000.

### Affected experiments
| Old ID | New ID | Fix |
|--------|--------|-----|
| `sft-acemath-10k` | `sft-acemath-10k-v2` | all 3 fixes above |

## Re-eval with EOS fix (TODO)

### Bug found (2026-03-20)
All existing eval results are unreliable — `run_hf.py` was passing `eos_token_id=<|endoftext|>` (151643)
but Qwen3 chat template ends turns with `<|im_end|>` (151645). Generation never stopped at the right
token, causing repetition loops that inflated `max_tokens` usage and hurt extraction rates.

**Fix applied**: `_eos_token_ids()` helper in `run_hf.py` now passes both IDs to `model.generate()`.
**Sweep function**: `run_eval_sweep` in `modal_jobs/train.py` — loads base model once, swaps adapters
in a loop on a single L4 GPU. Results saved as `eval_<id>_v2.json` to avoid overwriting old results.

### To re-run:
```bash
# All sft-* adapters in one job (~2.5h, ~$1.35 on L4)
uv run modal run --detach modal_jobs/train.py::run_eval_sweep

# Specific subset (comma-separated)
uv run modal run modal_jobs/train.py::run_eval_sweep \
  --experiments 'sft-dartmath-50k,sft-openmathinstruct-10k,sft-openmathinstruct-50k'

# Just acemath (already correct — EOS fix was applied before this run)
uv run modal run modal_jobs/train.py::run_eval_sweep --experiments 'sft-acemath-10k'
```

**Note**: `sft-acemath-10k` was evaluated after the EOS fix — results are reliable, re-eval optional.
All other adapters (dartmath, metamath, openmathinstruct, numinamath, stratos, mathinstruct) need re-eval.

### What we expect to change
- 50k models should improve significantly — their loops were truncating before `\boxed{}` on some problems
- Extraction rates should go up across the board (fewer incomplete generations)
- `sft-openmathinstruct-50k` (only 2 epochs, but correct math) may close the gap with the 10k model
- Relative rankings may shift — re-run `scripts/modal_status.sh` after to see updated leaderboard

## Agent workflow
1. Read `/results/experiment_registry.jsonl` to see completed experiments
2. Read this file for the experiment plan
3. Pick next un-run experiment from the current wave
4. Launch: `uv run modal run --detach modal_jobs/train.py::run_sft_lora ...`
5. Results auto-logged to registry on completion — includes all 4 eval tiers
6. After a wave completes: analyze findings, fill in `{best}` placeholders, update this file

## Architecture decisions
- **Qwen3-0.6B-Base**: best sub-1B base model, 36T pretraining tokens, Apache 2.0
- **LoRA (not full FT)**: faster iteration, lower memory, easy adapter comparison
- **chat_think prompt format**: Qwen3 thinking mode generalizes better across difficulty levels
- **transformers + peft (not TRL)**: simple SFT loop, TRL adds value for GRPO later
- **Fixed hyperparams**: focus on data recipe first, tune training later
- **4-tier eval**: SVAMP → GSM8K → MATH → AIME 2025 (uncontaminated, post-cutoff)
- **Modal A100**: bf16 support, good price/performance
- **Auto-eval + registry**: every experiment gets all 4 tiers, logged automatically
- **Batched eval on GPU**: batch_size=8 on Modal, sequential locally (MPS safe)
