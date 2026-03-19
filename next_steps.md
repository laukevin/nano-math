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
| `openthoughts3` | `open-thoughts/OpenThoughts-114k` | 114K | R1-style long CoT | DeepSeek-R1 distilled thinking traces |
| `stratos` | `bespokelabs/Bespoke-Stratos-17k` | 17K | R1-style long CoT | High-quality R1 distillation |
| `dartmath` | `hkust-nlp/dart-math-hard` | 585K | CoT | Difficulty-aware rejection sampling (NeurIPS'24) |
| `mathinstruct` | `TIGER-Lab/MathInstruct` | 260K | CoT + PoT | Hybrid natural language + Python (ICLR'24) |
| `acemath` | `nvidia/AceMath-Instruct-Training-Data` | varies | CoT | Two-stage SFT data (ACL'25) |

### Fixed hyperparams (don't sweep these yet)
- LoRA: rank 16, alpha 32, dropout 0.05
- LR: 2e-5, cosine schedule, warmup 0.03
- Batch size: 8, weight decay: 0.01
- Max seq len: 2048
- Eval: 100 problems per tier (30 for AIME), 1024 max tokens, batch=8 on GPU

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

### Wave 0: Baseline (in progress)
Confirm pipeline works E2E and get reference numbers on all 4 tiers.

| Exp ID | Data | Size | Status |
|--------|------|------|--------|
| `sft-gsm8k-full-v1` | gsm8k | 7473 | running on Modal |
| `base-no-sft` | (none) | — | TODO: eval base model with no SFT |

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
