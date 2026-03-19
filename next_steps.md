# Next Steps — math-nano (Seoul)

## Where we are (as of 2026-03-19)

### Phase 1: nanochat from-scratch (complete)
- **Model**: depth-8 (125M params), pretrained on FineWeb-edu (~524M tokens)
- **Best result**: SVAMP 6%, GSM8K 0-2% — ceiling hit at this model size
- **Lesson**: sub-200M models trained from scratch can't chain multi-step math reasoning

### Phase 2: Qwen3-0.6B + LoRA SFT (pipeline working, baseline in progress)
- **Base model**: Qwen/Qwen3-0.6B-Base (606M params, 36T tokens pretraining)
- **Method**: LoRA SFT (rank 16, 1.66% trainable params = 10M params)
- **Pipeline**: data prep → SFT → eval → registry, all automated on Modal
- **Prompt format**: `chat_think` (Qwen3 thinking mode) — best for generalization
- **Local test**: 1000 GSM8K, 2 epochs → 25% GSM8K accuracy (20 problems, slow MPS eval)
- **Baseline run**: full GSM8K (7473), 3 epochs, A100 — in progress on Modal

## Infrastructure (all working)

### Running an experiment
```bash
# Single experiment
uv run modal run modal_jobs/train.py::run_sft_lora \
  --experiment-id sft-gsm8k \
  --data-source gsm8k \
  --data-size -1 \
  --epochs 3

# With --detach for long runs
uv run modal run --detach modal_jobs/train.py::run_sft_lora \
  --experiment-id sft-openthoughts-10k \
  --data-source openthoughts3 \
  --data-size 10000

# Check registry
uv run modal volume get math-nano-results experiment_registry.jsonl
```

### Key files
- `scripts/train/sft_lora.py` — LoRA SFT training (transformers + peft)
- `scripts/data/normalize_dataset.py` — Download + normalize HF datasets
- `scripts/eval/run_hf.py` — Eval HF models on GSM8K/SVAMP/MATH (batched on GPU)
- `scripts/registry.py` — Experiment registry (append, read, leaderboard)
- `modal_jobs/train.py::run_sft_lora()` — Orchestrates: data → train → eval → registry

### Available datasets
| ID | HF Path | Size | Notes |
|----|---------|------|-------|
| `gsm8k` | `openai/gsm8k` | 7.5K | Grade-school, clean |
| `metamath` | `meta-math/MetaMathQA` | 395K | Augmented GSM8K+MATH |
| `numinamath` | `AI-MO/NuminaMath-CoT` | 860K | Competition math |
| `math` | `lighteval/MATH` | 7.5K | Competition math (AMC/AIME) |
| `openmathinstruct2` | `nvidia/OpenMathInstruct-2` | 14M | Nemotron-generated |
| `openthoughts3` | `open-thoughts/OpenThoughts-114k` | 114K | R1-style long CoT |
| `stratos` | `bespokelabs/Bespoke-Stratos-17k` | 17K | High-quality R1 distillation |

### Fixed hyperparams (don't sweep these yet)
- LoRA: rank 16, alpha 32, dropout 0.05
- LR: 2e-5, cosine schedule, warmup 0.03
- Batch size: 8, weight decay: 0.01
- Max seq len: 2048, epochs: 3
- Eval: 100 problems per benchmark, 1024 max tokens, batch=8 on GPU

## Experiment plan

### Wave 0: Baseline (in progress)
Confirm pipeline works E2E and get reference numbers.

| Exp ID | Data | Size | Status |
|--------|------|------|--------|
| `sft-gsm8k-full-v1` | gsm8k | 7473 | running on Modal |
| `base-no-sft` | (none) | — | TODO: eval base model with no SFT |

### Wave 1: Data source ablation (fixed size ~10K, 3 epochs)
**Goal**: Which data source gives the best GSM8K/SVAMP accuracy?
Each experiment uses the same hyperparams, only the data changes.

| Exp ID | Data | Size | Notes |
|--------|------|------|-------|
| `sft-gsm8k` | gsm8k | 7.5K | All GSM8K data (baseline) |
| `sft-math` | math | 7.5K | Competition-level MATH |
| `sft-stratos-10k` | stratos | 10K | High-quality R1 distillation |
| `sft-openthoughts-10k` | openthoughts3 | 10K | R1-style long CoT |
| `sft-metamath-10k` | metamath | 10K | Augmented/rephrased |
| `sft-numinamath-10k` | numinamath | 10K | Competition math CoT |
| `sft-openmathinstruct-10k` | openmathinstruct2 | 10K | Nemotron-generated solutions |

### Wave 2: Scaling + mixing (based on Wave 1 winners)
**Goal**: Does more data help? Do mixtures beat single sources?

**Size scaling** (top-2 datasets from Wave 1):
| Exp ID | Data | Size |
|--------|------|------|
| `sft-{best}-1k` | winner | 1K |
| `sft-{best}-5k` | winner | 5K |
| `sft-{best}-10k` | winner | 10K |
| `sft-{best}-30k` | winner | 30K |
| `sft-{best}-100k` | winner | 100K |

**Mixtures** (combine top sources):
| Exp ID | Data | Size | Ratio |
|--------|------|------|-------|
| `sft-mix-gsm8k-{best}` | gsm8k + winner | 15K | 50/50 |
| `sft-mix-easy-hard` | gsm8k + math | 15K | 50/50 |
| `sft-mix-triple` | gsm8k + stratos + openthoughts | 15K | 33/33/33 |

### Wave 3: Training recipe ablation (best data from Wave 2)
**Goal**: Does prompt format, epochs, or seq length matter?

| Exp ID | Variable | Value |
|--------|----------|-------|
| `sft-{best}-fewshot` | prompt_format | few_shot (vs chat_think) |
| `sft-{best}-1ep` | epochs | 1 |
| `sft-{best}-5ep` | epochs | 5 |
| `sft-{best}-seq4096` | max_seq_len | 4096 |
| `sft-{best}-rank32` | lora_rank | 32 |
| `sft-{best}-rank8` | lora_rank | 8 |

### Wave 4: Hard benchmark push
**Goal**: Move accuracy on MATH and eventually AIME.
- Add MATH benchmark to eval (already supported)
- Try curriculum: train on easy (GSM8K) first, then hard (MATH/NuminaMath)
- Try filtering: only keep samples with correct `\boxed{}` answers
- Explore longer CoT (OpenThoughts has 2K+ token solutions)

### Wave 5: GRPO/RL (after best SFT recipe found)
- Use best SFT model as starting point for GRPO
- Outcome reward: correct answer = +1, wrong = 0
- This is where we expect the biggest gains on MATH/AIME

## Agent workflow
1. Read `/results/experiment_registry.jsonl` to see completed experiments
2. Read this file for the experiment plan
3. Pick next un-run experiment from the current wave
4. Launch: `uv run modal run --detach modal_jobs/train.py::run_sft_lora ...`
5. Results auto-logged to registry on completion
6. After a wave completes: analyze findings, fill in `{best}` placeholders, update this file

## Architecture decisions
- **Qwen3-0.6B-Base**: best sub-1B base model, 36T pretraining tokens, Apache 2.0
- **LoRA (not full FT)**: faster iteration, lower memory, easy adapter comparison
- **chat_think prompt format**: Qwen3 thinking mode generalizes better than few-shot across difficulty levels
- **transformers + peft (not TRL)**: simple SFT loop, TRL adds value for GRPO later
- **Fixed hyperparams**: focus on data recipe first, tune training later
- **Modal A100**: bf16 support, good price/performance
- **Auto-eval + registry**: every experiment gets GSM8K + SVAMP eval, logged automatically
- **Batched eval on GPU**: batch_size=8 on Modal, sequential locally (MPS safe)
