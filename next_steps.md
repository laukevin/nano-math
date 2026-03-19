# Next Steps — math-nano (Seoul)

## Where we are (as of 2026-03-19)

### Phase 1: nanochat from-scratch (complete)
- **Model**: depth-8 (125M params), pretrained on FineWeb-edu (~524M tokens)
- **Best result**: SVAMP 6%, GSM8K 0-2% — ceiling hit at this model size
- **Lesson**: sub-200M models trained from scratch can't chain multi-step math reasoning

### Phase 2: Qwen3-0.6B + LoRA SFT (E2E pipeline working)
- **Base model**: Qwen/Qwen3-0.6B-Base (606M params, 36T tokens pretraining)
- **Method**: LoRA SFT (rank 16, 1.66% trainable params = 10M params)
- **Pipeline**: data prep → SFT → eval → registry, all automated on Modal
- **Smoke test passed**: 100 GSM8K samples, 1 epoch, loss 0.66→0.56, 12s on A100

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
- `scripts/eval/run_hf.py` — Eval HF models on GSM8K/SVAMP/MATH
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

## Immediate TODO

### Step 1: Run baseline eval (no SFT)
- [ ] Eval Qwen3-0.6B-Base on GSM8K, SVAMP, MATH to get baseline numbers
- [ ] This tells us where the base model starts before any SFT

### Step 2: First wave — data source comparison (fixed size 10K)
| Exp ID | Data | Size | Notes |
|--------|------|------|-------|
| `sft-gsm8k` | gsm8k | 7.5K | Smallest, simplest, all data |
| `sft-stratos` | stratos | 10K | High-quality R1 distillation |
| `sft-metamath-10k` | metamath | 10K | Augmented data |
| `sft-numinamath-10k` | numinamath | 10K | Competition math |
| `sft-openmathinstruct-10k` | openmathinstruct2 | 10K | Open-source solutions |
| `sft-openthoughts-10k` | openthoughts3 | 10K | R1-style long CoT |
| `sft-math` | math | 7.5K | Competition math, all data |

### Step 3: Second wave (based on first wave findings)
- **Size scaling**: top dataset at 1K, 5K, 10K, 30K, 100K, full
- **Mixing**: combine top-2 datasets at various ratios
- **Curriculum**: easy-to-hard vs random ordering

### Step 4: GRPO/RL (after finding best SFT recipe)
- Use SFT model as starting point for GRPO
- Outcome reward: correct answer = +1, wrong = 0
- This is where we expect the biggest gains

## Agent workflow
1. Read `/results/experiment_registry.jsonl` to see completed experiments
2. Read this file for the experiment plan
3. Pick next un-run experiment from the matrix
4. Launch: `uv run modal run --detach modal_jobs/train.py::run_sft_lora ...`
5. Results auto-logged to registry on completion
6. Analyze findings, propose next experiments, update this file

## Architecture decisions
- **Qwen3-0.6B-Base**: best sub-1B base model, 36T pretraining tokens, Apache 2.0
- **LoRA (not full FT)**: faster iteration (12s for 100 samples), lower memory, easy adapter comparison
- **Plain text tokenization**: Qwen3's thinking-mode chat template breaks loss masking, plain text is reliable
- **transformers + peft (not TRL)**: simple SFT loop, TRL adds value for GRPO later
- **Fixed hyperparams**: focus on data recipe first, tune training later
- **Modal A100**: bf16 support, good price/performance
- **Auto-eval + registry**: every experiment gets GSM8K + SVAMP eval, logged automatically
