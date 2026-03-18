# Spec 02: Pretraining Data

## Goal

Study how data mixture composition affects math capability at small scale.
The key question: does a small model benefit more from (a) general web data,
(b) pure math data, or (c) a tuned mixture?

## Data Sources

### Primary: General Web
- **FineWeb-Edu** (HuggingFace: HuggingFaceFW/fineweb-edu)
  - Already supported by nanochat
  - High-quality educational web text
  - Serves as the "general knowledge" component

### Math-Specific Sources

| Dataset | HuggingFace ID | Size | Content Type |
|---------|---------------|------|--------------|
| OpenMathReasoning | nvidia/OpenMathReasoning | ~30B tokens | Math reasoning traces, solutions |
| Proof-Pile-2 | EleutherAI/proof-pile-2 | ~55B tokens | Textbooks, papers, code (math-heavy) |
| OpenWebMath | open-web-math/open-web-math | ~14B tokens | Math web pages, filtered |
| MathPile | GAIR/MathPile | ~9B tokens | Curated math corpus (textbooks, wiki, etc.) |
| AutoMathText | Yuxiang-Wu/AutoMathText | ~200B tokens | Web text scored for math quality |

### Selection for Experiments

We don't use all of these. Pick **two** math sources to keep the mixture
space manageable:

1. **OpenWebMath** — broad math web content, good diversity
2. **OpenMathReasoning** — high-quality reasoning chains (closer to SFT data)

Rationale: OpenWebMath gives breadth (exposure to notation, concepts).
OpenMathReasoning gives depth (step-by-step reasoning patterns).
The contrast between them is interesting for studying what matters at small scale.

## Data Mixtures (Pretrain Experiments)

| Mixture ID | FineWeb-Edu | OpenWebMath | OpenMathReasoning | Hypothesis |
|------------|-------------|-------------|-------------------|------------|
| `mix-general` | 100% | 0% | 0% | Baseline: general web only |
| `mix-math-broad` | 50% | 40% | 10% | Balanced with math breadth |
| `mix-math-heavy` | 20% | 50% | 30% | Math-dominant |
| `mix-math-pure` | 0% | 60% | 40% | No general web at all |
| `mix-reasoning` | 30% | 20% | 50% | Reasoning-trace heavy |

**Note:** Percentages are by token count within each training batch.
Implementation: interleave shards from each source using nanochat's
dataloader with a weighted sampler.

## Tokenization Pipeline

All data must be tokenized into nanochat's shard format:

```
scripts/data/download_and_tokenize.py \
  --source openwebmath \
  --output data/openwebmath/ \
  --shard-size 100M  # tokens per shard, matching nanochat default
```

Output format per shard:
- `.bin` file: uint16 token IDs (nanochat uses GPT-2 BPE, max token < 50257)
- `.idx` file: document boundary offsets (for proper document masking)

**Validation:** After tokenization, load shards with nanochat's dataloader
and verify:
1. Token distribution looks reasonable (no pathological repeats)
2. Detokenized samples are readable
3. Shard boundaries don't split documents mid-sentence

## Data Mixture Implementation

Two approaches, pick one based on nanochat's dataloader flexibility:

### Option A: Pre-mixed shards
- Create mixed shards offline: sample from each source at mixture ratios
- Pro: Simple, no dataloader changes
- Con: Can't change mixture without re-sharding

### Option B: Multi-source dataloader
- Modify dataloader to accept multiple shard directories + weights
- Pro: Flexible, can sweep mixtures without re-tokenizing
- Con: Requires dataloader changes

**Recommendation:** Option B is worth the engineering cost. It enables
mixture sweeps without re-processing terabytes of data.

Dataloader config format:
```yaml
data_sources:
  - path: data/fineweb-edu/
    weight: 0.5
  - path: data/openwebmath/
    weight: 0.4
  - path: data/openmathreasoning/
    weight: 0.1
```

## Token Budget

Nanochat uses a compute-optimal D:N ratio by default. We override this
with `--token-multiplier` because:
- Chinchilla-optimal is for large models with expensive compute
- Small models benefit from overtraining (see: LLaMA, Phi, TinyLlama)
- Our multiplier: **50x** compute-optimal as default
- We may also experiment with 20x and 100x

For a 50M model, compute-optimal ~= 1B tokens.
At 50x multiplier: ~50B tokens seen during pretraining.

| Model | Compute-Optimal | 20x | 50x | 100x |
|-------|----------------|-----|-----|------|
| XS (50M) | ~1B | 20B | 50B | 100B |
| S (85M) | ~1.7B | 34B | 85B | 170B |
| M (130M) | ~2.6B | 52B | 130B | 260B |
| L (200M) | ~4B | 80B | 200B | 400B |
| XL (320M) | ~6.4B | 128B | 320B | 640B |

**Cost implications:** At 50x, the XL model sees 320B tokens. On a single
H100 at ~500K tok/s throughput (nanochat small models), that's ~178 hours
= ~7.4 days. This is our upper bound per run. Budget accordingly.

## Held-Out Validation Set

From each math source, hold out 10M tokens for validation.
val_bpb is computed on this held-out set (not on FineWeb val).
This gives us a math-specific loss signal separate from general perplexity.

## Data Integrity Checks

Before training begins:
- [ ] Total token count per source matches expected
- [ ] Shard format passes nanochat's built-in validation
- [ ] Detokenization spot check: 10 random samples per source look correct
- [ ] No data leakage: eval set problems (GSM8K, MATH500) are NOT in training data
  - Run exact-match deduplication of eval problems against training corpus
  - Log dedup stats to W&B
