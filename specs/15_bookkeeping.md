# Spec 15: Data & Model Bookkeeping

## Problem

Without rigorous bookkeeping, we'll lose track of:
- Which data went into which model
- Which checkpoint came from which parent
- Whether two "comparable" results actually used the same data
- Whether a data change invalidated downstream results

This spec defines the **provenance chain**: data → pretrain → SFT → RL → eval.
Every link is recorded. Nothing is implicit.

---

## Data Registry

### `data/registry.json`

Single source of truth for all data assets:

```json
{
  "version": "1.0",
  "datasets": {
    "fineweb-edu-v1": {
      "type": "pretrain",
      "source": "HuggingFaceFW/fineweb-edu",
      "hf_revision": "main@2024-06-15",
      "download_date": "2026-03-18",
      "raw_path": "data/raw/fineweb-edu/",
      "tokenized_path": "data/tokenized/fineweb-edu/",
      "tokenizer": "gpt2-bpe-50257",
      "n_shards": 42,
      "total_tokens": 5_200_000_000,
      "shard_checksums_file": "data/tokenized/fineweb-edu/checksums.sha256",
      "content_hash": "sha256:abc123..."
    },
    "openwebmath-v1": {
      "type": "pretrain",
      "source": "open-web-math/open-web-math",
      "hf_revision": "main@2024-03-01",
      "download_date": "2026-03-18",
      "raw_path": "data/raw/openwebmath/",
      "tokenized_path": "data/tokenized/openwebmath/",
      "tokenizer": "gpt2-bpe-50257",
      "n_shards": 28,
      "total_tokens": 3_400_000_000,
      "shard_checksums_file": "data/tokenized/openwebmath/checksums.sha256",
      "content_hash": "sha256:def456..."
    },
    "sft-concise-cot-v1": {
      "type": "sft",
      "recipe": "sft-concise-cot",
      "sources": ["metamath@2024-01"],
      "n_samples": 100000,
      "path": "data/sft/concise-cot/",
      "format": "chat-jsonl",
      "content_hash": "sha256:ghi789...",
      "stats": {
        "avg_tokens": 412,
        "max_tokens": 2048,
        "truncation_rate": 0.03
      }
    }
  },
  "eval_sets": {
    "gsm8k-test-v1": {
      "source": "openai/gsm8k",
      "split": "test",
      "n": 1319,
      "path": "data/eval/gsm8k_test.jsonl",
      "content_hash": "sha256:..."
    }
  },
  "mixtures": {
    "mix-math-broad": {
      "components": {
        "fineweb-edu-v1": 0.50,
        "openwebmath-v1": 0.40,
        "openmathreasoning-v1": 0.10
      },
      "description": "Balanced general + math breadth"
    }
  }
}
```

### Data Versioning Rules

1. **Immutable once created.** Never modify a dataset after it's registered.
   If you need different data, create a new version (`openwebmath-v2`).
2. **Content hash is king.** Two datasets with same content hash are identical,
   regardless of name or path.
3. **Checksums per shard.** Each tokenized shard has a SHA256. If any shard
   changes, the dataset version must change.
4. **Registry is version-controlled.** `data/registry.json` is committed to git.

### Data Lineage Queries

```python
# "What data was used to train this model?"
def get_data_lineage(model_id: str) -> dict:
    model = model_registry[model_id]
    if model["stage"] == "pretrain":
        return {"pretrain_data": model["data_mixture"]}
    elif model["stage"] == "sft":
        parent = get_data_lineage(model["parent_model"])
        parent["sft_data"] = model["sft_recipe"]
        return parent
    elif model["stage"] == "grpo":
        parent = get_data_lineage(model["parent_model"])
        parent["rl_data"] = model["rl_datasets"]
        return parent
```

---

## Model Registry

### `results/model_registry.json`

Every model checkpoint is registered with its full lineage:

```json
{
  "version": "1.0",
  "models": {
    "pt-s-broad-final": {
      "experiment_id": "pt-s-broad",
      "stage": "pretrain",
      "depth": 12,
      "params": 85_000_000,
      "checkpoint_path": "/checkpoints/d12/pretrain/pt-s-broad/final.pt",
      "checkpoint_hash": "sha256:...",
      "parent_model": null,
      "data": {
        "mixture": "mix-math-broad",
        "data_versions": ["fineweb-edu-v1", "openwebmath-v1", "openmathreasoning-v1"],
        "token_multiplier": 50,
        "tokens_seen": 4_250_000_000
      },
      "hyperparams": {
        "lr_peak": 0.001,
        "batch_size": 64,
        "seq_len": 1024,
        "warmup_steps": 500,
        "total_steps": 50000
      },
      "training": {
        "wall_clock_hours": 4.2,
        "cost_usd": 14.70,
        "final_train_loss": 2.31,
        "final_val_bpb_math": 2.81,
        "gpu": "H100",
        "wandb_run_id": "abc123"
      },
      "eval_results": {
        "gsm8k_pass1_greedy": 0.02,
        "math500_pass1_greedy": 0.01
      },
      "created_at": "2026-03-20T14:00:00Z",
      "git_hash": "deadbeef"
    },
    "sft-m-concise-best": {
      "experiment_id": "sft-m-concise",
      "stage": "sft",
      "depth": 16,
      "params": 130_000_000,
      "checkpoint_path": "/checkpoints/d16/sft/sft-m-concise/best_gsm8k.pt",
      "checkpoint_hash": "sha256:...",
      "parent_model": "pt-m-broad-final",
      "data": {
        "recipe": "sft-concise-cot",
        "data_version": "sft-concise-cot-v1",
        "epochs": 3,
        "samples": 100000
      },
      "hyperparams": {
        "lr": 2e-5,
        "batch_size": 32,
        "max_seq_len": 2048
      },
      "training": {
        "wall_clock_hours": 1.5,
        "cost_usd": 5.25,
        "best_gsm8k_step": 2400,
        "gpu": "H100",
        "wandb_run_id": "def456"
      },
      "eval_results": {
        "gsm8k_pass1_greedy": 0.34,
        "gsm8k_pass1_sampled": 0.38,
        "gsm8k_pass4": 0.52,
        "gsm8k_pass8": 0.61,
        "math500_pass1_greedy": 0.11
      },
      "created_at": "2026-03-22T10:00:00Z",
      "git_hash": "cafebabe"
    }
  }
}
```

### Lineage Chain Visualization

```
fineweb-edu-v1 ─┐
openwebmath-v1 ─┼─→ mix-math-broad ─→ pt-m-broad-final ─→ sft-m-concise-best ─→ grpo-m-easy2hard-best
omr-v1 ─────────┘                          (pretrain)           (sft)                   (grpo)
                                         130M, 50x tokens    concise-cot recipe     easy→hard curriculum
                                         GSM8K: 2%           GSM8K: 34%             GSM8K: 42%
```

Generate this with:
```bash
uv run python scripts/registry/show_lineage.py --model grpo-m-easy2hard-best
```

### Model Comparison

```bash
# Compare any two models side by side
uv run python scripts/registry/compare.py \
  --model-a sft-m-concise-best \
  --model-b sft-m-distill-r1-best

# Output:
# ┌──────────────────┬────────────────────┬─────────────────────┐
# │                  │ sft-m-concise-best │ sft-m-distill-r1-best│
# ├──────────────────┼────────────────────┼─────────────────────┤
# │ Parent           │ pt-m-broad-final   │ pt-m-broad-final    │
# │ SFT Recipe       │ concise-cot        │ distill-r1          │
# │ SFT Samples      │ 100K               │ 100K                │
# │ GSM8K pass@1     │ 0.34               │ 0.28                │
# │ MATH500 pass@1   │ 0.11               │ 0.08                │
# │ Cost             │ $5.25              │ $5.25               │
# └──────────────────┴────────────────────┴─────────────────────┘
```

### Registration Happens Automatically

Models are registered by the training harness at checkpoint save time.
No manual registration. The harness writes to `model_registry.json`.

```python
def register_model(experiment_id, stage, checkpoint_path, parent_model, config, metrics):
    registry = load_registry()
    model_id = f"{experiment_id}-{checkpoint_type}"  # e.g., "sft-m-concise-best"
    registry["models"][model_id] = {
        "experiment_id": experiment_id,
        "stage": stage,
        "checkpoint_path": checkpoint_path,
        "checkpoint_hash": sha256_file(checkpoint_path),
        "parent_model": parent_model,
        "data": config.data_config,
        "hyperparams": config.hparams,
        "eval_results": metrics,
        "created_at": datetime.utcnow().isoformat(),
        "git_hash": get_git_hash(),
    }
    save_registry(registry)
```

---

## Invalidation Tracking

When upstream data or code changes, downstream results may be invalid.

### What Invalidates What

```
Data change (e.g., re-tokenize openwebmath)
  └→ Invalidates: ALL pretrain models using that data
      └→ Invalidates: ALL SFT models built on those pretrain models
          └→ Invalidates: ALL GRPO models built on those SFT models
              └→ Invalidates: ALL eval results for those models

Eval code change (e.g., fix answer extraction bug)
  └→ Invalidates: ALL eval results (but not models themselves)

Training code change (e.g., fix bug in loss computation)
  └→ Invalidates: ALL models trained after the bug was introduced
```

### Invalidation Detection

```python
def check_invalidations():
    """Check if any registered results are invalidated by changes."""
    registry = load_registry()
    data_registry = load_data_registry()

    warnings = []
    for model_id, model in registry["models"].items():
        # Check data still matches
        for data_version in model["data"].get("data_versions", []):
            current_hash = data_registry["datasets"][data_version]["content_hash"]
            if current_hash != model.get("data_hash_at_train"):
                warnings.append(f"Data {data_version} changed since {model_id} was trained!")

        # Check parent still exists and hasn't been retrained
        parent = model.get("parent_model")
        if parent and parent not in registry["models"]:
            warnings.append(f"Parent {parent} of {model_id} no longer exists!")

    return warnings
```

Run this as part of every gate check:
```bash
uv run python scripts/registry/check_invalidations.py
```
