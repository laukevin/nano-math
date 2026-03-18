# Spec 09: Inference

## Use Cases

1. **Eval harness** — batch inference on eval sets (the primary use)
2. **Interactive testing** — manual spot-check of model outputs
3. **RL generation** — generating completions during GRPO training
4. **Demo / showcase** — showing model capabilities at end of project

## Batch Inference (Eval)

Used by the eval harness for all benchmark evaluations.

```python
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 1024,
    temperature: float = 0.0,  # greedy by default
    top_p: float = 1.0,
    batch_size: int = 32,
) -> list[str]:
    """Generate completions for a batch of prompts."""
    ...
```

### Prompt Formatting

For eval, format problems consistently:

```python
def format_eval_prompt(problem: str) -> str:
    return f"""Solve the following math problem step by step. Put your final answer in \\boxed{{}}.

Problem: {problem}

Solution:"""
```

### Performance Targets

These models are tiny. Inference should be fast:

| Model | Params | Expected tok/s (H100) | Expected tok/s (A100) | Expected tok/s (CPU) |
|-------|--------|----------------------|----------------------|---------------------|
| XS | 50M | ~10,000 | ~5,000 | ~500 |
| S | 85M | ~8,000 | ~4,000 | ~300 |
| M | 130M | ~6,000 | ~3,000 | ~200 |
| L | 200M | ~4,000 | ~2,000 | ~100 |
| XL | 320M | ~3,000 | ~1,500 | ~60 |

These are rough estimates. Actual throughput depends on sequence length,
batch size, and implementation. Log actual throughput in eval results.

### KV Cache

nanochat should support KV caching for autoregressive generation.
Verify this works correctly:
- Same output with and without KV cache
- Measurable speedup with KV cache

If nanochat doesn't support KV cache in generation, add it.
At these model sizes, it's a minor optimization, but it matters
for RL (generating 8 completions per prompt x thousands of prompts).

## Interactive Inference

For manual testing and debugging:

```bash
python scripts/inference/chat.py \
  --checkpoint results/sft-m-best/best_gsm8k.pt \
  --depth 16 \
  --device cpu  # or cuda
```

Interactive mode:
```
> What is 15% of 80?
To find 15% of 80:
15% = 15/100 = 0.15
0.15 × 80 = 12
The answer is \boxed{12}

> (Ctrl+D to exit)
```

### Sample Collection

During interactive testing, optionally save samples:
```bash
python scripts/inference/chat.py \
  --checkpoint $CKPT \
  --depth 16 \
  --save-samples results/samples/sft-m-best_interactive.jsonl
```

Saves each prompt + completion pair for manual review.

## RL Generation (GRPO)

During GRPO training, the model generates multiple completions per prompt.
This is the most performance-critical inference path.

Requirements:
- Generate 8 completions per prompt (group_size=8)
- Batch across prompts: 16 prompts * 8 completions = 128 generations per step
- Max new tokens: 1024
- Temperature: 0.7 (sampling, not greedy)

If using TRL's GRPOTrainer, generation is handled internally.
If implementing GRPO in nanochat, need efficient batched generation:

```python
def generate_groups(
    model,
    tokenizer,
    prompts: list[str],   # batch of prompts
    group_size: int = 8,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
) -> list[list[str]]:
    """Generate group_size completions for each prompt."""
    # Expand prompts: repeat each prompt group_size times
    expanded = [p for p in prompts for _ in range(group_size)]
    # Generate all at once
    outputs = generate_batch(model, tokenizer, expanded,
                             max_new_tokens=max_new_tokens,
                             temperature=temperature)
    # Reshape back to groups
    return [outputs[i*group_size:(i+1)*group_size] for i in range(len(prompts))]
```

## Model Export

For sharing or external evaluation, export models to standard formats:

### HuggingFace Format
```bash
python scripts/inference/export_hf.py \
  --checkpoint results/grpo-m-best/best_aime.pt \
  --depth 16 \
  --output models/hf/math-nano-130M/
```

Creates a standard HuggingFace model that works with `AutoModelForCausalLM.from_pretrained()`.

### GGUF (Optional, for local inference with llama.cpp)
```bash
python scripts/inference/export_gguf.py \
  --hf-model models/hf/math-nano-130M/ \
  --output models/gguf/math-nano-130M-Q8_0.gguf \
  --quantize Q8_0
```

Small models benefit little from quantization (already fast), but GGUF
enables easy local testing with llama.cpp / ollama.

## Generation Guardrails

During eval and RL, enforce these limits:
- **Max output tokens: 2048** — hard cap. If model hits this, truncate and attempt answer extraction
- **Repetition detection** — if model generates the same 20-token sequence 3 times in a row, stop generation early
- **EOS handling** — stop on EOS token (nanochat's tokenizer: token 50256)

Log metrics:
- `inference/truncated_pct` — % of generations hitting max length
- `inference/repetition_stopped_pct` — % stopped for repetition
- `inference/avg_output_tokens` — average completion length
