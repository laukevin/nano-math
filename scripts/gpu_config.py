"""GPU memory estimator and batch size tuner.

Estimates memory for training and eval, recommends batch sizes targeting
a given utilization. Uses empirical calibration from real A100 runs.

Qwen3-0.6B architecture:
    params=596M, hidden=1024, layers=28, heads=16, kv_heads=8,
    head_dim=128, intermediate=3072, vocab=151936

LoRA config (rank=16, 7 target modules per layer):
    trainable_params=9.18M
"""

from __future__ import annotations

import dataclasses


# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ModelSpec:
    name: str
    params: int           # total params
    hidden: int           # hidden dimension
    layers: int           # number of transformer layers
    heads: int            # attention heads
    kv_heads: int         # KV heads (GQA)
    head_dim: int         # per-head dimension
    intermediate: int     # MLP intermediate size
    vocab: int            # vocabulary size


QWEN3_06B = ModelSpec(
    name="Qwen3-0.6B",
    params=596_049_920,
    hidden=1024,
    layers=28,
    heads=16,
    kv_heads=8,
    head_dim=128,
    intermediate=3072,
    vocab=151936,
)

# GPU specs (memory in GB)
GPU_SPECS = {
    "T4": 16.0,
    "A10G": 24.0,
    "L4": 24.0,
    "A100-40GB": 40.0,
    "A100-80GB": 80.0,
    "H100": 80.0,
}


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

def estimate_lora_params(
    model: ModelSpec,
    rank: int = 16,
    n_target_modules: int = 7,
) -> int:
    """Estimate LoRA trainable parameters."""
    # Each target module: A (d_in × r) + B (r × d_out)
    # For simplicity, use hidden dim for attention and intermediate for MLP
    attn_per_layer = 4 * (model.hidden * rank + rank * model.hidden)  # q,k,v,o
    mlp_per_layer = 3 * (model.hidden * rank + rank * model.intermediate)  # gate,up: h->inter; down: inter->h
    # Correction: gate and up are h->inter, down is inter->h
    mlp_per_layer = 2 * (model.hidden * rank + rank * model.intermediate) + (model.intermediate * rank + rank * model.hidden)
    per_layer = attn_per_layer + mlp_per_layer
    return per_layer * model.layers


def estimate_training_memory_gb(
    model: ModelSpec = QWEN3_06B,
    batch_size: int = 8,
    seq_len: int = 2048,
    lora_rank: int = 16,
    gradient_checkpointing: bool = True,
    packing: bool = False,
    avg_seq_len: int = 0,
    bytes_per_param: int = 2,  # bf16
) -> dict:
    """Estimate GPU memory for LoRA training.

    Calibrated against real A100 observations:
    - batch=32, seq=2048, no packing (avg_seq~756): 37 GB observed
    - batch=32, seq=2048, packing (all seqs=2048): OOM at 39.5 GB (logits alone = 39.8 GB)
    - batch=8, eval: 3.6 GB observed

    Key insight: the DOMINANT memory cost is the logits tensor during loss
    computation: batch * effective_seq * vocab_size * 4 bytes (cast to float32).
    With vocab=151936, this is enormous and often exceeds activation memory.

    With packing, all sequences are padded to max_seq_len.
    Without packing, DataCollator pads to the longest in the batch (avg is shorter).
    """
    lora_params = estimate_lora_params(model, lora_rank)

    # Fixed costs (independent of batch size)
    model_weights = model.params * bytes_per_param / 1e9
    lora_weights = lora_params * bytes_per_param / 1e9
    lora_gradients = lora_params * bytes_per_param / 1e9
    # AdamW: momentum (fp32) + variance (fp32) = 8 bytes per trainable param
    optimizer_states = lora_params * 8 / 1e9
    framework_overhead = 1.0  # PyTorch allocator, CUDA context

    fixed = model_weights + lora_weights + lora_gradients + optimizer_states + framework_overhead

    # Effective sequence length for memory estimation
    # With packing: all sequences are max_seq_len
    # Without packing: collator pads to longest in batch, use avg * 1.3 as heuristic
    if packing:
        effective_seq = seq_len
    elif avg_seq_len > 0:
        effective_seq = min(int(avg_seq_len * 1.3), seq_len)
    else:
        # Default: assume ~40% of max_seq_len average (empirical from our datasets)
        effective_seq = int(seq_len * 0.4)

    # Logits tensor: batch * seq * vocab * 4 bytes (fp32 for loss computation)
    # This is the DOMINANT cost and the usual cause of OOM
    logits_gb = batch_size * effective_seq * model.vocab * 4 / 1e9

    # Activation memory (attention + MLP intermediates)
    # With gradient checkpointing, only ~sqrt(layers) worth stored at once
    # Empirical: ~0.3 GB per sample at seq=2048 (excluding logits)
    per_sample_activation = 0.3 * (effective_seq / 2048)
    if not gradient_checkpointing:
        per_sample_activation *= 2.0
    activations = batch_size * per_sample_activation

    total = fixed + logits_gb + activations

    return {
        "total_gb": total,
        "fixed_gb": fixed,
        "logits_gb": logits_gb,
        "activations_gb": activations,
        "effective_seq_len": effective_seq,
        "per_sample_gb": (logits_gb + activations) / batch_size if batch_size > 0 else 0,
        "breakdown": {
            "model_weights": model_weights,
            "lora_weights": lora_weights,
            "lora_gradients": lora_gradients,
            "optimizer_states": optimizer_states,
            "framework_overhead": framework_overhead,
            "logits": logits_gb,
            "activations": activations,
        },
        "lora_params": lora_params,
    }


def estimate_eval_memory_gb(
    model: ModelSpec = QWEN3_06B,
    batch_size: int = 32,
    seq_len: int = 1024,
    lora_rank: int = 16,
    bytes_per_param: int = 2,
) -> dict:
    """Estimate GPU memory for eval/inference (no gradients, no optimizer).

    KV cache is the main batch-dependent cost. Much cheaper than training.
    Calibrated: batch=8, seq=1024 → 3.6 GB observed on A100.
    """
    lora_params = estimate_lora_params(model, lora_rank)

    # Fixed costs
    model_weights = model.params * bytes_per_param / 1e9
    lora_weights = lora_params * bytes_per_param / 1e9
    framework_overhead = 0.8

    fixed = model_weights + lora_weights + framework_overhead

    # KV cache: batch * 2 (K+V) * layers * kv_heads * head_dim * seq * bytes
    kv_cache = (
        batch_size * 2 * model.layers * model.kv_heads * model.head_dim
        * seq_len * bytes_per_param / 1e9
    )

    # Attention computation buffer (temporary, for current batch)
    # batch * heads * seq * seq * bytes (but only for current generation step)
    # For autoregressive generation, this is batch * heads * seq * 1 * bytes per step
    # Much smaller than training since we only compute one position at a time
    attn_buffer = batch_size * model.heads * seq_len * bytes_per_param / 1e9

    activations = kv_cache + attn_buffer

    total = fixed + activations

    return {
        "total_gb": total,
        "fixed_gb": fixed,
        "activations_gb": activations,
        "per_sample_gb": activations / batch_size if batch_size > 0 else 0,
        "breakdown": {
            "model_weights": model_weights,
            "lora_weights": lora_weights,
            "framework_overhead": framework_overhead,
            "kv_cache": kv_cache,
            "attn_buffer": attn_buffer,
        },
    }


# ---------------------------------------------------------------------------
# Batch size recommendation
# ---------------------------------------------------------------------------

def recommend_batch_size(
    gpu: str = "A100-40GB",
    mode: str = "train",
    seq_len: int = 2048,
    model: ModelSpec = QWEN3_06B,
    lora_rank: int = 16,
    target_utilization: float = 0.80,
    gradient_checkpointing: bool = True,
    packing: bool = False,
    avg_seq_len: int = 0,
    memory_gb: float = 0.0,
) -> dict:
    """Recommend batch size for target GPU utilization.

    memory_gb: if > 0, use this instead of looking up gpu in GPU_SPECS.
               Useful for dynamic devices (MPS, unknown CUDA cards).

    Returns dict with recommended batch size, memory estimates, and safety margin.
    """
    gpu_mem = memory_gb if memory_gb > 0 else GPU_SPECS.get(gpu, 40.0)
    target_mem = gpu_mem * target_utilization

    # Binary search for max batch size within target
    lo, hi = 1, 512
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if mode == "train":
            est = estimate_training_memory_gb(
                model, mid, seq_len, lora_rank, gradient_checkpointing,
                packing=packing, avg_seq_len=avg_seq_len,
            )
        else:
            est = estimate_eval_memory_gb(model, mid, seq_len, lora_rank)

        if est["total_gb"] <= target_mem:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # Get final estimate at recommended batch size
    if mode == "train":
        final_est = estimate_training_memory_gb(
            model, best, seq_len, lora_rank, gradient_checkpointing,
            packing=packing, avg_seq_len=avg_seq_len,
        )
    else:
        final_est = estimate_eval_memory_gb(model, best, seq_len, lora_rank)

    return {
        "recommended_batch_size": best,
        "mode": mode,
        "gpu": gpu,
        "gpu_memory_gb": gpu_mem,
        "target_utilization": target_utilization,
        "estimated_memory_gb": final_est["total_gb"],
        "actual_utilization": final_est["total_gb"] / gpu_mem,
        "headroom_gb": gpu_mem - final_est["total_gb"],
        "detail": final_est,
    }


def print_recommendations(
    gpu: str = "A100-40GB",
    seq_len: int = 2048,
    lora_rank: int = 16,
):
    """Print batch size recommendations for both training and eval."""
    print(f"{'='*70}")
    print(f"GPU Memory Planner: Qwen3-0.6B + LoRA (rank={lora_rank})")
    print(f"GPU: {gpu} ({GPU_SPECS.get(gpu, '?')} GB)")
    print(f"Seq length: {seq_len}")
    print(f"{'='*70}")

    for pack_label, pack_val in [("no packing", False), ("with packing", True)]:
        print(f"\n  Training ({pack_label}):")
        for target in [0.60, 0.70, 0.80, 0.90]:
            rec = recommend_batch_size(
                gpu=gpu, mode="train", seq_len=seq_len,
                lora_rank=lora_rank, target_utilization=target,
                packing=pack_val,
            )
            eff_seq = rec["detail"].get("effective_seq_len", seq_len)
            print(
                f"    @ {target*100:.0f}% util: "
                f"batch={rec['recommended_batch_size']:>4d}, "
                f"est={rec['estimated_memory_gb']:.1f}GB, "
                f"eff_seq={eff_seq}, "
                f"headroom={rec['headroom_gb']:.1f}GB"
            )

    print(f"\n  Eval:")
    for target in [0.60, 0.70, 0.80, 0.90]:
        rec = recommend_batch_size(
            gpu=gpu, mode="eval", seq_len=seq_len,
            lora_rank=lora_rank, target_utilization=target,
        )
        print(
            f"    @ {target*100:.0f}% util: "
            f"batch={rec['recommended_batch_size']:>4d}, "
            f"est={rec['estimated_memory_gb']:.1f}GB, "
            f"headroom={rec['headroom_gb']:.1f}GB"
        )


if __name__ == "__main__":
    import sys
    gpu = sys.argv[1] if len(sys.argv) > 1 else "A100-40GB"
    seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    print_recommendations(gpu=gpu, seq_len=seq_len)
