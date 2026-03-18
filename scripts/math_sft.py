"""Simple math SFT: finetune a nanochat pretrained model on GSM8K math data.

Stripped-down SFT that avoids nanochat's complex data packing.
Each batch is simply: prompt + solution, padded to max_seq_len.
Loss is only computed on the solution tokens.

Usage (from vendor/nanochat dir):
    PYTHONPATH=../.. uv run python -m scripts.math_sft --model-tag d2 --num-steps 200
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
NANOCHAT_DIR = PROJECT_ROOT / "vendor" / "nanochat"
sys.path.insert(0, str(NANOCHAT_DIR))

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import autodetect_device_type
from nanochat.loss_eval import evaluate_bpb


def load_gsm8k_train(tokenizer, max_seq_len: int, n_samples: int = -1):
    """Load GSM8K train split, tokenize as prompt+solution pairs.

    Format: <|bos|><|user_start|>{question}<|user_end|><|assistant_start|>{solution}\n\nThe answer is \\boxed{answer}.<|assistant_end|>

    Returns list of (input_ids, loss_mask) where loss_mask=1 for assistant tokens.
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    if n_samples > 0:
        ds = ds.select(range(min(n_samples, len(ds))))

    bos_id = tokenizer.get_bos_token_id()
    samples = []
    skipped = 0

    for row in ds:
        question = row["question"]
        # Extract answer from GSM8K format (solution text #### answer)
        parts = row["answer"].split("####")
        solution = parts[0].strip()
        answer = parts[-1].strip()

        # Format as chat with \boxed{} answer
        user_msg = question
        assistant_msg = f"{solution}\n\nThe answer is \\boxed{{{answer}}}."

        # Tokenize each part
        # <|bos|><|user_start|>question<|user_end|><|assistant_start|>solution<|assistant_end|>
        user_start = tokenizer.encode_special("<|user_start|>")
        user_end = tokenizer.encode_special("<|user_end|>")
        asst_start = tokenizer.encode_special("<|assistant_start|>")
        asst_end = tokenizer.encode_special("<|assistant_end|>")

        user_ids = tokenizer.encode(user_msg)
        asst_ids = tokenizer.encode(assistant_msg)

        # Full sequence
        ids = [bos_id, user_start] + user_ids + [user_end, asst_start] + asst_ids + [asst_end]

        # Loss mask: 0 for everything up to and including <|assistant_start|>, 1 for solution + <|assistant_end|>
        prompt_len = 1 + 1 + len(user_ids) + 1 + 1  # bos + user_start + question + user_end + asst_start
        mask = [0] * prompt_len + [1] * (len(asst_ids) + 1)  # +1 for asst_end

        # Truncate to max_seq_len + 1 (for shifted targets)
        max_len = max_seq_len + 1
        if len(ids) > max_len:
            ids = ids[:max_len]
            mask = mask[:max_len]

        # Skip if too short (need at least some solution tokens)
        if sum(mask) < 3:
            skipped += 1
            continue

        samples.append((ids, mask))

    print(f"Loaded {len(samples)} GSM8K samples ({skipped} skipped as too long)")
    return samples


def make_batch(samples, batch_indices, max_seq_len, device):
    """Create a training batch from sample indices.

    Returns (inputs, targets) where targets=-1 for masked positions.
    """
    row_len = max_seq_len + 1
    bos_pad = 0  # pad with 0s (will be masked anyway)

    batch_ids = []
    batch_masks = []

    for idx in batch_indices:
        ids, mask = samples[idx]
        # Pad to row_len
        pad_len = row_len - len(ids)
        if pad_len > 0:
            ids = ids + [bos_pad] * pad_len
            mask = mask + [0] * pad_len
        batch_ids.append(ids[:row_len])
        batch_masks.append(mask[:row_len])

    batch_tensor = torch.tensor(batch_ids, dtype=torch.long, device=device)
    mask_tensor = torch.tensor(batch_masks, dtype=torch.int8, device=device)

    inputs = batch_tensor[:, :-1].to(dtype=torch.int32).contiguous()
    targets = batch_tensor[:, 1:].to(dtype=torch.int64).contiguous()

    # Apply loss mask: mask[1:] aligns with targets
    target_mask = mask_tensor[:, 1:]
    targets[target_mask == 0] = -1

    return inputs, targets


def main():
    parser = argparse.ArgumentParser(description="Simple math SFT on nanochat checkpoint")
    parser.add_argument("--model-tag", type=str, required=True, help="nanochat model tag (e.g. d2)")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step (default=latest)")
    parser.add_argument("--num-steps", type=int, default=500, help="number of training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--max-seq-len", type=int, default=512, help="max sequence length")
    parser.add_argument("--eval-every", type=int, default=50, help="eval every N steps")
    parser.add_argument("--save-every", type=int, default=100, help="save checkpoint every N steps")
    parser.add_argument("--n-samples", type=int, default=-1, help="limit training samples (-1=all)")
    parser.add_argument("--warmup-steps", type=int, default=20, help="LR warmup steps")
    args = parser.parse_args()

    device_type = autodetect_device_type()
    device = torch.device(device_type)
    print(f"Device: {device_type}")

    # Load pretrained model
    print(f"Loading pretrained model (tag={args.model_tag}, step={args.step})...")
    model, tokenizer, meta = load_model("base", device, phase="train",
                                         model_tag=args.model_tag, step=args.step)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")
    print(f"  Config: {meta.get('model_config', {})}")

    # Compile model
    model = torch.compile(model, dynamic=False)

    # Load data
    print(f"\nLoading GSM8K training data...")
    samples = load_gsm8k_train(tokenizer, args.max_seq_len, n_samples=args.n_samples)
    n_samples = len(samples)
    if n_samples == 0:
        print("ERROR: No training samples!")
        return

    # Simple AdamW optimizer (no Muon complexity)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    # Training loop
    print(f"\nStarting training: {args.num_steps} steps, batch_size={args.batch_size}, lr={args.lr}")
    print("-" * 70)

    step = 0
    epoch = 0
    sample_idx = 0
    best_loss = float("inf")

    while step < args.num_steps:
        epoch += 1
        # Shuffle at start of each epoch
        perm = torch.randperm(n_samples).tolist()
        sample_idx = 0

        while sample_idx + args.batch_size <= n_samples and step < args.num_steps:
            t0 = time.time()

            # LR warmup
            if step < args.warmup_steps:
                lr = args.lr * (step + 1) / args.warmup_steps
            else:
                # Cosine decay
                progress = (step - args.warmup_steps) / max(args.num_steps - args.warmup_steps, 1)
                lr = args.lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Make batch
            batch_idx = perm[sample_idx:sample_idx + args.batch_size]
            inputs, targets = make_batch(samples, batch_idx, args.max_seq_len, device)
            sample_idx += args.batch_size

            # Check for all-masked batch (skip to avoid NaN)
            if (targets == -1).all():
                continue

            # Forward + backward
            loss = model(inputs, targets)
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at step {step}, skipping")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            dt = time.time() - t0
            tokens = inputs.numel()
            tok_sec = tokens / dt

            step += 1

            if step % 10 == 0 or step == 1:
                print(f"step {step:05d}/{args.num_steps} | loss: {loss.item():.4f} | lr: {lr:.2e} | dt: {dt*1000:.0f}ms | tok/sec: {tok_sec:,.0f} | epoch: {epoch}")

            # Save checkpoint
            if step % args.save_every == 0 or step == args.num_steps:
                save_dir = Path.home() / ".cache" / "nanochat" / "mathsft_checkpoints" / args.model_tag
                save_dir.mkdir(parents=True, exist_ok=True)
                # Save model weights
                torch.save(model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict(),
                          save_dir / f"model_{step:06d}.pt")
                # Save metadata
                meta_out = {
                    "model_config": meta.get("model_config", {}),
                    "step": step,
                    "loss": loss.item(),
                    "lr": lr,
                    "sft_args": vars(args),
                }
                with open(save_dir / f"meta_{step:06d}.json", "w") as f:
                    json.dump(meta_out, f, indent=2)
                print(f"  Saved checkpoint at step {step} to {save_dir}")

    print(f"\nTraining complete. {step} steps, {epoch} epochs.")


if __name__ == "__main__":
    main()
