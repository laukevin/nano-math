"""LoRA SFT training on a HuggingFace model.

Usage:
    python scripts/train/sft_lora.py \
        --base-model Qwen/Qwen3-0.6B-Base \
        --data /data/sft/gsm8k/train.jsonl \
        --output-dir /checkpoints/sft-001

Data format: JSONL with {"problem": "...", "solution": "..."} per line.
Also accepts {"messages": [...]} chat format from prepare_sft.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

FEW_SHOT_PREFIX = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is \\boxed{6}.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is \\boxed{5}.

Q: """


def load_data(path: str, max_samples: int = -1) -> list[dict]:
    """Load JSONL data. Supports {problem, solution} and {messages} formats."""
    samples = []
    with open(path) as f:
        for line in f:
            if max_samples > 0 and len(samples) >= max_samples:
                break
            row = json.loads(line)
            if "messages" in row:
                problem = solution = ""
                for msg in row["messages"]:
                    if msg["role"] == "user":
                        problem = msg["content"]
                    elif msg["role"] == "assistant":
                        solution = msg["content"]
                if problem and solution:
                    samples.append({"problem": problem, "solution": solution})
            elif "problem" in row and "solution" in row:
                samples.append({"problem": row["problem"], "solution": row["solution"]})
    return samples



def tokenize_chat_think(
    sample: dict, tokenizer, max_seq_len: int
) -> dict | None:
    """Tokenize using Qwen3 chat template with thinking mode.

    Format: user asks problem → assistant produces <think>reasoning</think>\\boxed{answer}
    Loss is only on the assistant's response (after the generation prompt).

    Note: apply_chat_template(tokenize=True) returns a BatchEncoding dict,
    so we access .input_ids to get the token list.
    """
    problem = (
        "Solve the following math problem step by step. "
        "Put your final answer in \\boxed{}.\n\n" + sample["problem"]
    )
    solution = sample["solution"]

    msgs_full = [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": solution},
    ]
    msgs_prefix = [{"role": "user", "content": problem}]

    # Full conversation (thinking=True wraps solution in empty <think></think>)
    full_enc = tokenizer.apply_chat_template(
        msgs_full, tokenize=True, return_dict=True, enable_thinking=True
    )
    full_ids = full_enc["input_ids"]

    # Prefix = user message + generation prompt (assistant\n)
    prefix_enc = tokenizer.apply_chat_template(
        msgs_prefix, tokenize=True, return_dict=True,
        add_generation_prompt=True, enable_thinking=True,
    )
    prefix_ids = prefix_enc["input_ids"]
    prefix_len = len(prefix_ids)

    # Truncate
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]

    prefix_len = min(prefix_len, len(full_ids))
    labels = [-100] * prefix_len + list(full_ids[prefix_len:])
    full_ids = list(full_ids)

    n_loss_tokens = sum(1 for l in labels if l != -100)
    if n_loss_tokens < 5:
        return None

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def tokenize_few_shot(
    sample: dict, tokenizer, max_seq_len: int
) -> dict | None:
    """Tokenize using few-shot Q&A format (plain text, no chat template)."""
    solution = sample["solution"]
    prompt = FEW_SHOT_PREFIX + sample["problem"] + "\nA: "
    full_text = prompt + solution + tokenizer.eos_token
    full_ids = tokenizer.encode(full_text, add_special_tokens=True)
    prefix_ids = tokenizer.encode(prompt, add_special_tokens=True)
    prefix_len = len(prefix_ids)

    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]

    prefix_len = min(prefix_len, len(full_ids))
    labels = [-100] * prefix_len + full_ids[prefix_len:]

    n_loss_tokens = sum(1 for l in labels if l != -100)
    if n_loss_tokens < 5:
        return None

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser(description="LoRA SFT training")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--data", type=str, required=True, help="JSONL data path")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--data-size", type=int, default=-1)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--prompt-format", type=str, default="chat_think",
        choices=["chat_think", "few_shot"],
        help="Prompt format: chat_think (Qwen3 thinking mode) or few_shot (plain text Q&A)",
    )
    args = parser.parse_args()

    start_time = time.time()

    # Load model + tokenizer
    print(f"Loading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Data
    tokenize_fn = tokenize_chat_think if args.prompt_format == "chat_think" else tokenize_few_shot
    print(f"Prompt format: {args.prompt_format}")
    print(f"Loading data from {args.data}...")
    raw = load_data(args.data, args.data_size)
    print(f"  Raw samples: {len(raw)}")

    tokenized = []
    skipped = 0
    for i, s in enumerate(raw):
        tok = tokenize_fn(s, tokenizer, args.max_seq_len)
        if tok is not None:
            tokenized.append(tok)
            if i < 3:
                n_loss = sum(1 for l in tok["labels"] if l != -100)
                print(f"  Sample {i}: {len(tok['input_ids'])} tokens, {n_loss} loss tokens")
        else:
            skipped += 1
            if skipped <= 3:
                print(f"  SKIPPED sample {i}: problem={len(s['problem'])} chars, solution={len(s['solution'])} chars")
    print(f"  Tokenized: {len(tokenized)}, skipped: {skipped}")

    if not tokenized:
        print("ERROR: No valid training samples!")
        return

    dataset = Dataset.from_list(tokenized)

    # Training
    os.makedirs(args.output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, pad_to_multiple_of=8
        ),
    )

    print(f"\nTraining: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    trainer.train()

    # Save adapter
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    elapsed = time.time() - start_time

    # Extract final loss
    final_loss = None
    for entry in reversed(trainer.state.log_history):
        if "train_loss" in entry:
            final_loss = entry["train_loss"]
            break

    # Save metadata
    meta = {
        "base_model": args.base_model,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "data_path": args.data,
        "data_size": len(tokenized),
        "epochs": args.epochs,
        "lr": args.lr,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "final_loss": final_loss,
        "wall_clock_s": elapsed,
        "log_history": trainer.state.log_history,
    }
    with open(os.path.join(args.output_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nDone in {elapsed:.0f}s. Loss: {final_loss}. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
