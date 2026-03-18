"""Model loading and batch generation."""

from __future__ import annotations

import torch

from scripts.eval.data import EOS_TOKEN_ID, MAX_NEW_TOKENS


def resolve_device(device: str) -> str:
    """Resolve 'auto' device to actual device string."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint_path: str, depth: int, device: str) -> tuple:
    """Load model and tokenizer from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        depth: Model depth (transformer layers). Reserved for config override.
        device: Device string ('auto', 'cpu', 'cuda', 'mps').

    Returns:
        (model, tokenizer, device_str, model_params_count)
    """
    _ = depth  # reserved for future config override
    device = resolve_device(device)

    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    try:
        from nanochat.model import GPT, GPTConfig
    except ImportError:
        raise ImportError(
            "nanochat is required for model loading. "
            "Install with: pip install nanochat"
        )

    config = checkpoint.get("config", {})
    if isinstance(config, dict):
        model_config = GPTConfig(**config)
    else:
        model_config = config

    model = GPT(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    return model, tokenizer, device, n_params


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.0,
    top_p: float = 1.0,
    batch_size: int = 32,
    device: str = "cpu",
) -> tuple[list[str], dict]:
    """Generate completions for a batch of prompts.

    Returns:
        (completions, stats) where stats has keys:
            total_tokens, truncated, repetition_stopped
    """
    all_outputs: list[str] = []
    total_tokens = 0
    truncated = 0
    repetition_stopped = 0

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]

        input_ids_list = [tokenizer.encode(p) for p in batch]
        max_input_len = max(len(ids) for ids in input_ids_list)

        # Left-pad with EOS token
        padded = []
        for ids in input_ids_list:
            pad_len = max_input_len - len(ids)
            padded.append([EOS_TOKEN_ID] * pad_len + ids)

        generated = torch.tensor(padded, device=device)

        for _step in range(max_new_tokens):
            logits = model(generated)
            if isinstance(logits, tuple):
                logits = logits[0]
            next_logits = logits[:, -1, :]

            if temperature == 0.0:
                next_tokens = next_logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = next_logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(
                        scaled, descending=True
                    )
                    cum_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    mask = (
                        cum_probs - torch.softmax(sorted_logits, dim=-1)
                    ) >= top_p
                    sorted_logits[mask] = float("-inf")
                    scaled.scatter_(1, sorted_idx, sorted_logits)
                probs = torch.softmax(scaled, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_tokens], dim=-1)

            # Stop early if all sequences hit EOS
            if (next_tokens.squeeze(-1) == EOS_TOKEN_ID).all():
                break

        # Decode outputs
        for j in range(len(batch)):
            input_len = max_input_len
            output_ids = generated[j, input_len:].tolist()

            # Truncate at EOS
            hit_eos = False
            if EOS_TOKEN_ID in output_ids:
                output_ids = output_ids[: output_ids.index(EOS_TOKEN_ID)]
                hit_eos = True

            hit_rep = _detect_repetition(output_ids, window=20, repeats=3)
            if hit_rep:
                repetition_stopped += 1

            if not hit_eos and len(output_ids) >= max_new_tokens:
                truncated += 1

            total_tokens += len(output_ids)
            all_outputs.append(tokenizer.decode(output_ids))

    stats = {
        "total_tokens": total_tokens,
        "truncated": truncated,
        "repetition_stopped": repetition_stopped,
    }
    return all_outputs, stats


def _detect_repetition(
    token_ids: list[int], window: int = 20, repeats: int = 3
) -> bool:
    """Check if the same window-length sequence appears repeats times consecutively."""
    if len(token_ids) < window * repeats:
        return False
    for i in range(len(token_ids) - window * repeats + 1):
        pattern = token_ids[i : i + window]
        found = True
        for r in range(1, repeats):
            start = i + window * r
            if token_ids[start : start + window] != pattern:
                found = False
                break
        if found:
            return True
    return False
