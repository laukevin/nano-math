"""Convert between nanochat (.pt) and HuggingFace model formats.

Nanochat uses a custom GPT-2 style architecture. TRL/HuggingFace expects
a standard transformers model. This script bridges the two.

Usage:
    # nanochat → HF (for SFT/GRPO via TRL)
    python -m scripts.train.convert_to_hf \
        --checkpoint checkpoints/final.pt \
        --output hf_model/ \
        --depth 16 \
        --direction nanochat_to_hf

    # HF → nanochat (after TRL training, back to nanochat format)
    python -m scripts.train.convert_to_hf \
        --checkpoint hf_model/ \
        --output converted.pt \
        --depth 16 \
        --direction hf_to_nanochat
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Nanochat model configs by depth (matches spec 01)
DEPTH_CONFIGS = {
    10: {"n_layer": 10, "n_head": 8, "n_embd": 512, "vocab_size": 50257},
    12: {"n_layer": 12, "n_head": 12, "n_embd": 768, "vocab_size": 50257},
    16: {"n_layer": 16, "n_head": 12, "n_embd": 768, "vocab_size": 50257},
    20: {"n_layer": 20, "n_head": 16, "n_embd": 1024, "vocab_size": 50257},
    24: {"n_layer": 24, "n_head": 16, "n_embd": 1024, "vocab_size": 50257},
}

# Weight name mapping: nanochat key → HF GPT2LMHeadModel key
# nanochat follows nanoGPT conventions; HF uses transformer.h.{i}.* conventions
NANOCHAT_TO_HF_MAP = {
    "transformer.wte.weight": "transformer.wte.weight",
    "transformer.wpe.weight": "transformer.wpe.weight",
    "transformer.ln_f.weight": "transformer.ln_f.weight",
    "transformer.ln_f.bias": "transformer.ln_f.bias",
    "lm_head.weight": "lm_head.weight",
}


def _build_layer_map(n_layers: int) -> dict[str, str]:
    """Build per-layer weight name mapping."""
    mapping = dict(NANOCHAT_TO_HF_MAP)
    for i in range(n_layers):
        prefix = f"transformer.h.{i}"
        # Attention
        mapping[f"{prefix}.attn.c_attn.weight"] = f"{prefix}.attn.c_attn.weight"
        mapping[f"{prefix}.attn.c_attn.bias"] = f"{prefix}.attn.c_attn.bias"
        mapping[f"{prefix}.attn.c_proj.weight"] = f"{prefix}.attn.c_proj.weight"
        mapping[f"{prefix}.attn.c_proj.bias"] = f"{prefix}.attn.c_proj.bias"
        # MLP
        mapping[f"{prefix}.mlp.c_fc.weight"] = f"{prefix}.mlp.c_fc.weight"
        mapping[f"{prefix}.mlp.c_fc.bias"] = f"{prefix}.mlp.c_fc.bias"
        mapping[f"{prefix}.mlp.c_proj.weight"] = f"{prefix}.mlp.c_proj.weight"
        mapping[f"{prefix}.mlp.c_proj.bias"] = f"{prefix}.mlp.c_proj.bias"
        # LayerNorm
        mapping[f"{prefix}.ln_1.weight"] = f"{prefix}.ln_1.weight"
        mapping[f"{prefix}.ln_1.bias"] = f"{prefix}.ln_1.bias"
        mapping[f"{prefix}.ln_2.weight"] = f"{prefix}.ln_2.weight"
        mapping[f"{prefix}.ln_2.bias"] = f"{prefix}.ln_2.bias"
    return mapping


def nanochat_to_hf(checkpoint_path: str, output_dir: str, depth: int) -> None:
    """Convert a nanochat .pt checkpoint to HF GPT2LMHeadModel format."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = DEPTH_CONFIGS[depth]
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract model state dict (nanochat checkpoints store it under 'model_state_dict')
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # Create HF config
    hf_config = GPT2Config(
        vocab_size=cfg["vocab_size"],
        n_positions=1024,
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )

    # Create model and load weights
    model = GPT2LMHeadModel(hf_config)
    layer_map = _build_layer_map(cfg["n_layer"])

    # Map nanochat weights to HF format
    hf_state = {}
    for nc_key, hf_key in layer_map.items():
        if nc_key in state_dict:
            hf_state[hf_key] = state_dict[nc_key]

    model.load_state_dict(hf_state, strict=False)

    # Save in HF format
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)

    # Save conversion metadata
    meta = {
        "source": checkpoint_path,
        "depth": depth,
        "config": cfg,
        "direction": "nanochat_to_hf",
    }
    (out / "conversion_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Converted %s → %s", checkpoint_path, output_dir)


def hf_to_nanochat(hf_dir: str, output_path: str, depth: int) -> None:
    """Convert an HF GPT2LMHeadModel directory back to nanochat .pt format."""
    import torch
    from transformers import GPT2LMHeadModel

    cfg = DEPTH_CONFIGS[depth]
    model = GPT2LMHeadModel.from_pretrained(hf_dir)
    hf_state = model.state_dict()

    layer_map = _build_layer_map(cfg["n_layer"])
    # Reverse the mapping
    reverse_map = {v: k for k, v in layer_map.items()}

    nc_state = {}
    for hf_key, nc_key in reverse_map.items():
        if hf_key in hf_state:
            nc_state[nc_key] = hf_state[hf_key]

    # Save in nanochat format
    ckpt = {
        "model_state_dict": nc_state,
        "config": cfg,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)
    logger.info("Converted %s → %s", hf_dir, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert between nanochat and HF formats")
    parser.add_argument("--checkpoint", required=True, help="Source checkpoint/dir")
    parser.add_argument("--output", required=True, help="Output path/dir")
    parser.add_argument("--depth", required=True, type=int, choices=list(DEPTH_CONFIGS.keys()))
    parser.add_argument(
        "--direction",
        required=True,
        choices=["nanochat_to_hf", "hf_to_nanochat"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.direction == "nanochat_to_hf":
        nanochat_to_hf(args.checkpoint, args.output, args.depth)
    else:
        hf_to_nanochat(args.checkpoint, args.output, args.depth)


if __name__ == "__main__":
    main()
