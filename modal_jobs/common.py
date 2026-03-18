"""Shared Modal infrastructure: app, image, volumes, secrets."""

from __future__ import annotations

try:
    import modal
    from pathlib import Path

    app = modal.App("math-nano")

    # Volumes
    vol_checkpoints = modal.Volume.from_name("math-nano-checkpoints", create_if_missing=True)
    vol_data = modal.Volume.from_name("math-nano-data", create_if_missing=True)
    vol_results = modal.Volume.from_name("math-nano-results", create_if_missing=True)

    VOLUME_MOUNTS = {
        "/checkpoints": vol_checkpoints,
        "/data": vol_data,
        "/results": vol_results,
    }

    project_root = Path(__file__).parent.parent

    # Single image for all training jobs
    # Matches nanochat's pyproject.toml dependencies
    train_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch>=2.4.0",
            "transformers>=4.50.0",
            "datasets>=4.0.0",
            "wandb>=0.16.0",
            "tiktoken>=0.11.0",
            "tokenizers>=0.22.0",
            "rustbpe>=0.1.0",
            "numpy>=1.26.0",
            "huggingface-hub>=0.22.0",
            "scipy>=1.15.0",
            "tabulate>=0.9.0",
            "regex>=2024.0.0",
            "zstandard>=0.25.0",
        )
        .add_local_dir(
            str(project_root),
            remote_path="/root/math-nano",
            ignore=[
                "**/.venv/**", "**/__pycache__/**", "**/.git/**",
                "**/wandb/**", "**/checkpoints/**",
                "data/raw/**", "data/tokenized/**", "logs/**",
            ],
        )
        .add_local_dir(
            str(project_root / "modal_jobs"),
            remote_path="/root/modal_jobs",
        )
        .add_local_dir(
            str(Path.home() / ".cache" / "nanochat" / "tokenizer"),
            remote_path="/root/.cache/nanochat/tokenizer",
        )
    )

    # Set to None if you haven't created the secret on Modal yet.
    # Create at: https://modal.com/secrets/create?secret_name=wandb-secret
    WANDB_SECRET = None

except ImportError:
    # Modal not installed — stubs for import compatibility
    app = None  # type: ignore[assignment]
    vol_checkpoints = None
    vol_data = None
    vol_results = None
    VOLUME_MOUNTS = {}
    train_image = None
    WANDB_SECRET = None
