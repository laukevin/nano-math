"""Shared Modal infrastructure: app, image, volumes, secrets."""

from __future__ import annotations

try:
    import modal

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

    # Single image for all training jobs (includes eval deps)
    train_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch==2.4.0",
            "transformers>=4.40.0",
            "datasets>=2.18.0",
            "trl>=0.8.0",
            "wandb>=0.16.0",
            "tiktoken>=0.6.0",
            "numpy>=1.26.0",
            "huggingface-hub>=0.22.0",
        )
    )

    WANDB_SECRET = modal.Secret.from_name("wandb-secret")

except ImportError:
    # Modal not installed — stubs for import compatibility
    app = None  # type: ignore[assignment]
    vol_checkpoints = None
    vol_data = None
    vol_results = None
    VOLUME_MOUNTS = {}
    train_image = None
    WANDB_SECRET = None
