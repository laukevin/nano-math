"""Modal job: download and tokenize datasets.

GPU: None (CPU, high memory), Timeout: 4h
"""

from __future__ import annotations

import modal

from modal_jobs.common import (
    HF_SECRET,
    app,
    data_image,
    vol_data,
)


@app.function(
    image=data_image,
    cpu=4,
    memory=32768,  # 32GB
    timeout=4 * 3600,
    volumes={"/data": vol_data},
    secrets=[HF_SECRET],
)
def run_data_prep(
    source: str,
    output_name: str,
    max_shards: int | None = None,
    tokenizer: str = "gpt2",
    seq_len: int = 1024,
) -> dict:
    """Download, tokenize, and shard a dataset."""
    import subprocess
    from pathlib import Path

    output_dir = f"/data/tokenized/{output_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "scripts.data.download_and_tokenize",
        f"--source={source}",
        f"--output-dir={output_dir}",
        f"--tokenizer={tokenizer}",
        f"--seq-len={seq_len}",
    ]
    if max_shards is not None:
        cmd.append(f"--max-shards={max_shards}")

    print(f"[data_prep] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    vol_data.commit()

    # Count shards
    shards = list(Path(output_dir).glob("*.bin"))
    total_tokens = sum(p.stat().st_size // 2 for p in shards)  # uint16 tokens

    return {
        "output_dir": output_dir,
        "n_shards": len(shards),
        "total_tokens": total_tokens,
    }
