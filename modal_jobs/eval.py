"""Modal job: full eval on any checkpoint.

GPU: A100 40GB, Timeout: 1h
"""

from __future__ import annotations

import modal

from modal_jobs.common import (
    VOLUME_MOUNTS,
    app,
    eval_image,
    vol_results,
)


@app.function(
    image=eval_image,
    gpu=modal.gpu.A100(count=1, size="40GB"),
    timeout=1 * 3600,
    volumes=VOLUME_MOUNTS,
)
def run_eval(
    checkpoint: str,
    depth: int,
    experiment_id: str,
    datasets: str = "gsm8k,math500",
    mode: str = "full",
    samples: int = 1,
    device: str = "cuda",
) -> dict:
    """Run full eval suite on a checkpoint."""
    import json
    import subprocess
    from pathlib import Path

    eval_output = f"/results/eval_{experiment_id}.json"

    cmd = [
        "python", "-m", "scripts.eval.run_eval",
        f"--checkpoint={checkpoint}",
        f"--datasets={datasets}",
        f"--mode={mode}",
        f"--depth={depth}",
        f"--samples={samples}",
        f"--device={device}",
        f"--output={eval_output}",
    ]

    print(f"[eval] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    vol_results.commit()

    results = {}
    if Path(eval_output).exists():
        results = json.loads(Path(eval_output).read_text())

    return results
