"""W&B logging for eval runs."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def log_to_wandb(
    output_json: dict,
    output_path: Path,
    project: str = "math-nano",
) -> None:
    """Log eval results to W&B as type='eval' with artifact + per-problem Table."""
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed, skipping W&B logging")
        return

    run = wandb.init(
        project=project,
        job_type="eval",
        name=f"eval-{output_json.get('experiment_id', 'unknown')}",
        tags=[
            f"depth_{output_json['model_depth']}",
            f"suite_{output_json['eval_suite']}",
            f"stage_{output_json.get('stage', 'unknown')}",
        ],
        config={
            "checkpoint": output_json["checkpoint"],
            "model_depth": output_json["model_depth"],
            "model_params": output_json["model_params"],
            "eval_suite": output_json["eval_suite"],
            "n_samples": output_json["n_samples_per_problem"],
            "temperature": output_json["temperature"],
        },
    )

    # Log summary metrics
    summary = {}
    for ds_name, ds_result in output_json["results"].items():
        for key in [
            "pass_at_1_greedy",
            "pass_at_1_sampled",
            "pass_at_4_sampled",
            "pass_at_8_sampled",
        ]:
            if key in ds_result:
                summary[f"eval/{ds_name}_{key}"] = ds_result[key]

    if output_json.get("aggregate"):
        for key, val in output_json["aggregate"].items():
            summary[f"eval/{key}"] = val

    wandb.log(summary)

    # Upload JSON as artifact
    artifact = wandb.Artifact(
        name=f"eval-{output_json.get('experiment_id', 'results')}",
        type="eval",
    )
    artifact.add_file(str(output_path))
    run.log_artifact(artifact)

    # Log per-problem results as W&B Table
    for ds_name, ds_result in output_json["results"].items():
        per_problem = ds_result.get("per_problem", [])
        if per_problem:
            table = wandb.Table(
                columns=["problem_id", "correct_samples", "total_samples"],
                data=[
                    [r["id"], r["correct_samples"], r["total_samples"]]
                    for r in per_problem
                ],
            )
            wandb.log({f"eval/{ds_name}_per_problem": table})

    wandb.finish()
