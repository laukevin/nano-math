"""Eval runner: load checkpoint, generate completions, compute metrics, output JSON.

Usage:
    # Quick eval during training (greedy, small suite)
    python scripts/eval/run_eval.py --checkpoint $CKPT --depth 16 --suite small --mode greedy

    # Full blessed eval (sampled, all datasets)
    python scripts/eval/run_eval.py --checkpoint $CKPT --depth 16 --suite full --samples 16

    # Single dataset, custom k
    python scripts/eval/run_eval.py --checkpoint $CKPT --depth 16 --datasets gsm8k --samples 32

    # CPU test
    python scripts/eval/run_eval.py --checkpoint $CKPT --depth 10 --suite small --mode greedy --device cpu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from scripts.eval.extraction import extract_answer, normalize_answer
from scripts.eval.metrics import bootstrap_ci, compute_pass_at_k, pass_at_k

logger = logging.getLogger(__name__)

EVAL_VERSION = "1.0"
MAX_NEW_TOKENS = 1024
GREEDY_TEMPERATURE = 0.0
SAMPLED_TEMPERATURE = 0.7
BOOTSTRAP_SEED = 42
EOS_TOKEN_ID = 50256  # GPT-2 EOS

SUITE_DATASETS = {
    "small": ["gsm8k_mini", "math_mini"],
    "full": ["gsm8k", "math500", "amc", "aime", "minerva"],
}


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_eval_prompt(problem: str) -> str:
    """Format a math problem for evaluation."""
    return (
        "Solve the following math problem step by step. "
        "Put your final answer in \\boxed{}.\n\n"
        f"Problem: {problem}\n\n"
        "Solution:"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_dataset(dataset_name: str, data_dir: Path) -> list[dict]:
    """Load an eval dataset from JSONL, verifying checksum against manifest."""
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        datasets_info = manifest.get("datasets", {})
        if dataset_name in datasets_info:
            entry = datasets_info[dataset_name]
            file_path = data_dir / entry["file"]
            if not file_path.exists():
                raise FileNotFoundError(f"Eval dataset not found: {file_path}")

            content = file_path.read_bytes()
            actual_sha = hashlib.sha256(content).hexdigest()
            expected_sha = entry.get("sha256", "")
            if expected_sha and actual_sha != expected_sha:
                raise ValueError(
                    f"Checksum mismatch for {dataset_name}: "
                    f"expected {expected_sha[:16]}..., got {actual_sha[:16]}..."
                )

            problems = [
                json.loads(line)
                for line in content.decode().splitlines()
                if line.strip()
            ]
            expected_n = entry.get("n")
            if expected_n and len(problems) != expected_n:
                raise ValueError(
                    f"Count mismatch for {dataset_name}: "
                    f"expected {expected_n}, got {len(problems)}"
                )
            return problems

    # Fallback: try direct file load
    for pattern in [f"{dataset_name}.jsonl", f"{dataset_name}_test.jsonl"]:
        file_path = data_dir / pattern
        if file_path.exists():
            return [
                json.loads(line)
                for line in file_path.read_text().splitlines()
                if line.strip()
            ]

    raise FileNotFoundError(
        f"No eval data found for '{dataset_name}' in {data_dir}"
    )


def get_manifest_sha(data_dir: Path) -> str:
    """Return SHA256 of the manifest file itself."""
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        return hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return ""


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def resolve_device(device: str) -> str:
    """Resolve 'auto' device to actual device string."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    checkpoint_path: str, depth: int, device: str  # noqa: ARG001 — depth reserved for future config override
) -> tuple:
    """Load model and tokenizer from checkpoint.

    Returns:
        (model, tokenizer, device_str, model_params_count)
    """
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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

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
            input_len = max_input_len  # all padded to same length
            output_ids = generated[j, input_len:].tolist()

            # Truncate at EOS
            hit_eos = False
            if EOS_TOKEN_ID in output_ids:
                output_ids = output_ids[: output_ids.index(EOS_TOKEN_ID)]
                hit_eos = True

            # Detect repetition (same 20-token window 3x in a row)
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


# ---------------------------------------------------------------------------
# Pure evaluation logic (no model needed — testable)
# ---------------------------------------------------------------------------

def evaluate_completions(
    outputs: list[str] | list[list[str]],
    ground_truths: list[str],
    problem_ids: list[str],
    n_samples: int,
    dataset_name: str = "",  # noqa: ARG001
) -> dict:
    """Evaluate model outputs against ground truths.

    Args:
        outputs: If n_samples==1, list[str] (one per problem).
                 If n_samples>1, list[list[str]] (n_samples per problem).
        ground_truths: normalized answer strings, one per problem.
        problem_ids: unique ID per problem.
        n_samples: number of samples per problem.
        dataset_name: name of the dataset (for logging).

    Returns:
        Result dict matching spec 07 format.
    """
    n_problems = len(ground_truths)
    extraction_failures = 0

    if n_samples == 1:
        # Greedy mode
        per_problem = []
        greedy_correct: list[bool] = []

        for pid, output, gt in zip(problem_ids, outputs, ground_truths):
            predicted = extract_answer(output)  # type: ignore[arg-type]
            if predicted is None:
                extraction_failures += 1
                correct = False
            else:
                correct = predicted == gt
            greedy_correct.append(correct)
            per_problem.append(
                {"id": pid, "correct_samples": int(correct), "total_samples": 1}
            )

        mean, ci_low, ci_high = bootstrap_ci(
            greedy_correct, seed=BOOTSTRAP_SEED
        )

        result = {
            "n_problems": n_problems,
            "pass_at_1_greedy": mean,
            "pass_at_1_greedy_ci95": [ci_low, ci_high],
            "extraction_failures": extraction_failures,
            "extraction_failure_rate": (
                extraction_failures / n_problems if n_problems else 0.0
            ),
            "per_problem": per_problem,
        }

    else:
        # Sampled mode
        per_problem = []
        pass_k_inputs: list[dict] = []

        for idx, (pid, gt) in enumerate(zip(problem_ids, ground_truths)):
            sample_list = outputs[idx]  # type: ignore[index]
            n_correct = 0
            for output in sample_list:
                predicted = extract_answer(output)
                if predicted is None:
                    extraction_failures += 1
                elif predicted == gt:
                    n_correct += 1

            per_problem.append(
                {
                    "id": pid,
                    "correct_samples": n_correct,
                    "total_samples": n_samples,
                }
            )
            pass_k_inputs.append(
                {
                    "problem_id": pid,
                    "n_samples": n_samples,
                    "n_correct": n_correct,
                }
            )

        k_values = [k for k in [1, 4, 8] if k <= n_samples]
        pass_k_metrics = compute_pass_at_k(pass_k_inputs, k_values)

        result: dict = {
            "n_problems": n_problems,
            "n_samples_per_problem": n_samples,
            "extraction_failures": extraction_failures,
            "extraction_failure_rate": (
                extraction_failures / (n_problems * n_samples)
                if n_problems
                else 0.0
            ),
            "per_problem": per_problem,
        }

        # Add pass@k with bootstrap CIs
        for k in k_values:
            per_problem_scores = [
                pass_at_k(r["n_samples"], r["n_correct"], k)
                for r in pass_k_inputs
            ]
            result[f"pass_at_{k}_sampled"] = pass_k_metrics[f"pass@{k}"]

            # Bootstrap CI on per-problem pass@k scores
            _, ci_l, ci_h = bootstrap_ci(
                per_problem_scores, seed=BOOTSTRAP_SEED  # type: ignore[arg-type]
            )
            result[f"pass_at_{k}_sampled_ci95"] = [ci_l, ci_h]

    return result


# ---------------------------------------------------------------------------
# Full dataset evaluation (with generation)
# ---------------------------------------------------------------------------

def run_dataset_eval(
    model,
    tokenizer,
    problems: list[dict],
    dataset_name: str,
    n_samples: int = 1,
    temperature: float = 0.0,
    device: str = "cpu",
    batch_size: int = 32,
) -> dict:
    """Run eval on a single dataset: generate + evaluate."""
    prompts = [format_eval_prompt(p["problem"]) for p in problems]
    ground_truths = [normalize_answer(str(p["answer"])) for p in problems]
    problem_ids = [
        p.get("id", f"{dataset_name}_{i:04d}") for i, p in enumerate(problems)
    ]

    start_time = time.time()

    if n_samples == 1:
        outputs, gen_stats = generate_batch(
            model,
            tokenizer,
            prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            batch_size=batch_size,
            device=device,
        )
    else:
        # Expand: repeat each prompt n_samples times
        expanded = [p for p in prompts for _ in range(n_samples)]
        flat_outputs, gen_stats = generate_batch(
            model,
            tokenizer,
            expanded,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            batch_size=batch_size,
            device=device,
        )
        # Reshape into list-of-lists
        outputs = [
            flat_outputs[i * n_samples : (i + 1) * n_samples]
            for i in range(len(prompts))
        ]

    elapsed = time.time() - start_time
    n_total_gens = len(prompts) * n_samples

    result = evaluate_completions(
        outputs, ground_truths, problem_ids, n_samples, dataset_name
    )

    # Add timing and generation stats
    result["avg_output_tokens"] = (
        gen_stats["total_tokens"] / n_total_gens if n_total_gens else 0
    )
    result["avg_inference_ms"] = (
        (elapsed * 1000) / n_total_gens if n_total_gens else 0
    )

    return result


# ---------------------------------------------------------------------------
# Output JSON assembly
# ---------------------------------------------------------------------------

def build_output_json(
    checkpoint: str,
    depth: int,
    model_params: int,
    suite: str,
    n_samples: int,
    temperature: float,
    dataset_results: dict[str, dict],
    manifest_sha: str,
    experiment_id: str = "",
    stage: str = "",
) -> dict:
    """Build the full eval output JSON per spec 07."""
    # Aggregate metrics across datasets
    greedy_scores = []
    sampled_scores = []
    for ds_result in dataset_results.values():
        if "pass_at_1_greedy" in ds_result:
            greedy_scores.append(ds_result["pass_at_1_greedy"])
        if "pass_at_1_sampled" in ds_result:
            sampled_scores.append(ds_result["pass_at_1_sampled"])

    aggregate: dict = {}
    if greedy_scores:
        aggregate["avg_pass_at_1_greedy"] = float(np.mean(greedy_scores))
    if sampled_scores:
        aggregate["avg_pass_at_1_sampled"] = float(np.mean(sampled_scores))

    return {
        "eval_version": EVAL_VERSION,
        "checkpoint": checkpoint,
        "model_depth": depth,
        "model_params": model_params,
        "stage": stage,
        "experiment_id": experiment_id,
        "eval_suite": suite,
        "n_samples_per_problem": n_samples,
        "temperature": temperature,
        "max_new_tokens": MAX_NEW_TOKENS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_data_manifest_sha": manifest_sha,
        "results": dataset_results,
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# W&B integration
# ---------------------------------------------------------------------------

def log_to_wandb(
    output_json: dict,
    output_path: Path,
    project: str = "math-nano",
) -> None:
    """Log eval results to W&B."""
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run eval on a model checkpoint"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to checkpoint .pt file"
    )
    parser.add_argument(
        "--depth", type=int, required=True, help="Model depth (transformer layers)"
    )
    parser.add_argument(
        "--suite",
        choices=["small", "full"],
        default=None,
        help="Eval suite: small (300 problems) or full (2381 problems)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific dataset(s) to evaluate",
    )
    parser.add_argument(
        "--mode",
        choices=["greedy", "sampled"],
        default="greedy",
        help="Decoding mode",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=16,
        help="Number of samples per problem (sampled mode)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for inference",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Generation batch size"
    )
    parser.add_argument(
        "--output-dir",
        default="results/eval",
        help="Directory for output JSON files",
    )
    parser.add_argument(
        "--data-dir",
        default="data/eval",
        help="Directory containing eval JSONL files",
    )
    parser.add_argument(
        "--experiment-id", default="", help="Experiment identifier"
    )
    parser.add_argument(
        "--stage",
        default="",
        choices=["", "pretrain", "sft", "grpo"],
        help="Training stage",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Log results to W&B"
    )
    parser.add_argument(
        "--wandb-project", default="math-nano", help="W&B project name"
    )

    args = parser.parse_args(argv)

    # Resolve datasets
    if args.datasets:
        pass  # use explicit list
    elif args.suite:
        args.datasets = SUITE_DATASETS[args.suite]
    else:
        parser.error("Must specify --suite or --datasets")

    # Default suite label
    if not args.suite:
        args.suite = "custom"

    # Warn about full eval on CPU
    resolved_device = resolve_device(args.device)
    if resolved_device == "cpu" and args.suite == "full":
        logger.warning(
            "Running full eval suite on CPU will be slow. "
            "Consider --suite small for faster iteration."
        )

    return args


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load eval data
    logger.info("Loading eval datasets: %s", args.datasets)
    eval_data: dict[str, list[dict]] = {}
    for ds_name in args.datasets:
        eval_data[ds_name] = load_eval_dataset(ds_name, data_dir)
        logger.info("  %s: %d problems", ds_name, len(eval_data[ds_name]))

    manifest_sha = get_manifest_sha(data_dir)

    # Load model
    logger.info("Loading checkpoint: %s", args.checkpoint)
    model, tokenizer, device, model_params = load_model(
        args.checkpoint, args.depth, args.device
    )
    logger.info(
        "Model loaded: depth=%d, params=%d, device=%s",
        args.depth,
        model_params,
        device,
    )

    # Determine generation params
    if args.mode == "greedy":
        temperature = GREEDY_TEMPERATURE
        n_samples = 1
    else:
        temperature = SAMPLED_TEMPERATURE
        n_samples = args.samples

    # Evaluate each dataset
    dataset_results: dict[str, dict] = {}
    for ds_name, problems in eval_data.items():
        logger.info(
            "Evaluating %s (%d problems, n_samples=%d)...",
            ds_name,
            len(problems),
            n_samples,
        )
        dataset_results[ds_name] = run_dataset_eval(
            model,
            tokenizer,
            problems,
            ds_name,
            n_samples=n_samples,
            temperature=temperature,
            device=device,
            batch_size=args.batch_size,
        )
        # Log progress
        ds_r = dataset_results[ds_name]
        if "pass_at_1_greedy" in ds_r:
            logger.info(
                "  %s pass@1 (greedy): %.3f [%.3f, %.3f]",
                ds_name,
                ds_r["pass_at_1_greedy"],
                ds_r["pass_at_1_greedy_ci95"][0],
                ds_r["pass_at_1_greedy_ci95"][1],
            )
        if "pass_at_1_sampled" in ds_r:
            logger.info(
                "  %s pass@1 (sampled): %.3f",
                ds_name,
                ds_r["pass_at_1_sampled"],
            )

    # Build output
    output_json = build_output_json(
        checkpoint=args.checkpoint,
        depth=args.depth,
        model_params=model_params,
        suite=args.suite,
        n_samples=n_samples,
        temperature=temperature,
        dataset_results=dataset_results,
        manifest_sha=manifest_sha,
        experiment_id=args.experiment_id,
        stage=args.stage,
    )

    # Save
    exp_label = args.experiment_id or Path(args.checkpoint).stem
    output_path = output_dir / f"{exp_label}_{args.suite}.json"
    output_path.write_text(json.dumps(output_json, indent=2))
    logger.info("Results saved to %s", output_path)

    # W&B
    if args.wandb:
        log_to_wandb(output_json, output_path, project=args.wandb_project)
        logger.info("Results logged to W&B")


if __name__ == "__main__":
    main()
