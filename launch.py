"""launch.py — single entry point for all math-nano experiments.

Usage:
    uv run python launch.py run --config configs/experiments/sft-m-concise.yaml
    uv run python launch.py batch --configs configs/experiments/sft-m-*.yaml
    uv run python launch.py sweep --mixture mix-math-broad --stage pretrain
    uv run python launch.py status
    uv run python launch.py gate --check pretrain_to_sft
    uv run python launch.py eval --model sft-m-concise-best --suite full
    uv run python launch.py compile
    uv run python launch.py summarize --phase 1
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_config_from_yaml(path: str) -> "ExperimentConfig":
    """Load an ExperimentConfig from a YAML file."""
    import yaml

    from harness.config import ExperimentConfig

    with open(path) as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(**data)


def _load_config_from_args(args: argparse.Namespace) -> "ExperimentConfig":
    """Build an ExperimentConfig from CLI args."""
    from harness.config import ExperimentConfig

    kwargs = {}
    for field_name in [
        "experiment_id", "stage", "phase", "depth", "device",
        "mixture", "token_multiplier", "sft_recipe", "sft_epochs",
        "sft_lr", "sft_max_seq_len", "parent_checkpoint", "gpu",
        "timeout_hours", "wandb_mode", "eval_suite", "eval_every",
    ]:
        val = getattr(args, field_name, None)
        if val is not None:
            kwargs[field_name] = val

    if "experiment_id" not in kwargs:
        kwargs["experiment_id"] = getattr(args, "experiment", None) or "unnamed"
    if "phase" not in kwargs:
        kwargs["phase"] = "1a"

    return ExperimentConfig(**kwargs)


# ═══════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════


def cmd_run(args: argparse.Namespace) -> None:
    """Run a single experiment."""
    from harness.runner import ExperimentRunner

    if args.config:
        config = _load_config_from_yaml(args.config)
    else:
        config = _load_config_from_args(args)

    logger.info("Running experiment: %s", config.experiment_id)

    runner = ExperimentRunner(
        force=getattr(args, "force", False),
        dry_run=getattr(args, "dry_run", False),
    )
    result, eval_results = runner.run(config)

    logger.info("Experiment %s completed.", config.experiment_id)
    if eval_results:
        logger.info("Eval results: %s", json.dumps(eval_results, indent=2))


def cmd_batch(args: argparse.Namespace) -> None:
    """Run multiple experiments from config files."""
    from harness.experiment_state import ExperimentState
    from harness.runner import ExperimentRunner

    patterns = args.configs.split(",") if args.configs else []
    config_files = []
    for pattern in patterns:
        config_files.extend(sorted(glob.glob(pattern)))

    if not config_files:
        logger.error("No config files matched: %s", args.configs)
        sys.exit(1)

    logger.info("Batch: %d config files", len(config_files))

    configs = [_load_config_from_yaml(f) for f in config_files]

    # Show cost estimate
    from harness.runner import estimate_cost

    total_cost = sum(estimate_cost(c) for c in configs)
    logger.info("Estimated total cost: $%.2f", total_cost)

    if args.dry_run:
        for c in configs:
            cost = estimate_cost(c)
            print(f"  {c.experiment_id}: {c.stage} depth={c.depth} (~${cost:.2f})")
        return

    # Update state with pending
    state = ExperimentState.load()
    state.add_pending([c.experiment_id for c in configs])
    state.save()

    runner = ExperimentRunner(force=getattr(args, "force", False))
    for config in configs:
        try:
            state.mark_running(config.experiment_id)
            state.save()
            runner.run(config)
        except Exception as e:
            logger.error("Experiment %s failed: %s", config.experiment_id, e)
            state.mark_failed(config.experiment_id)
            state.save()
            if not args.continue_on_error:
                raise


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run a config across all model sizes."""
    from harness.config import VALID_DEPTHS
    from harness.runner import ExperimentRunner, estimate_cost

    depths = VALID_DEPTHS
    stage = args.stage

    configs = []
    for depth in depths:
        from harness.search import _depth_label

        eid = f"{stage[:2]}-{_depth_label(depth)}-{args.mixture or args.recipe or 'sweep'}"
        config_kwargs = {
            "experiment_id": eid,
            "stage": stage,
            "phase": args.phase or "1a",
            "depth": depth,
        }
        if args.mixture:
            config_kwargs["mixture"] = args.mixture
        if args.recipe:
            config_kwargs["sft_recipe"] = args.recipe
        if args.parent:
            config_kwargs["parent_checkpoint"] = args.parent

        from harness.config import ExperimentConfig

        configs.append(ExperimentConfig(**config_kwargs))

    total_cost = sum(estimate_cost(c) for c in configs)
    logger.info("Sweep: %d experiments, estimated $%.2f", len(configs), total_cost)

    if args.dry_run:
        for c in configs:
            cost = estimate_cost(c)
            print(f"  {c.experiment_id}: depth={c.depth} (~${cost:.2f})")
        return

    runner = ExperimentRunner(force=getattr(args, "force", False))
    for config in configs:
        try:
            runner.run(config)
        except Exception as e:
            logger.error("Sweep experiment %s failed: %s", config.experiment_id, e)


def cmd_status(args: argparse.Namespace) -> None:
    """Show experiment status."""
    from harness.experiment_state import ExperimentState

    state = ExperimentState.load()

    if args.experiment:
        # Show single experiment details
        eid = args.experiment
        if eid in state.completed_experiments:
            print(f"{eid}: completed")
        elif eid in state.running_experiments:
            print(f"{eid}: running")
        elif eid in state.pending_experiments:
            print(f"{eid}: pending")
        else:
            print(f"{eid}: not found in state")
        return

    print(state.summary())

    if state.completed_experiments:
        print(f"\nCompleted: {', '.join(state.completed_experiments)}")
    if state.running_experiments:
        print(f"Running: {', '.join(state.running_experiments)}")
    if state.pending_experiments:
        print(f"Pending: {', '.join(state.pending_experiments)}")


def cmd_gate(args: argparse.Namespace) -> None:
    """Check validation gates."""
    from harness.gates import check_preflight, check_pretrain_to_sft, check_sft_to_rl

    gates = {
        "preflight": check_preflight,
        "pretrain_to_sft": check_pretrain_to_sft,
        "sft_to_rl": check_sft_to_rl,
    }
    gate_fn = gates.get(args.check)
    if not gate_fn:
        print(f"Unknown gate '{args.check}'. Available: {list(gates.keys())}")
        sys.exit(1)
    result = gate_fn()
    print(result.summary())
    sys.exit(0 if result.passed else 1)


def cmd_eval(args: argparse.Namespace) -> None:
    """Run eval on a checkpoint."""
    print(f"[launch] eval: not yet implemented")
    print(f"  model={args.model}, suite={args.suite}")


def cmd_compile(args: argparse.Namespace) -> None:
    """Compile all results into summary CSVs."""
    from harness.bookkeeper import ModelRegistry

    registry = ModelRegistry()
    models = registry.models

    if not models:
        print("No models registered yet.")
        return

    # Build results table
    rows = []
    for model_id, model in models.items():
        row = {
            "model_id": model_id,
            "experiment_id": model.get("experiment_id"),
            "stage": model.get("stage"),
            "depth": model.get("depth"),
            "parent": model.get("parent_model"),
        }
        # Flatten eval results
        for k, v in model.get("eval_results", {}).items():
            row[k] = v
        # Training info
        training = model.get("training", {})
        row["cost_usd"] = training.get("cost_usd")
        row["wall_clock_hours"] = training.get("wall_clock_hours")
        rows.append(row)

    output_dir = Path("results/compiled")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "full_results.json"
    output_path.write_text(json.dumps(rows, indent=2) + "\n")
    logger.info("Compiled %d models → %s", len(rows), output_path)

    # Also write CSV if pandas available
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        csv_path = output_dir / "full_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info("CSV → %s", csv_path)
    except ImportError:
        pass


def cmd_summarize(args: argparse.Namespace) -> None:
    """Generate a summary for a phase."""
    from harness.bookkeeper import ModelRegistry
    from harness.experiment_state import ExperimentState

    state = ExperimentState.load()
    registry = ModelRegistry()

    phase = args.phase
    print(f"=== Phase {phase} Summary ===\n")
    print(state.summary())
    print()

    # List models from this phase
    models = registry.models
    phase_models = {
        mid: m for mid, m in models.items()
        if str(phase) in m.get("experiment_id", "")
    }

    if phase_models:
        print(f"Models registered: {len(phase_models)}")
        for mid, m in phase_models.items():
            eval_r = m.get("eval_results", {})
            gsm8k = eval_r.get("gsm8k_pass1_greedy", "n/a")
            print(f"  {mid}: gsm8k={gsm8k}")
    else:
        print("No models registered for this phase yet.")


def cmd_smoke_test(args: argparse.Namespace) -> None:
    """Run smoke tests."""
    print(f"[launch] smoke-test: not yet implemented")
    print(f"  depth={args.depth}, device={args.device}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="math-nano experiment launcher")
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run an experiment")
    p_run.add_argument("--config", help="YAML config file path")
    p_run.add_argument("--stage", choices=["pretrain", "sft", "grpo"])
    p_run.add_argument("--depth", type=int)
    p_run.add_argument("--experiment", help="Experiment ID")
    p_run.add_argument("--mixture", help="Data mixture (pretrain)")
    p_run.add_argument("--sft-recipe", dest="sft_recipe", help="SFT recipe")
    p_run.add_argument("--parent", dest="parent_checkpoint", help="Parent checkpoint")
    p_run.add_argument("--force", action="store_true", help="Skip validation")
    p_run.add_argument("--dry-run", action="store_true", dest="dry_run")
    p_run.add_argument("--wandb-mode", dest="wandb_mode", default="online")
    p_run.set_defaults(func=cmd_run)

    # batch
    p_batch = sub.add_parser("batch", help="Run multiple experiments")
    p_batch.add_argument("--configs", required=True, help="Glob pattern(s) for YAML configs")
    p_batch.add_argument("--dry-run", action="store_true", dest="dry_run")
    p_batch.add_argument("--force", action="store_true")
    p_batch.add_argument("--continue-on-error", action="store_true", dest="continue_on_error")
    p_batch.set_defaults(func=cmd_batch)

    # sweep
    p_sweep = sub.add_parser("sweep", help="Sweep a config across all model sizes")
    p_sweep.add_argument("--stage", required=True, choices=["pretrain", "sft", "grpo"])
    p_sweep.add_argument("--mixture", help="Data mixture (pretrain)")
    p_sweep.add_argument("--recipe", help="SFT recipe")
    p_sweep.add_argument("--parent", help="Parent checkpoint")
    p_sweep.add_argument("--phase", default="1a")
    p_sweep.add_argument("--dry-run", action="store_true", dest="dry_run")
    p_sweep.add_argument("--force", action="store_true")
    p_sweep.set_defaults(func=cmd_sweep)

    # status
    p_status = sub.add_parser("status", help="Show experiment status")
    p_status.add_argument("--experiment", help="Specific experiment ID")
    p_status.set_defaults(func=cmd_status)

    # gate
    p_gate = sub.add_parser("gate", help="Check validation gates")
    p_gate.add_argument("--check", required=True, help="Gate name to check")
    p_gate.set_defaults(func=cmd_gate)

    # eval
    p_eval = sub.add_parser("eval", help="Run eval on a checkpoint")
    p_eval.add_argument("--model", required=True, help="Model ID or checkpoint path")
    p_eval.add_argument("--suite", default="small", choices=["small", "full"])
    p_eval.add_argument("--samples", type=int, default=1)
    p_eval.add_argument("--device", default="auto")
    p_eval.set_defaults(func=cmd_eval)

    # compile
    p_compile = sub.add_parser("compile", help="Compile results into CSVs")
    p_compile.set_defaults(func=cmd_compile)

    # summarize
    p_summarize = sub.add_parser("summarize", help="Generate phase summary")
    p_summarize.add_argument("--phase", required=True, help="Phase number")
    p_summarize.set_defaults(func=cmd_summarize)

    # smoke-test
    p_smoke = sub.add_parser("smoke-test", help="Run smoke tests")
    p_smoke.add_argument("--depth", type=int, default=10)
    p_smoke.add_argument("--device", default="cpu")
    p_smoke.add_argument("--all", action="store_true")
    p_smoke.set_defaults(func=cmd_smoke_test)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
