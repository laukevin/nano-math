"""launch.py — single entry point for all math-nano experiments.

Usage:
    uv run python launch.py run --config configs/experiments/sft-m-concise.yaml
    uv run python launch.py batch --configs configs/experiments/sft-m-*.yaml
    uv run python launch.py sweep --mixture mix-math-broad --stage pretrain
    uv run python launch.py status
    uv run python launch.py gate --check pretrain_to_sft
    uv run python launch.py eval --checkpoint $CKPT --depth 16 --suite small
    uv run python launch.py compare --checkpoint-a $A --checkpoint-b $B --depth 16
    uv run python launch.py check-leakage --train-dir data/tokenized
    uv run python launch.py compile
    uv run python launch.py compile-eval --results-dir results/eval/
    uv run python launch.py plot --data results/compiled/full_results.csv --all
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
# Training commands
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


# ═══════════════════════════════════════════════════════════════════
# Status and gates
# ═══════════════════════════════════════════════════════════════════


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


def cmd_smoke_test(args: argparse.Namespace) -> None:
    """Run smoke tests."""
    print("[launch] smoke-test: not yet implemented")
    print(f"  depth={args.depth}, device={args.device}")


# ═══════════════════════════════════════════════════════════════════
# Eval
# ═══════════════════════════════════════════════════════════════════


def cmd_eval(args: argparse.Namespace) -> None:
    """Run eval on a checkpoint."""
    from scripts.eval.data import (
        GREEDY_TEMPERATURE,
        SAMPLED_TEMPERATURE,
        SUITE_DATASETS,
        get_manifest_sha,
        load_eval_dataset,
    )
    from scripts.eval.evaluate import build_output_json, run_dataset_eval
    from scripts.eval.inference import load_model, resolve_device
    from scripts.eval.wandb_logger import log_to_wandb

    datasets = args.datasets or SUITE_DATASETS[args.suite]

    resolved_device = resolve_device(args.device)
    if resolved_device == "cpu" and args.suite == "full":
        logger.warning(
            "Running full eval suite on CPU will be slow. "
            "Consider --suite small for faster iteration."
        )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading eval datasets: %s", datasets)
    eval_data: dict[str, list[dict]] = {}
    for ds_name in datasets:
        eval_data[ds_name] = load_eval_dataset(ds_name, data_dir)
        logger.info("  %s: %d problems", ds_name, len(eval_data[ds_name]))

    manifest_sha = get_manifest_sha(data_dir)

    logger.info("Loading checkpoint: %s", args.checkpoint)
    model, tokenizer, device, model_params = load_model(
        args.checkpoint, args.depth, args.device
    )
    logger.info(
        "Model loaded: depth=%d, params=%d, device=%s",
        args.depth, model_params, device,
    )

    if args.mode == "greedy":
        temperature = GREEDY_TEMPERATURE
        n_samples = 1
    else:
        temperature = SAMPLED_TEMPERATURE
        n_samples = args.samples

    dataset_results: dict[str, dict] = {}
    for ds_name, problems in eval_data.items():
        logger.info(
            "Evaluating %s (%d problems, n_samples=%d)...",
            ds_name, len(problems), n_samples,
        )
        dataset_results[ds_name] = run_dataset_eval(
            model, tokenizer, problems, ds_name,
            n_samples=n_samples, temperature=temperature,
            device=device, batch_size=args.batch_size,
        )
        ds_r = dataset_results[ds_name]
        if "pass_at_1_greedy" in ds_r:
            logger.info(
                "  %s pass@1 (greedy): %.3f [%.3f, %.3f]",
                ds_name, ds_r["pass_at_1_greedy"],
                ds_r["pass_at_1_greedy_ci95"][0],
                ds_r["pass_at_1_greedy_ci95"][1],
            )
        if "pass_at_1_sampled" in ds_r:
            logger.info("  %s pass@1 (sampled): %.3f", ds_name, ds_r["pass_at_1_sampled"])

    output_json = build_output_json(
        checkpoint=args.checkpoint, depth=args.depth,
        model_params=model_params, suite=args.suite,
        n_samples=n_samples, temperature=temperature,
        dataset_results=dataset_results, manifest_sha=manifest_sha,
        experiment_id=args.experiment_id, stage=args.stage,
    )

    exp_label = args.experiment_id or Path(args.checkpoint).stem
    output_path = output_dir / f"{exp_label}_{args.suite}.json"
    output_path.write_text(json.dumps(output_json, indent=2))
    logger.info("Results saved to %s", output_path)

    if args.wandb:
        log_to_wandb(output_json, output_path, project=args.wandb_project)
        logger.info("Results logged to W&B")


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two checkpoints."""
    from scripts.eval.compare import compare_checkpoints, format_comparison_table
    from scripts.eval.data import SUITE_DATASETS, load_eval_dataset
    from scripts.eval.inference import load_model

    datasets = args.datasets or SUITE_DATASETS[args.suite]
    data_dir = Path(args.data_dir)

    logger.info("Loading model A: %s", args.checkpoint_a)
    model_a, tok_a, device, _ = load_model(
        args.checkpoint_a, args.depth, args.device
    )
    logger.info("Loading model B: %s", args.checkpoint_b)
    model_b, tok_b, _, _ = load_model(
        args.checkpoint_b, args.depth, args.device
    )

    comparisons = []
    for ds_name in datasets:
        logger.info("Comparing on %s...", ds_name)
        problems = load_eval_dataset(ds_name, data_dir)
        comp = compare_checkpoints(
            model_a, tok_a, model_b, tok_b,
            problems, ds_name, device, args.batch_size,
        )
        comparisons.append(comp)

    table = format_comparison_table(
        comparisons, args.checkpoint_a, args.checkpoint_b
    )
    print(table)

    if args.output:
        output = {
            "checkpoint_a": args.checkpoint_a,
            "checkpoint_b": args.checkpoint_b,
            "depth": args.depth,
            "comparisons": comparisons,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, indent=2))
        logger.info("Comparison saved to %s", args.output)


def cmd_check_leakage(args: argparse.Namespace) -> None:
    """Check for eval-vs-training data leakage."""
    from scripts.eval.leakage import check_leakage, load_eval_problems, load_train_texts

    eval_dir = Path(args.eval_dir)
    train_path = Path(args.train_dir)

    if not eval_dir.exists():
        logger.error("Eval directory not found: %s", eval_dir)
        return
    if not train_path.exists():
        logger.error("Training data not found: %s", train_path)
        return

    logger.info("Loading eval problems from %s", eval_dir)
    eval_problems = load_eval_problems(eval_dir)
    for ds_name, probs in eval_problems.items():
        logger.info("  %s: %d problems", ds_name, len(probs))

    logger.info("Loading training texts from %s", train_path)
    train_texts = load_train_texts(train_path)
    logger.info("  %d unique training texts loaded", len(train_texts))

    report = check_leakage(eval_problems, train_texts)

    print(f"\n{'Dataset':<20} {'Problems':>10} {'Matches':>10} {'Rate':>8}")
    print("-" * 50)
    for ds_name, ds_report in report["datasets"].items():
        print(
            f"{ds_name:<20} "
            f"{ds_report['n_problems']:>10} "
            f"{ds_report['n_matches']:>10} "
            f"{ds_report['match_rate']:>8.2%}"
        )
    print("-" * 50)
    print(
        f"{'TOTAL':<20} "
        f"{report['total_eval_problems']:>10} "
        f"{report['total_matches']:>10}"
    )

    if report["total_matches"] > 0:
        print(f"\nWARNING: {report['total_matches']} eval problems found in training data!")
    else:
        print("\nNo leakage detected.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    logger.info("Report saved to %s", output_path)


# ═══════════════════════════════════════════════════════════════════
# Results & visualization
# ═══════════════════════════════════════════════════════════════════


def cmd_compile(args: argparse.Namespace) -> None:
    """Compile all results from model registry into summary CSVs."""
    from harness.bookkeeper import ModelRegistry

    registry = ModelRegistry()
    models = registry.models

    if not models:
        print("No models registered yet.")
        return

    rows = []
    for model_id, model in models.items():
        row = {
            "model_id": model_id,
            "experiment_id": model.get("experiment_id"),
            "stage": model.get("stage"),
            "depth": model.get("depth"),
            "parent": model.get("parent_model"),
        }
        for k, v in model.get("eval_results", {}).items():
            row[k] = v
        training = model.get("training", {})
        row["cost_usd"] = training.get("cost_usd")
        row["wall_clock_hours"] = training.get("wall_clock_hours")
        rows.append(row)

    output_dir = Path("results/compiled")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "full_results.json"
    output_path.write_text(json.dumps(rows, indent=2) + "\n")
    logger.info("Compiled %d models → %s", len(rows), output_path)

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        csv_path = output_dir / "full_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info("CSV → %s", csv_path)
    except ImportError:
        pass


def cmd_compile_eval(args: argparse.Namespace) -> None:
    """Compile eval JSON files into flat CSV."""
    from scripts.results.compile import compile_results

    df = compile_results(Path(args.results_dir), Path(args.output))
    if not df.empty:
        print(f"\nCompiled {len(df)} rows to {args.output}")
        if "dataset" in df.columns:
            print(f"Datasets: {sorted(df['dataset'].unique())}")
        if "pass_at_1_greedy" in df.columns:
            print("\npass@1 (greedy) by dataset:")
            print(
                df.groupby("dataset")["pass_at_1_greedy"]
                .agg(["mean", "min", "max", "count"])
                .to_string()
            )


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate result plots."""
    import pandas as pd
    from scripts.results.plot import PLOT_REGISTRY

    df = pd.read_csv(args.data)
    logger.info("Loaded %d rows from %s", len(df), args.data)

    if args.all:
        out_dir = Path(args.output_dir)
        for name, fn in PLOT_REGISTRY.items():
            out = str(out_dir / f"{name}.png")
            logger.info("Generating %s...", name)
            if name == "rl_dynamics":
                fn(df, out)
            else:
                fn(df, out, metric=args.metric, dataset=args.dataset)
    elif args.plot:
        fn = PLOT_REGISTRY[args.plot]
        if args.plot == "rl_dynamics":
            fn(df, args.output)
        else:
            fn(df, args.output, metric=args.metric, dataset=args.dataset)
    else:
        print("Specify --plot or --all")
        sys.exit(1)


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

    # smoke-test
    p_smoke = sub.add_parser("smoke-test", help="Run smoke tests")
    p_smoke.add_argument("--depth", type=int, default=10)
    p_smoke.add_argument("--device", default="cpu")
    p_smoke.add_argument("--all", action="store_true")
    p_smoke.set_defaults(func=cmd_smoke_test)

    # eval
    p_eval = sub.add_parser("eval", help="Run eval on a checkpoint")
    p_eval.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    p_eval.add_argument("--depth", type=int, required=True, help="Model depth")
    p_eval.add_argument("--suite", choices=["small", "full"], default="small")
    p_eval.add_argument("--datasets", nargs="+", default=None)
    p_eval.add_argument("--mode", choices=["greedy", "sampled"], default="greedy")
    p_eval.add_argument("--samples", type=int, default=16)
    p_eval.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p_eval.add_argument("--batch-size", type=int, default=32)
    p_eval.add_argument("--output-dir", default="results/eval")
    p_eval.add_argument("--data-dir", default="data/eval")
    p_eval.add_argument("--experiment-id", default="")
    p_eval.add_argument("--stage", default="", choices=["", "pretrain", "sft", "grpo"])
    p_eval.add_argument("--wandb", action="store_true")
    p_eval.add_argument("--wandb-project", default="math-nano")
    p_eval.set_defaults(func=cmd_eval)

    # compare
    p_cmp = sub.add_parser("compare", help="Compare two checkpoints")
    p_cmp.add_argument("--checkpoint-a", required=True)
    p_cmp.add_argument("--checkpoint-b", required=True)
    p_cmp.add_argument("--depth", type=int, required=True)
    p_cmp.add_argument("--suite", choices=["small", "full"], default="small")
    p_cmp.add_argument("--datasets", nargs="+", default=None)
    p_cmp.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p_cmp.add_argument("--batch-size", type=int, default=32)
    p_cmp.add_argument("--data-dir", default="data/eval")
    p_cmp.add_argument("--output", default=None, help="Output JSON path")
    p_cmp.set_defaults(func=cmd_compare)

    # check-leakage
    p_leak = sub.add_parser("check-leakage", help="Check eval-vs-training data leakage")
    p_leak.add_argument("--eval-dir", default="data/eval")
    p_leak.add_argument("--train-dir", required=True)
    p_leak.add_argument("--output", default="results/leakage_report.json")
    p_leak.set_defaults(func=cmd_check_leakage)

    # compile (from model registry)
    p_compile = sub.add_parser("compile", help="Compile results from model registry")
    p_compile.set_defaults(func=cmd_compile)

    # compile-eval (from eval JSONs)
    p_comp_eval = sub.add_parser("compile-eval", help="Compile eval JSON files into CSV")
    p_comp_eval.add_argument("--results-dir", default="results/eval")
    p_comp_eval.add_argument("--output", default="results/compiled/full_results.csv")
    p_comp_eval.set_defaults(func=cmd_compile_eval)

    # plot
    p_plot = sub.add_parser("plot", help="Generate result plots")
    p_plot.add_argument("--data", required=True, help="Path to compiled CSV")
    p_plot.add_argument(
        "--plot",
        choices=["scaling_curve", "mixture_comparison", "cost_efficiency",
                 "rl_dynamics", "recipe_comparison"],
        default=None,
    )
    p_plot.add_argument("--all", action="store_true")
    p_plot.add_argument("--output", default="results/plots/plot.png")
    p_plot.add_argument("--output-dir", default="results/plots")
    p_plot.add_argument("--metric", default="pass_at_1_greedy")
    p_plot.add_argument("--dataset", default="gsm8k")
    p_plot.set_defaults(func=cmd_plot)

    # summarize
    p_summarize = sub.add_parser("summarize", help="Generate phase summary")
    p_summarize.add_argument("--phase", required=True, help="Phase number")
    p_summarize.set_defaults(func=cmd_summarize)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
