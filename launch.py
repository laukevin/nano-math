"""launch.py — single entry point for all math-nano experiments.

Usage:
    uv run python launch.py run --config configs/experiments/sft-m-concise.yaml
    uv run python launch.py smoke-test --depth 10
    uv run python launch.py status
    uv run python launch.py gate --check pretrain_to_sft
    uv run python launch.py eval --checkpoint $CKPT --depth 16 --suite small
    uv run python launch.py compare --checkpoint-a $A --checkpoint-b $B --depth 16
    uv run python launch.py check-leakage --train-dir data/tokenized
    uv run python launch.py compile --results-dir results/eval/
    uv run python launch.py plot --data results/compiled/full_results.csv --all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Existing commands (stubs)
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> None:
    """Run an experiment."""
    print("[launch] run: not yet implemented")
    print(f"  config={args.config}, stage={args.stage}, depth={args.depth}")


def cmd_smoke_test(args: argparse.Namespace) -> None:
    """Run smoke tests."""
    print("[launch] smoke-test: not yet implemented")
    print(f"  depth={args.depth}, device={args.device}")


def cmd_status(args: argparse.Namespace) -> None:
    """Show experiment status."""
    print("[launch] status: not yet implemented")


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


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


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

    # Resolve datasets
    datasets = args.datasets or SUITE_DATASETS[args.suite]

    # Warn about full eval on CPU
    resolved_device = resolve_device(args.device)
    if resolved_device == "cpu" and args.suite == "full":
        logger.warning(
            "Running full eval suite on CPU will be slow. "
            "Consider --suite small for faster iteration."
        )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load eval data
    logger.info("Loading eval datasets: %s", datasets)
    eval_data: dict[str, list[dict]] = {}
    for ds_name in datasets:
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
        args.depth, model_params, device,
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

    # Build output
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


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Check leakage
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------


def cmd_compile(args: argparse.Namespace) -> None:
    """Compile eval results into CSV."""
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


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="math-nano experiment launcher")
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run an experiment")
    p_run.add_argument("--config", help="YAML config file path")
    p_run.add_argument("--stage", choices=["pretrain", "sft", "grpo"])
    p_run.add_argument("--depth", type=int)
    p_run.add_argument("--experiment", help="Experiment ID")
    p_run.set_defaults(func=cmd_run)

    # smoke-test
    p_smoke = sub.add_parser("smoke-test", help="Run smoke tests")
    p_smoke.add_argument("--depth", type=int, default=10)
    p_smoke.add_argument("--device", default="cpu")
    p_smoke.add_argument("--all", action="store_true")
    p_smoke.set_defaults(func=cmd_smoke_test)

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

    # compile
    p_comp = sub.add_parser("compile", help="Compile eval results into CSV")
    p_comp.add_argument("--results-dir", default="results/eval")
    p_comp.add_argument("--output", default="results/compiled/full_results.csv")
    p_comp.set_defaults(func=cmd_compile)

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

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
