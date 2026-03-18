"""launch.py — single entry point for all math-nano experiments.

Usage:
    uv run python launch.py run --config configs/experiments/sft-m-concise.yaml
    uv run python launch.py smoke-test --depth 10
    uv run python launch.py status
    uv run python launch.py gate --check pretrain_to_sft
    uv run python launch.py eval --model sft-m-concise-best --suite full
"""

from __future__ import annotations

import argparse
import sys


def cmd_run(args: argparse.Namespace) -> None:
    """Run an experiment."""
    print(f"[launch] run: not yet implemented")
    print(f"  config={args.config}, stage={args.stage}, depth={args.depth}")


def cmd_smoke_test(args: argparse.Namespace) -> None:
    """Run smoke tests."""
    print(f"[launch] smoke-test: not yet implemented")
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


def cmd_eval(args: argparse.Namespace) -> None:
    """Run eval on a checkpoint."""
    print(f"[launch] eval: not yet implemented")
    print(f"  model={args.model}, suite={args.suite}")


def main() -> None:
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
    p_eval.add_argument("--model", required=True, help="Model ID or checkpoint path")
    p_eval.add_argument("--suite", default="small", choices=["small", "full"])
    p_eval.add_argument("--samples", type=int, default=1)
    p_eval.add_argument("--device", default="auto")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
