"""Aggregate all eval JSON files into a single CSV + summary tables.

Usage:
    python scripts/results/compile.py \
      --results-dir results/eval/ \
      --output results/compiled/full_results.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_eval_jsons(results_dir: Path) -> list[dict]:
    """Load all eval JSON files from a directory."""
    jsons = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            data["_source_file"] = str(path)
            jsons.append(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Skipping %s: %s", path, e)
    return jsons


def flatten_results(eval_jsons: list[dict]) -> pd.DataFrame:
    """Flatten eval JSONs into one row per (experiment, dataset) pair.

    Schema matches spec 10:
        experiment_id, model_depth, model_params, stage, dataset,
        pass_at_1_greedy, pass_at_1_sampled, pass_at_4_sampled,
        pass_at_8_sampled, n_problems, n_samples, temperature, timestamp
    """
    rows = []
    for ej in eval_jsons:
        base = {
            "experiment_id": ej.get("experiment_id", ""),
            "model_depth": ej.get("model_depth"),
            "model_params": ej.get("model_params"),
            "stage": ej.get("stage", ""),
            "eval_suite": ej.get("eval_suite", ""),
            "n_samples_per_problem": ej.get("n_samples_per_problem"),
            "temperature": ej.get("temperature"),
            "timestamp": ej.get("timestamp", ""),
            "checkpoint": ej.get("checkpoint", ""),
        }

        results = ej.get("results", {})
        for ds_name, ds_result in results.items():
            row = {**base, "dataset": ds_name}

            # Extract all metrics present
            for key in [
                "n_problems",
                "pass_at_1_greedy",
                "pass_at_1_sampled",
                "pass_at_4_sampled",
                "pass_at_8_sampled",
                "extraction_failures",
                "extraction_failure_rate",
                "avg_output_tokens",
                "avg_inference_ms",
            ]:
                if key in ds_result:
                    row[key] = ds_result[key]

            # Extract CI bounds as separate columns
            for ci_key in [
                "pass_at_1_greedy_ci95",
                "pass_at_1_sampled_ci95",
                "pass_at_4_sampled_ci95",
                "pass_at_8_sampled_ci95",
            ]:
                if ci_key in ds_result:
                    ci = ds_result[ci_key]
                    row[f"{ci_key}_low"] = ci[0]
                    row[f"{ci_key}_high"] = ci[1]

            rows.append(row)

    return pd.DataFrame(rows)


def generate_scaling_csv(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate scaling_curve.csv: params vs pass@1 by stage."""
    if df.empty:
        return

    metric_col = (
        "pass_at_1_greedy" if "pass_at_1_greedy" in df.columns else None
    )
    if metric_col is None:
        return

    cols = ["model_params", "model_depth", "stage", "dataset", metric_col]
    cols = [c for c in cols if c in df.columns]
    scaling = df[cols].dropna(subset=metric_col)

    if not scaling.empty:
        path = output_dir / "scaling_curve.csv"
        scaling.to_csv(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(scaling))


def generate_mixture_csv(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate mixture_comparison.csv if mixture data is available."""
    if "pretrain_mixture" not in df.columns or df.empty:
        return

    metric_col = "pass_at_1_greedy"
    if metric_col not in df.columns:
        return

    cols = [
        "pretrain_mixture",
        "model_params",
        "model_depth",
        "dataset",
        metric_col,
    ]
    cols = [c for c in cols if c in df.columns]
    mix = df[cols].dropna(subset=metric_col)

    if not mix.empty:
        path = output_dir / "mixture_comparison.csv"
        mix.to_csv(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(mix))


def compile_results(
    results_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Main compilation: load JSONs, flatten, write CSV + summaries."""
    eval_jsons = load_eval_jsons(results_dir)
    if not eval_jsons:
        logger.warning("No eval JSON files found in %s", results_dir)
        return pd.DataFrame()

    logger.info("Loaded %d eval JSON files", len(eval_jsons))

    df = flatten_results(eval_jsons)
    if df.empty:
        logger.warning("No results to compile")
        return df

    # Write full results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %s (%d rows)", output_path, len(df))

    # Generate summary CSVs
    output_dir = output_path.parent
    generate_scaling_csv(df, output_dir)
    generate_mixture_csv(df, output_dir)

    # Print summary
    print(f"\nCompiled {len(eval_jsons)} eval files -> {len(df)} rows")
    print(f"\nDatasets: {sorted(df['dataset'].unique())}")
    if "stage" in df.columns:
        print(f"Stages: {sorted(df['stage'].unique())}")
    if "model_depth" in df.columns:
        print(f"Depths: {sorted(df['model_depth'].dropna().unique())}")

    # Quick stats
    if "pass_at_1_greedy" in df.columns:
        print("\npass@1 (greedy) by dataset:")
        summary = (
            df.groupby("dataset")["pass_at_1_greedy"]
            .agg(["mean", "min", "max", "count"])
        )
        print(summary.to_string())

    return df


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Compile eval results into CSV"
    )
    parser.add_argument(
        "--results-dir",
        default="results/eval",
        help="Directory containing eval JSON files",
    )
    parser.add_argument(
        "--output",
        default="results/compiled/full_results.csv",
        help="Output CSV path",
    )

    args = parser.parse_args(argv)
    compile_results(Path(args.results_dir), Path(args.output))


if __name__ == "__main__":
    main()
