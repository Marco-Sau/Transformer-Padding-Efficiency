"""
Main entry point for running experiments.
"""

# ── Hotfix: inject missing clear_device_cache before peft tries to import it ──
# Colab ships an old accelerate that lacks this symbol; peft>=0.10 requires it.
# We add a safe no-op so the import chain doesn't crash.
try:
    from accelerate.utils.memory import clear_device_cache  # noqa: F401
except ImportError:
    import torch
    import accelerate.utils.memory as _accel_mem
    def _clear_device_cache(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _accel_mem.clear_device_cache = _clear_device_cache
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import yaml

from src.experiment_runner import (
    aggregate_results,
    run_experiment_with_seeds,
    save_experiment_results,
)
from src.visualization import (
    plot_length_distributions,
    plot_model_comparison,
    plot_tradeoff_analysis,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config



def run_architecture_comparison(config: dict, output_dir: str = "./results"):
    """Run architecture comparison experiments."""
    logger.info("Running architecture comparison experiments...")

    all_results = []

    models = [m["name"] for m in config["models"]]
    padding_strategies = config["padding_strategies"]
    seeds = config["seeds"]
    dataset_name = "imdb"  # Start with IMDb

    for model_name in models:
        for padding_strategy in padding_strategies:
            logger.info(f"Running: {model_name} | {padding_strategy}")

            results_df = run_experiment_with_seeds(
                model_name=model_name,
                dataset_name=dataset_name,
                padding_strategy=padding_strategy,
                seeds=seeds,
                config=config,
                output_dir=output_dir,
            )

            all_results.append(results_df)

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Save combined results
    comparison_path = os.path.join(output_dir, "tables", "architecture_comparison.csv")
    combined_df.to_csv(comparison_path, index=False)
    logger.info(f"Architecture comparison results saved to {comparison_path}")

    # Generate visualizations
    logger.info("Generating visualizations...")
    plot_model_comparison(
        combined_df,
        metric="accuracy",
        save_path=os.path.join(output_dir, "figures", "model_comparison_accuracy.png"),
    )
    plot_model_comparison(
        combined_df,
        metric="total_time_minutes",
        save_path=os.path.join(output_dir, "figures", "model_comparison_time.png"),
    )
    plot_tradeoff_analysis(
        combined_df,
        x_metric="total_time_minutes",
        y_metric="accuracy",
        save_path=os.path.join(output_dir, "figures", "tradeoff_accuracy_vs_time.png"),
    )

    return combined_df


def run_padding_comparison(config: dict, output_dir: str = "./results", dataset_name: str = "imdb"):
    """Run padding strategy comparison experiments."""
    logger.info(f"Running padding strategy comparison experiments on {dataset_name}...")

    model_name = "bert-base-uncased"
    padding_strategies = config["padding_strategies"]
    seeds = config["seeds"]

    all_results = []

    for padding_strategy in padding_strategies:
        results_df = run_experiment_with_seeds(
            model_name=model_name,
            dataset_name=dataset_name,
            padding_strategy=padding_strategy,
            seeds=seeds,
            config=config,
            output_dir=output_dir,
        )
        all_results.append(results_df)

    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Save results
    padding_path = os.path.join(output_dir, "tables", f"padding_comparison_{dataset_name}.csv")
    combined_df.to_csv(padding_path, index=False)
    logger.info(f"Padding comparison results saved to {padding_path}")

    return combined_df


def run_emotion_validation(config: dict, output_dir: str = "./results"):
    """Run Emotion dataset validation experiments (optional, if resources permit)."""
    logger.info("Running Emotion dataset validation experiments...")
    logger.info("Note: This validates findings on uniform-length text (short sequences)")
    
    return run_padding_comparison(config, output_dir, dataset_name="emotion")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run LLM training efficiency experiments")
    parser.add_argument(
        "--experiment",
        choices=["architecture", "padding", "emotion", "all"],
        default="architecture",
        help="Type of experiment to run (emotion is optional validation)",
    )
    parser.add_argument(
        "--dataset",
        choices=["imdb", "emotion", "both"],
        default="imdb",
        help="Dataset to use",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to use (filters models defined in config)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Random seeds to use",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/base_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "metrics"), exist_ok=True)

    # Load configuration
    config = load_config(args.config)

    # Override config with command-line arguments if provided
    if args.models:
        if "models" in config:
            config["models"] = [m for m in config["models"] if m["name"] in args.models]
        else:
            config["models"] = [{"name": m, "max_length": config["datasets"]["imdb"].get("max_length", 256)} for m in args.models]
    if args.seeds:
        config["seeds"] = args.seeds

    logger.info(f"Running experiment: {args.experiment}")
    logger.info(f"Output directory: {args.output_dir}")

    # Run experiments
    if args.experiment == "architecture":
        run_architecture_comparison(config, args.output_dir)
    elif args.experiment == "padding":
        run_padding_comparison(config, args.output_dir, dataset_name="imdb")
    elif args.experiment == "emotion":
        run_emotion_validation(config, args.output_dir)
    elif args.experiment == "all":
        logger.info("Running all primary experiments...")
        run_architecture_comparison(config, args.output_dir)
        run_padding_comparison(config, args.output_dir, dataset_name="imdb")
        logger.info("Primary experiments completed. Emotion validation is optional.")

    logger.info("All experiments completed!")


if __name__ == "__main__":
    main()

