"""
Visualization module for creating plots and figures.
"""

import logging
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11


def plot_training_curves(
    trainer_logs: List[Dict],
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot training curves from trainer logs.

    Args:
        trainer_logs: List of log dictionaries from trainer
        save_path: Path to save figure
        show: Whether to display the plot
    """
    if not trainer_logs:
        logger.warning("No training logs provided")
        return

    # Extract metrics
    steps = [log.get("step", 0) for log in trainer_logs if "loss" in log]
    train_losses = [log["loss"] for log in trainer_logs if "loss" in log]
    eval_losses = [log.get("eval_loss", None) for log in trainer_logs if "eval_loss" in log]
    eval_accuracies = [
        log.get("eval_accuracy", None) for log in trainer_logs if "eval_accuracy" in log
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot losses
    axes[0].plot(steps, train_losses, label="Train Loss", marker="o", markersize=3)
    if any(eval_losses):
        eval_steps = [log.get("step", 0) for log in trainer_logs if "eval_loss" in log]
        axes[0].plot(eval_steps, eval_losses, label="Eval Loss", marker="s", markersize=3)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    if any(eval_accuracies):
        eval_steps = [log.get("step", 0) for log in trainer_logs if "eval_accuracy" in log]
        axes[1].plot(eval_steps, eval_accuracies, label="Eval Accuracy", marker="s", markersize=3)
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Training curves saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "accuracy",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot bar chart comparing models on a metric.

    Args:
        results_df: DataFrame with results
        metric: Metric to compare
        save_path: Path to save figure
        show: Whether to display the plot
    """
    if metric not in results_df.columns:
        logger.warning(f"Metric {metric} not found in results")
        return

    # Group by model and padding strategy
    if "model_name" in results_df.columns and "padding_strategy" in results_df.columns:
        grouped = results_df.groupby(["model_name", "padding_strategy"])[metric].agg(
            ["mean", "std"]
        ).reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(grouped))
        width = 0.35

        models = grouped["model_name"].unique()
        padding_strategies = grouped["padding_strategy"].unique()

        for i, padding in enumerate(padding_strategies):
            mask = grouped["padding_strategy"] == padding
            means = grouped[mask]["mean"]
            stds = grouped[mask]["std"]
            ax.bar(
                x[mask] + i * width,
                means,
                width,
                yerr=stds,
                label=f"{padding} padding",
                alpha=0.8,
            )

        ax.set_xlabel("Model")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Model Comparison: {metric.replace('_', ' ').title()}")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(grouped["model_name"].unique(), rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    else:
        # Simple bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        if "model_name" in results_df.columns:
            grouped = results_df.groupby("model_name")[metric].agg(["mean", "std"]).reset_index()
            ax.bar(grouped["model_name"], grouped["mean"], yerr=grouped["std"], alpha=0.8)
            ax.set_xticklabels(grouped["model_name"], rotation=45, ha="right")
        else:
            ax.bar(range(len(results_df)), results_df[metric], alpha=0.8)

        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} Comparison")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Model comparison plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_tradeoff_analysis(
    results_df: pd.DataFrame,
    x_metric: str = "total_time_minutes",
    y_metric: str = "accuracy",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot trade-off analysis between two metrics.

    Args:
        results_df: DataFrame with results
        x_metric: Metric for x-axis
        y_metric: Metric for y-axis
        save_path: Path to save figure
        show: Whether to display the plot
    """
    if x_metric not in results_df.columns or y_metric not in results_df.columns:
        logger.warning(f"Metrics not found: {x_metric}, {y_metric}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by model and padding
    if "model_name" in results_df.columns and "padding_strategy" in results_df.columns:
        for model in results_df["model_name"].unique():
            for padding in results_df["padding_strategy"].unique():
                mask = (results_df["model_name"] == model) & (
                    results_df["padding_strategy"] == padding
                )
                subset = results_df[mask]

                if len(subset) > 0:
                    x_mean = subset[x_metric].mean()
                    y_mean = subset[y_metric].mean()
                    x_std = subset[x_metric].std()
                    y_std = subset[y_metric].std()

                    ax.scatter(x_mean, y_mean, s=200, alpha=0.7, label=f"{model} ({padding})")
                    ax.errorbar(
                        x_mean,
                        y_mean,
                        xerr=x_std,
                        yerr=y_std,
                        alpha=0.3,
                        fmt="none",
                    )
    else:
        ax.scatter(results_df[x_metric], results_df[y_metric], s=100, alpha=0.7)

    ax.set_xlabel(x_metric.replace("_", " ").title())
    ax.set_ylabel(y_metric.replace("_", " ").title())
    ax.set_title(f"Trade-off: {y_metric.replace('_', ' ').title()} vs {x_metric.replace('_', ' ').title()}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Trade-off plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_length_distributions(
    imdb_lengths: List[int],
    emotion_lengths: List[int],
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot length distributions for datasets.

    Args:
        imdb_lengths: List of sequence lengths from IMDb
        emotion_lengths: List of sequence lengths from Emotion
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(imdb_lengths, bins=50, alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Sequence Length")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("IMDb Dataset - Sequence Length Distribution")
    axes[0].axvline(np.mean(imdb_lengths), color="r", linestyle="--", label=f"Mean: {np.mean(imdb_lengths):.1f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(emotion_lengths, bins=50, alpha=0.7, edgecolor="black", color="orange")
    axes[1].set_xlabel("Sequence Length")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Emotion Dataset - Sequence Length Distribution")
    axes[1].axvline(np.mean(emotion_lengths), color="r", linestyle="--", label=f"Mean: {np.mean(emotion_lengths):.1f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Length distribution plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_padding_waste(
    static_stats: Dict,
    dynamic_stats: Dict,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot padding waste analysis.

    Args:
        static_stats: Statistics for static padding
        dynamic_stats: Statistics for dynamic padding
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Mean Length", "Max Length", "Padding Waste %"]
    static_values = [
        static_stats.get("mean_length", 0),
        static_stats.get("max_length", 0),
        static_stats.get("padding_waste_percent", 0),
    ]
    dynamic_values = [
        dynamic_stats.get("mean_length", 0),
        dynamic_stats.get("max_length", 0),
        dynamic_stats.get("padding_waste_percent", 0),
    ]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width / 2, static_values, width, label="Static Padding", alpha=0.8)
    ax.bar(x + width / 2, dynamic_values, width, label="Dynamic Padding", alpha=0.8)

    ax.set_ylabel("Value")
    ax.set_title("Padding Strategy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Padding waste plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

