"""
Experiment runner for orchestrating experiments with multiple seeds.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import pandas as pd
import torch
from datasets import DatasetDict
from tqdm.auto import tqdm

# Handle imports for both Colab (src/ in path) and local (project root in path)
try:
    # Try absolute import (when project root is in path)
    from src.data_processing import (
        get_data_collator,
        load_and_preprocess_emotion,
        load_and_preprocess_imdb,
    )
    from src.evaluation import (
        compute_confidence_interval,
        evaluate_model_comprehensive,
        measure_inference_latency,
    )
    from src.model_utils import get_device, load_model, load_tokenizer
    from src.training import (
        create_training_args,
        get_gpu_memory_usage,
        measure_training_time,
        train_model,
    )
except ImportError:
    from src.data_processing import (
        get_data_collator,
        load_and_preprocess_emotion,
        load_and_preprocess_imdb,
    )
    from src.evaluation import (
        compute_confidence_interval,
        evaluate_model_comprehensive,
        measure_inference_latency,
    )
    from src.model_utils import get_device, load_model, load_tokenizer
    from src.training import (
        create_training_args,
        get_gpu_memory_usage,
        measure_training_time,
        train_model,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_single_experiment(
    model_name: str,
    dataset_name: str,
    padding_strategy: str,
    seed: int,
    config: Dict,
    output_dir: str = "./experiments",
) -> Dict:
    """
    Run a single experiment with specified parameters.

    Args:
        model_name: Name of the model to use
        dataset_name: Name of the dataset ("imdb" or "emotion")
        padding_strategy: "static" or "dynamic"
        seed: Random seed
        config: Configuration dictionary
        output_dir: Base output directory

    Returns:
        Dictionary with all experiment results
    """
    logger.info(
        f"Running experiment: {model_name} | {dataset_name} | {padding_strategy} | seed={seed}"
    )

    # Set all random seeds
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device()

    # Load tokenizer and model
    tokenizer = load_tokenizer(model_name)

    # Get dataset configuration
    if dataset_name == "imdb":
        num_labels = config["datasets"]["imdb"]["num_labels"]
        max_length = config["datasets"]["imdb"]["max_length"]
        dataset = load_and_preprocess_imdb(
            tokenizer=tokenizer,
            max_length=max_length,
            padding_strategy=padding_strategy,
            val_split=config["datasets"]["imdb"].get("val_split", 0.1),
            seed=seed,
        )
    elif dataset_name == "emotion":
        num_labels = config["datasets"]["emotion"]["num_labels"]
        max_length = config["datasets"]["emotion"]["max_length"]
        dataset = load_and_preprocess_emotion(
            tokenizer=tokenizer,
            max_length=max_length,
            padding_strategy=padding_strategy,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load model
    model = load_model(model_name, num_labels=num_labels, seed=seed)

    # Get data collator
    data_collator = get_data_collator(
        tokenizer=tokenizer,
        padding_strategy=padding_strategy,
        max_length=max_length,
    )

    # Create training arguments
    training_config = config["training"]
    experiment_output_dir = os.path.join(
        output_dir,
        f"{model_name.replace('/', '_')}_{dataset_name}_{padding_strategy}_seed{seed}",
    )
    os.makedirs(experiment_output_dir, exist_ok=True)

    logger.info(f"Training config: {training_config['num_epochs']} epochs, batch_size={training_config['batch_size']}")
    training_args = create_training_args(
        output_dir=experiment_output_dir,
        num_epochs=int(training_config["num_epochs"]),
        batch_size=int(training_config["batch_size"]),
        learning_rate=float(training_config["learning_rate"]),
        seed=int(seed),
        warmup_steps=int(training_config.get("warmup_steps", 500)),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
        fp16=bool(training_config.get("fp16", True)),
        gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 1)),
    )
    logger.info(f"TrainingArguments created with num_train_epochs={training_args.num_train_epochs}")

    # Create compute_metrics function
    try:
        from src.evaluation import compute_metrics
    except ImportError:
        from evaluation import compute_metrics

    compute_metrics_fn = lambda pred: compute_metrics(pred, num_labels=num_labels)

    # Get initial GPU memory
    initial_memory = get_gpu_memory_usage()

    # Train model
    trainer = train_model(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        training_args=training_args,
        compute_metrics_fn=compute_metrics_fn,
    )

    # Get training time
    training_time = measure_training_time(trainer)

    # Get peak GPU memory during training
    peak_memory = get_gpu_memory_usage()

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = evaluate_model_comprehensive(
        model=trainer.model,
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        batch_size=training_config["batch_size"],
        device=device,
    )

    # Measure inference latency
    logger.info("Measuring inference latency...")
    latency_results = measure_inference_latency(
        model=trainer.model,
        dataset=dataset["test"],
        data_collator=data_collator,
        num_samples=min(100, len(dataset["test"])),
        batch_size=training_config["batch_size"],
        device=device,
    )

    # Compile all results
    results = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "padding_strategy": padding_strategy,
        "seed": seed,
        "num_labels": num_labels,
        "max_length": max_length,
        **training_time,
        **test_results,
        **latency_results,
        "gpu_initial_memory_mb": initial_memory.get("allocated_mb", 0.0),
        "gpu_peak_memory_mb": peak_memory.get("max_allocated_mb", 0.0),
        "experiment_dir": experiment_output_dir,
    }

    # Clear GPU cache (CUDA or MPS)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()

    logger.info(f"Experiment completed: Accuracy = {test_results['accuracy']:.4f}")
    return results


def run_experiment_with_seeds(
    model_name: str,
    dataset_name: str,
    padding_strategy: str,
    seeds: List[int],
    config: Dict,
    output_dir: str = "./experiments",
) -> pd.DataFrame:
    """
    Run experiment with multiple seeds and aggregate results.

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        padding_strategy: Padding strategy
        seeds: List of random seeds
        config: Configuration dictionary
        output_dir: Output directory

    Returns:
        DataFrame with results from all seeds
    """
    all_results = []

    for seed in tqdm(seeds, desc=f"Running {model_name} on {dataset_name} ({padding_strategy})"):
        result = run_single_experiment(
            model_name=model_name,
            dataset_name=dataset_name,
            padding_strategy=padding_strategy,
            seed=seed,
            config=config,
            output_dir=output_dir,
        )
        all_results.append(result)

        # Save individual result
        result_file = os.path.join(
            output_dir,
            "logs",
            f"{model_name.replace('/', '_')}_{dataset_name}_{padding_strategy}_seed{seed}.json",
        )
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

    return pd.DataFrame(all_results)



def aggregate_results(results_df: pd.DataFrame) -> Dict:
    """
    Aggregate results across seeds.

    Args:
        results_df: DataFrame with results from multiple seeds

    Returns:
        Dictionary with aggregated statistics
    """
    if len(results_df) == 0:
        return {}

    # Metrics to aggregate
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_macro",
        "f1_micro",
        "f1",
        "total_time_minutes",
        "inference_time_per_sample_ms",
        "gpu_peak_memory_mb",
    ]

    aggregated = {}

    for metric in metrics:
        if metric in results_df.columns:
            values = results_df[metric].dropna().tolist()
            if values:
                mean, lower, upper = compute_confidence_interval(values)
                aggregated[f"{metric}_mean"] = mean
                aggregated[f"{metric}_std"] = results_df[metric].std()
                aggregated[f"{metric}_ci_lower"] = lower
                aggregated[f"{metric}_ci_upper"] = upper
                aggregated[f"{metric}_min"] = results_df[metric].min()
                aggregated[f"{metric}_max"] = results_df[metric].max()

    # Add metadata
    aggregated["num_seeds"] = len(results_df)
    aggregated["model_name"] = results_df["model_name"].iloc[0]
    aggregated["dataset_name"] = results_df["dataset_name"].iloc[0]
    aggregated["padding_strategy"] = results_df["padding_strategy"].iloc[0]

    return aggregated


def save_experiment_results(results: Dict, output_path: str):
    """
    Save experiment results to JSON file.

    Args:
        results: Results dictionary
        output_path: Path to save results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


# Wrapper functions for compatibility with notebook imports
def setup_output_directories(output_dir: str = "./results"):
    """
    Setup output directories for experiments.
    
    Args:
        output_dir: Base output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    logger.info(f"Output directories created at: {output_dir}")


def run_experiment_with_multiple_seeds(
    model_name: str,
    dataset_name: str,
    padding_strategy: str,
    seeds: List[int],
    config: Dict,
    output_dir: str = "./experiments",
) -> pd.DataFrame:
    """
    Alias for run_experiment_with_seeds for notebook compatibility.
    """
    return run_experiment_with_seeds(
        model_name=model_name,
        dataset_name=dataset_name,
        padding_strategy=padding_strategy,
        seeds=seeds,
        config=config,
        output_dir=output_dir,
    )


def create_comparison_table(results_df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    """
    Create a comparison table from results DataFrame.
    
    Args:
        results_df: DataFrame with experiment results
        output_path: Optional path to save the table
    
    Returns:
        Aggregated comparison DataFrame
    """
    if len(results_df) == 0:
        logger.warning("Empty results DataFrame")
        return pd.DataFrame()
    
    # Group by model and padding strategy, aggregate metrics
    grouped = results_df.groupby(["model_name", "padding_strategy"]).agg({
        "accuracy": ["mean", "std"],
        "f1_macro": ["mean", "std"],
        "total_time_minutes": ["mean", "std"],
        "gpu_peak_memory_mb": ["mean", "std"],
    }).round(4)
    
    if output_path:
        grouped.to_csv(output_path)
        logger.info(f"Comparison table saved to {output_path}")
    
    return grouped


def run_architecture_comparison(
    config: Dict = None,
    output_dir: str = "./results",
    dataset_name: str = None,
    padding_strategy: str = None,
    configs: Dict = None  # Alias for config for notebook compatibility
) -> pd.DataFrame:
    """
    Run architecture comparison experiments (BERT vs DistilBERT).
    
    Args:
        config: Configuration dictionary (or use configs parameter)
        output_dir: Output directory for results
        dataset_name: Dataset to use (defaults to "imdb" or from config)
        padding_strategy: Specific padding strategy to use (defaults to all from config)
        configs: Alias for config parameter (for notebook compatibility)
    
    Returns:
        DataFrame with all results
    """
    # Handle configs alias - prioritize configs if provided
    if configs is not None:
        config = configs
    elif config is None:
        raise ValueError("Either 'config' or 'configs' must be provided")
    
    try:
        from visualization import plot_model_comparison, plot_tradeoff_analysis
    except ImportError:
        try:
            from src.visualization import plot_model_comparison, plot_tradeoff_analysis
        except ImportError:
            logger.warning("Visualization module not available, skipping plots")
            plot_model_comparison = None
            plot_tradeoff_analysis = None
    
    logger.info("Running architecture comparison experiments...")
    
    all_results = []
    
    models = [m["name"] for m in config["models"]]
    padding_strategies = [padding_strategy] if padding_strategy else config["padding_strategies"]
    seeds = config["seeds"]
    dataset_name = dataset_name or "imdb"
    
    for model_name in models:
        for padding_strategy in tqdm(padding_strategies, desc=f"Padding Strategies for {model_name}"):
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
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    comparison_path = os.path.join(output_dir, "tables", "architecture_comparison.csv")
    combined_df.to_csv(comparison_path, index=False)
    logger.info(f"Architecture comparison results saved to {comparison_path}")
    
    # Generate visualizations if available
    if plot_model_comparison:
        try:
            os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
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
            if plot_tradeoff_analysis:
                plot_tradeoff_analysis(
                    combined_df,
                    x_metric="total_time_minutes",
                    y_metric="accuracy",
                    save_path=os.path.join(output_dir, "figures", "tradeoff_accuracy_vs_time.png"),
                )
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")
    
    return combined_df


def run_padding_comparison(config: Dict, output_dir: str = "./results", dataset_name: str = "imdb") -> pd.DataFrame:
    """
    Run padding strategy comparison experiments.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory for results
        dataset_name: Dataset to use ("imdb" or "emotion")
    
    Returns:
        DataFrame with all results
    """
    logger.info(f"Running padding strategy comparison experiments on {dataset_name}...")
    
    models = [m["name"] for m in config["models"]] if "models" in config else ["bert-base-uncased"]
    padding_strategies = config["padding_strategies"]
    seeds = config["seeds"]
    
    all_results = []
    
    for model_name in models:
        for padding_strategy in tqdm(padding_strategies, desc=f"Evaluating Padding Strategies on {dataset_name} for {model_name}"):
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
    
    # Save results - Append if file exists to support sequential cell execution
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    padding_path = os.path.join(output_dir, "tables", f"padding_comparison_{dataset_name}.csv")
    
    if os.path.exists(padding_path):
        try:
            old_df = pd.read_csv(padding_path)
            # Remove any existing rows with the exact same model_name and dataset to prevent duplicates if cell is re-run
            existing_models = combined_df['model_name'].unique()
            old_df = old_df[~old_df['model_name'].isin(existing_models)]
            combined_df = pd.concat([old_df, combined_df], ignore_index=True)
        except Exception as e:
            logger.warning(f"Could not read existing CSV to append: {e}")
            
    combined_df.to_csv(padding_path, index=False)
    logger.info(f"Padding comparison results saved to {padding_path}")
    
    return combined_df

