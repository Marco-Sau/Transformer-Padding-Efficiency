"""
Evaluation module for computing metrics and statistical tests.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred, num_labels: int = 2) -> Dict[str, float]:
    """
    Compute metrics for use with Trainer API.

    Args:
        eval_pred: Tuple of (predictions, labels) from Trainer
        num_labels: Number of classification labels

    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred

    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_micro = f1_score(labels, predictions, average="micro")

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
    }

    # For binary classification, add binary F1
    if num_labels == 2:
        f1_binary = f1_score(labels, predictions, average="binary")
        metrics["f1"] = f1_binary

    # For multi-class, add per-class metrics
    if num_labels > 2:
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        for i in range(num_labels):
            metrics[f"f1_class_{i}"] = float(f1[i])
            metrics[f"precision_class_{i}"] = float(precision[i])
            metrics[f"recall_class_{i}"] = float(recall[i])

    return metrics


def get_device() -> torch.device:
    """Get the appropriate device (CUDA/MPS/CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_model_comprehensive(
    model: PreTrainedModel,
    eval_dataset: Dataset,
    data_collator: callable,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Comprehensive evaluation of a model.

    Args:
        model: Model to evaluate
        eval_dataset: Evaluation dataset
        data_collator: Data collator for batching
        tokenizer: Tokenizer instance
        batch_size: Batch size for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with accuracy, F1, inference time, and memory usage
    """
    if device is None:
        device = get_device()

    model.eval()
    model.to(device)

    # Clear GPU cache (CUDA and MPS)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        # MPS cache clearing (available in newer PyTorch versions)
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

    all_predictions = []
    all_labels = []

    # Measure inference time
    inference_times = []
    start_time = time.time()

    with torch.no_grad():
        for i in tqdm(range(0, len(eval_dataset), batch_size), desc="Evaluating Batches"):
            # HuggingFace dataset slices return dict-of-lists; convert to list-of-dicts for collator
            batch_dict = eval_dataset[i : i + batch_size]
            n = len(batch_dict[list(batch_dict.keys())[0]])
            batch_list = [{k: batch_dict[k][j] for k in batch_dict} for j in range(n)]
            batch = data_collator(batch_list)

            # Separate labels before moving to device (don't pass to model to avoid internal loss)
            labels_tensor = batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            batch_start = time.time()
            outputs = model(**batch)
            batch_end = time.time()

            inference_times.append(batch_end - batch_start)

            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            if labels_tensor is not None:
                all_labels.extend(labels_tensor.numpy())

    total_time = time.time() - start_time

    # Compute metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_predictions, average="macro", zero_division=0)
    f1_micro = f1_score(all_labels, all_predictions, average="micro")

    num_labels = len(np.unique(all_labels))
    if num_labels == 2:
        precision_bin, recall_bin, f1_binary, _ = precision_recall_fscore_support(all_labels, all_predictions, average="binary", zero_division=0)
    else:
        f1_binary = None
        precision_bin = None
        recall_bin = None

    # Get memory usage
    memory_stats = {}
    if torch.cuda.is_available():
        memory_stats = {
            "gpu_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "gpu_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "gpu_max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }
    elif device.type == "mps":
        # MPS memory tracking (limited compared to CUDA)
        if hasattr(torch.mps, 'current_allocated_memory'):
            memory_stats = {
                "gpu_allocated_mb": torch.mps.current_allocated_memory() / 1024**2,
            }
        else:
            # Fallback for older PyTorch versions
            memory_stats = {
                "device_type": "mps",
                "note": "MPS memory tracking limited",
            }

    return {
        "accuracy": float(accuracy),
        "precision": float(precision_bin) if precision_bin is not None else float(precision_macro),
        "recall": float(recall_bin) if recall_bin is not None else float(recall_macro),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "f1": float(f1_binary) if f1_binary is not None else float(f1_macro),
        "inference_time_seconds": float(total_time),
        "inference_time_per_sample_ms": float(total_time / len(eval_dataset) * 1000),
        "num_samples": len(eval_dataset),
        **memory_stats,
    }


def measure_inference_latency(
    model: PreTrainedModel,
    dataset: Dataset,
    data_collator: callable,
    num_samples: int = 100,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Measure inference latency on a subset of data.

    Args:
        model: Model to evaluate
        dataset: Dataset to use
        data_collator: Data collator
        num_samples: Number of samples to test
        batch_size: Batch size
        device: Device to run on

    Returns:
        Dictionary with latency statistics
    """
    if device is None:
        device = get_device()

    model.eval()
    model.to(device)

    # Use subset of dataset
    subset = dataset.select(range(min(num_samples, len(dataset))))

    latencies = []

    with torch.no_grad():
        for i in tqdm(range(0, len(subset), batch_size), desc="Measuring Latency"):
            # HuggingFace dataset slices return dict-of-lists; convert to list-of-dicts for collator
            batch_dict = subset[i : i + batch_size]
            n = len(batch_dict[list(batch_dict.keys())[0]])
            batch_list = [{k: batch_dict[k][j] for k in batch_dict} for j in range(n)]
            batch = data_collator(batch_list)

            # Strip labels - not needed for latency measurement
            batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Warmup (skip first batch)
            if i == 0:
                _ = model(**batch)
                # Synchronize device
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elif device.type == "mps" and hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
                continue

            # Measure latency
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif device.type == "mps" and hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()

            start = time.time()
            _ = model(**batch)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif device.type == "mps" and hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()

            end = time.time()
            latencies.append((end - start) / len(batch) * 1000)  # ms per sample

    latencies = np.array(latencies)

    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
    }


def statistical_comparison(
    results_1: List[Dict],
    results_2: List[Dict],
    metric: str = "accuracy",
) -> Dict:
    """
    Perform statistical comparison between two sets of results.

    Args:
        results_1: List of result dictionaries from first condition
        results_2: List of result dictionaries from second condition
        metric: Metric name to compare

    Returns:
        Dictionary with statistical test results
    """
    values_1 = [r[metric] for r in results_1 if metric in r]
    values_2 = [r[metric] for r in results_2 if metric in r]

    if len(values_1) != len(values_2):
        logger.warning(
            f"Unequal sample sizes: {len(values_1)} vs {len(values_2)}. "
            "Using independent samples test."
        )
        # Independent samples t-test
        t_stat, p_value = stats.ttest_ind(values_1, values_2)
        test_type = "independent_ttest"
    else:
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(values_1, values_2)
        test_type = "paired_ttest"

    # Also perform Wilcoxon signed-rank test (non-parametric)
    if len(values_1) == len(values_2):
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(values_1, values_2)
        except ValueError:
            wilcoxon_stat, wilcoxon_p = None, None
    else:
        wilcoxon_stat, wilcoxon_p = None, None

    return {
        "metric": metric,
        "test_type": test_type,
        "mean_1": float(np.mean(values_1)),
        "mean_2": float(np.mean(values_2)),
        "std_1": float(np.std(values_1)),
        "std_2": float(np.std(values_2)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "wilcoxon_statistic": float(wilcoxon_stat) if wilcoxon_stat is not None else None,
        "wilcoxon_p_value": float(wilcoxon_p) if wilcoxon_p is not None else None,
    }


def compute_confidence_interval(
    values: List[float], confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for a set of values.

    Args:
        values: List of values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample standard deviation
    n = len(values)

    # t-distribution for small samples, normal for large
    if n < 30:
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin = t_critical * (std / np.sqrt(n))
    else:
        z_critical = stats.norm.ppf((1 + confidence) / 2)
        margin = z_critical * (std / np.sqrt(n))

    return (float(mean), float(mean - margin), float(mean + margin))

