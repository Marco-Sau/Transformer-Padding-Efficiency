"""
Training module using Hugging Face Trainer API.
"""

import logging
import time
from typing import Callable, Dict, Optional

import torch
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device_type() -> str:
    """Get the current device type (cuda, mps, or cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def create_training_args(
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    seed: int = 42,
    **kwargs,
) -> TrainingArguments:
    """
    Create TrainingArguments with sensible defaults.
    
    Automatically adapts for:
    - CUDA (NVIDIA GPU): fp16 enabled
    - MPS (Apple Silicon): No mixed precision (not fully supported)
    - CPU: No mixed precision

    Args:
        output_dir: Directory to save checkpoints and logs
        num_epochs: Number of training epochs
        batch_size: Training batch size per device
        learning_rate: Learning rate
        seed: Random seed
        **kwargs: Additional arguments to override defaults

    Returns:
        TrainingArguments instance
    """
    device_type = get_device_type()
    
    # Extract and validate optional kwargs
    gradient_accumulation_steps = kwargs.pop("gradient_accumulation_steps", 1)
    warmup_steps_override = kwargs.pop("warmup_steps", 500)
    weight_decay_override = kwargs.pop("weight_decay", 0.01)
    
    defaults = {
        "output_dir": output_dir,
        "num_train_epochs": int(num_epochs),
        "per_device_train_batch_size": int(batch_size),
        "per_device_eval_batch_size": int(batch_size),  # Keep same as train to prevent OOM with max_length=512
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay_override),
        "warmup_steps": int(warmup_steps_override),
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_accuracy",
        "greater_is_better": True,
        "save_total_limit": 2,  # Keep only last 2 checkpoints
        "logging_steps": 100,
        "logging_dir": f"{output_dir}/logs",
        "report_to": "none",  # Disable wandb/tensorboard by default
        "seed": int(seed),
        "dataloader_num_workers": 0,  # Set to 0 to avoid compatibility issues
        "dataloader_pin_memory": False,  # Disable pin_memory when num_workers=0
        "optim": "adamw_torch",
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "eval_accumulation_steps": 1,
    }
    
    # Device-specific settings
    if device_type == "cuda":
        # NVIDIA GPU: Enable FP16 mixed precision
        defaults["fp16"] = True
        defaults["bf16"] = False
        logger.info("CUDA detected: Enabling FP16 mixed precision training")
    elif device_type == "mps":
        # Apple Silicon: Disable mixed precision (not fully supported on MPS)
        defaults["fp16"] = False
        defaults["bf16"] = False
        # Note: use_mps_device is deprecated in newer transformers versions
        # MPS is now automatically detected
        logger.info("MPS detected: Using Apple Silicon GPU (mixed precision disabled)")
    else:
        # CPU: No mixed precision
        defaults["fp16"] = False
        defaults["bf16"] = False
        logger.info("CPU mode: Mixed precision disabled")

    # Override with any provided kwargs
    defaults.update(kwargs)
    
    # Safety check: Ensure fp16 is disabled if no CUDA
    if defaults.get("fp16", False) and device_type != "cuda":
        logger.warning("FP16 requested but CUDA not available. Disabling FP16.")
        defaults["fp16"] = False

    return TrainingArguments(**defaults)


def train_model(
    model: PreTrainedModel,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    data_collator: Callable,
    tokenizer: PreTrainedTokenizer,
    training_args: TrainingArguments,
    compute_metrics_fn: Optional[Callable] = None,
) -> Trainer:
    """
    Train a model using the Trainer API.

    Args:
        model: Model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator for batching
        tokenizer: Tokenizer instance
        training_args: Training arguments
        compute_metrics_fn: Optional function to compute metrics

    Returns:
        Trained Trainer instance
    """
    logger.info("Initializing Trainer...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )


    logger.info("Starting training...")
    train_start_time = time.time()

    trainer.train()

    train_end_time = time.time()
    total_time = train_end_time - train_start_time
    logger.info(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    return trainer


def measure_training_time(trainer: Trainer) -> Dict[str, float]:
    """
    Measure training time from trainer logs.

    Args:
        trainer: Trained Trainer instance

    Returns:
        Dictionary with timing information
    """
    if not hasattr(trainer, "state") or not trainer.state.log_history:
        return {"total_time_minutes": 0.0, "epochs": 0}

    log_history = trainer.state.log_history

    # Extract epoch times
    epoch_times = []
    epoch_start = None

    for log_entry in log_history:
        if "epoch" in log_entry:
            if epoch_start is not None:
                epoch_time = log_entry.get("epoch", 0) - epoch_start
                epoch_times.append(epoch_time)
            epoch_start = log_entry.get("epoch", 0)

    # Get total training time
    total_time_seconds = 0.0
    for log_entry in log_history:
        if "train_runtime" in log_entry:
            total_time_seconds = log_entry["train_runtime"]
            break

    if total_time_seconds == 0 and epoch_times:
        # Estimate from epoch times
        total_time_seconds = sum(epoch_times) * 60  # Rough estimate

    return {
        "total_time_minutes": total_time_seconds / 60.0,
        "total_time_seconds": total_time_seconds,
        "num_epochs": len(epoch_times) if epoch_times else 0,
        "avg_epoch_time_minutes": (total_time_seconds / 60.0) / max(len(epoch_times), 1),
    }


def get_gpu_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage (CUDA or MPS).

    Returns:
        Dictionary with memory usage in MB
    """
    device_type = get_device_type()
    
    if device_type == "cuda":
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "device_type": "cuda",
        }
    elif device_type == "mps":
        # MPS memory tracking (available in PyTorch 2.1+)
        if hasattr(torch.mps, 'current_allocated_memory'):
            return {
                "allocated_mb": torch.mps.current_allocated_memory() / 1024**2,
                "device_type": "mps",
            }
        else:
            return {
                "allocated_mb": 0.0,
                "device_type": "mps",
                "note": "MPS memory tracking requires PyTorch 2.1+",
            }
    else:
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
            "device_type": "cpu",
        }

