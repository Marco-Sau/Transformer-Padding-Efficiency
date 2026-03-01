"""
Model utilities for loading and configuring transformer models.
"""

import logging
from typing import Dict

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """
    Load tokenizer for a given model.

    Args:
        model_name: Name of the pretrained model

    Returns:
        Tokenizer instance
    """
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def load_model(
    model_name: str,
    num_labels: int,
    seed: int = 42,
) -> PreTrainedModel:
    """
    Load pretrained model for sequence classification.

    Args:
        model_name: Name of the pretrained model
        num_labels: Number of classification labels
        seed: Random seed for reproducibility

    Returns:
        Model instance
    """
    logger.info(f"Loading model: {model_name} with {num_labels} labels")
    
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    logger.info(f"Model loaded: {count_parameters(model)}")
    return model


def count_parameters(model: PreTrainedModel) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: Model instance

    Returns:
        Dictionary with total, trainable, and non-trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
    }


def get_model_size_mb(model: PreTrainedModel) -> float:
    """
    Calculate model size in megabytes.

    Args:
        model: Model instance

    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def get_device() -> torch.device:
    """
    Get the appropriate device (GPU/MPS/CPU).
    
    Priority:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU)
    3. CPU (fallback)

    Returns:
        torch.device instance
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        logger.info("Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")
    
    return device


def get_device_info() -> Dict:
    """
    Get detailed device information.
    
    Returns:
        Dictionary with device info
    """
    info = {"device_type": "cpu", "device_name": "CPU"}
    
    if torch.cuda.is_available():
        info = {
            "device_type": "cuda",
            "device_name": torch.cuda.get_device_name(0),
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        info = {
            "device_type": "mps",
            "device_name": "Apple Silicon GPU",
            # MPS doesn't expose memory info directly
        }
    
    return info

