"""
Data processing module for loading and preprocessing datasets.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StaticPaddingCollator:
    """Data collator that pads all sequences to a fixed max_length."""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 256):
        """
        Initialize static padding collator.

        Args:
            tokenizer: Tokenizer instance
            max_length: Fixed length to pad all sequences to
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Pad features to fixed max_length.

        Args:
            features: List of tokenized examples

        Returns:
            Batch with padded tensors
        """
        batch = {}
        for key in features[0].keys():
            if key in ["input_ids", "attention_mask", "token_type_ids"]:
                # Pad sequences
                sequences = [f[key] for f in features]
                padded = []
                for seq in sequences:
                    if len(seq) < self.max_length:
                        pad_length = self.max_length - len(seq)
                        if key == "input_ids":
                            pad_value = self.tokenizer.pad_token_id
                        elif key == "attention_mask":
                            pad_value = 0
                        else:  # token_type_ids
                            pad_value = 0
                        padded_seq = seq + [pad_value] * pad_length
                    elif len(seq) > self.max_length:
                        padded_seq = seq[: self.max_length]
                    else:
                        padded_seq = seq
                    padded.append(padded_seq)
                batch[key] = torch.tensor(padded)
            else:
                # Labels or other non-sequence fields
                batch[key] = torch.tensor([f[key] for f in features])
        return batch


def load_and_preprocess_imdb(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    padding_strategy: str = "dynamic",
    val_split: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    Load and preprocess IMDb dataset.

    Args:
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        padding_strategy: "static" or "dynamic"
        val_split: Fraction of training data to use for validation
        seed: Random seed for splitting

    Returns:
        DatasetDict with train, validation, and test splits
    """
    logger.info("Loading IMDb dataset...")
    dataset = load_dataset("imdb")

    # Create validation split from training data
    if val_split > 0:
        train_val_split = dataset["train"].train_test_split(
            test_size=val_split, seed=seed
        )
        dataset["train"] = train_val_split["train"]
        dataset["validation"] = train_val_split["test"]
    else:
        # Use test set as validation if no split specified
        dataset["validation"] = dataset["test"]

    # Tokenize function
    def tokenize_function(examples: Dict) -> Dict:
        """Tokenize text examples."""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length if padding_strategy == "static" else None,
            padding=False,  # We'll pad in collator
        )

    logger.info("Tokenizing IMDb dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    # Rename label -> labels (HuggingFace Trainer requires plural form)
    for split in tokenized_dataset:
        if "label" in tokenized_dataset[split].column_names and "labels" not in tokenized_dataset[split].column_names:
            tokenized_dataset[split] = tokenized_dataset[split].rename_column("label", "labels")

    logger.info(f"IMDb dataset loaded: {len(tokenized_dataset['train'])} train, "
                f"{len(tokenized_dataset['validation'])} val, "
                f"{len(tokenized_dataset['test'])} test")

    return tokenized_dataset


def load_and_preprocess_emotion(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128,
    padding_strategy: str = "dynamic",
    seed: int = 42,
) -> DatasetDict:
    """
    Load and preprocess Emotion dataset.

    Args:
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        padding_strategy: "static" or "dynamic"
        seed: Random seed (for reproducibility)

    Returns:
        DatasetDict with train, validation, and test splits
    """
    logger.info("Loading Emotion dataset...")
    dataset = load_dataset("emotion")

    # Emotion dataset already has train, validation, test splits
    # Tokenize function
    def tokenize_function(examples: Dict) -> Dict:
        """Tokenize text examples."""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length if padding_strategy == "static" else None,
            padding=False,  # We'll pad in collator
        )

    logger.info("Tokenizing Emotion dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    # Rename label -> labels (HuggingFace Trainer requires plural form)
    for split in tokenized_dataset:
        if "label" in tokenized_dataset[split].column_names and "labels" not in tokenized_dataset[split].column_names:
            tokenized_dataset[split] = tokenized_dataset[split].rename_column("label", "labels")

    logger.info(f"Emotion dataset loaded: {len(tokenized_dataset['train'])} train, "
                f"{len(tokenized_dataset['validation'])} val, "
                f"{len(tokenized_dataset['test'])} test")

    return tokenized_dataset


def get_data_collator(
    tokenizer: PreTrainedTokenizer,
    padding_strategy: str,
    max_length: int,
):
    """
    Get appropriate data collator based on padding strategy.

    Args:
        tokenizer: Tokenizer instance
        padding_strategy: "static" or "dynamic"
        max_length: Maximum sequence length (used for static padding)

    Returns:
        Data collator instance
    """
    if padding_strategy == "static":
        return StaticPaddingCollator(tokenizer, max_length=max_length)
    else:
        return DataCollatorWithPadding(tokenizer=tokenizer)


def compute_length_statistics(dataset: Union[Dataset, DatasetDict]) -> Dict:
    """
    Compute length statistics for a dataset.

    Args:
        dataset: Dataset or DatasetDict to analyze

    Returns:
        Dictionary with length statistics
    """
    if isinstance(dataset, DatasetDict):
        stats = {}
        for split_name, split_data in dataset.items():
            stats[split_name] = _compute_split_length_stats(split_data)
        return stats
    else:
        return _compute_split_length_stats(dataset)


def _compute_split_length_stats(dataset: Dataset) -> Dict:
    """Compute length statistics for a single dataset split."""
    lengths = [len(example["input_ids"]) for example in dataset]

    stats = {
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "median": float(np.median(lengths)),
        "percentile_25": float(np.percentile(lengths, 25)),
        "percentile_75": float(np.percentile(lengths, 75)),
        "percentile_90": float(np.percentile(lengths, 90)),
        "percentile_95": float(np.percentile(lengths, 95)),
        "percentile_99": float(np.percentile(lengths, 99)),
    }

    return stats

