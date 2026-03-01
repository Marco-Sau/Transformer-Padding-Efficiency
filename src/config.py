"""
Configuration loading utilities.
"""

import os
import yaml
from typing import Dict


def get_default_configs(config_path: str = None) -> Dict:
    """
    Load default configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
    
    Returns:
        Configuration dictionary with structure:
        {
            'training': {...},
            'models': [...],
            'datasets': {...},
            'seeds': [...],
            'padding_strategies': [...]
        }
    """
    if config_path is None:
        # Try to find config file relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # Go up from src/ to project root
        config_path = os.path.join(project_root, "configs", "base_config.yaml")
        
        # If not found, try common locations
        if not os.path.exists(config_path):
            possible_paths = [
                os.path.join(os.getcwd(), "configs", "base_config.yaml"),
                os.path.join("/content/drive/MyDrive/llm_efficiency_study", "configs", "base_config.yaml"),
                "configs/base_config.yaml",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Please specify the correct path to configs/base_config.yaml"
        )
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def print_config_summary(config: Dict):
    """
    Print a summary of the configuration in a readable format.
    
    Args:
        config: Configuration dictionary
    """
    print("Configuration:")
    print(f"  Models: {[m['name'] for m in config.get('models', [])]}")
    print(f"  Seeds: {config.get('seeds', [])}")
    print(f"  Padding strategies: {config.get('padding_strategies', [])}")
    print(f"  Epochs: {config.get('training', {}).get('num_epochs', 'N/A')}")
    print(f"  Batch size: {config.get('training', {}).get('batch_size', 'N/A')}")
    print(f"  Learning rate: {config.get('training', {}).get('learning_rate', 'N/A')}")

