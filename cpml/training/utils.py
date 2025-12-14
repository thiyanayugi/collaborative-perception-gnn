#!/usr/bin/env python3

import os
import random
import logging
import yaml
import torch
import numpy as np
from typing import Dict, Any

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        
    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        PyTorch device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def create_directory(path: str):
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)

def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def print_model_summary(model: torch.nn.Module, input_shape: tuple = None):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (optional)
    """
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    if input_shape:
        print(f"Input shape: {input_shape}")
    
    print("\nModel architecture:")
    print(model)

def save_metrics(metrics: Dict[str, float], save_path: str):
    """
    Save metrics to a file.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save metrics
    """
    import json
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_metrics(load_path: str) -> Dict[str, float]:
    """
    Load metrics from a file.
    
    Args:
        load_path: Path to load metrics from
        
    Returns:
        Dictionary of metrics
    """
    import json
    
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

class EarlyStopping:
    """
    Early stopping utility class.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            score: Current score (loss or accuracy)
            
        Returns:
            True if training should be stopped
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        """Check if the score is an improvement."""
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

def get_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: Tensor of labels
        
    Returns:
        Tensor of class weights
    """
    unique_labels, counts = torch.unique(labels, return_counts=True)
    total_samples = len(labels)
    
    weights = total_samples / (len(unique_labels) * counts.float())
    
    return weights
