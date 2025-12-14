"""
CPML Training Module

Graph Neural Network training pipeline including:
- Model architectures (GATv2, ECC)
- Training loops and optimization
- Evaluation metrics
- Data loading utilities
- Ablation studies
"""

from cpml.training import (
    model,
    train,
    evaluate,
    data_loader,
    utils,
    ablation,
    class_weights,
)

__all__ = [
    "model",
    "train",
    "evaluate",
    "data_loader",
    "utils",
    "ablation",
    "class_weights",
]
