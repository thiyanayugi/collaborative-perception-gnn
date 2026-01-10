"""
Training module for GNN-based collaborative perception models.

This module contains model architectures, training pipelines, evaluation tools,
and utilities for training Graph Neural Networks on collaborative perception data.
"""

from cpml.training import (
    model,
    train,
    evaluate,
    data_loader,
    utils,
    class_weights,
    ablation
)

__all__ = [
    'model',
    'train',
    'evaluate',
    'data_loader',
    'utils',
    'class_weights',
    'ablation'
]
