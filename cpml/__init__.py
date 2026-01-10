"""
Collaborative Perception Management Layer (CPML).

A comprehensive framework for multi-robot collaborative perception using
Graph Neural Networks in warehouse environments.

Modules:
    - preprocessing: Data preprocessing pipeline for multi-robot sensor data
    - training: GNN model training and evaluation
    - visualization: Interactive visualization tools
"""

__version__ = '1.0.0'
__author__ = 'Thiyanayugi Mariraj'
__email__ = 'yugimariraj01@gmail.com'

# Package-level imports for convenience
from cpml import preprocessing, training, visualization

__all__ = ['preprocessing', 'training', 'visualization']
