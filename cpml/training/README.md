# Training Module

This module contains GNN model architectures and training utilities for collaborative perception.

## Components

- `model.py`: GNN model architectures (GATv2, ECC, GraphSAGE)
- `train.py`: Training pipeline and optimization
- `evaluate.py`: Model evaluation and metrics
- `data_loader.py`: Data loading utilities
- `utils.py`: Training helper functions
- `class_weights.py`: Class balancing utilities
- `ablation.py`: Ablation study framework

## Supported Models

- **GATv2**: Graph Attention Networks v2
- **ECC**: Edge-Conditioned Convolution
- **GraphSAGE**: Sampling and Aggregating

## Usage

Train a model using the configuration files in `configs/models/`.
