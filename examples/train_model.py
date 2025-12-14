# Example: Training a GNN Model

This example demonstrates how to train a Graph Neural Network model for collaborative perception.

## Prerequisites

- CPML installed (`pip install -e .`)
- Dataset prepared in `data/graph_frames/`
- Configuration file ready

## Step 1: Prepare Configuration

Create or modify a configuration file in `configs/models/`:

```yaml
# configs/models/my_config.yaml

# Data Configuration
data:
  data_dir: "data/graph_frames/"
  batch_size: 32
  num_workers: 4
  temporal_windows: [3]

# Model Configuration
model:
  name: "OccupancyGNN"
  input_dim: 16
  hidden_dim: 64
  output_dim: 1
  num_layers: 3
  dropout: 0.2
  gnn_type: "gatv2"
  attention_heads: 4

# Training Configuration
training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  device: "cuda"  # or "cpu"
  checkpoint_dir: "models/my_experiment/"
```

## Step 2: Train the Model

### Using Python API

```python
from cpml.training import main, model
import yaml

# Load configuration
with open("configs/models/my_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create and train model
gnn_model = model.create_model(config)
main.train_model(gnn_model, config)
```

### Using Command Line

```bash
cpml-train --config configs/models/my_config.yaml
```

## Step 3: Monitor Training

Training progress will be logged to the console and saved to checkpoint directory:

```
Epoch 1/100: Loss=0.4523, Acc=0.7821
Epoch 2/100: Loss=0.3891, Acc=0.8234
...
Best model saved to models/my_experiment/best_model.pth
```

## Step 4: Evaluate the Model

```python
from cpml.training import evaluate

# Evaluate on test set
results = evaluate.evaluate_model(
    model_path="models/my_experiment/best_model.pth",
    test_data_dir="data/graph_frames/test/temporal_3/",
    config_path="configs/models/my_config.yaml"
)

print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Test F1-Score: {results['f1']:.4f}")
```

## Step 5: Visualize Results

```python
from cpml.training import evaluate
import matplotlib.pyplot as plt

# Generate confusion matrix
evaluate.plot_confusion_matrix(
    model_path="models/my_experiment/best_model.pth",
    test_data_dir="data/graph_frames/test/temporal_3/",
    save_path="results/confusion_matrix.png"
)

# Plot training curves
evaluate.plot_training_curves(
    checkpoint_dir="models/my_experiment/",
    save_path="results/training_curves.png"
)
```

## Advanced: Hyperparameter Tuning

```python
from cpml.training import main
import itertools

# Define hyperparameter grid
learning_rates = [0.001, 0.0001]
hidden_dims = [32, 64, 128]
num_layers = [2, 3, 4]

# Grid search
best_acc = 0
best_params = None

for lr, hidden, layers in itertools.product(learning_rates, hidden_dims, num_layers):
    config['training']['learning_rate'] = lr
    config['model']['hidden_dim'] = hidden
    config['model']['num_layers'] = layers
    
    print(f"Training with lr={lr}, hidden={hidden}, layers={layers}")
    
    model = create_model(config)
    results = main.train_model(model, config)
    
    if results['val_accuracy'] > best_acc:
        best_acc = results['val_accuracy']
        best_params = (lr, hidden, layers)

print(f"Best parameters: lr={best_params[0]}, hidden={best_params[1]}, layers={best_params[2]}")
print(f"Best validation accuracy: {best_acc:.4f}")
```

## Complete Training Script

```python
#!/usr/bin/env python3
"""
Complete training script for CPML GNN models
"""

import argparse
import yaml
from pathlib import Path
from cpml.training import main, model, evaluate

def train_and_evaluate(config_path: str):
    """Train and evaluate a GNN model"""
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"Training with configuration: {config_path}")
    print(f"Model: {config['model']['gnn_type']}")
    print(f"Temporal window: {config['data']['temporal_windows'][0]}")
    
    # Create model
    gnn_model = model.create_model(config)
    print(f"Model parameters: {gnn_model.count_parameters():,}")
    
    # Train model
    print("\nStarting training...")
    train_results = main.train_model(gnn_model, config)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = evaluate.evaluate_model(
        model_path=Path(config['training']['checkpoint_dir']) / "best_model.pth",
        test_data_dir=f"{config['data']['data_dir']}/test/temporal_{config['data']['temporal_windows'][0]}/",
        config_path=config_path
    )
    
    # Print results
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Best Validation Accuracy: {train_results['best_val_acc']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test Precision: {test_results['precision']:.4f}")
    print(f"Test Recall: {test_results['recall']:.4f}")
    print(f"Test F1-Score: {test_results['f1']:.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CPML GNN model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    train_and_evaluate(args.config)
```

Save this as `examples/train_model.py` and run:

```bash
python examples/train_model.py --config configs/models/my_config.yaml
```

## Next Steps

- Try different model architectures (GATv2, ECC)
- Experiment with temporal windows (1, 3, 5)
- Perform ablation studies
- Fine-tune on your own data
