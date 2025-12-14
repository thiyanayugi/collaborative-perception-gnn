# Quick Start Guide

This guide will help you get started with CPML in minutes.

## Prerequisites

- Python 3.9 or higher
- ROS2 Humble (for data extraction from ROS bags)
- 8GB+ RAM recommended
- GPU optional but recommended for training

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/thiyanayugi/collaborative-perception-gnn.git
cd collaborative-perception-gnn
```

### 2. Set Up Environment

Choose one of the following methods:

**Using pip:**

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

**Using conda:**

```bash
conda env create -f environment.yml
conda activate cpml
```

**Using Docker:**

```bash
docker-compose up -d
docker exec -it cpml_container bash
```

## Basic Workflow

### Step 1: Prepare Your Data

If you have ROS2 bag files with radar data:

```python
from cpml.preprocessing import radar_extraction

radar_extraction.process_rosbag(
    bag_path="/path/to/your/rosbag",
    robot_id="ep03",
    output_dir="data/01_extracted"
)
```

### Step 2: Process the Data

Run the complete preprocessing pipeline:

```bash
# Synchronize multi-robot data
python -m cpml.preprocessing.synchronization \
    --robot1 data/01_extracted/robot1_radar_points.csv \
    --robot2 data/01_extracted/robot2_radar_points.csv \
    --output data/02_synchronized/

# Transform coordinates
python -m cpml.preprocessing.coordinate_transform \
    --input data/02_synchronized/ \
    --output data/03_transformed/

# Clean point clouds
python -m cpml.preprocessing.point_cloud_cleaner \
    --input data/03_transformed/ \
    --output data/04_cleaned/

# Label data
python -m cpml.preprocessing.point_cloud_labeler \
    --input data/04_cleaned/ \
    --output data/05_labelled/

# Convert to graphs
python -m cpml.preprocessing.graph_converter \
    --input data/05_labelled/ \
    --output data/graph_frames/
```

### Step 3: Train a Model

```python
from cpml.training import main

# Train with default configuration
main.main(config_path="configs/models/config_standard_gatv2_t3.yaml")
```

Or use the command-line interface:

```bash
cpml-train --config configs/models/config_standard_gatv2_t3.yaml
```

### Step 4: Evaluate Results

```python
from cpml.training import evaluate

# Evaluate trained model
evaluate.evaluate_model(
    model_path="models/checkpoints_standard_gatv2_t3/best_model.pth",
    test_data_dir="data/graph_frames/test/temporal_3/"
)
```

## Using Pre-trained Models

If you want to use our pre-trained models:

```python
from cpml.training import model

# Load pre-trained model
gnn_model = model.load_pretrained("models/checkpoints_standard_gatv2_t3/")

# Make predictions
predictions = gnn_model.predict(test_data)
```

## Visualization

Explore your data interactively:

```python
from cpml.visualization import point_cloud_explorer

# Launch interactive explorer
point_cloud_explorer.launch_explorer(
    data_path="data/05_labelled/annotated_dataset.csv"
)
```

## Next Steps

- Read the [Architecture Documentation](architecture.md) to understand the system design
- Check out [examples/](../examples/) for more detailed usage examples
- See the [API Reference](api_reference.md) for detailed function documentation

## Troubleshooting

**Issue: CUDA out of memory**

- Reduce batch size in config file
- Use CPU instead: set `device: "cpu"` in config

**Issue: ROS2 bag not found**

- Ensure ROS2 Humble is installed
- Check bag file path is correct
- Verify bag contains radar topics

**Issue: Import errors**

- Ensure package is installed: `pip install -e .`
- Activate virtual environment
- Check Python version >= 3.9

For more help, see [CONTRIBUTING.md](../CONTRIBUTING.md) or open an issue.
