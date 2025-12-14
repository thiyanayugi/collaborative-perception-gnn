#  Dataset - GNN based Collaborative Perception Management Layer (CPML)

## Overview

This dataset contains comprehensive robotics data collected from a multi-robot system across three different layouts. The data includes radar point clouds, motion capture data, and workspace configuration information for robotics research and development.

## Dataset Structure

```
data_full/
├── 01_extracted/          # Raw extracted sensor data
├── 02_synchronized/       # Time-synchronized multi-sensor data
├── 03_transformed/        # Coordinate-transformed data
├── 04_cleaned/           # Noise-filtered and cleaned data
├── 05_labelled/          # Annotated data for machine learning
├── graph_frames/          # Graph neural network frames for collaborative perception
│   ├── train/            # Training set (42,936 frames, 72.1%)
│   ├── val/              # Validation set (8,802 frames, 14.8%)
│   ├── test/             # Test set (7,803 frames, 13.1%)
│   ├── split_statistics.txt    # Human-readable split information
│   
└── README.md             # This file
```

## Processing Pipeline

The dataset follows a systematic 5-stage processing pipeline:

1. **Extraction (01_extracted)**: Raw sensor data extraction from ROS bags
2. **Synchronization (02_synchronized)**: Temporal alignment of multi-sensor streams
3. **Transformation (03_transformed)**: Coordinate system transformations and calibration
4. **Cleaning (04_cleaned)**: Noise filtering and boundary detection
5. **Labelling (05_labelled)**: Ground truth annotations for ML applications

## Layout Configurations

### Layout_01
- **Sessions**: 20 datasets (dataset_20250219_113400 to dataset_20250306_124800)
- **Robot Configurations**: Standard, Robot 1 Vertical, Robot 1 Horizontal, Robot 1 Diagonal
- **Movement Patterns**: Horizontal, Vertical, Diagonal trajectories
- **Anchor Stations**: AS_1, AS_3, AS_4, AS_5, AS_6

### Layout_02  
- **Sessions**: 9 datasets (dataset_144830 to dataset_162330)
- **Robot Configurations**: Standard configuration
- **Movement Patterns**: Horizontal, Vertical, Mixed patterns, Docking, Random
- **Anchor Stations**: AS_1, AS_3, AS_4, AS_5, AS_6

### Layout_03
- **Sessions**: 5 datasets (dataset_164720 to dataset_174030)  
- **Robot Configurations**: Standard configuration
- **Movement Patterns**: Horizontal, Vertical, Mixed patterns, Random
- **Anchor Stations**: AS_1, AS_3, AS_4, AS_5, AS_6 + KLT tracking system

## File Structure

### Extracted Data (01_extracted)
Each layout contains:
```
Layout_XX/
├── dataset_YYYYMMDD_HHMMSS/
│   ├── robot1_radar_points.csv      # Robot 1 radar point cloud data
│   ├── robot2_radar_points.csv      # Robot 2 radar point cloud data
│   └── vicon_motion_capture.csv     # Motion capture ground truth
└── layoutX_edges.csv                # Workspace boundary definitions
```

### Processed Data (02_synchronized through 05_labelled)
Each processing stage contains:
```
Layout_XX/
├── dataset_YYYYMMDD_HHMMSS_synchronized.csv    # Stage 2
├── transformed_dataset_YYYYMMDD_HHMMSS.csv     # Stage 3  
├── cleaned_dataset_YYYYMMDD_HHMMSS.csv         # Stage 4
└── annotated_dataset_YYYYMMDD_HHMMSS.csv       # Stage 5
```

## Data Formats

### Radar Point Cloud Data
- **Format**: CSV with columns [timestamp, x, y, z, intensity, ...]
- **Coordinate System**: Local robot frame
- **Frequency**: Variable (typically 10-20 Hz)

### Motion Capture Data  
- **Format**: CSV with columns [timestamp, x, y, z, qx, qy, qz, qw, ...]
- **Coordinate System**: Global Vicon frame
- **Frequency**: 100 Hz (downsampled as needed)

### Layout Edges
- **Format**: CSV with anchor station boundary coordinates
- **Columns**: AS_X_[bottom|right|top|left]_[x1|y1|x2|y2]
- **Purpose**: Workspace boundary definition for path planning

## Dataset Statistics

| Layout | Sessions  | Date Range | 
|--------|----------|------------|
| Layout_01 | 20  | Feb-Mar 2025 | 
| Layout_02 | 9  | May 2025 |
| Layout_03 | 5 | May 2025 |

## Graph Frames for Collaborative Perception

### Overview
The dataset includes **59,541 graph neural network frames** designed for collaborative perception research. These frames represent temporal multi-robot sensor data as graph structures, enabling advanced machine learning approaches for multi-agent systems.

### Graph Frame Structure
```
graph_frames/
├── train/                    # Training set (22 datasets, 42,936 frames)
│   ├── temporal_1/          # Highest temporal resolution (14,356 frames)
│   ├── temporal_3/          # Medium temporal resolution (14,312 frames)
│   └── temporal_5/          # Lower temporal resolution (14,268 frames)
├── val/                     # Validation set (5 datasets, 8,802 frames)
│   ├── temporal_1/          # Multi-scale validation frames
│   ├── temporal_3/          # Temporal validation data
│   └── temporal_5/          # Coarse temporal validation
├── test/                    # Test set (7 datasets, 7,803 frames)
│   ├── temporal_1/          # High-resolution test frames
│   ├── temporal_3/          # Medium-resolution test frames
│   └── temporal_5/          # Low-resolution test frames
├── split_statistics.txt     # Human-readable split information
└── split_statistics.json    # Machine-readable metadata
```

### Temporal Multi-Scale Design
- **temporal_1**: Highest temporal resolution for fine-grained analysis
- **temporal_3**: Medium temporal resolution for balanced performance
- **temporal_5**: Lower temporal resolution for computational efficiency

### Frame Format
- **File Format**: PyTorch tensors (`.pt` files)
- **Naming Convention**: `{timestamp}.pt` (Unix timestamp with microseconds)
- **Content**: Graph representations of multi-robot sensor data
- **Node Features**: Robot poses, sensor measurements, spatial relationships
- **Edge Features**: Inter-robot communication, spatial proximity, temporal connections

### Dataset Splits

#### Training Set (72.1% - 42,936 frames)
**Layout_01 (14 datasets):**
- dataset_20250219_174900, dataset_20250306_123000, dataset_20250221_111900
- dataset_20250221_124800, dataset_20250306_124600, dataset_20250221_121100
- dataset_20250306_120600, dataset_20250219_121000, dataset_20250221_122700
- dataset_20250306_115400, dataset_20250221_120000, dataset_20250306_111700
- dataset_20250221_104600, dataset_20250306_112000

**Layout_02 (6 datasets):**
- dataset_145530, dataset_162330, dataset_155450
- dataset_151420, dataset_144830, dataset_160600

**Layout_03 (2 datasets):**
- dataset_164720, dataset_173220

#### Validation Set (14.8% - 8,802 frames)
**Layout_01 (3 datasets):**
- dataset_20250306_124800, dataset_20250219_180400, dataset_20250219_113400

**Layout_02 (1 dataset):**
- dataset_154310

**Layout_03 (1 dataset):**
- dataset_174030

#### Test Set (13.1% - 7,803 frames)
**Layout_01 (3 datasets):**
- dataset_20250219_171800, dataset_20250219_120000, dataset_20250221_125900

**Layout_02 (2 datasets):**
- dataset_150340, dataset_161700

**Layout_03 (2 datasets):**
- dataset_170610, dataset_165730

