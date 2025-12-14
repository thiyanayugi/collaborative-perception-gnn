# Dataset Download Instructions

## Overview

The CPML dataset is **5.0 GB** and contains 59,541 graph neural network frames from multi-robot warehouse experiments. Due to GitHub's file size limitations, the dataset is **not included in this repository**.

## Dataset Statistics

- **Total Size**: 5.0 GB
- **Graph Frames**: 59,541
- **Layouts**: 3 warehouse configurations
- **Sessions**: 34 experimental runs
- **Train/Val/Test Split**: 72.1% / 14.8% / 13.1%

## Download Options

### Option 1: Google Drive (Recommended)

**Coming Soon**: The dataset will be made available via Google Drive.

```bash
# Download link will be provided here
# After downloading, extract to the data/ directory:
unzip cpml_dataset.zip -d data/
```

### Option 2: Zenodo (Academic Archive)

**Coming Soon**: The dataset will be archived on Zenodo with a DOI for academic citation.

### Option 3: Request from Author

If you need immediate access to the dataset, please contact:

- **Email**: yugimariraj01@gmail.com
- **GitHub**: [@thiyanayugi](https://github.com/thiyanayugi)

## Dataset Structure

Once downloaded, the data directory should have this structure:

```
data/
├── 01_extracted/          # Raw sensor data from ROS2 bags
├── 02_synchronized/       # Temporally aligned multi-robot data
├── 03_transformed/        # Global coordinate frame data
├── 04_cleaned/           # Filtered and noise-removed data
├── 05_labelled/          # Semantically annotated data
├── graph_frames/         # GNN-ready graph structures
│   ├── train/            # 42,936 training frames
│   ├── val/              # 8,802 validation frames
│   └── test/             # 7,803 test frames
└── README.md             # Dataset documentation
```

## Verification

After downloading, verify the dataset:

```bash
# Check directory structure
ls -la data/

# Verify frame counts
find data/graph_frames/train -name "*.pt" | wc -l  # Should be ~42,936
find data/graph_frames/val -name "*.pt" | wc -l    # Should be ~8,802
find data/graph_frames/test -name "*.pt" | wc -l   # Should be ~7,803
```

## Alternative: Generate Your Own Data

If you have ROS2 bag files with mmWave radar and Vicon data, you can generate the dataset using our preprocessing pipeline:

```bash
# 1. Extract radar data
python -m cpml.preprocessing.radar_extraction --bag your_rosbag/

# 2. Synchronize multi-robot data
python -m cpml.preprocessing.synchronization --input data/01_extracted/

# 3. Transform coordinates
python -m cpml.preprocessing.coordinate_transform --input data/02_synchronized/

# 4. Clean point clouds
python -m cpml.preprocessing.point_cloud_cleaner --input data/03_transformed/

# 5. Label data
python -m cpml.preprocessing.point_cloud_labeler --input data/04_cleaned/

# 6. Convert to graphs
python -m cpml.preprocessing.graph_converter --input data/05_labelled/
```

See [docs/quickstart.md](docs/quickstart.md) for detailed instructions.

## Dataset License

The dataset is released under the same MIT License as the code. If you use this dataset in your research, please cite:

```bibtex
@mastersthesis{Mariraj2025CPML,
  author       = {Thiyanayugi Mariraj},
  title        = {Development of a Framework for Collaborative Perception
                  Management Layer for Future 6G-Enabled Robotic Systems},
  school       = {Your University Name},
  year         = {2025},
  type         = {Master's Thesis},
  note         = {Student ID: 241940}
}
```

## Support

For questions about the dataset:

- Open an issue: [GitHub Issues](https://github.com/thiyanayugi/collaborative-perception-gnn/issues)
- Email: your.email@example.com

---

**Note**: Once you obtain the dataset, place it in the `data/` directory at the root of this repository. The training scripts expect this structure.
