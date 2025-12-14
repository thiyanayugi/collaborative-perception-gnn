#!/usr/bin/env python3

import os
import torch
import yaml
import numpy as np
from torch_geometric.data import Data, Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union
import glob
from tqdm import tqdm
import random

# Add safe globals for PyTorch serialization
torch.serialization.add_safe_globals([
    'torch_geometric.data.data.Data',
    'torch_geometric.data.data.DataEdgeAttr',
    'torch_geometric.data.data.DataNodeAttr',
    'torch_geometric.data.storage.EdgeStorage',
    'torch_geometric.data.storage.NodeStorage',
])


class OccupancyDataset(Dataset):
    """
    PyTorch Geometric dataset for occupancy prediction.

    This dataset loads .pt files from the specified directory and converts
    the 5-class labels to binary labels (occupied vs. unoccupied).
    """

    def __init__(
        self,
        root: str,
        split: str,
        temporal_window: int = 1,
        transform=None,
        pre_transform=None,
        binary_mapping: Dict[str, List[int]] = None,
        augmentation: Dict = None,
    ):
        """
        Initialize the dataset.

        Args:
            root: Root directory where the data is stored
            split: Data split ('train', 'val', or 'test')
            temporal_window: Size of temporal window (1, 3, or 5)
            transform: Transform to apply to each data sample
            pre_transform: Transform to apply to the entire dataset
            binary_mapping: Mapping from binary labels to original labels
            augmentation: Augmentation parameters
        """
        self.root = root
        self.split = split
        self.temporal_window = temporal_window
        self.binary_mapping = binary_mapping or {
            "occupied": [1, 2, 3, 4],  # Workstation, Robot, Boundary, KLT
            "unoccupied": [0],  # Unknown
        }
        self.augmentation = augmentation or {
            "rotation_angle": 15,  # degrees (±15°)
            "scaling_range": [0.9, 1.1],  # 90-110%
        }

        # Set up paths
        self.data_dir = os.path.join(root, split, f"temporal_{temporal_window}")

        # Get all .pt files
        self.file_paths = sorted(glob.glob(os.path.join(self.data_dir, "*.pt")))

        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.file_paths)

    def get(self, idx):
        """
        Get a data sample by index.

        Args:
            idx: Index of the data sample

        Returns:
            Data sample
        """
        try:
            # Load data with weights_only=False to handle PyTorch Geometric objects
            data = torch.load(self.file_paths[idx], weights_only=False)

            # Convert to binary labels
            y_binary = torch.zeros_like(data.y)
            for label in self.binary_mapping["occupied"]:
                y_binary[data.y == label] = 1

            # Update labels
            data.y = y_binary

            # Handle collaborative dataset with 16 features
            if data.x.size(1) == 16:  # Collaborative dataset format
                # Keep all 16 features for collaborative perception
                # Features 0-14: Original features + collaborative features (Robot 1/2 point counts)
                # Feature 15: Additional collaborative feature if present
                pass  # Keep all features as-is
            elif data.x.size(1) >= 13:  # Original format with 13+ features
                # Create new feature tensor with only X and Y related features
                new_x = torch.zeros((data.x.size(0), 9), dtype=data.x.dtype, device=data.x.device)

                # Copy normalized X and Y (0-1)
                new_x[:, 0:2] = data.x[:, 0:2]

                # Copy raw X and Y (2-3)
                new_x[:, 2:4] = data.x[:, 3:5]

                # Copy relative X and Y (4-5)
                new_x[:, 4:6] = data.x[:, 6:8]

                # Copy origin X and Y (6-7)
                new_x[:, 6:8] = data.x[:, 9:11]

                # Copy distance to center (8)
                new_x[:, 8] = data.x[:, 12]

                # Add temporal feature if needed
                if self.temporal_window > 1 and data.x.size(1) >= 14:
                    # Create tensor with one more column for temporal feature
                    temp_x = torch.zeros((data.x.size(0), 10), dtype=data.x.dtype, device=data.x.device)
                    temp_x[:, :9] = new_x
                    temp_x[:, 9] = data.x[:, 13]  # Copy temporal feature
                    new_x = temp_x

                # Replace original features with new ones
                data.x = new_x

            # Apply augmentation if in training mode
            if self.split == "train" and self.augmentation:
                data = self._augment_data(data)

            return data
        except Exception as e:
            print(f"Error loading file {self.file_paths[idx]}: {e}")
            # Return a simple empty data object as fallback
            # Determine feature dimension based on data directory
            feature_dim = 16 if "COLLABORATIVE" in self.root else (10 if self.temporal_window > 1 else 9)
            return Data(
                x=torch.zeros((1, feature_dim)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                y=torch.zeros(1, dtype=torch.long),
                pos=torch.zeros((1, 3))  # Keep 3D position for compatibility
            )

    def _augment_data(self, data: Data) -> Data:
        """
        Apply augmentation to the data.

        Args:
            data: Data sample

        Returns:
            Augmented data sample
        """
        # Random rotation around z-axis (±15°)
        if random.random() > 0.5:
            angle = np.random.uniform(
                -self.augmentation["rotation_angle"],
                self.augmentation["rotation_angle"]
            )
            angle_rad = np.deg2rad(angle)

            # Rotation matrix around z-axis
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)
            rotation_matrix = torch.tensor([
                [cos_angle, -sin_angle, 0],
                [sin_angle, cos_angle, 0],
                [0, 0, 1]
            ], dtype=torch.float)

            # Apply rotation to positions
            pos_rotated = torch.matmul(data.pos, rotation_matrix.T)
            data.pos = pos_rotated

            # Update position-related features in x
            if data.x.size(1) >= 9:  # Check if we have enough features
                # Update normalized position (0-1) - only X and Y
                pos_min = data.pos[:, :2].min(dim=0)[0]  # Only X and Y
                pos_max = data.pos[:, :2].max(dim=0)[0]  # Only X and Y
                pos_range = pos_max - pos_min
                pos_range[pos_range == 0] = 1  # Avoid division by zero
                data.x[:, 0:2] = (data.pos[:, :2] - pos_min) / pos_range

                # Update raw position (2-3) - only X and Y
                data.x[:, 2:4] = data.pos[:, :2]

                # Update position relative to center (4-5) - only X and Y
                center = data.pos[:, :2].mean(dim=0)
                data.x[:, 4:6] = data.pos[:, :2] - center

                # Update position relative to origin (6-7) - only X and Y
                data.x[:, 6:8] = data.pos[:, :2]

                # Update distance to center (8) - only X and Y distance
                data.x[:, 8] = torch.norm(data.pos[:, :2] - center, dim=1)

        # Random scaling (90-110%)
        if random.random() > 0.5:
            scale = np.random.uniform(
                self.augmentation["scaling_range"][0],
                self.augmentation["scaling_range"][1]
            )

            # Apply scaling to positions
            data.pos = data.pos * scale

            # Update position-related features in x
            if data.x.size(1) >= 9:  # Check if we have enough features
                # Update raw position (2-3) - only X and Y
                data.x[:, 2:4] = data.pos[:, :2]

                # Update position relative to center (4-5) - only X and Y
                center = data.pos[:, :2].mean(dim=0)
                data.x[:, 4:6] = data.pos[:, :2] - center

                # Update position relative to origin (6-7) - only X and Y
                data.x[:, 6:8] = data.pos[:, :2]

                # Update distance to center (8) - only X and Y distance
                data.x[:, 8] = torch.norm(data.pos[:, :2] - center, dim=1)

        return data


def create_data_loaders(config: Dict) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of data loaders
    """
    data_loaders = {}

    for temporal_window in config["data"]["temporal_windows"]:
        data_loaders[f"temporal_{temporal_window}"] = {}

        for split in ["train", "val", "test"]:
            # Create dataset
            dataset = OccupancyDataset(
                root=config["data"]["data_dir"],
                split=split,
                temporal_window=temporal_window,
                binary_mapping=config["data"]["binary_mapping"],
                augmentation=config["data"]["augmentation"] if split == "train" else None,
            )

            # Create data loader
            data_loaders[f"temporal_{temporal_window}"][split] = DataLoader(
                dataset,
                batch_size=config["data"]["batch_size"],
                shuffle=(split == "train"),
                num_workers=config["data"]["num_workers"],
                pin_memory=True,
            )

    return data_loaders


if __name__ == "__main__":
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create data loaders
    data_loaders = create_data_loaders(config)

    # Print dataset statistics
    for temporal_key, loaders in data_loaders.items():
        print(f"\n{temporal_key} statistics:")
        for split, loader in loaders.items():
            print(f"  {split}: {len(loader.dataset)} samples, {len(loader)} batches")

            # Get a sample batch
            for batch in loader:
                print(f"  Sample batch: {batch}")
                print(f"  Batch size: {batch.num_graphs}")
                print(f"  Features shape: {batch.x.shape}")
                print(f"  Labels shape: {batch.y.shape}")
                print(f"  Labels distribution: {torch.bincount(batch.y)}")
                break
