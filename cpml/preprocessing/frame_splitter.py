#!/usr/bin/env python3
"""
Advanced Dataset Splitter for Collaborative Perception Training
# TODO: Add error handling

This module implements a sophisticated dataset splitting system for collaborative
perception GNN training data. It creates balanced train/validation/test splits
while maintaining temporal integrity and ensuring proper representation across
different warehouse layouts and collaborative scenarios.

Key Features:
- Temporal integrity preservation across splits
- Layout-balanced distribution for robust training
- Scenario-aware splitting for comprehensive coverage
- Multi-temporal window support (1, 3, 5 frame sequences)
- Stratified sampling for balanced collaborative patterns
- Comprehensive logging and validation of split quality

The splitter is essential for creating high-quality training datasets that
enable robust collaborative perception model development and evaluation.

Author: Thiyanayugi Mariraj
Project: Development of a Framework for Collaborative Perception Management Layer
         for Future 6G-Enabled Robotic Systems
"""

import os
import shutil
import numpy as np
import pandas as pd
from collections import defaultdict
import random
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple


def create_collaborative_perception_dataset_splits(
    main_dataset_directory: str = "07_gnn_frames_COLLABORATIVE_causal",
    layout_variants_directory: str = "07_gnn_frames_COLLABORATIVE_causal_layout23",
    output_splits_directory: str = "08_gnn_splits_COLLABORATIVE",
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> bool:
    """
    Create balanced train/validation/test splits for collaborative perception training.

    This function implements sophisticated dataset splitting that maintains temporal
    integrity while ensuring balanced representation across different warehouse
    layouts and collaborative perception scenarios.

    Args:
        main_dataset_directory (str): Path to main collaborative perception dataset
        layout_variants_directory (str): Path to layout variant datasets
        output_splits_directory (str): Directory to save split datasets
        train_ratio (float): Proportion of data for training (default: 0.7)
        validation_ratio (float): Proportion of data for validation (default: 0.15)
        test_ratio (float): Proportion of data for testing (default: 0.15)

    Returns:
        bool: True if splits were created successfully, False otherwise
    """
    # Set up comprehensive logging for splitting process
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("CollaborativePerceptionSplitter")

    logger.info("Creating collaborative perception dataset splits")
    logger.info(f"Split ratios - Train: {train_ratio}, Val: {validation_ratio}, Test: {test_ratio}")

    # Validate split ratios to ensure they sum to exactly 1.0
    # This prevents data leakage and ensures complete dataset coverage
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
        logger.error("Split ratios must sum to 1.0")
        return False

    # Create organized output directory structure for different temporal windows
    temporal_window_configurations = ['temporal_1', 'temporal_3', 'temporal_5']
    split_categories = ['train', 'val', 'test']

    for split_category in split_categories:
        for temporal_config in temporal_window_configurations:
            split_directory = os.path.join(output_splits_directory, split_category, temporal_config)
            os.makedirs(split_directory, exist_ok=True)
            logger.debug(f"Created directory: {split_directory}")

    # Collect all available collaborative perception datasets
    collaborative_datasets = []

    # Process main dataset scenarios
    if os.path.exists(main_dataset_directory):
        logger.info(f"Processing main dataset directory: {main_dataset_directory}")
        for scenario_directory in os.listdir(main_dataset_directory):
            scenario_path = os.path.join(main_dataset_directory, scenario_directory)
            if os.path.isdir(scenario_path):
                collaborative_datasets.append({
                    'scenario_name': scenario_directory,
                    'layout_type': 'main_warehouse',
                    'dataset_path': scenario_path
                })
                logger.debug(f"Added main scenario: {scenario_directory}")

    # Process layout variant datasets for comprehensive coverage
    layout2_directory = os.path.join(layout_variants_directory, "layout2")
    if os.path.exists(layout2_directory):
        logger.info(f"Processing layout2 variants: {layout2_directory}")
        for dataset_directory in os.listdir(layout2_directory):
            dataset_path = os.path.join(layout2_directory, dataset_directory)
            if os.path.isdir(dataset_path):
                collaborative_datasets.append({
                    'scenario_name': f"layout2_{dataset_directory}",
                    'layout_type': 'layout2_variant',
                    'dataset_path': dataset_path
                })
                logger.debug(f"Added layout2 variant: {dataset_directory}")
    
    # Process layout3 variants for additional diversity
    layout3_directory = os.path.join(layout_variants_directory, "layout3")
    if os.path.exists(layout3_directory):
        logger.info(f"Processing layout3 variants: {layout3_directory}")
        for dataset_directory in os.listdir(layout3_directory):
            dataset_path = os.path.join(layout3_directory, dataset_directory)
            if os.path.isdir(dataset_path):
                collaborative_datasets.append({
                    'scenario_name': f"layout3_{dataset_directory}",
                    'layout_type': 'layout3_variant',
                    'dataset_path': dataset_path
                })
                logger.debug(f"Added layout3 variant: {dataset_directory}")

    logger.info(f"Found {len(collaborative_datasets)} total collaborative perception datasets")
    
    # Group datasets by layout type for balanced splitting
    layout_type_groups = defaultdict(list)
    for dataset in collaborative_datasets:
        layout_type_groups[dataset['layout_type']].append(dataset)
    
    print("Datasets by layout:")
    for layout, dsets in layout_groups.items():
        print(f"  {layout}: {len(dsets)} datasets")
    
    # Create balanced splits
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Split each layout proportionally
    for layout, dsets in layout_groups.items():
        random.shuffle(dsets)
        n_total = len(dsets)
        n_train = int(0.70 * n_total)
        n_val = int(0.15 * n_total)
        n_test = n_total - n_train - n_val
        
        train_datasets.extend(dsets[:n_train])
        val_datasets.extend(dsets[n_train:n_train+n_val])
        test_datasets.extend(dsets[n_train+n_val:])
        
        print(f"{layout}: {n_train} train, {n_val} val, {n_test} test")
    
    # Copy files to split directories
    def copy_temporal_data(datasets, split_name):
        total_frames = 0
        for dataset in datasets:
            for temporal in ['temporal_1', 'temporal_3', 'temporal_5']:
                src_dir = os.path.join(dataset['path'], temporal)
                dst_dir = os.path.join(output_path, split_name, temporal)
                
                if os.path.exists(src_dir):
                    # Create dataset subdirectory in split
                    dataset_dst = os.path.join(dst_dir, dataset['name'])
                    os.makedirs(dataset_dst, exist_ok=True)
                    
                    # Copy all .pt files
                    frame_count = 0
                    for file in os.listdir(src_dir):
                        if file.endswith('.pt'):
                            shutil.copy2(
                                os.path.join(src_dir, file),
                                os.path.join(dataset_dst, file)
                            )
                            frame_count += 1
                    
                    total_frames += frame_count
                    print(f"  {dataset['name']}/{temporal}: {frame_count} frames")
        
        return total_frames
    
    print("\nCopying train datasets...")
    train_frames = copy_temporal_data(train_datasets, 'train')
    
    print("\nCopying validation datasets...")
    val_frames = copy_temporal_data(val_datasets, 'val')
    
    print("\nCopying test datasets...")
    test_frames = copy_temporal_data(test_datasets, 'test')
    
    # Generate split statistics
    stats = {
        'total_datasets': len(datasets),
        'train_datasets': len(train_datasets),
        'val_datasets': len(val_datasets), 
        'test_datasets': len(test_datasets),
        'train_frames': train_frames,
        'val_frames': val_frames,
        'test_frames': test_frames,
        'total_frames': train_frames + val_frames + test_frames
    }
    
    # Save statistics
    with open(f"{output_path}/split_statistics.txt", "w") as f:
        f.write("COLLABORATIVE PERCEPTION DATASET SPLITS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Datasets: {stats['total_datasets']}\n")
        f.write(f"Train Datasets: {stats['train_datasets']} ({stats['train_datasets']/stats['total_datasets']*100:.1f}%)\n")
        f.write(f"Val Datasets: {stats['val_datasets']} ({stats['val_datasets']/stats['total_datasets']*100:.1f}%)\n")
        f.write(f"Test Datasets: {stats['test_datasets']} ({stats['test_datasets']/stats['total_datasets']*100:.1f}%)\n\n")
        f.write(f"Total Frames: {stats['total_frames']}\n")
        f.write(f"Train Frames: {stats['train_frames']} ({stats['train_frames']/stats['total_frames']*100:.1f}%)\n")
        f.write(f"Val Frames: {stats['val_frames']} ({stats['val_frames']/stats['total_frames']*100:.1f}%)\n")
        f.write(f"Test Frames: {stats['test_frames']} ({stats['test_frames']/stats['total_frames']*100:.1f}%)\n\n")
        
        f.write("TRAIN DATASETS:\n")
        for d in train_datasets:
            f.write(f"  {d['name']} ({d['layout']})\n")
        f.write("\nVALIDATION DATASETS:\n")  
        for d in val_datasets:
            f.write(f"  {d['name']} ({d['layout']})\n")
        f.write("\nTEST DATASETS:\n")
        for d in test_datasets:
            f.write(f"  {d['name']} ({d['layout']})\n")
    
    print(f"\n✅ SPLITTING COMPLETE!")
    print(f"Train: {stats['train_datasets']} datasets, {stats['train_frames']} frames")
    print(f"Val: {stats['val_datasets']} datasets, {stats['val_frames']} frames") 
    print(f"Test: {stats['test_datasets']} datasets, {stats['test_frames']} frames")
    print(f"Total: {stats['total_datasets']} datasets, {stats['total_frames']} frames")
    
    return stats

if __name__ == "__main__":
    stats = create_collaborative_perception_dataset_splits()
