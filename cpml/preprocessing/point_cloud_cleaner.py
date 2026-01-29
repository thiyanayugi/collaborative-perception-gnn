#!/usr/bin/env python3
"""
Advanced Point Cloud Cleaner for Collaborative Perception

This module implements a comprehensive point cloud cleaning and filtering system
for collaborative perception applications in warehouse robotics. It processes
mmWave radar point clouds to remove noise, outliers, and invalid measurements
while preserving high-quality data for collaborative perception training.

Key Features:
- Multi-stage filtering pipeline with configurable parameters
- Memory-efficient processing for large datasets
# TODO: Improve documentation
- Statistical outlier detection and removal
- Boundary-based spatial filtering for warehouse environments
- Signal-to-noise ratio (SNR) filtering for quality assurance
- Field-of-view (FOV) filtering for sensor-specific constraints
- Height-based filtering for ground-level object detection
- Comprehensive logging and progress tracking

The cleaner is essential for preparing high-quality point cloud data for
collaborative perception graph generation and GNN training.

Author: Thiyanayugi Mariraj
Project: Development of a Framework for Collaborative Perception Management Layer
         for Future 6G-Enabled Robotic Systems
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import gc
import math
import logging
from typing import Optional, Dict, List, Tuple
from sklearn.neighbors import NearestNeighbors
import psutil
import time


class CollaborativePerceptionPointCloudCleaner:
    """
    Advanced Point Cloud Cleaner for Collaborative Perception Systems.

    This class provides comprehensive point cloud cleaning and filtering
    capabilities specifically designed for collaborative perception applications.
    It implements a multi-stage filtering pipeline to remove noise, outliers,
    and invalid measurements while preserving high-quality sensor data.

    Key Filtering Stages:
    - Spatial boundary filtering for warehouse environment constraints
    - Signal quality filtering based on SNR thresholds
    - Height-based filtering for ground-level object detection
    - Field-of-view filtering for sensor-specific constraints
    - Statistical outlier detection and removal
    - Memory-efficient processing for large-scale datasets
    """

    def __init__(
        self,
        input_data_path: str,
        output_data_path: Optional[str] = None,
        cleaning_configuration: Optional[Dict] = None
    ):
        """
        Initialize the collaborative perception point cloud cleaner.

        Args:
            input_data_path (str): Path to input point cloud data file
            output_data_path (str, optional): Path for cleaned output data
            cleaning_configuration (dict, optional): Custom cleaning parameters
        """
        self.input_data_path = input_data_path
        self.output_data_path = output_data_path

        # Setup comprehensive logging for cleaning process
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("CollaborativePerceptionCleaner")

        # Define default cleaning configuration for warehouse environments
        self.default_cleaning_configuration = {
            'spatial_boundaries': {
                'x_coordinate_min': -10.0, 'x_coordinate_max': 10.5,
                'y_coordinate_min': -5.0, 'y_coordinate_max': 6.0,
                'z_coordinate_min': -1.5, 'z_coordinate_max': 4.0
            },
            'signal_quality': {
                'minimum_snr_threshold': 5.0,
                'filtering_enabled': True
            },
            'height_filtering': {
                'minimum_height_meters': 0.05,
                'maximum_height_meters': 2.5,
                'filtering_enabled': True
            },
            'field_of_view': {
                'sensor_angle_degrees': 120.0,
                'filtering_enabled': True
            },
            'statistical_outlier_detection': {
                'k_neighbors': 20,
                'std_deviation': 3.0,
                'enabled': True
            },
            'processing': {
                'chunk_size': 'auto',  # 'auto' or integer
                'memory_threshold_gb': 2.0,  # Switch to chunked processing above this
                'max_chunk_size': 100000,
                'min_chunk_size': 10000
            },
            'visualization': {
                'enabled': True,
                'dpi': 150,
                'figsize': (14, 10)
            }
        }
        
        # Update with user configuration
        if cleaning_configuration:
            self._update_config(cleaning_configuration)
        
        # Processing stats
        self.stats = {
            'original_points': 0,
            'final_points': 0,
            'processing_time': 0,
            'memory_usage_mb': 0,
            'filters_applied': []
        }
    
    def _update_config(self, user_config: Dict):
        """Recursively update configuration"""
        def update_dict(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    update_dict(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        update_dict(self.config, user_config)
    
    def _estimate_file_size_and_memory_needs(self) -> Tuple[int, float]:
        """Estimate file size and memory requirements"""
        try:
            file_size_mb = os.path.getsize(self.input_path) / (1024 * 1024)
            
            # Sample first 1000 rows to estimate structure
            sample = pd.read_csv(self.input_path, nrows=1000)
            estimated_rows = int((file_size_mb * 1024 * 1024) / (len(sample) * sample.memory_usage(deep=True).sum() / len(sample)))
            
            # Estimate memory needs (rough calculation)
            estimated_memory_gb = file_size_mb / 1024 * 2.5  # Factor for pandas overhead
            
            return estimated_rows, estimated_memory_gb
        
        except Exception as e:
            self.logger.warning(f"Could not estimate file size: {e}")
            return 50000, 1.0
    
    def _determine_processing_strategy(self) -> Dict:
        """Determine optimal processing strategy based on file size and available memory"""
        estimated_rows, estimated_memory_gb = self._estimate_file_size_and_memory_needs()
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        strategy = {
            'use_chunking': False,
            'chunk_size': estimated_rows,
            'estimated_memory_gb': estimated_memory_gb,
            'available_memory_gb': available_memory_gb
        }
        
        # Use chunking if file is large or memory is limited
        if (estimated_memory_gb > self.config['processing']['memory_threshold_gb'] or 
            estimated_memory_gb > available_memory_gb * 0.5):
            
            strategy['use_chunking'] = True
            
            # Calculate optimal chunk size
            if self.config['processing']['chunk_size'] == 'auto':
                # Target using ~25% of available memory per chunk
                target_memory_per_chunk = available_memory_gb * 0.25
                chunk_size = int((target_memory_per_chunk / estimated_memory_gb) * estimated_rows)
                
                # Clamp to reasonable bounds
                chunk_size = max(self.config['processing']['min_chunk_size'], 
                               min(chunk_size, self.config['processing']['max_chunk_size']))
            else:
                chunk_size = self.config['processing']['chunk_size']
            
            strategy['chunk_size'] = chunk_size
        
        self.logger.info(f"Processing strategy: {strategy}")
        return strategy
    
    def _apply_boundary_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply spatial boundary filters"""
        self.logger.info("Applying boundary filters...")
        filtered = data.copy()
        original_len = len(filtered)
        
        for robot_id in ['robot_1', 'robot_2']:
            x_col = f'{robot_id}_global_x_radar'
            y_col = f'{robot_id}_global_y_radar'
            z_col = f'{robot_id}_global_z_radar'
            
            if not all(col in filtered.columns for col in [x_col, y_col, z_col]):
                continue
            
            # Vectorized boundary filtering
            boundary_mask = (
                filtered[x_col].between(self.config['boundary']['x_min'], 
                                      self.config['boundary']['x_max']) &
                filtered[y_col].between(self.config['boundary']['y_min'], 
                                      self.config['boundary']['y_max']) &
                filtered[z_col].between(self.config['boundary']['z_min'], 
                                      self.config['boundary']['z_max'])
            )
            
            # Apply mask to all radar columns for this robot
            radar_cols = [col for col in filtered.columns 
                         if col.startswith(f'{robot_id}_') and 'radar' in col]
            
            for col in radar_cols:
                filtered.loc[~boundary_mask, col] = np.nan
        
        remaining_points = self._count_valid_points(filtered)
        self.logger.info(f"Boundary filter: {original_len} -> {remaining_points} points")
        return filtered
    
    def _apply_snr_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply SNR (Signal-to-Noise Ratio) filters"""
        if not self.config['snr']['enabled']:
            return data
            
        self.logger.info("Applying SNR filters...")
        filtered = data.copy()
        
        for robot_id in ['robot_1', 'robot_2']:
            snr_col = f'{robot_id}_global_snr_radar'
            
            if snr_col not in filtered.columns:
                continue
            
            # Create SNR mask
            snr_mask = filtered[snr_col] >= self.config['snr']['min_value']
            
            # Apply to all radar columns for this robot
            radar_cols = [col for col in filtered.columns 
                         if col.startswith(f'{robot_id}_') and 'radar' in col]
            
            for col in radar_cols:
                filtered.loc[~snr_mask, col] = np.nan
        
        remaining_points = self._count_valid_points(filtered)
        self.logger.info(f"SNR filter: remaining {remaining_points} points")
        return filtered
    
    def _apply_height_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply height-based filters"""
        if not self.config['height']['enabled']:
            return data
            
        self.logger.info("Applying height filters...")
        filtered = data.copy()
        
        for robot_id in ['robot_1', 'robot_2']:
            z_col = f'{robot_id}_global_z_radar'
            
            if z_col not in filtered.columns:
                continue
            
            # Height filter mask
            height_mask = filtered[z_col].between(
                self.config['height']['min_height'],
                self.config['height']['max_height']
            )
            
            # Apply to all radar columns for this robot
            radar_cols = [col for col in filtered.columns 
                         if col.startswith(f'{robot_id}_') and 'radar' in col]
            
            for col in radar_cols:
                filtered.loc[~height_mask, col] = np.nan
        
        remaining_points = self._count_valid_points(filtered)
        self.logger.info(f"Height filter: remaining {remaining_points} points")
        return filtered
    
    def _apply_fov_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Field-of-View filters with optimized vectorized calculation"""
        if not self.config['fov']['enabled']:
            return data
            
        self.logger.info("Applying FOV filters...")
        filtered = data.copy()
        half_fov_rad = math.radians(self.config['fov']['angle_degrees'] / 2)
        
        for robot_id in ['robot_1', 'robot_2']:
            # Required columns
            x_radar_col = f'{robot_id}_global_x_radar'
            y_radar_col = f'{robot_id}_global_y_radar'
            x_robot_col = f'{robot_id}_global_x'
            y_robot_col = f'{robot_id}_global_y'
            yaw_col = f'{robot_id}_yaw'
            
            required_cols = [x_radar_col, y_radar_col, x_robot_col, y_robot_col, yaw_col]
            
            if not all(col in filtered.columns for col in required_cols):
                continue
            
            # Get valid data mask (non-NaN)
            valid_mask = ~filtered[required_cols].isna().any(axis=1)
            
            if not valid_mask.any():
                continue
            
            # Vectorized FOV calculation for valid points
            valid_data = filtered[valid_mask]
            
            # Calculate relative positions
            dx = valid_data[x_radar_col] - valid_data[x_robot_col]
            dy = valid_data[y_radar_col] - valid_data[y_robot_col]
            
            # Calculate angles from robot to radar points
            point_angles = np.arctan2(dy, dx)
            
            # Calculate relative angles (difference from robot heading)
            relative_angles = point_angles - valid_data[yaw_col]
            
            # Normalize angles to [-π, π]
            relative_angles = np.mod(relative_angles + math.pi, 2 * math.pi) - math.pi
            
            # Check if within FOV
            within_fov = np.abs(relative_angles) <= half_fov_rad
            
            # Create full FOV mask
            fov_mask = pd.Series(False, index=filtered.index)
            fov_mask.loc[valid_mask] = within_fov
            fov_mask.loc[~valid_mask] = True  # Keep invalid points for other filters
            
            # Apply FOV mask to radar columns
            radar_cols = [col for col in filtered.columns 
                         if col.startswith(f'{robot_id}_') and 'radar' in col]
            
            for col in radar_cols:
                filtered.loc[~fov_mask, col] = np.nan
        
        remaining_points = self._count_valid_points(filtered)
        self.logger.info(f"FOV filter: remaining {remaining_points} points")
        return filtered
    
    def _apply_statistical_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply statistical outlier removal"""
        if not self.config['statistical_filter']['enabled']:
            return data
            
        self.logger.info("Applying statistical outlier removal...")
        filtered = data.copy()
        
        for robot_id in ['robot_1', 'robot_2']:
            x_col = f'{robot_id}_global_x_radar'
            y_col = f'{robot_id}_global_y_radar'
            z_col = f'{robot_id}_global_z_radar'
            
            if not all(col in filtered.columns for col in [x_col, y_col, z_col]):
                continue
            
            # Get valid points for this robot
            valid_mask = ~filtered[[x_col, y_col, z_col]].isna().any(axis=1)
            
            if valid_mask.sum() < self.config['statistical_filter']['k_neighbors'] + 1:
                self.logger.warning(f"Not enough points for {robot_id} statistical filtering")
                continue
            
            try:
                # Extract valid points
                valid_points = filtered.loc[valid_mask, [x_col, y_col, z_col]].values
                
                # Compute k-nearest neighbors
                k = min(self.config['statistical_filter']['k_neighbors'], len(valid_points) - 1)
                nn = NearestNeighbors(n_neighbors=k)
                nn.fit(valid_points)
                distances, _ = nn.kneighbors(valid_points)
                
                # Calculate mean distance to neighbors
                mean_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self (distance 0)
                
                # Calculate outlier threshold
                global_mean = np.mean(mean_distances)
                global_std = np.std(mean_distances)
                threshold = global_mean + self.config['statistical_filter']['std_deviation'] * global_std
                
                # Identify outliers
                outlier_mask_valid = mean_distances > threshold
                
                # Map back to full dataframe
                valid_indices = filtered.index[valid_mask]
                outlier_indices = valid_indices[outlier_mask_valid]
                
                # Remove outliers by setting to NaN
                radar_cols = [col for col in filtered.columns 
                             if col.startswith(f'{robot_id}_') and 'radar' in col]
                
                for col in radar_cols:
                    filtered.loc[outlier_indices, col] = np.nan
                
                outliers_removed = len(outlier_indices)
                self.logger.info(f"Statistical filter for {robot_id}: removed {outliers_removed} outliers")
                
            except Exception as e:
                self.logger.error(f"Statistical filtering failed for {robot_id}: {e}")
        
        remaining_points = self._count_valid_points(filtered)
        self.logger.info(f"Statistical filter: remaining {remaining_points} points")
        return filtered
    
    def _count_valid_points(self, data: pd.DataFrame) -> int:
        """Count valid radar points across all robots"""
        total_points = 0
        
        for robot_id in ['robot_1', 'robot_2']:
            x_col = f'{robot_id}_global_x_radar'
            y_col = f'{robot_id}_global_y_radar'
            z_col = f'{robot_id}_global_z_radar'
            
            if all(col in data.columns for col in [x_col, y_col, z_col]):
                valid_mask = ~data[[x_col, y_col, z_col]].isna().any(axis=1)
                total_points += valid_mask.sum()
        
        return total_points
    
    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk with all filters"""
        # Apply filters in sequence
        filtered = chunk.copy()
        
        # Apply each filter
        filtered = self._apply_boundary_filter(filtered)
        filtered = self._apply_snr_filter(filtered)
        filtered = self._apply_height_filter(filtered)
        filtered = self._apply_fov_filter(filtered)
        # Note: Statistical filtering needs to be applied globally, not per chunk
        
        return filtered
    
    def process(self) -> bool:
        """Main processing pipeline"""
        start_time = time.time()
        
        self.logger.info(f"Starting processing of {self.input_path}")
        
        # Determine output path
        if not self.output_path:
            path_obj = Path(self.input_path)
            suffix = f"_cleaned_fov{self.config['fov']['angle_degrees']}"
            self.output_path = str(path_obj.parent / f"{path_obj.stem}{suffix}{path_obj.suffix}")
        
        # Create output directory
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Determine processing strategy
        strategy = self._determine_processing_strategy()
        
        try:
            if strategy['use_chunking']:
                result = self._process_chunked(strategy['chunk_size'])
            else:
                result = self._process_full()
            
            if result is not None:
                # Apply statistical filtering (requires full dataset)
                if self.config['statistical_filter']['enabled']:
                    result = self._apply_statistical_filter(result)
                
                # Update statistics
                self.stats['final_points'] = self._count_valid_points(result)
                self.stats['processing_time'] = time.time() - start_time
                self.stats['memory_usage_mb'] = psutil.Process().memory_info().rss / (1024 * 1024)
                
                # Save results
                result.to_csv(self.output_path, index=False)
                self.logger.info(f"Saved cleaned data to {self.output_path}")
                
                # Generate visualization
                if self.config['visualization']['enabled']:
                    self._create_visualization(result)
                
                # Print summary
                self._print_summary()
                
                return True
            else:
                self.logger.error("Processing failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _process_full(self) -> Optional[pd.DataFrame]:
        """Process entire file at once"""
        self.logger.info("Processing entire file at once")
        
        # Load data
        data = pd.read_csv(self.input_path)
        self.stats['original_points'] = self._count_valid_points(data)
        self.logger.info(f"Loaded {len(data)} rows with {self.stats['original_points']} valid points")
        
        # Process all filters
        result = self._process_chunk(data)
        
        return result
    
    def _process_chunked(self, chunk_size: int) -> Optional[pd.DataFrame]:
        """Process file in chunks"""
        self.logger.info(f"Processing file in chunks of {chunk_size}")
        
        chunks = []
        total_original_points = 0
        
        try:
            chunk_iterator = pd.read_csv(self.input_path, chunksize=chunk_size)
            
            for chunk_idx, chunk in enumerate(chunk_iterator):
                self.logger.info(f"Processing chunk {chunk_idx + 1} ({len(chunk)} rows)")
                
                # Count original points in this chunk
                total_original_points += self._count_valid_points(chunk)
                
                # Process chunk
                filtered_chunk = self._process_chunk(chunk)
                chunks.append(filtered_chunk)
                
                # Periodic garbage collection
                if chunk_idx % 10 == 0:
                    gc.collect()
            
            # Combine chunks
            if chunks:
                self.logger.info(f"Combining {len(chunks)} chunks...")
                result = pd.concat(chunks, ignore_index=True)
                self.stats['original_points'] = total_original_points
                return result
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Chunked processing failed: {e}")
            return None
    
    def _create_visualization(self, data: pd.DataFrame):
        """Create comprehensive visualization"""
        vis_path = os.path.join(
            os.path.dirname(self.output_path),
            f"visualization_{os.path.basename(self.output_path)}.png"
        )
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=self.config['visualization']['figsize'])
            
            # Top view (X-Y)
            self._plot_top_view(axes[0, 0], data, "Top View (X-Y)")
            
            # Side view (X-Z)
            self._plot_side_view(axes[0, 1], data, "Side View (X-Z)")
            
            # Height distribution
            self._plot_height_distribution(axes[1, 0], data, "Height Distribution")
            
            # Point density
            self._plot_point_density(axes[1, 1], data, "Point Density")
            
            plt.suptitle(f"Cleaned Radar Data Analysis\nFOV: {self.config['fov']['angle_degrees']}°, Points: {self.stats['final_points']}")
            plt.tight_layout()
            
            plt.savefig(vis_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved visualization to {vis_path}")
            
        except Exception as e:
            self.logger.warning(f"Visualization failed: {e}")
    
    def _plot_top_view(self, ax, data: pd.DataFrame, title: str):
        """Plot top view with boundary and robot positions"""
        colors = {'robot_1': 'blue', 'robot_2': 'red'}
        
        for robot_id, color in colors.items():
            x_col = f'{robot_id}_global_x_radar'
            y_col = f'{robot_id}_global_y_radar'
            
            if all(col in data.columns for col in [x_col, y_col]):
                valid_mask = ~data[[x_col, y_col]].isna().any(axis=1)
                if valid_mask.any():
                    valid_data = data[valid_mask]
                    ax.scatter(valid_data[x_col], valid_data[y_col], 
                              c=color, alpha=0.6, s=2, label=robot_id.replace('_', ' ').title())
        
        # Plot boundary
        boundary = self.config['boundary']
        boundary_x = [boundary['x_min'], boundary['x_max'], boundary['x_max'], 
                     boundary['x_min'], boundary['x_min']]
        boundary_y = [boundary['y_min'], boundary['y_min'], boundary['y_max'], 
                     boundary['y_max'], boundary['y_min']]
        ax.plot(boundary_x, boundary_y, 'k-', linewidth=2, label='Boundary')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
    
    def _plot_side_view(self, ax, data: pd.DataFrame, title: str):
        """Plot side view with height constraints"""
        colors = {'robot_1': 'blue', 'robot_2': 'red'}
        
        for robot_id, color in colors.items():
            x_col = f'{robot_id}_global_x_radar'
            z_col = f'{robot_id}_global_z_radar'
            
            if all(col in data.columns for col in [x_col, z_col]):
                valid_mask = ~data[[x_col, z_col]].isna().any(axis=1)
                if valid_mask.any():
                    valid_data = data[valid_mask]
                    ax.scatter(valid_data[x_col], valid_data[z_col], 
                              c=color, alpha=0.6, s=2, label=robot_id.replace('_', ' ').title())
        
        # Plot height constraints
        if self.config['height']['enabled']:
            ax.axhline(self.config['height']['min_height'], color='r', linestyle='--', 
                      alpha=0.7, label='Min Height')
            ax.axhline(self.config['height']['max_height'], color='r', linestyle='--', 
                      alpha=0.7, label='Max Height')
        
        # Plot boundary
        ax.axvline(self.config['boundary']['x_min'], color='k', linestyle='--', alpha=0.5)
        ax.axvline(self.config['boundary']['x_max'], color='k', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_height_distribution(self, ax, data: pd.DataFrame, title: str):
        """Plot height distribution histogram"""
        colors = {'robot_1': 'blue', 'robot_2': 'red'}
        
        for robot_id, color in colors.items():
            z_col = f'{robot_id}_global_z_radar'
            
            if z_col in data.columns:
                valid_data = data[z_col].dropna()
                if not valid_data.empty:
                    ax.hist(valid_data, bins=50, alpha=0.6, color=color, 
                           label=robot_id.replace('_', ' ').title(), density=True)
        
        ax.set_xlabel('Height (m)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_point_density(self, ax, data: pd.DataFrame, title: str):
        """Plot 2D point density heatmap"""
        # Combine all valid points
        all_x, all_y = [], []
        
        for robot_id in ['robot_1', 'robot_2']:
            x_col = f'{robot_id}_global_x_radar'
            y_col = f'{robot_id}_global_y_radar'
            
            if all(col in data.columns for col in [x_col, y_col]):
                valid_mask = ~data[[x_col, y_col]].isna().any(axis=1)
                if valid_mask.any():
                    valid_data = data[valid_mask]
                    all_x.extend(valid_data[x_col].tolist())
                    all_y.extend(valid_data[y_col].tolist())
        
        if all_x and all_y:
            h = ax.hist2d(all_x, all_y, bins=50, cmap='YlOrRd')
            plt.colorbar(h[3], ax=ax, label='Point Count')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.set_aspect('equal')
    
    def _print_summary(self):
        """Print processing summary"""
        self.logger.info("\n" + "="*60)
        self.logger.info("PROCESSING SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Input file: {self.input_path}")
        self.logger.info(f"Output file: {self.output_path}")
        self.logger.info(f"Original points: {self.stats['original_points']:,}")
        self.logger.info(f"Final points: {self.stats['final_points']:,}")
        
        if self.stats['original_points'] > 0:
            reduction_pct = ((self.stats['original_points'] - self.stats['final_points']) / 
                           self.stats['original_points']) * 100
            self.logger.info(f"Points removed: {reduction_pct:.1f}%")
        
        self.logger.info(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        self.logger.info(f"Memory usage: {self.stats['memory_usage_mb']:.1f} MB")
        
        # Filter configuration summary
        self.logger.info("\nFilter Configuration:")
        self.logger.info(f"  Boundary: X[{self.config['boundary']['x_min']:.1f}, {self.config['boundary']['x_max']:.1f}], "
                        f"Y[{self.config['boundary']['y_min']:.1f}, {self.config['boundary']['y_max']:.1f}], "
                        f"Z[{self.config['boundary']['z_min']:.1f}, {self.config['boundary']['z_max']:.1f}]")
        self.logger.info(f"  FOV: {self.config['fov']['angle_degrees']}° ({'enabled' if self.config['fov']['enabled'] else 'disabled'})")
        self.logger.info(f"  SNR: ≥{self.config['snr']['min_value']} ({'enabled' if self.config['snr']['enabled'] else 'disabled'})")
        self.logger.info(f"  Height: [{self.config['height']['min_height']:.2f}, {self.config['height']['max_height']:.2f}] m ({'enabled' if self.config['height']['enabled'] else 'disabled'})")
        self.logger.info(f"  Statistical: k={self.config['statistical_filter']['k_neighbors']}, std={self.config['statistical_filter']['std_deviation']} ({'enabled' if self.config['statistical_filter']['enabled'] else 'disabled'})")
        self.logger.info("="*60)


def process_directory(input_dir: str, output_dir: Optional[str] = None, 
                     config: Optional[Dict] = None, pattern: str = "transformed_*.csv") -> Dict:
    """Process all files in a directory with comprehensive error handling and progress tracking"""
    
    if output_dir is None:
        fov_angle = config.get('fov', {}).get('angle_degrees', 120) if config else 120
        output_dir = os.path.join(os.path.dirname(input_dir), f"Cleaned_Data_FOV{fov_angle}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all files to process
    files_to_process = []
    for path in Path(input_dir).rglob(pattern):
        files_to_process.append(str(path))
    
    logging.info(f"Found {len(files_to_process)} files to process")
    
    # Processing results
    results = {
        'total_files': len(files_to_process),
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'failed_files': [],
        'total_processing_time': 0,
        'total_original_points': 0,
        'total_final_points': 0
    }
    
    start_time = time.time()
    
    for i, file_path in enumerate(files_to_process):
        logging.info(f"\n[{i+1}/{len(files_to_process)}] Processing: {os.path.basename(file_path)}")
        
        try:
            # Determine output path
            rel_path = os.path.relpath(os.path.dirname(file_path), input_dir)
            output_subdir = os.path.join(output_dir, rel_path)
            os.makedirs(output_subdir, exist_ok=True)
            
            # Generate output filename
            base_name = os.path.basename(file_path)
            if config and 'fov' in config:
                fov_suffix = f"_fov{config['fov']['angle_degrees']}"
            else:
                fov_suffix = "_fov120"
            
            output_filename = base_name.replace("transformed_", f"cleaned{fov_suffix}_")
            output_path = os.path.join(output_subdir, output_filename)
            
            # Skip if already exists
            if os.path.exists(output_path):
                logging.info("Output file already exists, skipping...")
                results['skipped'] += 1
                continue
            
            # Process file
            cleaner = CollaborativePerceptionPointCloudCleaner(file_path, output_path, config)
            success = cleaner.process()
            
            if success:
                results['successful'] += 1
                results['total_original_points'] += cleaner.stats['original_points']
                results['total_final_points'] += cleaner.stats['final_points']
                results['total_processing_time'] += cleaner.stats['processing_time']
            else:
                results['failed'] += 1
                results['failed_files'].append(file_path)
            
            # Cleanup
            del cleaner
            gc.collect()
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            results['failed'] += 1
            results['failed_files'].append(file_path)
    
    # Print final summary
    total_time = time.time() - start_time
    logging.info(f"\n{'='*80}")
    logging.info("BATCH PROCESSING SUMMARY")
    logging.info(f"{'='*80}")
    logging.info(f"Total files: {results['total_files']}")
    logging.info(f"Successful: {results['successful']}")
    logging.info(f"Failed: {results['failed']}")
    logging.info(f"Skipped: {results['skipped']}")
    logging.info(f"Total processing time: {total_time:.2f} seconds")
    logging.info(f"Average time per file: {total_time/max(1, results['successful']):.2f} seconds")
    
    if results['total_original_points'] > 0:
        overall_reduction = ((results['total_original_points'] - results['total_final_points']) / 
                           results['total_original_points']) * 100
        logging.info(f"Total original points: {results['total_original_points']:,}")
        logging.info(f"Total final points: {results['total_final_points']:,}")
        logging.info(f"Overall reduction: {overall_reduction:.1f}%")
    
    if results['failed_files']:
        logging.info(f"\nFailed files ({len(results['failed_files'])}):")
        for failed_file in results['failed_files']:
            logging.info(f"  - {failed_file}")
    
    logging.info(f"{'='*80}")
    
    return results


def create_config_from_args(args) -> Dict:
    """Create configuration dictionary from command line arguments"""
    config = {}
    
    # FOV configuration
    if hasattr(args, 'fov') and args.fov is not None:
        config['fov'] = {'angle_degrees': args.fov, 'enabled': True}
    
    # SNR configuration
    if hasattr(args, 'snr') and args.snr is not None:
        config['snr'] = {'min_value': args.snr, 'enabled': True}
    
    # Height configuration
    if hasattr(args, 'min_height') and args.min_height is not None:
        if 'height' not in config:
            config['height'] = {'enabled': True}
        config['height']['min_height'] = args.min_height
    
    if hasattr(args, 'max_height') and args.max_height is not None:
        if 'height' not in config:
            config['height'] = {'enabled': True}
        config['height']['max_height'] = args.max_height
    
    # Boundary configuration
    if hasattr(args, 'x_min') and args.x_min is not None:
        if 'boundary' not in config:
            config['boundary'] = {}
        config['boundary']['x_min'] = args.x_min
    
    if hasattr(args, 'x_max') and args.x_max is not None:
        if 'boundary' not in config:
            config['boundary'] = {}
        config['boundary']['x_max'] = args.x_max
    
    # Processing configuration
    if hasattr(args, 'chunk_size') and args.chunk_size is not None:
        config['processing'] = {'chunk_size': args.chunk_size}
    
    if hasattr(args, 'memory_threshold') and args.memory_threshold is not None:
        if 'processing' not in config:
            config['processing'] = {}
        config['processing']['memory_threshold_gb'] = args.memory_threshold
    
    # Statistical filter configuration
    if hasattr(args, 'disable_statistical') and args.disable_statistical:
        config['statistical_filter'] = {'enabled': False}
    
    # Visualization configuration
    if hasattr(args, 'no_viz') and args.no_viz:
        config['visualization'] = {'enabled': False}
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Unified Radar Data Cleaner - Memory efficient with comprehensive filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python unified_radar_cleaner.py -i data.csv -o cleaned_data.csv --fov 120

  # Process directory with custom parameters
  python unified_radar_cleaner.py --batch -i /path/to/data -o /path/to/output --snr 8.0 --fov 90

  # Process with memory constraints
  python unified_radar_cleaner.py --batch -i /path/to/data --chunk-size 50000 --memory-threshold 1.5
        """
    )
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input CSV file or directory')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file or directory (auto-generated if not specified)')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process entire directory structure')
    
    # Filter parameters
    parser.add_argument('--fov', '-f', type=float, default=120.0,
                       help='Field of view angle in degrees (default: 120)')
    parser.add_argument('--snr', type=float, default=5.0,
                       help='Minimum SNR threshold (default: 5.0)')
    parser.add_argument('--min-height', type=float, default=0.05,
                       help='Minimum height in meters (default: 0.05)')
    parser.add_argument('--max-height', type=float, default=2.5,
                       help='Maximum height in meters (default: 2.5)')
    parser.add_argument('--x-min', type=float, help='Minimum X boundary')
    parser.add_argument('--x-max', type=float, help='Maximum X boundary')
    
    # Processing parameters
    parser.add_argument('--chunk-size', type=int, default=None,
                       help='Chunk size for processing (default: auto)')
    parser.add_argument('--memory-threshold', type=float, default=2.0,
                       help='Memory threshold in GB for chunked processing (default: 2.0)')
    
    # Feature toggles
    parser.add_argument('--disable-statistical', action='store_true',
                       help='Disable statistical outlier removal')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization generation')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration from arguments
    config = create_config_from_args(args)
    
    try:
        if args.batch:
            # Process directory
            if not os.path.isdir(args.input):
                logging.error(f"Input path is not a directory: {args.input}")
                return 1
            
            results = process_directory(args.input, args.output, config)
            
            # Return appropriate exit code
            if results['failed'] > 0:
                return 1 if results['successful'] == 0 else 2  # 1 = total failure, 2 = partial failure
            else:
                return 0
                
        else:
            # Process single file
            if not os.path.isfile(args.input):
                logging.error(f"Input file does not exist: {args.input}")
                return 1
            
            cleaner = CollaborativePerceptionPointCloudCleaner(args.input, args.output, config)
            success = cleaner.process()
            
            return 0 if success else 1
            
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        return 130
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
