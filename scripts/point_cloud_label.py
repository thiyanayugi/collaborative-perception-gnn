#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Point Cloud Annotator for Collaborative Perception

This module implements a comprehensive point cloud annotation system for
collaborative perception applications in warehouse robotics. It automatically
annotates mmWave radar point clouds with semantic labels including workstations,
robots, boundaries, and unknown objects for training collaborative perception models.

Key Features:
- Automated semantic annotation of point cloud data
- Workstation detection and labeling (AS_1, AS_3, AS_4, AS_5, AS_6)
- Robot detection and multi-robot identification
- Arena boundary detection and classification
- Unknown object categorization for comprehensive coverage
- Visualization capabilities for annotation validation
- Batch processing for large datasets

The annotator is essential for creating labeled training datasets for
collaborative perception GNN models in warehouse environments.

Author: Thiyanayugi Mariraj
Project: Development of a Framework for Collaborative Perception Management Layer
         for Future 6G-Enabled Robotic Systems
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import math
import logging
from datetime import datetime
from tqdm import tqdm
import re
from typing import Dict, List, Optional, Tuple, Any

# Configure comprehensive logging for annotation process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("collaborative_perception_annotation_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CollaborativePerceptionAnnotator")


class CollaborativePerceptionPointCloudAnnotator:
    """
    Advanced Point Cloud Annotator for Collaborative Perception Systems.

    This class provides comprehensive semantic annotation capabilities for point
    cloud data used in collaborative perception applications. It automatically
    identifies and labels various objects in warehouse environments including
    workstations, robots, boundaries, and unknown objects.

    Key Annotation Categories:
    - Workstations: Automated detection of known workstation positions
    - Robots: Multi-robot identification and tracking
    - Boundaries: Arena and structural boundary detection
    - Unknown: Comprehensive coverage for unclassified objects

    The annotator supports batch processing, visualization validation, and
    quality assurance for creating high-quality training datasets.
    """

    def __init__(
        self,
        input_data_directory: str,
        output_data_directory: str,
        workstation_positions_file: str,
        visualization_directory: Optional[str] = None
    ):
        """
        Initialize the collaborative perception point cloud annotator.

        Args:
            input_data_directory (str): Directory containing cleaned point cloud CSV files
            output_data_directory (str): Directory to save annotated CSV files
            workstation_positions_file (str): Path to workstation position Excel file
            visualization_directory (str, optional): Directory to save annotation visualizations
        """
        self.input_data_directory = input_data_directory
        self.output_data_directory = output_data_directory
        self.workstation_positions_file = workstation_positions_file
        self.visualization_directory = visualization_directory

        # Create output directories if they don't exist
        os.makedirs(output_data_directory, exist_ok=True)
        if visualization_directory:
            os.makedirs(visualization_directory, exist_ok=True)

        # Load workstation positions
        self.ws_positions = pd.read_excel(ws_position_file)
        logger.info(f"Loaded {len(self.ws_positions)} workstation position entries")

        # Arena boundary
        self.arena_boundary = {
            'x_min': -9.1, 'x_max': 10.2,
            'y_min': -4.42, 'y_max': 5.5
        }

        # Tolerances
        self.ws_tolerance = 0.200  # 12.5 cm (middle of 10-15cm range)
        self.robot_tolerance = 0.12  # 7.5 cm (middle of 5-10cm range)
        self.boundary_tolerance = 0.25  # 15 cm for straight edges
        self.corner_tolerance = 0.35  # 25 cm for corners

        # Workstation dimensions (in meters)
        self.ws_length = 1.0
        self.ws_width = 0.65

        # Robot dimensions (in meters)
        self.robot_length = 0.32  # 320mm
        self.robot_width = 0.24   # 240mm

        # Colors for visualization
        self.colors = {
            'AS_1': 'red',
            'AS_3': 'blue',
            'AS_4': 'green',
            'AS_5': 'purple',
            'AS_6': 'orange',
            'robot': 'cyan',
            'boundary': 'brown',
            'unknown': 'gray'
        }

        # Note: Only AS4 is vertical (rotated 90 degrees), others are horizontal

    def find_matching_ws_positions(self, dataset_name):
        """
        Find the matching workstation positions for a given dataset.

        Args:
            dataset_name (str): Name of the dataset

        Returns:
            pd.Series: Row containing workstation positions for the dataset
        """
        # Extract date and time from dataset name using regex
        match = re.search(r'(\d{8})_(\d{6})', dataset_name)
        if not match:
            logger.warning(f"Could not extract date and time from dataset name: {dataset_name}")
            # Use the first entry as fallback
            return self.ws_positions.iloc[0]

        # Convert extracted date and time to match format in Excel file
        date_str = f"2025-{match.group(1)[4:6]}-{match.group(1)[6:8]}"
        time_str = f"{match.group(2)[:2]}:{match.group(2)[2:4]}:{match.group(2)[4:6]}"

        logger.debug(f"Looking for workstation positions for date: {date_str}, time: {time_str}")

        # Find the closest matching entry in the workstation positions file
        # First try to match by exact dataset name
        exact_match = self.ws_positions[self.ws_positions['Dataset'].str.contains(dataset_name, na=False)]
        if not exact_match.empty:
            logger.info(f"Found exact dataset match for {dataset_name}")
            return exact_match.iloc[0]

        # Then try to match by date and approximate time
        date_matches = self.ws_positions[self.ws_positions['Dataset Date'] == date_str]
        if date_matches.empty:
            logger.warning(f"No workstation positions found for date: {date_str}")
            # Use the first entry as fallback
            return self.ws_positions.iloc[0]

        # Convert time to datetime for comparison
        dataset_time = datetime.strptime(time_str, "%H:%M:%S").time()

        # Find the closest time match
        closest_match = None
        min_time_diff = float('inf')

        for _, row in date_matches.iterrows():
            ws_time = datetime.strptime(row['Dataset Time'], "%H:%M:%S").time()
            ws_datetime = datetime.combine(datetime.today(), ws_time)
            dataset_datetime = datetime.combine(datetime.today(), dataset_time)
            time_diff = abs((ws_datetime - dataset_datetime).total_seconds())

            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_match = row

        logger.info(f"Found closest workstation position match for {dataset_name} with time difference of {min_time_diff} seconds")
        return closest_match

    def get_rotated_rectangle_corners(self, center_x, center_y, length, width, yaw, ws_name=None):
        """
        Calculate the corners of a rotated rectangle.

        Args:
            center_x (float): X coordinate of the center
            center_y (float): Y coordinate of the center
            length (float): Length of the rectangle
            width (float): Width of the rectangle
            yaw (float): Rotation angle in radians
            ws_name (str, optional): Workstation name for special handling

        Returns:
            list: List of (x, y) coordinates for the four corners
        """
        # For AS4, ensure it's vertical (rotated 90 degrees)
        # For other workstations, ensure they're horizontal
        if ws_name:
            if ws_name == 'AS_4':
                # AS4 should be vertical (rotated 90 degrees)
                yaw = math.pi/2  # Set to exactly 90 degrees
            else:
                # Other workstations should be horizontal (0 degrees)
                yaw = 0.0  # Set to exactly 0 degrees

        # Calculate half dimensions
        half_length = length / 2
        half_width = width / 2

        # Calculate corners relative to center (before rotation)
        corners_rel = [
            (-half_length, -half_width),  # Bottom left
            (half_length, -half_width),   # Bottom right
            (half_length, half_width),    # Top right
            (-half_length, half_width)    # Top left
        ]

        # Apply rotation and translation
        corners = []
        for x_rel, y_rel in corners_rel:
            # Rotate
            x_rot = x_rel * math.cos(yaw) - y_rel * math.sin(yaw)
            y_rot = x_rel * math.sin(yaw) + y_rel * math.cos(yaw)

            # Translate
            x = center_x + x_rot
            y = center_y + y_rot

            corners.append((x, y))

        return corners

    def is_point_near_rectangle_edge(self, point_x, point_y, center_x, center_y, length, width, yaw, tolerance=0, ws_name=None):
        """
        Check if a point is near the edge of a rotated rectangle (with optional tolerance).

        Args:
            point_x (float): X coordinate of the point
            point_y (float): Y coordinate of the point
            center_x (float): X coordinate of the rectangle center
            center_y (float): Y coordinate of the rectangle center
            length (float): Length of the rectangle
            width (float): Width of the rectangle
            yaw (float): Rotation angle in radians
            tolerance (float): Tolerance in meters
            ws_name (str, optional): Workstation name for special handling

        Returns:
            bool: True if the point is near the edge of the rectangle, False otherwise
        """
        # For AS4, ensure it's vertical (rotated 90 degrees)
        # For other workstations, ensure they're horizontal
        if ws_name:
            if ws_name == 'AS_4':
                # AS4 should be vertical (rotated 90 degrees)
                # If the provided yaw is not close to 90 degrees (π/2), adjust it
                if abs(yaw - math.pi/2) > 0.1:  # If not close to 90 degrees
                    yaw = math.pi/2  # Set to exactly 90 degrees
            else:
                # Other workstations should be horizontal (0 degrees)
                # If the provided yaw is not close to 0 or 180 degrees, adjust it
                if abs(yaw) > 0.1 and abs(yaw - math.pi) > 0.1:
                    yaw = 0.0  # Set to exactly 0 degrees

        # Translate point to origin
        translated_x = point_x - center_x
        translated_y = point_y - center_y

        # Rotate point in the opposite direction of rectangle rotation
        rotated_x = translated_x * math.cos(-yaw) - translated_y * math.sin(-yaw)
        rotated_y = translated_x * math.sin(-yaw) + translated_y * math.cos(-yaw)

        # Calculate half dimensions
        half_length = length / 2
        half_width = width / 2

        # Check if the point is inside the outer rectangle (with tolerance)
        inside_outer = (
            -half_length - tolerance <= rotated_x <= half_length + tolerance and
            -half_width - tolerance <= rotated_y <= half_width + tolerance
        )

        # Check if the point is outside the inner rectangle (without tolerance)
        outside_inner = not (
            -half_length + tolerance <= rotated_x <= half_length - tolerance and
            -half_width + tolerance <= rotated_y <= half_width - tolerance
        )

        # The point is near the edge if it's inside the outer rectangle but outside the inner rectangle
        return inside_outer and outside_inner

    def is_point_near_boundary(self, point_x, point_y):
        """
        Check if a point is near the arena boundary.

        Args:
            point_x (float): X coordinate of the point
            point_y (float): Y coordinate of the point

        Returns:
            tuple: (bool, str) - True if near boundary and which boundary (north, south, east, west, corner)
        """
        x_min = self.arena_boundary['x_min']
        x_max = self.arena_boundary['x_max']
        y_min = self.arena_boundary['y_min']
        y_max = self.arena_boundary['y_max']

        # Check if point is near corners (using corner_tolerance)
        corner_dist_sw = math.sqrt((point_x - x_min)**2 + (point_y - y_min)**2)
        corner_dist_se = math.sqrt((point_x - x_max)**2 + (point_y - y_min)**2)
        corner_dist_nw = math.sqrt((point_x - x_min)**2 + (point_y - y_max)**2)
        corner_dist_ne = math.sqrt((point_x - x_max)**2 + (point_y - y_max)**2)

        if corner_dist_sw <= self.corner_tolerance:
            return True, "boundary_corner_sw"
        if corner_dist_se <= self.corner_tolerance:
            return True, "boundary_corner_se"
        if corner_dist_nw <= self.corner_tolerance:
            return True, "boundary_corner_nw"
        if corner_dist_ne <= self.corner_tolerance:
            return True, "boundary_corner_ne"

        # Check if point is near edges (using boundary_tolerance)
        if abs(point_x - x_min) <= self.boundary_tolerance and y_min <= point_y <= y_max:
            return True, "boundary_west"
        if abs(point_x - x_max) <= self.boundary_tolerance and y_min <= point_y <= y_max:
            return True, "boundary_east"
        if abs(point_y - y_min) <= self.boundary_tolerance and x_min <= point_x <= x_max:
            return True, "boundary_south"
        if abs(point_y - y_max) <= self.boundary_tolerance and x_min <= point_x <= x_max:
            return True, "boundary_north"

        return False, None

    def is_point_near_robot(self, point_x, point_y, robot_x, robot_y, robot_yaw):
        """
        Check if a point is near a robot's edge.

        Args:
            point_x (float): X coordinate of the point
            point_y (float): Y coordinate of the point
            robot_x (float): X coordinate of the robot center
            robot_y (float): Y coordinate of the robot center
            robot_yaw (float): Rotation angle of the robot in radians

        Returns:
            bool: True if the point is near the robot's edge, False otherwise
        """
        return self.is_point_near_rectangle_edge(
            point_x, point_y,
            robot_x, robot_y,
            self.robot_length, self.robot_width,
            robot_yaw,
            self.robot_tolerance
        )

    def annotate_file(self, file_path, visualize_frames=0):
        """
        Annotate a single CSV file with workstation, robot, boundary, and unknown labels.

        Args:
            file_path (str): Path to the CSV file
            visualize_frames (int): Number of frames to visualize (evenly distributed)

        Returns:
            str: Path to the annotated file
        """
        logger.info(f"Annotating file: {file_path}")

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Extract dataset name from file path
        dataset_dir = os.path.basename(os.path.dirname(file_path))

        # Find matching workstation positions
        ws_pos = self.find_matching_ws_positions(dataset_dir)

        # Initialize annotation columns
        df['annotation_specific'] = 'unknown'
        df['annotation_general'] = 'unknown'

        # Skip frame visualization to prevent PC from hanging
        timestamps_to_visualize = []

        # Process each timestamp
        for timestamp in tqdm(df['vicon_timestamp'].unique(), desc="Processing timestamps"):
            # Get data for this timestamp
            timestamp_df = df[df['vicon_timestamp'] == timestamp]

            # Get robot positions for this timestamp
            robot1_x = timestamp_df['robot_1_global_x'].iloc[0]
            robot1_y = timestamp_df['robot_1_global_y'].iloc[0]
            robot1_yaw = timestamp_df['robot_1_yaw'].iloc[0]

            robot2_x = timestamp_df['robot_2_global_x'].iloc[0]
            robot2_y = timestamp_df['robot_2_global_y'].iloc[0]
            robot2_yaw = timestamp_df['robot_2_yaw'].iloc[0]

            # Process each point in this timestamp
            for idx in timestamp_df.index:
                point_x = df.loc[idx, 'robot_1_global_x_radar']
                point_y = df.loc[idx, 'robot_1_global_y_radar']

                # Check if point is near a workstation
                ws_found = False
                for ws_name in ['AS_1', 'AS_3', 'AS_4', 'AS_5', 'AS_6']:
                    ws_x = ws_pos[f'{ws_name}_neu_X']
                    ws_y = ws_pos[f'{ws_name}_neu_Y']
                    ws_yaw = ws_pos[f'{ws_name}_neu_Yaw']

                    if self.is_point_near_rectangle_edge(
                        point_x, point_y,
                        ws_x, ws_y,
                        self.ws_length, self.ws_width,
                        ws_yaw,
                        self.ws_tolerance,
                        ws_name
                    ):
                        df.loc[idx, 'annotation_specific'] = ws_name
                        df.loc[idx, 'annotation_general'] = 'workstation'
                        ws_found = True
                        break

                if ws_found:
                    continue

                # Check if point is near a robot
                if self.is_point_near_robot(point_x, point_y, robot1_x, robot1_y, robot1_yaw) or \
                   self.is_point_near_robot(point_x, point_y, robot2_x, robot2_y, robot2_yaw):
                    df.loc[idx, 'annotation_specific'] = 'robot'
                    df.loc[idx, 'annotation_general'] = 'robot'
                    continue

                # Check if point is near boundary
                is_boundary, boundary_type = self.is_point_near_boundary(point_x, point_y)
                if is_boundary:
                    df.loc[idx, 'annotation_specific'] = boundary_type
                    df.loc[idx, 'annotation_general'] = 'boundary'
                    continue

                # If none of the above, it's unknown (already set as default)

            # Visualize this timestamp if needed
            if timestamp in timestamps_to_visualize and self.visualize_dir:
                self.visualize_frame(df[df['vicon_timestamp'] == timestamp],
                                    ws_pos,
                                    dataset_dir,
                                    timestamp)

        # Create output directory structure
        rel_path = os.path.relpath(os.path.dirname(file_path), self.input_dir)
        out_dir = os.path.join(self.output_dir, rel_path)
        os.makedirs(out_dir, exist_ok=True)

        # Save annotated file
        output_file = os.path.join(out_dir, f"annotated_{os.path.basename(file_path)}")
        df.to_csv(output_file, index=False)

        logger.info(f"Saved annotated file to: {output_file}")
        return output_file

    def visualize_frame(self, frame_df, ws_pos, dataset_name, timestamp):
        """
        Visualize a single frame with annotations.

        Args:
            frame_df (pd.DataFrame): DataFrame containing points for a single timestamp
            ws_pos (pd.Series): Workstation positions
            dataset_name (str): Name of the dataset
            timestamp (float): Timestamp of the frame
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot arena boundary
        x_min = self.arena_boundary['x_min']
        x_max = self.arena_boundary['x_max']
        y_min = self.arena_boundary['y_min']
        y_max = self.arena_boundary['y_max']

        boundary_rect = Rectangle((x_min, y_min),
                                 x_max - x_min,
                                 y_max - y_min,
                                 fill=False,
                                 edgecolor='black',
                                 linestyle='--',
                                 linewidth=2)
        ax.add_patch(boundary_rect)

        # Plot workstations
        for ws_name in ['AS_1', 'AS_3', 'AS_4', 'AS_5', 'AS_6']:
            ws_x = ws_pos[f'{ws_name}_neu_X']
            ws_y = ws_pos[f'{ws_name}_neu_Y']
            ws_yaw = ws_pos[f'{ws_name}_neu_Yaw']

            # Get corners of the workstation
            corners = self.get_rotated_rectangle_corners(
                ws_x, ws_y,
                self.ws_length, self.ws_width,
                ws_yaw,
                ws_name
            )

            # Create polygon
            polygon = Polygon(corners,
                             fill=False,
                             edgecolor=self.colors[ws_name],
                             linewidth=2,
                             label=ws_name)
            ax.add_patch(polygon)

        # Plot robot positions
        robot1_x = frame_df['robot_1_global_x'].iloc[0]
        robot1_y = frame_df['robot_1_global_y'].iloc[0]
        robot1_yaw = frame_df['robot_1_yaw'].iloc[0]

        robot2_x = frame_df['robot_2_global_x'].iloc[0]
        robot2_y = frame_df['robot_2_global_y'].iloc[0]
        robot2_yaw = frame_df['robot_2_yaw'].iloc[0]

        # Get corners of the robots
        robot1_corners = self.get_rotated_rectangle_corners(
            robot1_x, robot1_y,
            self.robot_length, self.robot_width,
            robot1_yaw
        )

        robot2_corners = self.get_rotated_rectangle_corners(
            robot2_x, robot2_y,
            self.robot_length, self.robot_width,
            robot2_yaw
        )

        # Create polygons
        robot1_polygon = Polygon(robot1_corners,
                                fill=False,
                                edgecolor='black',
                                linewidth=2,
                                label='Robot 1')
        ax.add_patch(robot1_polygon)

        robot2_polygon = Polygon(robot2_corners,
                                fill=False,
                                edgecolor='black',
                                linewidth=2,
                                label='Robot 2')
        ax.add_patch(robot2_polygon)

        # Plot points with colors based on annotations
        for annotation in frame_df['annotation_specific'].unique():
            points = frame_df[frame_df['annotation_specific'] == annotation]

            if annotation.startswith('AS_'):
                color = self.colors[annotation]
                label = annotation
            elif annotation == 'robot':
                color = self.colors['robot']
                label = 'Robot Points'
            elif annotation.startswith('boundary'):
                color = self.colors['boundary']
                label = 'Boundary Points'
            else:
                color = self.colors['unknown']
                label = 'Unknown Points'

            ax.scatter(points['robot_1_global_x_radar'],
                      points['robot_1_global_y_radar'],
                      c=color,
                      s=10,
                      alpha=0.7,
                      label=label)

        # Set title and labels
        ax.set_title(f"Dataset: {dataset_name}\nTimestamp: {timestamp:.2f}\nWorkstation Edges Only")
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')

        # Set equal aspect ratio
        ax.set_aspect('equal')

        # Set limits with some padding
        padding = 1.0  # 1 meter padding
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

        # Add legend
        ax.legend(loc='upper right')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Save figure
        vis_dir = os.path.join(self.visualize_dir, dataset_name)
        os.makedirs(vis_dir, exist_ok=True)

        fig_path = os.path.join(vis_dir, f"frame_{timestamp:.2f}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved visualization to: {fig_path}")

    def create_dataset_summary(self, dataset_name, annotated_file):
        """
        Create a summary visualization for a dataset showing all annotated points.

        Args:
            dataset_name (str): Name of the dataset
            annotated_file (str): Path to the annotated CSV file
        """
        logger.info(f"Creating summary visualization for dataset: {dataset_name}")

        # Read the annotated file
        df = pd.read_csv(annotated_file)

        # Find matching workstation positions
        ws_pos = self.find_matching_ws_positions(dataset_name)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))

        # Plot arena boundary
        x_min = self.arena_boundary['x_min']
        x_max = self.arena_boundary['x_max']
        y_min = self.arena_boundary['y_min']
        y_max = self.arena_boundary['y_max']

        boundary_rect = Rectangle((x_min, y_min),
                                 x_max - x_min,
                                 y_max - y_min,
                                 fill=False,
                                 edgecolor='black',
                                 linestyle='--',
                                 linewidth=2)
        ax.add_patch(boundary_rect)

        # Plot workstations
        for ws_name in ['AS_1', 'AS_3', 'AS_4', 'AS_5', 'AS_6']:
            ws_x = ws_pos[f'{ws_name}_neu_X']
            ws_y = ws_pos[f'{ws_name}_neu_Y']
            ws_yaw = ws_pos[f'{ws_name}_neu_Yaw']

            # Get corners of the workstation
            corners = self.get_rotated_rectangle_corners(
                ws_x, ws_y,
                self.ws_length, self.ws_width,
                ws_yaw,
                ws_name
            )

            # Create polygon
            polygon = Polygon(corners,
                             fill=False,
                             edgecolor=self.colors[ws_name],
                             linewidth=2,
                             label=ws_name)
            ax.add_patch(polygon)

        # Sample points for visualization (to avoid overcrowding)
        # If there are too many points, sample a subset
        if len(df) > 10000:
            sample_size = 10000
            sampled_df = df.sample(sample_size, random_state=42)
        else:
            sampled_df = df

        # Plot points with colors based on annotations
        for annotation in sampled_df['annotation_specific'].unique():
            points = sampled_df[sampled_df['annotation_specific'] == annotation]

            if annotation.startswith('AS_'):
                color = self.colors[annotation]
                label = annotation
            elif annotation == 'robot':
                color = self.colors['robot']
                label = 'Robot Points'
            elif annotation.startswith('boundary'):
                color = self.colors['boundary']
                label = 'Boundary Points'
            else:
                color = self.colors['unknown']
                label = 'Unknown Points'

            ax.scatter(points['robot_1_global_x_radar'],
                      points['robot_1_global_y_radar'],
                      c=color,
                      s=5,  # Smaller point size for summary
                      alpha=0.5,
                      label=label)

        # Set title and labels
        ax.set_title(f"Dataset Summary: {dataset_name}\nTotal Points: {len(df)}\nWorkstation Edges Only", fontsize=14)
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)

        # Set equal aspect ratio
        ax.set_aspect('equal')

        # Set limits with some padding
        padding = 1.0  # 1 meter padding
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

        # Add legend
        ax.legend(loc='upper right', fontsize=10)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add annotation statistics
        annotation_stats = df['annotation_general'].value_counts()
        stats_text = "Annotation Statistics:\n"
        for category, count in annotation_stats.items():
            percentage = 100 * count / len(df)
            stats_text += f"{category}: {count} ({percentage:.1f}%)\n"

        # Add text box with statistics
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        # Save figure in the main visualization directory (not in a subdirectory)
        os.makedirs(self.visualize_dir, exist_ok=True)

        fig_path = os.path.join(self.visualize_dir, f"summary_{dataset_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved summary visualization to: {fig_path}")

    def process_all_files(self, visualize_frames_per_dataset=0):
        """
        Process all CSV files in the input directory.

        Args:
            visualize_frames_per_dataset (int): Number of frames to visualize per dataset
        """
        logger.info(f"Starting annotation process for files in: {self.input_dir}")

        # Find all CSV files
        csv_files = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.csv') and 'boundary_only_cleaned' in file:
                    csv_files.append(os.path.join(root, file))

        logger.info(f"Found {len(csv_files)} CSV files to process")

        # Get list of already processed datasets
        processed_datasets = []
        for root, _, files in os.walk(self.output_dir):
            for file in files:
                if file.startswith('annotated_') and file.endswith('.csv'):
                    dataset_dir = os.path.basename(os.path.dirname(os.path.join(root, file)))
                    processed_datasets.append(dataset_dir)

        logger.info(f"Found {len(processed_datasets)} already processed datasets")

        # Process each file and create summary visualizations
        for file_path in csv_files:
            # Get dataset name from file path
            dataset_dir = os.path.basename(os.path.dirname(file_path))

            # Skip if already processed
            if dataset_dir in processed_datasets:
                logger.info(f"Skipping already processed dataset: {dataset_dir}")
                continue

            # Annotate the file
            annotated_file = self.annotate_file(file_path, visualize_frames=visualize_frames_per_dataset)

            # Create summary visualization for the dataset
            self.create_dataset_summary(dataset_dir, annotated_file)

        logger.info("Annotation process completed")


def main():
    """Main function to run the annotation process."""
    # Input and output directories
    input_dir = "data/Cleaned_data_variations/boundary_only_improved"
    output_dir = "data/Annotated_data/boundary_only_improved_annotated"
    ws_position_file = "data/workstation_position_with_edges_rotated.xlsx"
    visualize_dir = "data/Annotated_data/visualizations"

    # Create annotator
    annotator = PointCloudAnnotator(
        input_dir=input_dir,
        output_dir=output_dir,
        ws_position_file=ws_position_file,
        visualize_dir=visualize_dir
    )

    # Process all files without frame visualization to prevent PC from hanging
    annotator.process_all_files(visualize_frames_per_dataset=0)


if __name__ == "__main__":
    main()
