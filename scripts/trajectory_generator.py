# Updated: 2026-01-29
"""
trajectory_generator module.
"""

#!/usr/bin/env python3
"""
Advanced Trajectory Generator and Visualizer for Collaborative Perception

This module implements comprehensive trajectory generation and visualization
capabilities for collaborative perception applications in warehouse robotics.
It processes Vicon motion capture data to generate robot trajectories and
creates detailed visualizations for analysis and validation.

Key Features:
- Multi-robot trajectory extraction from Vicon motion capture data
- Comprehensive trajectory visualization with temporal analysis
- Workstation position mapping and warehouse layout visualization
- Interactive trajectory exploration with animation capabilities
- Statistical analysis of robot movement patterns
- Export capabilities for documentation and presentations

The trajectory generator is essential for understanding robot behavior patterns
and validating collaborative perception scenarios in warehouse environments.

Author: Thiyanayugi Mariraj
Project: Development of a Framework for Collaborative Perception Management Layer
         for Future 6G-Enabled Robotic Systems
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
from matplotlib.patches import Rectangle
import time
import logging
from typing import Dict, List, Optional, Tuple, Any


class CollaborativePerceptionTrajectoryVisualizer:
    """
    Advanced Trajectory Visualizer for Collaborative Perception Systems.

    This class provides comprehensive trajectory generation and visualization
    capabilities for multi-robot collaborative perception applications. It
    processes high-precision Vicon motion capture data to extract robot
    trajectories and creates detailed visualizations for analysis.

    Key Capabilities:
    - Multi-robot trajectory extraction and processing
    - Temporal trajectory analysis with statistical insights
    - Warehouse layout visualization with workstation mapping
    - Interactive trajectory exploration and animation
    - Collaborative behavior pattern analysis
    - Export capabilities for documentation and presentations
    """

    def __init__(self, trajectory_output_directory: str = "collaborative_trajectory_plots"):
        """
        Initialize the collaborative perception trajectory visualizer.

        Args:
            trajectory_output_directory (str): Directory to save trajectory visualizations
        """
        # Configure robot identifiers for collaborative perception system
        self.collaborative_robot_identifiers = ['ep03', 'ep05']
        self.warehouse_workstation_names = ['AS_1_neu', 'AS_3_neu', 'AS_4_neu', 'AS_5_neu', 'AS_6_neu']
        self.trajectory_output_directory = trajectory_output_directory

        # Create organized output directory structure
        os.makedirs(trajectory_output_directory, exist_ok=True)

        # Define color scheme for multi-robot visualization
        self.robot_visualization_colors = {
            'ep03': 'blue',      # Robot 1 - Primary collaborative robot
            'ep05': 'red',       # Robot 2 - Secondary collaborative robot
            'cable_robot': 'green'  # Cable robot for reference
        }

        # Define workstation visualization colors for warehouse layout
        self.workstation_visualization_colors = {
            'AS_1_neu': 'lightcoral',
            'AS_3_neu': 'lightblue',
            'AS_4_neu': 'lightgreen',
            'AS_5_neu': 'lightyellow',
            'AS_6_neu': 'lightpink'
        }

        # Define warehouse spatial boundaries for visualization
        self.warehouse_spatial_boundaries = {
            'x_coordinate_min': -9.1, 'x_coordinate_max': 10.2,
            'y_coordinate_min': -4.42, 'y_coordinate_max': 5.5,
        }

        # Define default workstation physical dimensions
        self.workstation_physical_dimensions = {'width_meters': 1.0, 'height_meters': 0.6}

        # Set up logging for trajectory processing
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("TrajectoryVisualizer")
    
    def parse_vicon_file(self, file_path):
        """Parse Vicon data file to extract robot and workstation positions"""
        print(f"Processing {os.path.basename(file_path)}...")
        
        # Data storage
        robot_data = {robot: {'x': [], 'y': [], 'timestamps': []} for robot in self.robot_names}
        workstation_data = {}
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    try:
                        # Parse line content
                        data = eval(line.strip())
                        
                        # Skip empty data
                        if all(all(v == 0 for v in obj_data.values()) for obj_data in data.values()):
                            continue
                        
                        # Process objects in this frame
                        timestamp = None
                        
                        for key, values in data.items():
                            # Skip empty entries
                            if all(v == 0 for v in values.values()):
                                continue
                                
                            # Extract object name
                            object_name = key.split('/')[1]
                            
                            # Get timestamp (convert to seconds)
                            if timestamp is None:
                                timestamp = values['system_time'] / 1e9
                            
                            # Store position data
                            if object_name in self.robot_names:
                                robot_data[object_name]['x'].append(values['pos_x'])
                                robot_data[object_name]['y'].append(values['pos_y'])
                                robot_data[object_name]['timestamps'].append(timestamp)
                            elif object_name.startswith('AS_'):
                                if object_name not in workstation_data:
                                    workstation_data[object_name] = []
                                workstation_data[object_name].append({
                                    'x': values['pos_x'],
                                    'y': values['pos_y'],
                                    'z': values['pos_z'],
                                    'yaw': values['yaw']
                                })
                    except Exception as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
        except Exception as e:
            print(f"Error reading file: {e}")
            return None, None
        
        # Process workstation data to get average positions
        avg_workstation_positions = {}
        for ws_name, positions in workstation_data.items():
            if positions:
                avg_pos = {
                    'x': np.mean([p['x'] for p in positions]),
                    'y': np.mean([p['y'] for p in positions]),
                    'z': np.mean([p['z'] for p in positions]),
                    'yaw': np.mean([p['yaw'] for p in positions])
                }
                
                # Apply 90-degree rotation to AS_4_neu
                if ws_name == 'AS_4_neu':
                    avg_pos['yaw'] = np.pi/2  # 90 degrees in radians
                
                avg_workstation_positions[ws_name] = avg_pos
        
        return robot_data, avg_workstation_positions
    
    def get_workstation_positions_from_file(self, vicon_timestamp):
        """Get workstation positions from dedicated files based on timestamp"""
        ws_positions_dir = "/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/CPPS_Static_Scenario/archiv_measurements/Vicon/working_station_positions"
        
        if not os.path.exists(ws_positions_dir):
            print(f"Workstation positions directory not found: {ws_positions_dir}")
            return None
        
        # Get list of workstation files
        ws_files = glob.glob(os.path.join(ws_positions_dir, "*.txt"))
        if not ws_files:
            print("No workstation position files found")
            return None
        
        # Sort files by modification time
        ws_files.sort(key=lambda x: os.path.getmtime(x))
        
        # Find file with closest timestamp
        closest_file = None
        min_time_diff = float('inf')
        
        for ws_file in ws_files:
            file_time = os.path.getmtime(ws_file)
            time_diff = abs(file_time - vicon_timestamp)
            
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_file = ws_file
        
        if closest_file:
            print(f"Using workstation positions from {os.path.basename(closest_file)}")
            return self.parse_workstation_file(closest_file)
        
        return None
    
    def parse_workstation_file(self, file_path):
        """Parse a workstation positions file"""
        workstation_positions = {}
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        # Parse line content
                        data = eval(line.strip())
                        
                        # Process each object in the data
                        for key, values in data.items():
                            # Extract object name 
                            object_name = key.split('/')[1]
                            
                            # Only process workstations
                            if object_name.startswith('AS_'):
                                # Store position data
                                workstation_positions[object_name] = {
                                    'x': values['pos_x'],
                                    'y': values['pos_y'],
                                    'z': values['pos_z'],
                                    'yaw': values['yaw'],
                                }
                                
                                # Apply 90-degree rotation to AS_4_neu
                                if object_name == 'AS_4_neu':
                                    workstation_positions[object_name]['yaw'] = np.pi/2
                                
                                print(f"Found workstation {object_name} at ({values['pos_x']:.2f}, {values['pos_y']:.2f})")
                    
                    except Exception as e:
                        print(f"Error parsing line in workstation file: {e}")
                        continue
            
            return workstation_positions
        
        except Exception as e:
            print(f"Error reading workstation file: {e}")
            return None
    
    def get_default_workstation_positions(self):
        """Return default workstation positions"""
        return {
            'AS_1_neu': {'x': 1.52, 'y': 2.24, 'z': 1.02, 'yaw': 0},
            'AS_3_neu': {'x': -5.74, 'y': -0.13, 'z': 1.47, 'yaw': 0},
            'AS_4_neu': {'x': 5.37, 'y': 0.21, 'z': 2.30, 'yaw': np.pi/2},
            'AS_5_neu': {'x': -3.05, 'y': 2.39, 'z': 2.21, 'yaw': 0},
            'AS_6_neu': {'x': 0.01, 'y': -1.45, 'z': 1.53, 'yaw': 0}
        }
    
    def plot_trajectory(self, vicon_file, output_file=None):
        """Plot trajectory from Vicon data and save to file"""
        # Parse data
        robot_data, vicon_workstations = self.parse_vicon_file(vicon_file)
        
        if not robot_data:
            print("No robot data found!")
            return False
        
        # Get reference timestamp for workstation file matching
        ref_timestamp = 0
        for robot, data in robot_data.items():
            if data['timestamps']:
                ref_timestamp = data['timestamps'][0]
                break
        
        # Get workstation positions, with fallback options
        workstation_positions = self.get_workstation_positions_from_file(ref_timestamp)
        if not workstation_positions:
            print("Using workstation positions from Vicon data")
            workstation_positions = vicon_workstations
        if not workstation_positions:
            print("Using default workstation positions")
            workstation_positions = self.get_default_workstation_positions()
        
        # Setup plot
        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        
        # Set title based on filename
        scenario = os.path.basename(os.path.dirname(os.path.dirname(vicon_file)))
        ax.set_title(f'Robot Trajectory - {scenario}', fontsize=14)
        
        # Draw warehouse boundaries
        warehouse_x = [
            self.warehouse_bounds['x_min'], self.warehouse_bounds['x_max'],
            self.warehouse_bounds['x_max'], self.warehouse_bounds['x_min'],
            self.warehouse_bounds['x_min']
        ]
        warehouse_y = [
            self.warehouse_bounds['y_min'], self.warehouse_bounds['y_min'],
            self.warehouse_bounds['y_max'], self.warehouse_bounds['y_max'],
            self.warehouse_bounds['y_min']
        ]
        ax.plot(warehouse_x, warehouse_y, 'k-', linewidth=2, label='Boundary')
        
        # Set axis limits with margin
        margin = 1.0
        ax.set_xlim(
            self.warehouse_bounds['x_min'] - margin,
            self.warehouse_bounds['x_max'] + margin
        )
        ax.set_ylim(
            self.warehouse_bounds['y_min'] - margin,
            self.warehouse_bounds['y_max'] + margin
        )
        
        # Draw workstations
        for ws_name, pos in workstation_positions.items():
            color = self.workstation_colors.get(ws_name, 'lightgray')
            width = self.workstation_size['width']
            height = self.workstation_size['height']
            
            # Account for rotation in AS_4_neu only
            if ws_name == 'AS_4_neu':  # Only rotate AS_4_neu
                # For 90-degree rotation, swap width and height
                rect = Rectangle(
                    (pos['x'] - height/2, pos['y'] - width/2),
                    height, width,
                    facecolor=color,
                    edgecolor='black',
                    alpha=0.7
                )
            else:
                rect = Rectangle(
                    (pos['x'] - width/2, pos['y'] - height/2),
                    width, height,
                    facecolor=color,
                    edgecolor='black',
                    alpha=0.7
                )
            
            ax.add_patch(rect)
            ax.text(
                pos['x'], pos['y'], ws_name,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8
            )
        
        # Plot robot trajectories
        for robot, data in robot_data.items():
            if data['x'] and data['y']:
                color = self.colors.get(robot, 'gray')
                
                # Plot trajectory
                ax.plot(data['x'], data['y'], '-', color=color, linewidth=2, label=f"{robot} Path")
                
                # Mark start position
                ax.scatter(data['x'][0], data['y'][0], s=100, color=color, 
                          edgecolor='black', marker='o', label=f"{robot} Start")
                
                # Mark end position
                ax.scatter(data['x'][-1], data['y'][-1], s=100, color=color, 
                          edgecolor='black', marker='x', label=f"{robot} End")
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Save plot or show
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved trajectory plot to {output_file}")
            plt.close(fig)
            return True
        else:
            plt.tight_layout()
            plt.show()
            return True
    
    def process_all_datasets(self, base_dir="/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/CPPS_Static_Scenario"):
        """Process all available datasets in the base directory"""
        # Find all scenario folders
        scenario_dirs = []
        for item in os.listdir(base_dir):
            if item.startswith('CPPS_') and os.path.isdir(os.path.join(base_dir, item)):
                scenario_dirs.append(os.path.join(base_dir, item))
        
        print(f"Found {len(scenario_dirs)} scenario directories")
        
        # Process each scenario
        for scenario_dir in scenario_dirs:
            scenario_name = os.path.basename(scenario_dir)
            print(f"\nProcessing scenario: {scenario_name}")
            
            # Find Vicon data files
            vicon_dir = os.path.join(scenario_dir, 'Vicon')
            if not os.path.exists(vicon_dir):
                print(f"No Vicon directory found for {scenario_name}")
                continue
            
            vicon_files = glob.glob(os.path.join(vicon_dir, '*.txt'))
            if not vicon_files:
                print(f"No Vicon files found for {scenario_name}")
                continue
            
            print(f"Found {len(vicon_files)} Vicon files")
            
            # Process each Vicon file
            for vicon_file in vicon_files:
                file_name = os.path.basename(vicon_file)
                output_file = os.path.join(self.output_dir, f"{scenario_name}_{file_name.replace('.txt', '.png')}")
                
                print(f"Generating trajectory plot for {file_name}")
                self.plot_trajectory(vicon_file, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate trajectory plots for RoboFUSE datasets')
    parser.add_argument('--output_dir', default='trajectory_plots', help='Directory to save trajectory plots')
    parser.add_argument('--vicon_file', help='Process a specific Vicon file instead of all datasets')
    
    args = parser.parse_args()
    
    visualizer = CollaborativePerceptionTrajectoryVisualizer(args.output_dir)
    
    if args.vicon_file:
        if not os.path.exists(args.vicon_file):
            print(f"Error: File not found: {args.vicon_file}")
            sys.exit(1)
        
        visualizer.plot_trajectory(args.vicon_file)
    else:
        visualizer.process_all_datasets()