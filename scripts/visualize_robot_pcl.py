#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import glob


def find_transformed_datasets(base_dir):
    """Find all transformed dataset files in the directory structure."""
    pattern = os.path.join(base_dir, "**", "transformed_*.csv")
    return glob.glob(pattern, recursive=True)


def visualize_robot_pcl(data_path, output_dir=None, show_plot=True):
    """
    Visualize the point cloud data from each robot separately.
    
    Args:
        data_path: Path to the transformed data CSV file
        output_dir: Directory to save the visualization images
        show_plot: Whether to display the plots interactively
    """
    # Define warehouse boundaries
    warehouse_bounds = {
        'x_min': -9.1, 'x_max': 10.2,
        'y_min': -4.42, 'y_max': 5.5,
    }
    
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load the transformed data
    print(f"Loading data from: {data_path}")
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    print(f"Loaded data with {len(data)} rows")
    
    # Extract scenario name from the path
    scenario = os.path.basename(os.path.dirname(os.path.dirname(data_path)))
    dataset_name = os.path.basename(os.path.dirname(data_path))
    
    # Check for required columns
    required_robot1_cols = ['robot_1_global_x_radar', 'robot_1_global_y_radar', 'robot_1_global_z_radar']
    required_robot2_cols = ['robot_2_global_x_radar', 'robot_2_global_y_radar', 'robot_2_global_z_radar']
    
    has_robot1_data = all(col in data.columns for col in required_robot1_cols)
    has_robot2_data = all(col in data.columns for col in required_robot2_cols)
    
    if not (has_robot1_data or has_robot2_data):
        print("Error: No radar data found in the dataset")
        return False
    
    # Filter out NaN values
    if has_robot1_data:
        robot1_data = data.dropna(subset=required_robot1_cols)
        print(f"Robot 1 has {len(robot1_data)} valid points")
    
    if has_robot2_data:
        robot2_data = data.dropna(subset=required_robot2_cols)
        print(f"Robot 2 has {len(robot2_data)} valid points")
    
    # Extract robot positions if available
    robot_positions = {}
    
    if 'robot_1_global_x' in data.columns and 'robot_1_global_y' in data.columns:
        robot_positions['robot_1'] = {
            'x': data['robot_1_global_x'].mean(),
            'y': data['robot_1_global_y'].mean()
        }
    
    if 'robot_2_global_x' in data.columns and 'robot_2_global_y' in data.columns:
        robot_positions['robot_2'] = {
            'x': data['robot_2_global_x'].mean(),
            'y': data['robot_2_global_y'].mean()
        }
    
    # Function to draw warehouse boundary
    def draw_warehouse_boundary(ax):
        # Draw warehouse boundary as a rectangle
        boundary_x = [
            warehouse_bounds['x_min'], warehouse_bounds['x_max'],
            warehouse_bounds['x_max'], warehouse_bounds['x_min'],
            warehouse_bounds['x_min']
        ]
        boundary_y = [
            warehouse_bounds['y_min'], warehouse_bounds['y_min'],
            warehouse_bounds['y_max'], warehouse_bounds['y_max'],
            warehouse_bounds['y_min']
        ]
        ax.plot(boundary_x, boundary_y, 'k-', linewidth=2, label='Warehouse Boundary')
        
        # Set axis limits to show the entire warehouse with a margin
        margin = 1.0
        ax.set_xlim(warehouse_bounds['x_min'] - margin, warehouse_bounds['x_max'] + margin)
        ax.set_ylim(warehouse_bounds['y_min'] - margin, warehouse_bounds['y_max'] + margin)
    
    # Create separate plots for each robot
    for robot_id, required_cols in [
        ('robot_1', required_robot1_cols),
        ('robot_2', required_robot2_cols)
    ]:
        if robot_id == 'robot_1' and has_robot1_data:
            robot_data = robot1_data
            color = 'blue'
            alpha = 0.6
        elif robot_id == 'robot_2' and has_robot2_data:
            robot_data = robot2_data
            color = 'red'
            alpha = 0.6
        else:
            continue
        
        # Create figure for this robot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw warehouse boundary
        draw_warehouse_boundary(ax)
        
        # Plot the point cloud
        ax.scatter(
            robot_data[f'{robot_id}_global_x_radar'],
            robot_data[f'{robot_id}_global_y_radar'],
            c=color, alpha=alpha, s=5,
            label=f'{robot_id.capitalize()} Radar Points'
        )
        
        # Plot robot positions if available
        for r_id, pos in robot_positions.items():
            marker = '^' if r_id == robot_id else 'o'
            size = 150 if r_id == robot_id else 100
            r_color = 'blue' if r_id == 'robot_1' else 'red'
            
            ax.scatter(
                pos['x'], pos['y'],
                color=r_color, marker=marker, s=size, edgecolor='black',
                label=f'{r_id.capitalize()} Position'
            )
        
        # Add title and labels
        ax.set_title(f'{robot_id.capitalize()} Point Cloud - {scenario} - {dataset_name}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True)
        ax.legend()
        
        # Make axes equal for proper scaling
        ax.set_aspect('equal')
        
        # Save figure if output directory provided
        if output_dir:
            output_file = os.path.join(output_dir, f"{robot_id}_pcl_{dataset_name}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved {robot_id} visualization to: {output_file}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
    
    # Create a combined plot with both robots
    if has_robot1_data and has_robot2_data:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw warehouse boundary
        draw_warehouse_boundary(ax)
        
        # Plot the point clouds with different colors
        ax.scatter(
            robot1_data['robot_1_global_x_radar'],
            robot1_data['robot_1_global_y_radar'],
            c='blue', alpha=0.5, s=5,
            label='Robot 1 Radar Points'
        )
        
        ax.scatter(
            robot2_data['robot_2_global_x_radar'],
            robot2_data['robot_2_global_y_radar'],
            c='red', alpha=0.5, s=5,
            label='Robot 2 Radar Points'
        )
        
        # Plot robot positions
        for r_id, pos in robot_positions.items():
            r_color = 'blue' if r_id == 'robot_1' else 'red'
            ax.scatter(
                pos['x'], pos['y'],
                color=r_color, marker='^', s=150, edgecolor='black',
                label=f'{r_id.capitalize()} Position'
            )
        
        # Add title and labels
        ax.set_title(f'Combined Point Cloud - {scenario} - {dataset_name}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True)
        ax.legend()
        
        # Make axes equal for proper scaling
        ax.set_aspect('equal')
        
        # Save figure if output directory provided
        if output_dir:
            output_file = os.path.join(output_dir, f"combined_pcl_{dataset_name}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved combined visualization to: {output_file}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Visualize robot point clouds separately')
    parser.add_argument('--input', '-i', type=str, help='Path to transformed data CSV file')
    parser.add_argument('--dir', '-d', type=str, help='Directory to search for all transformed datasets')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='Directory to save visualization images')
    parser.add_argument('--no-show', action='store_true', 
                        help='Do not display plots interactively')
    
    args = parser.parse_args()
    
    if args.input:
        # Visualize a single dataset
        visualize_robot_pcl(args.input, args.output, not args.no_show)
    elif args.dir:
        # Find and visualize all datasets
        datasets = find_transformed_datasets(args.dir)
        print(f"Found {len(datasets)} transformed datasets")
        
        for dataset in datasets:
            print(f"\nProcessing dataset: {dataset}")
            visualize_robot_pcl(dataset, args.output, not args.no_show)
    else:
        print("Error: Please provide either --input or --dir argument")
        return 1
    
    return 0


if __name__ == "__main__":
    main()