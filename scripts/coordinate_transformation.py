"""
coordinate_transformation module.
"""

#!/usr/bin/env python3
"""
Coordinate Transformation for Collaborative Perception

This module implements coordinate frame transformations for collaborative
perception in warehouse robotics. It transforms local radar point cloud data
from multiple robots to a unified global coordinate frame using high-precision
Vicon motion capture data as ground truth.

Key Features:
- Multi-robot coordinate frame transformation (robot-local to global)
- Vicon motion capture integration for precise positioning
- Sensor offset compensation for accurate spatial alignment
- Temporal synchronization handling for multi-robot data
- Comprehensive validation and visualization capabilities

The transformation enables true collaborative perception by aligning sensor
data from multiple robots in a common spatial reference frame, essential
for multi-robot sensor fusion and collaborative occupancy prediction.

Author: Thiyanayugi Mariraj
Project: Development of a Framework for Collaborative Perception Management Layer
         for Future 6G-Enabled Robotic Systems
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path


class CollaborativePerceptionCoordinateTransformer:
    """
    Coordinate Frame Transformer for Multi-Robot Collaborative Perception.

    This class handles the transformation of local radar point cloud data from
    multiple autonomous robots to a unified global coordinate frame. It uses
    high-precision Vicon motion capture data to establish accurate spatial
    relationships between robots and transform their sensor measurements.

    The transformer is essential for collaborative perception as it enables
    the fusion of sensor data from multiple robots by aligning their local
    coordinate frames to a common global reference frame.

    Key Capabilities:
    - Robot-local to global coordinate transformation
    - Sensor mounting offset compensation
    - Temporal synchronization handling
    - Transformation validation and visualization
    - Support for multiple robot configurations
    """

    def __init__(self, synchronized_data_path=None):
        """
        Initialize the coordinate transformer for collaborative perception.

        Sets up the transformer with synchronized multi-robot data and defines
        sensor mounting configurations for accurate coordinate transformations.

        Args:
            synchronized_data_path (str, optional): Path to synchronized CSV file
                                                   containing multi-robot sensor data
        """
        self.data_file_path = synchronized_data_path
        self.raw_synchronized_data = None
        self.globally_transformed_data = None

        # Define sensor mounting offsets relative to robot center
        # These offsets account for physical sensor placement on each robot
        self.robot_sensor_mounting_offsets = {
            'robot_1': {
                'translation_x_meters': 0.0,    # Forward/backward offset
                'translation_y_meters': 0.0,    # Left/right offset
                'translation_z_meters': 0.1,    # Vertical offset (sensor height)
                'rotation_roll_radians': 0.0,   # Roll rotation offset
                'rotation_pitch_radians': 0.0,  # Pitch rotation offset
                'rotation_yaw_radians': 0.0,    # Yaw rotation offset
            },
            'robot_2': {
                'translation_x_meters': 0.0,    # Forward/backward offset
                'translation_y_meters': 0.0,    # Left/right offset
                'translation_z_meters': 0.1,    # Vertical offset (sensor height)
                'rotation_roll_radians': 0.0,   # Roll rotation offset
                'rotation_pitch_radians': 0.0,  # Pitch rotation offset
                'rotation_yaw_radians': 0.0,    # Yaw rotation offset
            }
        }
        
        # Mapping from dataset column names to our standardized column names
        self.column_mapping = {
            # Robot 1 (ep03)
            'ep03_pos_x': 'robot_1_global_x',
            'ep03_pos_y': 'robot_1_global_y',
            'ep03_pos_z': 'robot_1_global_z',
            'ep03_yaw': 'robot_1_yaw',
            'ep03_rot_x': 'robot_1_roll',
            'ep03_rot_y': 'robot_1_pitch',
            'ep03_rot_z': 'robot_1_rot_z',  # Additional rotation component
            
            # Robot 2 (ep05)
            'ep05_pos_x': 'robot_2_global_x',
            'ep05_pos_y': 'robot_2_global_y',
            'ep05_pos_z': 'robot_2_global_z',
            'ep05_yaw': 'robot_2_yaw',
            'ep05_rot_x': 'robot_2_roll',
            'ep05_rot_y': 'robot_2_pitch',
            'ep05_rot_z': 'robot_2_rot_z',  # Additional rotation component
            
            # Radar points from robot 1
            'robot_1_x': 'robot_1_local_x',
            'robot_1_y': 'robot_1_local_y',
            'robot_1_z': 'robot_1_local_z',
            
            # Radar points from robot 2
            'robot_2_x': 'robot_2_local_x',
            'robot_2_y': 'robot_2_local_y',
            'robot_2_z': 'robot_2_local_z',
        }
        
    def load_data(self, data_path=None):
        """
        Load synchronized data from CSV file.
        
        Args:
            data_path (str, optional): Path to override the initialized path
            
        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        if data_path:
            self.data_path = data_path
            
        if not self.data_path:
            print("Error: No data path provided")
            return False
        
        try:
            print(f"Loading synchronized data from: {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            
            # Rename columns for consistency
            rename_dict = {}
            for old_col, new_col in self.column_mapping.items():
                if old_col in self.data.columns:
                    rename_dict[old_col] = new_col
            
            if rename_dict:
                self.data = self.data.rename(columns=rename_dict)
                
            print(f"Loaded {len(self.data)} synchronized data points")
            print(f"Available columns: {', '.join(self.data.columns)}")
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
            
    def create_rotation_matrix(self, roll, pitch, yaw):
        """
        Create a 3x3 rotation matrix from roll, pitch, and yaw angles.
        
        Args:
            roll (float): Roll angle in radians
            pitch (float): Pitch angle in radians
            yaw (float): Yaw angle in radians
            
        Returns:
            numpy.ndarray: 3x3 rotation matrix
        """
        # Convert to shorter variable names for readability
        r, p, y = roll, pitch, yaw
        
        # Calculate trigonometric functions
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        
        # Construct rotation matrix (Z-Y-X order: yaw, pitch, roll)
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        
        return R
        
    def create_2d_rotation_matrix(self, yaw):
        """
        Create a 2x2 rotation matrix from yaw angle.
        
        Args:
            yaw (float): Yaw angle in radians
                
        Returns:
            numpy.ndarray: 2x2 rotation matrix
        """
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        # 2D rotation matrix
        R = np.array([
            [cy, -sy],
            [sy, cy]
        ])
        
        return R
        
    def transform_point_cloud(self, robot_id):
        """
        Transform local radar point cloud from a robot to the global frame.
        
        Args:
            robot_id (str): Identifier for the robot ('robot_1' or 'robot_2')
            
        Returns:
            pandas.DataFrame: DataFrame with transformed points added
        """
        if self.data is None:
            print("Error: No data loaded")
            return None
            
        # Create new columns for transformed points
        self.data[f'{robot_id}_global_x_radar'] = np.nan
        self.data[f'{robot_id}_global_y_radar'] = np.nan
        self.data[f'{robot_id}_global_z_radar'] = np.nan
        
        # Get required column names
        local_x_col = f'{robot_id}_local_x'
        local_y_col = f'{robot_id}_local_y'
        local_z_col = f'{robot_id}_local_z'
        
        global_x_col = f'{robot_id}_global_x'
        global_y_col = f'{robot_id}_global_y'
        global_z_col = f'{robot_id}_global_z'
        
        roll_col = f'{robot_id}_roll'
        pitch_col = f'{robot_id}_pitch'
        yaw_col = f'{robot_id}_yaw'
        
        # Check if required columns exist
        required_cols = [local_x_col, local_y_col, local_z_col, 
                        global_x_col, global_y_col, global_z_col,
                        roll_col, pitch_col, yaw_col]
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {', '.join(missing_cols)}")
            return None
        
        # Get sensor offsets for this robot
        offset = self.sensor_offsets[robot_id]
        
        # Transform each point
        rows_transformed = 0
        
        for i, row in self.data.iterrows():
            # Skip rows with missing data
            if (pd.isna(row[local_x_col]) or pd.isna(row[global_x_col]) or 
                pd.isna(row[roll_col]) or pd.isna(row[yaw_col])):
                continue
                
            # Create rotation matrix
            R = self.create_rotation_matrix(
                roll=row[roll_col], 
                pitch=row[pitch_col], 
                yaw=row[yaw_col]
            )
            
            # Local point relative to sensor
            p_local = np.array([
                row[local_x_col],
                row[local_y_col],
                row[local_z_col]
            ])
            
            # Account for sensor offset in robot frame
            sensor_offset_vector = np.array([
                offset['x'],
                offset['y'],
                offset['z']
            ])
            
            # Apply sensor offset in local frame
            p_local_adjusted = p_local + sensor_offset_vector
            
            # Apply rotation
            p_rotated = R @ p_local_adjusted
            
            # Global translation (robot position in global frame)
            T = np.array([
                row[global_x_col],
                row[global_y_col],
                row[global_z_col]
            ])
            
            # Apply translation to get global coordinates
            p_global = p_rotated + T
            
            # Store transformed coordinates
            self.data.at[i, f'{robot_id}_global_x_radar'] = p_global[0]
            self.data.at[i, f'{robot_id}_global_y_radar'] = p_global[1]
            self.data.at[i, f'{robot_id}_global_z_radar'] = p_global[2]
            
            rows_transformed += 1
            
        print(f"Transformed {rows_transformed} points for {robot_id}")
        return self.data
        
    def transform_point_cloud_2d(self, robot_id):
        """
        Transform local radar point cloud from a robot to the global frame using 2D transformation.
        
        Args:
            robot_id (str): Identifier for the robot ('robot_1' or 'robot_2')
            
        Returns:
            pandas.DataFrame: DataFrame with transformed points added
        """
        if self.data is None:
            print("Error: No data loaded")
            return None
            
        # Create new columns for transformed points
        self.data[f'{robot_id}_global_x_radar'] = np.nan
        self.data[f'{robot_id}_global_y_radar'] = np.nan
        self.data[f'{robot_id}_global_z_radar'] = np.nan
        
        # Add radar-specific columns
        radar_specific_cols = ['range', 'azimuth', 'elevation', 'snr']
        for col in radar_specific_cols:
            source_col = f'{robot_id}_{col}'
            global_col = f'{robot_id}_global_{col}_radar'
            
            if source_col in self.data.columns:
                # For non-azimuth values, copy directly
                if col != 'azimuth':
                    self.data[global_col] = self.data[source_col]
        
        # Get required column names - use the actual columns in your dataset
        local_x_col = f'{robot_id}_local_x'
        local_y_col = f'{robot_id}_local_y'
        local_z_col = f'{robot_id}_local_z'
        
        global_x_col = f'{robot_id}_global_x'
        global_y_col = f'{robot_id}_global_y'
        global_z_col = f'{robot_id}_global_z'
        
        yaw_col = f'{robot_id}_yaw'
        
        # Check if required columns exist
        required_cols = [local_x_col, local_y_col, local_z_col, 
                        global_x_col, global_y_col, yaw_col]
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {', '.join(missing_cols)}")
            return None
        
        # Get sensor offsets for this robot
        offset = self.sensor_offsets[robot_id]
        
        # Transform each point
        rows_transformed = 0
        
        for i, row in self.data.iterrows():
            # Skip rows with missing data
            if (pd.isna(row[local_x_col]) or pd.isna(row[global_x_col]) or 
                pd.isna(row[yaw_col])):
                continue
                
            # Create 2D rotation matrix
            yaw = row[yaw_col]
            cy, sy = np.cos(yaw), np.sin(yaw)
            R_2d = np.array([[cy, -sy], [sy, cy]])
            
            # Local point relative to sensor (2D)
            p_local_2d = np.array([
                row[local_x_col] + offset['x'],
                row[local_y_col] + offset['y']
            ])
            
            # Apply 2D rotation
            p_rotated_2d = R_2d @ p_local_2d
            
            # Global translation (robot position in global frame)
            T_2d = np.array([
                row[global_x_col],
                row[global_y_col]
            ])
            
            # Apply translation to get global coordinates (2D)
            p_global_2d = p_rotated_2d + T_2d
            
            # For Z coordinate, we simply add the local z to the robot's global z
            z_global = row[global_z_col] + row[local_z_col] + offset['z']
            
            # Store transformed coordinates
            self.data.at[i, f'{robot_id}_global_x_radar'] = p_global_2d[0]
            self.data.at[i, f'{robot_id}_global_y_radar'] = p_global_2d[1]
            self.data.at[i, f'{robot_id}_global_z_radar'] = z_global
            
            # Transform azimuth if that column exists
            azimuth_col = f'{robot_id}_azimuth'
            if azimuth_col in self.data.columns and not pd.isna(row[azimuth_col]):
                # Convert from degrees to radians if needed
                local_azimuth = np.radians(row[azimuth_col])
                global_azimuth = local_azimuth + yaw
                # Normalize to -π to π range
                global_azimuth = np.arctan2(np.sin(global_azimuth), np.cos(global_azimuth))
                # Convert back to degrees if your original values were in degrees
                self.data.at[i, f'{robot_id}_global_azimuth_radar'] = np.degrees(global_azimuth)
            
            rows_transformed += 1
            
        print(f"Transformed {rows_transformed} points for {robot_id} using 2D transformation")
        return self.data
        
    def transform_all_robots(self):
        """
        Transform point clouds from all robots using 2D transformation.
        
        Returns:
            pandas.DataFrame: DataFrame with all transformed points
        """
        self.transform_point_cloud_2d('robot_1')
        self.transform_point_cloud_2d('robot_2')
        
        # Store transformed data
        self.transformed_data = self.data.copy()
        
        return self.transformed_data
    
    def save_transformed_data(self, output_path=None):
        """
        Save transformed data to a CSV file.
        
        Args:
            output_path (str, optional): Path to save the transformed data
            
        Returns:
            str: Path where data was saved
        """
        if self.transformed_data is None:
            print("Error: No transformed data to save")
            return None
            
        if output_path is None:
            # Create output path based on input path
            if self.data_path:
                path_obj = Path(self.data_path)
                output_path = str(path_obj.parent / f"transformed_{path_obj.name}")
            else:
                output_path = "transformed_data.csv"
                
        # Save to CSV
        self.transformed_data.to_csv(output_path, index=False)
        print(f"Saved transformed data to: {output_path}")
        
        return output_path
    
    def validate_transformation(self):
        """
        Validate the transformation by comparing points that should coincide.
        
        Returns:
            dict: Validation metrics
        """
        if self.transformed_data is None:
            print("Error: No transformed data to validate")
            return None
            
        # For validation, we'll use points from both robots that should be
        # the same physical object in the environment
        # In real-world usage, you'd have correspondence information
        # For now, just measure overall distribution of distances
        
        # Check if we have the necessary columns
        required_cols = [
            'robot_1_global_x_radar', 'robot_1_global_y_radar', 'robot_1_global_z_radar',
            'robot_2_global_x_radar', 'robot_2_global_y_radar', 'robot_2_global_z_radar'
        ]
        
        missing_cols = [col for col in required_cols if col not in self.transformed_data.columns]
        if missing_cols:
            print(f"Error: Missing required columns for validation: {', '.join(missing_cols)}")
            return None
            
        # Calculate distances between robot positions in global frame
        distances = np.sqrt(
            (self.transformed_data['robot_1_global_x'] - self.transformed_data['robot_2_global_x'])**2 +
            (self.transformed_data['robot_1_global_y'] - self.transformed_data['robot_2_global_y'])**2 +
            (self.transformed_data['robot_1_global_z'] - self.transformed_data['robot_2_global_z'])**2
        )
        
        # Calculate statistics
        metrics = {
            'mean_robot_distance': distances.mean(),
            'std_robot_distance': distances.std(),
            'min_robot_distance': distances.min(),
            'max_robot_distance': distances.max(),
        }
        
        print(f"Validation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
            
        return metrics
    
    def visualize_transformation(self, plot_type='2d'):
        """
        Visualize the transformed point clouds.
        
        Args:
            plot_type (str): Type of plot ('2d' or '3d')
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if self.transformed_data is None:
            print("Error: No transformed data to visualize")
            return None
            
        # Filter out rows with NaN values in transformed coordinates
        data = self.transformed_data.dropna(subset=[
            'robot_1_global_x_radar', 'robot_1_global_y_radar', 'robot_1_global_z_radar',
            'robot_2_global_x_radar', 'robot_2_global_y_radar', 'robot_2_global_z_radar'
        ])
        
        if len(data) == 0:
            print("Error: No valid data points for visualization")
            return None
            
        if plot_type == '3d':
            # Create 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot robot positions
            ax.scatter(data['robot_1_global_x'], data['robot_1_global_y'], data['robot_1_global_z'],
                      color='blue', marker='^', s=100, label='Robot 1 Position')
            ax.scatter(data['robot_2_global_x'], data['robot_2_global_y'], data['robot_2_global_z'],
                      color='red', marker='^', s=100, label='Robot 2 Position')
            
            # Plot transformed radar points
            ax.scatter(data['robot_1_global_x_radar'], data['robot_1_global_y_radar'], data['robot_1_global_z_radar'],
                      color='lightblue', alpha=0.5, s=20, label='Robot 1 Radar Points')
            ax.scatter(data['robot_2_global_x_radar'], data['robot_2_global_y_radar'], data['robot_2_global_z_radar'],
                      color='salmon', alpha=0.5, s=20, label='Robot 2 Radar Points')
                      
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            
        else:
            # Create 2D plot (top-down view)
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot robot positions
            ax.scatter(data['robot_1_global_x'], data['robot_1_global_y'],
                      color='blue', marker='^', s=100, label='Robot 1 Position')
            ax.scatter(data['robot_2_global_x'], data['robot_2_global_y'],
                      color='red', marker='^', s=100, label='Robot 2 Position')
            
            # Plot transformed radar points
            ax.scatter(data['robot_1_global_x_radar'], data['robot_1_global_y_radar'],
                      color='lightblue', alpha=0.5, s=20, label='Robot 1 Radar Points')
            ax.scatter(data['robot_2_global_x_radar'], data['robot_2_global_y_radar'],
                      color='salmon', alpha=0.5, s=20, label='Robot 2 Radar Points')
                      
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            
        # Add legend and title
        ax.legend()
        plt.title('Transformed Radar Points in Global Coordinate Frame')
        
        # Add grid
        ax.grid(True)
        
        return fig
    
    def create_global_point_cloud(self):
        """
        Combine the transformed point clouds from all robots into a single global point cloud.
        
        Returns:
            pandas.DataFrame: Combined point cloud in global frame
        """
        if self.transformed_data is None:
            print("Error: No transformed data available")
            return None
            
        # Extract global radar points from each robot
        robot1_points = self.transformed_data[
            ['vicon_timestamp', 'robot_1_global_x_radar', 'robot_1_global_y_radar', 'robot_1_global_z_radar']
        ].rename(columns={
            'robot_1_global_x_radar': 'global_x',
            'robot_1_global_y_radar': 'global_y',
            'robot_1_global_z_radar': 'global_z'
        })
        robot1_points = robot1_points.dropna(subset=['global_x', 'global_y', 'global_z'])
        robot1_points['source'] = 'robot_1'
        
        robot2_points = self.transformed_data[
            ['vicon_timestamp', 'robot_2_global_x_radar', 'robot_2_global_y_radar', 'robot_2_global_z_radar']
        ].rename(columns={
            'robot_2_global_x_radar': 'global_x',
            'robot_2_global_y_radar': 'global_y',
            'robot_2_global_z_radar': 'global_z'
        })
        robot2_points = robot2_points.dropna(subset=['global_x', 'global_y', 'global_z'])
        robot2_points['source'] = 'robot_2'
        
        # Print diagnostic info
        print(f"Found {len(robot1_points)} valid points from robot_1")
        print(f"Found {len(robot2_points)} valid points from robot_2")
        
        # Combine points
        global_point_cloud = pd.concat([robot1_points, robot2_points])
        
        # Check and report any remaining NaN values
        nan_count = global_point_cloud.isna().sum().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values remain in global point cloud")
            global_point_cloud = global_point_cloud.dropna()
        
        print(f"Created global point cloud with {len(global_point_cloud)} points")
        return global_point_cloud


def main():
    """Main function to run the coordinate transformation."""
    parser = argparse.ArgumentParser(description='Transform radar data to global coordinate frame')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to synchronized data CSV file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save transformed data (default: auto-generated)')
    parser.add_argument('--plot', '-p', type=str, choices=['2d', '3d', 'none'], default='2d',
                        help='Type of visualization plot (default: 2d)')
    parser.add_argument('--save-plot', '-s', type=str, default=None,
                        help='Path to save visualization plot (default: not saved)')
    
    args = parser.parse_args()
    
    # Create coordinate transformer for collaborative perception
    transformer = CollaborativePerceptionCoordinateTransformer()
    
    # Load data
    if not transformer.load_data(args.input):
        return 1
    
    # Transform point clouds
    transformer.transform_all_robots()
    
    # Validate transformation
    transformer.validate_transformation()
    
    # Save transformed data
    transformer.save_transformed_data(args.output)
    
    # Create global point cloud
    global_point_cloud = transformer.create_global_point_cloud()
    
    # Visualization
    if args.plot != 'none':
        fig = transformer.visualize_transformation(plot_type=args.plot)
        
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {args.save_plot}")
        else:
            plt.show()
    
    return 0


if __name__ == "__main__":
    main()