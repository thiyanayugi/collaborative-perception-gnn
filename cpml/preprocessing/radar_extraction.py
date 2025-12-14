#!/usr/bin/env python3
"""
mmWave Radar Data Extractor for Collaborative Perception

This module extracts and processes mmWave radar point cloud data from ROS2 bag files
for collaborative perception applications in warehouse robotics. It handles real-time
extraction of radar measurements from multiple robots and prepares the data for
collaborative perception processing.

Key Features:
- Real-time radar point cloud extraction from ROS2 topics
- Multi-robot data handling with robot identification
- Temporal synchronization with bag file timestamps
- Structured data output for collaborative perception pipeline
- Comprehensive logging and progress tracking

The extractor processes TI mmWave radar sensor data and outputs structured
point clouds suitable for collaborative perception graph generation.

Author: Thiyanayugi Mariraj
Project: Development of a Framework for Collaborative Perception Management Layer
         for Future 6G-Enabled Robotic Systems
"""

import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import os
import struct
from datetime import datetime
import subprocess
import threading
import pathlib
import json
import tempfile
from typing import Dict, List, Any, Optional
import re
import sqlite3
import glob
import math

from sensor_msgs.msg import PointCloud2


class CollaborativeRadarDataExtractor(Node):
    """
    mmWave Radar Data Extractor for Multi-Robot Collaborative Perception.

    This class extracts radar point cloud data from ROS2 bag files containing
    recordings from multiple autonomous robots equipped with TI mmWave radar
    sensors. The extracted data is structured for collaborative perception
    processing and graph neural network training.

    The extractor handles temporal synchronization, robot identification,
    and data formatting to support collaborative perception algorithms.
    """

    def __init__(
        self,
        robot_identifier: str = 'ep03',
        output_directory: Optional[str] = None,
        scenario_name: Optional[str] = None,
        bag_file_timestamps: Optional[Dict] = None
    ):
        """
        Initialize the collaborative radar data extractor.

        Args:
            robot_identifier (str): Unique identifier for the robot (e.g., 'ep03', 'ep04')
            output_directory (str, optional): Directory for saving extracted data
            scenario_name (str, optional): Name of the warehouse scenario being processed
            bag_file_timestamps (dict, optional): Timestamp mapping from bag file metadata
        """
        super().__init__('collaborative_radar_data_extractor')

        # Robot and scenario configuration
        self.robot_id = robot_identifier
        self.scenario_name = scenario_name or "unknown_warehouse_scenario"
        self.output_directory = (output_directory or
                               os.path.expanduser(f'~/collaborative_radar_data/{self.scenario_name}'))

        # Data storage containers
        self.extracted_radar_metadata = []  # Metadata for each radar message
        self.extracted_radar_points = []    # Actual point cloud data
        self.bag_timestamp_mapping = bag_file_timestamps or {}  # Temporal synchronization data
        self.processed_message_count = 0    # Counter for message processing

        # ROS2 subscription for radar data
        radar_topic_name = f'/{robot_identifier}/ti_mmwave/radar_scan_pcl'
        self.radar_data_subscription = self.create_subscription(
            PointCloud2,
            radar_topic_name,
            self._process_radar_message_callback,
            10  # Queue size
        )

        # Logging and timing
        self.get_logger().info(f'Initialized radar extractor for robot: {robot_identifier}')
        self.get_logger().info(f'Processing scenario: {self.scenario_name}')
        self.get_logger().info(f'Output directory: {self.output_directory}')
        self.get_logger().info(f'Subscribed to topic: {radar_topic_name}')

        self.message_counter = 0
        self.extraction_start_time = datetime.now().timestamp()

    def _extract_radar_points_from_pointcloud2(self, radar_pointcloud_message):
        """
        Extract spatial coordinates and intensity from radar PointCloud2 message.

        Processes ROS2 PointCloud2 messages from TI mmWave radar sensors to extract
        3D spatial coordinates (x, y, z) and signal intensity values. The method
        handles the binary data format and field offsets to properly decode radar
        measurements for collaborative perception processing.

        Args:
            radar_pointcloud_message (sensor_msgs.msg.PointCloud2): ROS2 radar message
                                                                   containing point cloud data

        Returns:
            List[List[float]]: List of radar points, each containing [x, y, z, intensity]
        """
        extracted_radar_points = []
        x_coordinate_offset = y_coordinate_offset = z_coordinate_offset = None
        signal_intensity_offset = None

        # Parse PointCloud2 field structure to find data offsets
        for field_descriptor in radar_pointcloud_message.fields:
            if field_descriptor.name == 'x':
                x_coordinate_offset = field_descriptor.offset
            elif field_descriptor.name == 'y':
                y_coordinate_offset = field_descriptor.offset
            elif field_descriptor.name == 'z':
                z_coordinate_offset = field_descriptor.offset
            elif field_descriptor.name == 'intensity':
                signal_intensity_offset = field_descriptor.offset

        # Verify that all required coordinate fields are present
        if None in (x_coordinate_offset, y_coordinate_offset, z_coordinate_offset):
            self.get_logger().warning("Missing required coordinate fields in radar point cloud")
            return extracted_radar_points

        # Extract point data using field offsets
        point_data_step = radar_pointcloud_message.point_step

        for byte_index in range(0, len(radar_pointcloud_message.data), point_data_step):
            try:
                # Extract spatial coordinates
                x_coord = struct.unpack_from('f', radar_pointcloud_message.data,
                                           byte_index + x_coordinate_offset)[0]
                y_coord = struct.unpack_from('f', radar_pointcloud_message.data,
                                           byte_index + y_coordinate_offset)[0]
                z_coord = struct.unpack_from('f', radar_pointcloud_message.data,
                                           byte_index + z_coordinate_offset)[0]

                # Extract signal intensity if available
                if signal_intensity_offset is not None:
                    signal_intensity = struct.unpack_from('f', radar_pointcloud_message.data,
                                                        byte_index + signal_intensity_offset)[0]
                else:
                    signal_intensity = 0.0  # Default intensity value

                extracted_radar_points.append([x_coord, y_coord, z_coord, signal_intensity])
            except struct.error as e:
                self.get_logger().warning(f"Error unpacking radar point data: {e}")
                continue
            except Exception as e:
                self.get_logger().warning(f"Unexpected error processing radar point: {e}")
                continue

        return extracted_radar_points

    def _process_radar_message_callback(self, radar_message):
        """
        Process incoming radar point cloud messages for collaborative perception.

        This callback function handles real-time radar data from ROS2 topics,
        extracts point cloud information, and stores it for collaborative
        perception processing. The method maintains temporal synchronization
        and robot identification for multi-robot scenarios.

        Args:
            radar_message (sensor_msgs.msg.PointCloud2): Incoming radar point cloud message
        """
        # Extract timestamp from radar message header
        message_timestamp = (radar_message.header.stamp.sec +
                            radar_message.header.stamp.nanosec * 1e-9)

        # Create unique message identifier for tracking
        message_id = f"{radar_message.header.stamp.sec}.{radar_message.header.stamp.nanosec}"

        # Look up synchronized timestamp from bag file metadata
        synchronized_timestamp = self.bag_timestamp_mapping.get(
            self.processed_message_count, message_timestamp
        )

        # Determine standardized robot name based on robot identifier
        if self.robot_id == 'ep03':
            standardized_robot_name = 'Robot_1'
        elif self.robot_id == 'ep05':
            standardized_robot_name = 'Robot_2'
        else:
            standardized_robot_name = 'Cable_Robot'

        # Create metadata record for this radar message
        radar_message_metadata = {
            'synchronized_timestamp': synchronized_timestamp,
            'original_timestamp': message_timestamp,
            'frame_id': radar_message.header.frame_id,
            'point_cloud_height': radar_message.height,
            'point_cloud_width': radar_message.width,
            'total_point_count': radar_message.width * radar_message.height,
            'robot_identifier': self.robot_id,
            'robot_name': standardized_robot_name,
            'scenario_name': self.scenario_name,
            'message_index': self.processed_message_count,
        }

        # Store metadata for this radar message
        self.extracted_radar_metadata.append(radar_message_metadata)

        # Extract actual point cloud data from the radar message
        radar_points = self._extract_radar_points_from_pointcloud2(radar_message)
        
        # Process each extracted radar point for collaborative perception
        for radar_point in radar_points:
            x_coordinate = radar_point[0]
            y_coordinate = radar_point[1]
            z_coordinate = radar_point[2]
            signal_intensity = radar_point[3]

            # Calculate range (distance from sensor origin)
            range_distance = math.sqrt(
                (x_coordinate * x_coordinate) +
                (y_coordinate * y_coordinate) +
                (z_coordinate * z_coordinate)
            )

            # Calculate azimuth angle from x, y coordinates
            if y_coordinate == 0:
                if x_coordinate >= 0:
                    detected_azimuth_degrees = 90
                else:
                    detected_azimuth_degrees = -90
            else:
                detected_azimuth_degrees = math.atan(x_coordinate/y_coordinate) * 180 / np.pi
                detected_azimuth_degrees = np.round(detected_azimuth_degrees, 3)

            # Calculate elevation angle from x, y, z coordinates
            if x_coordinate == 0 and y_coordinate == 0:
                if z_coordinate >= 0:
                    detected_elevation_degrees = 90
                else:
                    detected_elevation_degrees = -90
            else:
                horizontal_distance = math.sqrt((x_coordinate * x_coordinate) + (y_coordinate * y_coordinate))
                detected_elevation_degrees = math.atan(z_coordinate/horizontal_distance) * 180 / np.pi
                detected_elevation_degrees = np.round(detected_elevation_degrees, 3)

            # Store processed radar point data for collaborative perception
            processed_radar_point = {
                f'{standardized_robot_name}_timestamp': synchronized_timestamp,
                'range_meters': range_distance,
                'azimuth_degrees': detected_azimuth_degrees,
                'elevation_degrees': detected_elevation_degrees,
                'x_coordinate': x_coordinate,
                'y_coordinate': y_coordinate,
                'z_coordinate': z_coordinate,
                'signal_intensity': signal_intensity,
                'robot_identifier': self.robot_id,
                'robot_name': standardized_robot_name,
                'scenario_name': self.scenario_name,
                'message_index': self.processed_message_count,
            }

            self.extracted_radar_points.append(processed_radar_point)

        # Update message counters for progress tracking
        self.message_counter += 1
        self.processed_message_count += 1

        # Log progress periodically
        if self.message_counter % 50 == 0:
            elapsed_time = datetime.now().timestamp() - self.extraction_start_time
            processing_rate = self.message_counter / elapsed_time if elapsed_time > 0 else 0
            self.get_logger().info(
                f'Processed {self.message_counter} radar messages, '
                f'{len(self.extracted_radar_points)} points '
                f'({processing_rate:.1f} msgs/sec)'
            )

    def save_extracted_data(self):
        """
        Save extracted radar data to structured CSV files for collaborative perception.

        Saves the extracted radar metadata and point cloud data to organized CSV files
        that can be used in the collaborative perception pipeline. Also saves timestamp
        mapping for temporal synchronization across multiple robots.
        """
        # Create output directory structure
        os.makedirs(self.output_directory, exist_ok=True)

        # Save radar message metadata
        if self.extracted_radar_metadata:
            metadata_filename = f'{self.robot_id}_radar_metadata.csv'
            metadata_path = os.path.join(self.output_directory, metadata_filename)
            pd.DataFrame(self.extracted_radar_metadata).to_csv(metadata_path, index=False)
            self.get_logger().info(f"Saved radar metadata: {metadata_path}")

        # Save extracted radar point cloud data
        if self.extracted_radar_points:
            points_filename = f'{self.robot_id}_radar_points.csv'
            points_path = os.path.join(self.output_directory, points_filename)
            pd.DataFrame(self.extracted_radar_points).to_csv(points_path, index=False)
            self.get_logger().info(f"Saved radar points: {points_path}")

        # Save timestamp mapping for temporal synchronization
        if self.bag_timestamp_mapping:
            timestamps_filename = f'{self.robot_id}_bag_timestamps.json'
            timestamps_path = os.path.join(self.output_directory, timestamps_filename)
            with open(timestamps_path, 'w') as f:
                json.dump(self.bag_timestamp_mapping, f, indent=2)
            self.get_logger().info(f"Saved timestamp mapping: {timestamps_path}")

        # Log extraction summary
        total_messages = len(self.extracted_radar_metadata)
        total_points = len(self.extracted_radar_points)
        self.get_logger().info(f"Extraction complete: {total_messages} messages, {total_points} points")


def extract_timestamps_from_rosbag_database(db3_file_path, radar_topic_name):
    """
    Extract timestamps from ROS2 bag SQLite database for temporal synchronization.

    This function reads the SQLite database file from a ROS2 bag to extract precise
    timestamps for radar messages. These timestamps are essential for temporal
    synchronization in collaborative perception scenarios where multiple robots
    need to be temporally aligned.

    Args:
        db3_file_path (str): Path to the ROS2 bag .db3 SQLite database file
        radar_topic_name (str): Name of the radar topic to extract timestamps for
                               (e.g., '/ep03/ti_mmwave/radar_scan_pcl')

    Returns:
        Dict[int, float]: Dictionary mapping message indices to timestamps in seconds
    """
    timestamp_mapping = {}
    
    try:
        # Connect to the ROS2 bag SQLite database
        database_connection = sqlite3.connect(db3_file_path)
        database_cursor = database_connection.cursor()

        # Query for the topic ID corresponding to the radar topic
        database_cursor.execute("SELECT id FROM topics WHERE name = ?;", (radar_topic_name,))
        topic_query_result = database_cursor.fetchone()

        if not topic_query_result:
            print(f"Radar topic {radar_topic_name} not found in database {db3_file_path}")
            database_connection.close()
            return timestamp_mapping

        radar_topic_id = topic_query_result[0]

        # Extract all message timestamps for the radar topic
        try:
            # Try the 'messages' table first (newer ROS2 bag format)
            database_cursor.execute(
                "SELECT timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp;",
                (radar_topic_id,)
            )
            timestamp_query_results = database_cursor.fetchall()
        except sqlite3.Error:
            try:
                # Try the 'message' table (older ROS2 bag format)
                database_cursor.execute(
                    "SELECT timestamp FROM message WHERE topic_id = ? ORDER BY timestamp;",
                    (radar_topic_id,)
                )
                timestamp_query_results = database_cursor.fetchall()
            except sqlite3.Error:
                print(f"Could not query message table in database {db3_file_path}")
                database_connection.close()
                return timestamp_mapping

        # Process timestamp results and create mapping
        for message_index, (timestamp_nanoseconds,) in enumerate(timestamp_query_results):
            # Convert nanoseconds to seconds for standard timestamp format
            timestamp_mapping[message_index] = timestamp_nanoseconds / 1e9

        database_connection.close()
        print(f"Extracted {len(timestamp_mapping)} timestamps for {radar_topic_name} from {db3_file_path}")
        return timestamp_mapping
        
    except Exception as e:
        print(f"Error extracting timestamps from {db3_file_path}: {e}")
        return {}


def process_collaborative_rosbag(rosbag_directory_path, robot_identifier, output_directory, scenario_name):
    """
    Process a single ROS2 bag file for collaborative radar data extraction.

    This function handles the complete workflow of extracting radar data from
    a ROS2 bag file, including timestamp extraction, bag playback, and data
    collection for collaborative perception processing.

    Args:
        rosbag_directory_path (str): Path to the ROS2 bag directory
        robot_identifier (str): Unique identifier for the robot (e.g., 'ep03')
        output_directory (str): Directory to save extracted data
        scenario_name (str): Name of the warehouse scenario
    """
    # Locate the SQLite database file in the bag directory
    db3_database_files = glob.glob(os.path.join(rosbag_directory_path, "*.db3"))
    if not db3_database_files:
        print(f"No .db3 database files found in {rosbag_directory_path}")
        return

    bag_database_file = db3_database_files[0]
    radar_topic_name = f'/{robot_identifier}/ti_mmwave/radar_scan_pcl'

    print(f"Extracting timestamps for {radar_topic_name} from {bag_database_file}...")
    synchronized_timestamps = extract_timestamps_from_rosbag_database(bag_database_file, radar_topic_name)

    if not synchronized_timestamps:
        print(f"No timestamps found for {radar_topic_name} in {bag_database_file}")
        return

    # Start ROS2 bag playback process
    print(f"Starting bag playback and radar data extraction...")
    bag_playback_process = subprocess.Popen(
        ['ros2', 'bag', 'play', rosbag_directory_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Initialize ROS2 and create radar data extractor
    rclpy.init()
    radar_extractor = CollaborativeRadarDataExtractor(
        robot_identifier, output_directory, scenario_name, synchronized_timestamps
    )

    def monitor_bag_playback():
        """Monitor bag playback completion and shutdown ROS2."""
        bag_playback_process.wait()
        rclpy.shutdown()

    # Start monitoring thread for bag playback
    monitoring_thread = threading.Thread(target=monitor_bag_playback, daemon=True)
    monitoring_thread.start()

    try:
        # Process incoming radar messages
        rclpy.spin(radar_extractor)
    except KeyboardInterrupt:
        print("Extraction interrupted by user")
    finally:
        # Save extracted data and cleanup
        radar_extractor.save_extracted_data()
        radar_extractor.destroy_node()

        # Terminate bag playback if still running
        if bag_playback_process.poll() is None:
            bag_playback_process.terminate()
            bag_playback_process.wait()

        # Shutdown ROS2 context safely
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except RuntimeError:
            print("ROS2 context already shut down")


def extract_all_radar_data(dataset_root):
    """Process all rosbags in the dataset"""
    rosbag_dirs = []
    scenario_types = ['CPPS_Diagonal', 'CPPS_Horizontal', 
    'CPPS_Vertical',
    'CPPS_Diagonal_Horizontal',
    'CPPS_Horizontal_Diagonal',
    'CPPS_Horizontal_Vertical',
    'CPPS_Vertical_Horizontal']
    
    for scenario in scenario_types:
        scenario_path = pathlib.Path(dataset_root) / scenario
        if not scenario_path.exists():
            continue
        
        for robot_type in ['Robot_1', 'Robot_2']:
            rosbag_path = scenario_path / robot_type / 'rosbag'
            if not rosbag_path.exists():
                continue
            
            for item in rosbag_path.iterdir():
                if item.is_dir() and any(f.suffix == '.db3' for f in item.iterdir()):
                    rosbag_dirs.append({'path': str(item), 'scenario': scenario, 'robot_type': robot_type})
    
    print(f"Found {len(rosbag_dirs)} rosbag directories to process")
    
    for idx, rosbag_info in enumerate(rosbag_dirs):
        rosbag_path, scenario, robot_type = rosbag_info.values()
        robot_id = 'ep03' if robot_type == 'Robot_1' else 'ep05' if robot_type == 'Robot_2' else 'cable_robot'
        output_dir = os.path.join('/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/Extracted_Radar_Data', 
                                scenario, robot_type, os.path.basename(rosbag_path))
        
        print(f"\nProcessing {idx+1}/{len(rosbag_dirs)}: {rosbag_path}")
        process_collaborative_rosbag(rosbag_path, robot_id, output_dir, scenario)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        process_collaborative_rosbag(sys.argv[1],
                                     sys.argv[2] if len(sys.argv) > 2 else 'ep03',
                                     '/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/ExtractedRadar_Data',
                                     "single_rosbag")
    else:
        extract_all_radar_data('/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/CPPS_Static_Scenario')