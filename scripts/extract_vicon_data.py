#!/usr/bin/env python3
"""
Vicon Motion Capture Data Extractor for Collaborative Perception

This module extracts and processes Vicon motion capture data for collaborative
perception applications in warehouse robotics. The Vicon system provides
high-precision ground truth positioning data for multiple robots, which is
essential for coordinate frame transformations and collaborative perception.

# TODO: Consider edge cases
Key Features:
- High-precision 6DOF pose extraction (position + orientation)
- Multi-robot trajectory processing
- Temporal synchronization with sensor data
- Structured CSV output for collaborative perception pipeline
- Comprehensive logging and progress tracking

The extracted Vicon data serves as ground truth for robot positioning and
enables accurate coordinate transformations in collaborative perception tasks.

Author: Thiyanayugi Mariraj
Project: Development of a Framework for Collaborative Perception Management Layer
         for Future 6G-Enabled Robotic Systems
"""

import os
import pandas as pd
import glob
import argparse
from tqdm import tqdm
import logging
import re


def extract_collaborative_vicon_data(dataset_root_directory, output_directory, target_scenario=None):
    """
    Extract Vicon motion capture data for collaborative perception applications.

    This function processes Vicon motion capture text files to extract precise
    6DOF pose information (position and orientation) for multiple robots in
    warehouse scenarios. The extracted data provides ground truth positioning
    for coordinate frame transformations in collaborative perception.

    Args:
        dataset_root_directory (str): Root directory of the RoboFUSE dataset
        output_directory (str): Directory to save processed CSV files
        target_scenario (str, optional): Specific scenario to process (e.g., "CPPS_Horizontal")
                                       If None, processes all available scenarios
    """
    # Create output directory structure for Vicon data
    os.makedirs(output_directory, exist_ok=True)

    # Set up comprehensive logging for Vicon extraction process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_directory, 'vicon_extraction_log.txt'), 'w')
        ]
    )
    logger = logging.getLogger('collaborative-vicon-extractor')

    # Determine which warehouse scenarios to process
    scenarios_to_process = []
    if target_scenario:
        # Process only the specified scenario if it exists
        scenario_directory_path = os.path.join(dataset_root_directory, target_scenario)
        if os.path.exists(scenario_directory_path):
            scenarios_to_process = [target_scenario]
            logger.info(f"Processing single scenario: {target_scenario}")
        else:
            logger.error(f"Target scenario {target_scenario} not found in {dataset_root_directory}")
            return
    else:
        # Find all available CPPS warehouse scenarios
        logger.info(f"Scanning for scenarios in {dataset_root_directory}")
        for directory_item in os.listdir(dataset_root_directory):
            item_full_path = os.path.join(dataset_root_directory, directory_item)
            if os.path.isdir(item_full_path) and directory_item.startswith('CPPS_'):
                scenarios_to_process.append(directory_item)
        logger.info(f"Found {len(scenarios_to_process)} scenarios to process")
    
    logger.info(f"Processing scenarios: {', '.join(scenarios_to_process)}")

    # Process each warehouse scenario
    for scenario_name in scenarios_to_process:
        logger.info(f"Processing scenario: {scenario_name}")

        scenario_directory_path = os.path.join(dataset_root_directory, scenario_name)
        vicon_data_path = os.path.join(scenario_directory_path, 'Vicon')

        if not os.path.exists(vicon_data_path):
            logger.warning(f"No Vicon folder found for scenario {scenario_name}")
            continue

        # Create output directory for this scenario's processed data
        scenario_output_directory = os.path.join(output_directory, scenario_name)
        os.makedirs(scenario_output_directory, exist_ok=True)
        logger.info(f"Created output directory: {scenario_output_directory}")
        
        # Find all Vicon motion capture data files
        vicon_data_files = []
        for file_extension in ['*.txt']:
            vicon_data_files.extend(glob.glob(os.path.join(vicon_data_path, file_extension)))
        
        if not vicon_data_files:
            logger.warning(f"No Vicon data files found for scenario {scenario_name}")
            continue

        logger.info(f"Processing {len(vicon_data_files)} Vicon files for scenario {scenario_name}")

        # Process each Vicon motion capture file
        for vicon_data_file in vicon_data_files:
            file_basename = os.path.basename(vicon_data_file)
            file_name = os.path.splitext(file_basename)[0]

            # Skip processing if the file name indicates it's a failed measurement
            if "failed" in file_name.lower():
                logger.info(f"Skipping failed measurement file: {file_basename}")
                continue

            logger.info(f"Processing Vicon file: {file_basename}")

            try:
                # Parse Vicon motion capture data file
                data_rows = parse_vicon_file(vicon_data_file, logger)
                
                if not data_rows:
                    logger.warning(f"No valid data extracted from {file_basename}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(data_rows)
                
                # Save to CSV
                csv_path = os.path.join(scenario_output_directory, f"{file_name}.csv")
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved {len(df)} data points to {csv_path}")
                
                # Create summary file for robot positions
                if 'object_name' in df.columns:
                    # Find all unique robot IDs
                    robots = df['object_name'].unique()
                    robot_summary = []
                    
                    for robot in robots:
                        if robot in ['ep03', 'ep05', 'cable_robot', 'radar_TI_1', 'radar_TI_2', 'radar_TI_3']:
                            robot_data = df[df['object_name'] == robot]
                            
                            # Calculate average position and orientation
                            position_cols = [col for col in df.columns if col.endswith('_pos_x') or col.endswith('_pos_y') or col.endswith('_pos_z')]
                            orientation_cols = [col for col in df.columns if col.endswith('_yaw') or col.endswith('_rot_x') or col.endswith('_rot_y') or col.endswith('_rot_z')]
                            
                            if position_cols and len(robot_data) > 0:
                                avg_pos = {col: robot_data[col].mean() for col in position_cols}
                                avg_orient = {col: robot_data[col].mean() for col in orientation_cols}
                                
                                summary_row = {'robot_id': robot}
                                summary_row.update(avg_pos)
                                summary_row.update(avg_orient)
                                robot_summary.append(summary_row)
                    
                    if robot_summary:
                        summary_df = pd.DataFrame(robot_summary)
                        summary_path = os.path.join(scenario_output_directory, f"{file_name}_robot_positions.csv")
                        summary_df.to_csv(summary_path, index=False)
                        logger.info(f"Saved robot position summary to {summary_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_basename}: {e}", exc_info=True)
    
    logger.info("Vicon data extraction complete!")

def parse_vicon_file(file_path, logger):
    """
    Parse a Vicon data file and extract data points.
    
    Args:
        file_path (str): Path to the Vicon data file
        logger: Logger object
        
    Returns:
        list: List of dictionaries containing parsed data
    """
    data_rows = []
    line_number = 0
    
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc=f"Processing {os.path.basename(file_path)}"):
            line_number += 1
            line = line.strip()
            
            if not line:
                continue
                
            try:
                # The file contains dictionary-like data, which we can parse using eval
                # This is generally unsafe but we know our data format
                vicon_dict = eval(line)
                
                # Special handling for different file formats
                if isinstance(vicon_dict, dict):
                    # Process data dictionary line by line
                    if all(key.startswith('Object/') for key in vicon_dict.keys()):
                        # This is the standard format with Object/name/Position entries
                        row_data = process_standard_format(vicon_dict)
                        if row_data:
                            data_rows.append(row_data)
                    else:
                        # Handle other dictionary formats
                        logger.warning(f"Unknown dictionary format in line {line_number}")
                        
            except Exception as e:
                logger.warning(f"Error parsing line {line_number}: {e}")
                continue
    
    return data_rows

def process_standard_format(vicon_dict):
    """
    Process Vicon data in the standard format.
    
    Args:
        vicon_dict (dict): Dictionary containing Vicon data
        
    Returns:
        dict: Processed data row or None if no valid data
    """
    # Initialize a data row with common timestamp
    row_data = {}
    timestamp = None
    
    # Process each object in the dictionary
    object_count = 1
    
    for key, value in vicon_dict.items():
        # Extract object name from key (e.g., "Object/ep03/Position")
        parts = key.split('/')
        if len(parts) < 2:
            continue
            
        object_name = parts[1]
        
        # Skip invalid data
        if all(v == 0 for v in value.values()):
            continue
            
        # Extract system_time for timestamp
        if 'system_time' in value:
            # Convert to seconds (from nanoseconds)
            if timestamp is None:
                timestamp = value['system_time'] / 1e9
                row_data['vicon_timestamp'] = timestamp
            
            # Store the timestamp for this object
            row_data[f'object_name_{object_count}_timestamp'] = value['system_time'] / 1e9
        
        # Store object name
        row_data[f'object_name_{object_count}'] = object_name
        
        # Extract position and orientation data with object_name prefix
        for field in ['pos_x', 'pos_y', 'pos_z', 'yaw', 'rot_x', 'rot_y', 'rot_z']:
            if field in value:
                row_data[f'{object_name}_{field}'] = value[field]
        
        # For the first object, also store as object_name_1
        if object_count == 1:
            row_data['object_name'] = object_name
        
        object_count += 1
    
    return row_data if row_data and 'vicon_timestamp' in row_data else None

def main():
    parser = argparse.ArgumentParser(description='Extract Vicon data from RoboFUSE dataset')
    parser.add_argument('--input_dir', type=str, default='/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/CPPS_Static_Scenario',
                       help='Root directory of the RoboFUSE dataset')
    parser.add_argument('--output_dir', type=str, default='./vicon_data',
                       help='Directory to save CSV files')
    parser.add_argument('--scenario', type=str, default=None,
                       help='Specific scenario to process (e.g., "CPPS_Horizontal")')
    
    args = parser.parse_args()
    
    extract_collaborative_vicon_data(args.input_dir, args.output_dir, args.scenario)

if __name__ == "__main__":
    main()