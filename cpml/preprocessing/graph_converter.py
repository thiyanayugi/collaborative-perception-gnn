#!/usr/bin/env python3
"""
Point Cloud to Graph Converter for Collaborative Perception

This module converts synchronized point cloud data from multiple robots into
graph-structured representations suitable for Graph Neural Network training.
The converter processes mmWave radar data from collaborative perception scenarios
and generates PyTorch Geometric Data objects with collaborative features.

Key Features:
- Multi-robot point cloud processing and voxelization
- Collaborative feature extraction (robot contributions, spatial overlaps)
- Temporal window support for multi-frame processing
- Graph generation with spatial relationships as edges
- Comprehensive metadata tracking for analysis

The generated graphs contain nodes representing spatial voxels with collaborative
features and edges encoding spatial relationships between voxels.

Author: Thiyanayugi Mariraj
Project: Development of a Framework for Collaborative Perception Management Layer
         for Future 6G-Enabled Robotic Systems
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import glob
from tqdm import tqdm
import time
import csv

# Semantic label mapping for warehouse environment objects
WAREHOUSE_OBJECT_LABELS = {
    'unknown': 0,        # Unclassified or background
    'workstation': 1,    # Work areas and stations
    'robot': 2,          # Autonomous mobile robots
    'boundary': 3,       # Walls and structural boundaries
    'KLT': 4            # Small load carriers (Kleinladungsträger)
}

def create_collaborative_gnn_frame(dataframe, timestamp, voxel_size_meters=0.1):
    """
    Create a GNN frame from collaborative multi-robot data at a specific timestamp.

    This function processes synchronized point cloud data from multiple robots
    and generates a graph representation with collaborative perception features.
    Each node represents a spatial voxel containing radar points from one or
    both robots, with features capturing collaborative sensing patterns.

    Args:
        dataframe (pd.DataFrame): Synchronized point cloud data from both robots
                                 containing spatial coordinates and annotations
        timestamp (float): Target timestamp for frame generation
        voxel_size_meters (float): Spatial voxel size in meters for discretization

    Returns:
        torch_geometric.data.Data: Graph object with collaborative features
                                  containing nodes (voxels) and edges (spatial relationships)
    """
    # Extract available column names for robot data identification
    available_columns = dataframe.columns.tolist()

    # Initialize containers for multi-robot data aggregation
    aggregated_positions = []
    aggregated_annotations = []
    aggregated_robot_identifiers = []

    # Extract Robot 1 radar data (handle various naming conventions)
    robot1_x_columns = [col for col in available_columns
                       if 'robot_1' in col.lower() and 'x' in col and 'radar' in col]
    robot1_y_columns = [col for col in available_columns
                       if 'robot_1' in col.lower() and 'y' in col and 'radar' in col]
    robot1_z_columns = [col for col in available_columns
                       if 'robot_1' in col.lower() and 'z' in col and 'radar' in col]

    robot1_point_count = 0
    if robot1_x_columns and robot1_y_columns and robot1_z_columns:
        # Extract spatial coordinates for Robot 1
        robot1_positions = dataframe[[robot1_x_columns[0], robot1_y_columns[0], robot1_z_columns[0]]].values
        # Filter out invalid/NaN measurements
        # Filter out invalid measurements (NaN values)
        robot1_valid_mask = ~np.isnan(robot1_positions).any(axis=1)
        robot1_positions = robot1_positions[robot1_valid_mask]

        if len(robot1_positions) > 0:
            # Extract corresponding annotations for valid Robot 1 points
            robot1_annotations = dataframe['annotation_general'].values[robot1_valid_mask]
            robot1_identifiers = np.ones(len(robot1_positions), dtype=int)  # Robot 1 ID

            # Aggregate Robot 1 data
            aggregated_positions.append(robot1_positions)
            aggregated_annotations.append(robot1_annotations)
            aggregated_robot_identifiers.append(robot1_identifiers)
            robot1_point_count = len(robot1_positions)

    # Extract Robot 2 radar data (handle various naming conventions)
    robot2_x_columns = [col for col in available_columns
                       if 'robot_2' in col.lower() and 'x' in col and 'radar' in col]
    robot2_y_columns = [col for col in available_columns
                       if 'robot_2' in col.lower() and 'y' in col and 'radar' in col]
    robot2_z_columns = [col for col in available_columns
                       if 'robot_2' in col.lower() and 'z' in col and 'radar' in col]

    robot2_point_count = 0
    if robot2_x_columns and robot2_y_columns and robot2_z_columns:
        # Extract spatial coordinates for Robot 2
        robot2_positions = dataframe[[robot2_x_columns[0], robot2_y_columns[0], robot2_z_columns[0]]].values
        # Filter out invalid measurements (NaN values)
        robot2_valid_mask = ~np.isnan(robot2_positions).any(axis=1)
        robot2_positions = robot2_positions[robot2_valid_mask]

        if len(robot2_positions) > 0:
            # Extract corresponding annotations for valid Robot 2 points
            robot2_annotations = dataframe['annotation_general'].values[robot2_valid_mask]
            robot2_identifiers = np.full(len(robot2_positions), 2, dtype=int)  # Robot 2 ID

            # Aggregate Robot 2 data
            aggregated_positions.append(robot2_positions)
            aggregated_annotations.append(robot2_annotations)
            aggregated_robot_identifiers.append(robot2_identifiers)
            robot2_point_count = len(robot2_positions)

    # Combine multi-robot collaborative data
    if aggregated_positions:
        # Stack all robot positions into unified array
        combined_positions = np.vstack(aggregated_positions)
        combined_annotations = np.concatenate(aggregated_annotations)
        combined_robot_ids = np.concatenate(aggregated_robot_identifiers)

        print(f"  Collaborative data: Robot 1: {robot1_point_count} points, "
              f"Robot 2: {robot2_point_count} points, Total: {len(combined_positions)} points")
    else:
        # No valid data found for this timestamp
        raise ValueError(f"No robot data found in timestamp {timestamp}!")

    # Convert semantic annotations to numeric labels for training
    numeric_labels = np.zeros(len(combined_annotations), dtype=np.int64)
    for idx, annotation in enumerate(combined_annotations):
        numeric_labels[idx] = WAREHOUSE_OBJECT_LABELS.get(annotation, 0)

    # Perform spatial voxelization for graph node generation
    # Each voxel becomes a node in the collaborative perception graph
    voxel_indices = np.floor(combined_positions / voxel_size_meters).astype(np.int64)
    
    # Create a dictionary to store collaborative voxel data
    collaborative_voxels = {}

    # Assign points to voxels and compute collaborative features
    for point_idx, (position, label, robot_id) in enumerate(zip(combined_positions, numeric_labels, combined_robot_ids)):
        voxel_coordinate = tuple(voxel_indices[point_idx])

        # Initialize voxel data structure if not seen before
        if voxel_coordinate not in collaborative_voxels:
            collaborative_voxels[voxel_coordinate] = {
                'positions': [],
                'labels': [],
                'robot_ids': []
            }

        # Add point data to corresponding voxel
        collaborative_voxels[voxel_coordinate]['positions'].append(position)
        collaborative_voxels[voxel_coordinate]['labels'].append(label)
        collaborative_voxels[voxel_coordinate]['robot_ids'].append(robot_id)
    
    # Compute collaborative voxel features and labels
    voxel_positions = []
    voxel_labels = []
    voxel_robot_counts = []
    voxel_collaboration_scores = []

    # Process each voxel to extract collaborative perception features
    for voxel_coordinate, voxel_data in collaborative_voxels.items():
        # Compute voxel center (mean position of all points in the voxel)
        voxel_pos = np.mean(voxel_data['positions'], axis=0)
        voxel_positions.append(voxel_pos)
        
        # Assign label based on majority vote
        labels_array = np.array(voxel_data['labels'])
        unique_labels, counts = np.unique(labels_array, return_counts=True)
        voxel_label = unique_labels[np.argmax(counts)]
        voxel_labels.append(voxel_label)
        
        # Count robot contributions to this voxel
        robot_ids_array = np.array(voxel_data['robot_ids'])
        robot1_count = np.sum(robot_ids_array == 1)
        robot2_count = np.sum(robot_ids_array == 2)
        voxel_robot_counts.append([robot1_count, robot2_count])
        
        # Calculate collaboration score (how much both robots contribute)
        total_points = robot1_count + robot2_count
        if total_points > 1:
            collaboration_score = min(robot1_count, robot2_count) / total_points
        else:
            collaboration_score = 0.0
        voxel_collaboration_scores.append(collaboration_score)
    
    # Convert to numpy arrays
    voxel_positions = np.array(voxel_positions)
    voxel_labels = np.array(voxel_labels)
    voxel_robot_counts = np.array(voxel_robot_counts)
    voxel_collaboration_scores = np.array(voxel_collaboration_scores)
    
    # Create enhanced node features (16 features per node) - ADDED COLLABORATIVE FEATURES
    num_nodes = len(voxel_positions)
    node_features = np.zeros((num_nodes, 16))
    
    # Features 0-2: Normalized position (x, y, z)
    pos_min = np.min(voxel_positions, axis=0)
    pos_max = np.max(voxel_positions, axis=0)
    pos_range = pos_max - pos_min
    pos_range[pos_range == 0] = 1  # Avoid division by zero
    node_features[:, 0:3] = (voxel_positions - pos_min) / pos_range
    
    # Features 3-5: Raw position (x, y, z)
    node_features[:, 3:6] = voxel_positions
    
    # Features 6-8: Position relative to center (x, y, z)
    center = np.mean(voxel_positions, axis=0)
    node_features[:, 6:9] = voxel_positions - center
    
    # Features 9-11: Position relative to origin (x, y, z)
    node_features[:, 9:12] = voxel_positions
    
    # Feature 12: Distance to center
    node_features[:, 12] = np.linalg.norm(voxel_positions - center, axis=1)
    
    # Features 13-15: Multi-robot collaboration features (NEW!)
    node_features[:, 13] = voxel_robot_counts[:, 0]  # Robot 1 point count in voxel
    node_features[:, 14] = voxel_robot_counts[:, 1]  # Robot 2 point count in voxel
    node_features[:, 15] = voxel_collaboration_scores  # Collaboration score (0-0.5)
    
    # Create edges (fully connected graph)
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # Exclude self-loops
                edge_index.append([i, j])
    
    # Convert to PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.tensor(voxel_labels, dtype=torch.long)
    pos = torch.tensor(voxel_positions, dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y, pos=pos)
    
    # Add metadata
    data.timestamp = timestamp
    data.robot1_points = robot1_count
    data.robot2_points = robot2_count
    data.collaboration_voxels = np.sum(voxel_collaboration_scores > 0)
    
    return data

def create_temporal_frame(frames, window_size, center_idx):
    """
    Create a temporal frame by aggregating multiple frames (CAUSAL VERSION - NO FUTURE FRAMES).

    Args:
        frames: List of frames
        window_size: Size of the temporal window
        center_idx: Index of the center frame

    Returns:
        PyTorch Geometric Data object with temporal and collaborative features
    """
    # Get the center frame
    center_frame = frames[center_idx]
    
    # CAUSAL: Only use past frames + current frame (NO FUTURE FRAMES)
    start_idx = max(0, center_idx - (window_size - 1))
    end_idx = center_idx + 1
    window_frames = frames[start_idx:end_idx]
    
    # Extract node features, positions, and labels from all frames in the window
    all_x = []
    all_pos = []
    all_y = []
    all_metadata = []
    
    for i, frame in enumerate(window_frames):
        # Calculate temporal offset (negative for past frames, 0 for current)
        temporal_offset = i - (center_idx - start_idx)
        
        # Add temporal offset to node features (now 17 features total)
        x_with_offset = torch.cat([frame.x, torch.full((frame.num_nodes, 1), temporal_offset, dtype=torch.float)], dim=1)
        
        all_x.append(x_with_offset)
        all_pos.append(frame.pos)
        all_y.append(frame.y)
        
        # Collect metadata for summary
        all_metadata.append({
            'robot1_points': getattr(frame, 'robot1_points', 0),
            'robot2_points': getattr(frame, 'robot2_points', 0),
            'collaboration_voxels': getattr(frame, 'collaboration_voxels', 0)
        })
    
    # Concatenate node features, positions, and labels
    x = torch.cat(all_x, dim=0)
    pos = torch.cat(all_pos, dim=0)
    y = torch.cat(all_y, dim=0)
    
    # Create edges (fully connected graph)
    num_nodes = x.size(0)
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # Exclude self-loops
                edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y, pos=pos)
    
    # Add metadata
    data.timestamp = center_frame.timestamp
    data.window_size = window_size
    data.total_robot1_points = sum(m['robot1_points'] for m in all_metadata)
    data.total_robot2_points = sum(m['robot2_points'] for m in all_metadata)
    data.total_collaboration_voxels = sum(m['collaboration_voxels'] for m in all_metadata)
    
    return data

def process_csv_file(csv_file, output_dir, voxel_size=0.1, temporal_windows=[1, 3, 5]):
    """
    Process a CSV file and create GNN frames with multi-robot collaborative perception.

    Args:
        csv_file: Path to the CSV file
        output_dir: Directory to save the GNN frames
        voxel_size: Size of voxels for discretizing the point cloud
        temporal_windows: List of temporal window sizes
    """
    try:
        # Get dataset name from file path (handle nested structure)
        csv_filename = os.path.basename(csv_file)
        dataset_name = os.path.splitext(csv_filename)[0]
        
        # Create output directory for this dataset
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Create directories for each temporal window
        for window_size in temporal_windows:
            os.makedirs(os.path.join(dataset_output_dir, f"temporal_{window_size}"), exist_ok=True)
        
        # Read CSV file
        print(f"Reading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        
        print(f"CSV columns: {df.columns.tolist()}")
        
        # Group by timestamp
        grouped = df.groupby('vicon_timestamp')
        
        # Create frames for each timestamp
        frames = []
        timestamps = []
        
        print(f"Processing {len(grouped)} timestamps...")
        for timestamp, group in tqdm(grouped, desc=f"Creating multi-robot frames for {dataset_name}"):
            try:
                # Create GNN frame with collaborative multi-robot data
                frame = create_collaborative_gnn_frame(group, timestamp, voxel_size)
                
                # Add to list
                frames.append(frame)
                timestamps.append(timestamp)
            except Exception as e:
                print(f"Error processing timestamp {timestamp}: {e}")
                continue
        
        print(f"Created {len(frames)} collaborative frames")
        
        # Calculate collaboration statistics
        total_robot1 = sum(getattr(f, 'robot1_points', 0) for f in frames)
        total_robot2 = sum(getattr(f, 'robot2_points', 0) for f in frames)
        total_collab_voxels = sum(getattr(f, 'collaboration_voxels', 0) for f in frames)
        
        print(f"Collaboration stats: Robot1={total_robot1}, Robot2={total_robot2}, Collaborative_voxels={total_collab_voxels}")
        
        # Save non-temporal frames (window size 1)
        print("Saving temporal_1 frames...")
        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            output_path = os.path.join(dataset_output_dir, f"temporal_1", f"{timestamp}.pt")
            torch.save(frame, output_path)
        
        # Create and save temporal frames
        for window_size in [3, 5]:
            if window_size not in temporal_windows:
                continue
            
            print(f"Creating temporal_{window_size} frames...")
            saved_count = 0
            
            for i in range(len(frames)):
                # CAUSAL: Skip if not enough PAST frames for the window
                if i < window_size - 1:  # Need window_size-1 past frames
                    continue
                
                try:
                    # Create temporal frame
                    temporal_frame = create_temporal_frame(frames, window_size, i)
                    
                    # Save frame
                    output_path = os.path.join(dataset_output_dir, f"temporal_{window_size}", f"{timestamps[i]}.pt")
                    torch.save(temporal_frame, output_path)
                    saved_count += 1
                except Exception as e:
                    print(f"Error creating temporal frame at index {i}: {e}")
                    continue
            
            print(f"Saved {saved_count} temporal_{window_size} frames (skipped first {window_size-1} frames - CAUSAL)")
        
        return True
    
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        import traceback
        traceback.print_exc()
        return False

def collect_statistics(output_dir):
    """
    Collect statistics about the generated collaborative frames.

    Args:
        output_dir: Directory containing the GNN frames
    """
    # Initialize statistics
    total_frames = 0
    total_nodes = 0
    total_edges = 0
    total_robot1_points = 0
    total_robot2_points = 0
    total_collab_voxels = 0
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    # Find all .pt files
    pt_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.pt'):
                pt_files.append(os.path.join(root, file))
    
    # Process each .pt file
    for pt_file in tqdm(pt_files, desc="Collecting collaborative statistics"):
        try:
            # Load the data
            data = torch.load(pt_file, map_location='cpu', weights_only=False)
            
            # Update statistics
            total_frames += 1
            total_nodes += data.num_nodes
            total_edges += data.num_edges
            
            # Multi-robot statistics
            total_robot1_points += getattr(data, 'robot1_points', 0)
            total_robot2_points += getattr(data, 'robot2_points', 0)
            total_collab_voxels += getattr(data, 'collaboration_voxels', 0)
            
            # Update label counts
            for label in range(5):
                label_counts[label] += (data.y == label).sum().item()
        except Exception as e:
            print(f"Error processing {pt_file}: {e}")
    
    # Calculate averages
    avg_nodes_per_frame = total_nodes / total_frames if total_frames > 0 else 0
    avg_edges_per_frame = total_edges / total_frames if total_frames > 0 else 0
    avg_robot1_per_frame = total_robot1_points / total_frames if total_frames > 0 else 0
    avg_robot2_per_frame = total_robot2_points / total_frames if total_frames > 0 else 0
    avg_collab_per_frame = total_collab_voxels / total_frames if total_frames > 0 else 0
    
    # Calculate label percentages
    total_labels = sum(label_counts.values())
    label_percentages = {label: count / total_labels * 100 if total_labels > 0 else 0 for label, count in label_counts.items()}
    
    # Create summary
    summary = f"""
    MULTI-ROBOT COLLABORATIVE GNN Frame Statistics (CAUSAL TEMPORAL WINDOWS)
    ========================================================================
    Total frames: {total_frames}
    Total nodes: {total_nodes}
    Total edges: {total_edges}
    Average nodes per frame: {avg_nodes_per_frame:.2f}
    Average edges per frame: {avg_edges_per_frame:.2f}
    
    MULTI-ROBOT COLLABORATION STATISTICS:
    ====================================
    Total Robot 1 points: {total_robot1_points}
    Total Robot 2 points: {total_robot2_points}
    Total collaborative voxels: {total_collab_voxels}
    Average Robot 1 points per frame: {avg_robot1_per_frame:.2f}
    Average Robot 2 points per frame: {avg_robot2_per_frame:.2f}
    Average collaborative voxels per frame: {avg_collab_per_frame:.2f}
    Robot collaboration ratio: {total_robot2_points / (total_robot1_points + total_robot2_points) * 100:.1f}% Robot 2 contribution

    Label distribution:
    ==================
    Unknown (0): {label_counts[0]} ({label_percentages[0]:.2f}%)
    Workstation (1): {label_counts[1]} ({label_percentages[1]:.2f}%)
    Robot (2): {label_counts[2]} ({label_percentages[2]:.2f}%)
    Boundary (3): {label_counts[3]} ({label_percentages[3]:.2f}%)
    KLT (4): {label_counts[4]} ({label_percentages[4]:.2f}%)
    """
    
    # Save summary
    with open(os.path.join(output_dir, "collaborative_summary.txt"), "w") as f:
        f.write(summary)
    
    print(summary)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert annotated CSV files to COLLABORATIVE GNN frames (CAUSAL VERSION)')
    parser.add_argument('--input_dir', type=str,
                        default='05_annotated/boundary_only_improved_annotated_ws300_rob75_bound150_corner250',
                        help='Directory containing annotated CSV files')
    parser.add_argument('--output_dir', type=str,
                        default='07_gnn_frames_COLLABORATIVE_causal',
                        help='Directory to save GNN frames')
    parser.add_argument('--voxel_size', type=float, default=0.1,
                        help='Size of voxels for discretizing the point cloud')
    parser.add_argument('--temporal_windows', type=int, nargs='+', default=[1, 3, 5],
                        help='List of temporal window sizes')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all CSV files
    csv_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f"🤝 MULTI-ROBOT COLLABORATIVE PERCEPTION DATA GENERATION")
    print(f"📁 Found {len(csv_files)} CSV files to process")
    print(f"🎯 Target: TRUE collaborative perception with both robots")
    print(f"⏰ Temporal windows: CAUSAL (no future frames)")
    print(f"🔧 Features: 16 spatial + 1 temporal = 17 total per node")
    print("="*60)
    
    # Process each CSV file
    success_count = 0
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"🤖 Processing {csv_file}...")
        if process_csv_file(csv_file, args.output_dir, args.voxel_size, args.temporal_windows):
            success_count += 1
        print(f"✅ Completed {success_count}/{len(csv_files)} files")
    
    print(f"\n{'='*60}")
    print(f"🎉 Processing complete: {success_count}/{len(csv_files)} files successful")
    
    # Collect statistics
    print("📊 Collecting collaborative statistics...")
    collect_statistics(args.output_dir)
    
    print("\n🚀 MULTI-ROBOT COLLABORATIVE DATA READY FOR TRAINING!")

if __name__ == "__main__":
    main()
