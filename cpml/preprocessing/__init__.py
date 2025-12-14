"""
CPML Preprocessing Module

Data preprocessing pipeline for collaborative perception including:
- Radar and Vicon data extraction
- Multi-robot synchronization
- Coordinate transformations
- Point cloud cleaning and labeling
- Graph conversion for GNN training
"""

from cpml.preprocessing import (
    radar_extraction,
    vicon_extraction,
    synchronization,
    coordinate_transform,
    point_cloud_cleaner,
    point_cloud_labeler,
    graph_converter,
    frame_splitter,
)

__all__ = [
    "radar_extraction",
    "vicon_extraction",
    "synchronization",
    "coordinate_transform",
    "point_cloud_cleaner",
    "point_cloud_labeler",
    "graph_converter",
    "frame_splitter",
]
