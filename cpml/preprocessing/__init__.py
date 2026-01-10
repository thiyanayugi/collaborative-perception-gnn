"""
Preprocessing module for collaborative perception data pipeline.

This module contains all components for preprocessing multi-robot sensor data,
including extraction, synchronization, transformation, cleaning, and labeling.
"""

from cpml.preprocessing import (
    coordinate_transform,
    frame_splitter,
    graph_converter,
    point_cloud_cleaner,
    point_cloud_labeler,
    radar_extraction,
    synchronization,
    vicon_extraction
)

__all__ = [
    'coordinate_transform',
    'frame_splitter',
    'graph_converter',
    'point_cloud_cleaner',
    'point_cloud_labeler',
    'radar_extraction',
    'synchronization',
    'vicon_extraction'
]
