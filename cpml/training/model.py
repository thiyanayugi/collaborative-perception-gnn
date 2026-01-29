#!/usr/bin/env python3
"""
Graph Neural Network Models for Collaborative Perception

This module implements various Graph Neural Network architectures specifically designed
for collaborative perception tasks in warehouse robotics. The models process graph-structured
data representing spatial voxels from mmWave radar sensors mounted on multiple robots.

Supported Architectures:
- GATv2 (Graph Attention Networks v2): Attention-based message passing
- ECC (Edge-Conditioned Convolution): Edge-feature conditioned convolutions
- GraphSAGE: Sampling and aggregating for large graphs

The models are optimized for occupancy prediction tasks using collaborative features
# TODO: Add unit tests
that capture spatial relationships and multi-robot sensor fusion patterns.

Author: Thiyanayugi Mariraj
Project: Development of a Framework for Collaborative Perception Management Layer
         for Future 6G-Enabled Robotic Systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GraphSAGE,
    GATv2Conv,
    EdgeConv,
    global_mean_pool,
    global_max_pool,
    BatchNorm
)
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from typing import Dict, List, Optional, Tuple, Union, Callable
import yaml


class OptimizedECCConv(MessagePassing):
    """
    Optimized Edge-Conditioned Convolution Layer for Collaborative Perception.

    This implementation provides an efficient Edge-Conditioned Convolution (ECC) layer
    specifically designed for collaborative perception tasks. The layer processes
    spatial relationships between voxels using edge features (spatial distances)
    to condition the convolution operations.

    Key Features:
    - Compact edge network design for computational efficiency (~2.5M parameters total)
    - Spatial distance conditioning for warehouse environment understanding
    - Optimized for collaborative multi-robot perception tasks
    - Memory-efficient implementation suitable for real-time applications

    The ECC layer is particularly effective for spatial reasoning tasks where
    edge relationships (distances between voxels) provide important contextual
    information for occupancy prediction.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int = 1,
        aggregation_method: str = 'mean',
    ):
        """
        Initialize the optimized ECC layer for collaborative perception.

        Args:
            in_channels (int): Number of input node feature channels
            out_channels (int): Number of output node feature channels
            edge_dim (int): Dimension of edge features (default: 1 for spatial distance)
            aggregation_method (str): Message aggregation method ('add', 'mean', or 'max')
                                    'mean' is recommended for stable training
        """
        super().__init__(aggr=aggregation_method)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim

        # Compact edge network for computational efficiency
        # This network generates transformation weights based on edge features (spatial distances)
        self.edge_network = nn.Sequential(
            nn.Linear(edge_dim, 32),  # Compact hidden layer (reduced from in*out dimensions)
            nn.ReLU(inplace=True),    # Memory-efficient activation
            nn.Linear(32, in_channels * out_channels)  # Generate transformation matrix weights
        )

        # Linear transformation for self-connections (residual-like connections)
        self.self_transform = Linear(in_channels, out_channels)

        # Initialize network parameters
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initialize network parameters using appropriate initialization schemes.

        Uses Xavier/Glorot initialization for linear layers to ensure stable
        gradient flow during training of the collaborative perception model.
        This initialization is particularly important for deep GNN architectures.
        """
        for layer in self.edge_network:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.self_transform.reset_parameters()

    def forward(
        self,
        x: Union[torch.Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Edge-Conditioned Convolution layer.

        Processes node features through edge-conditioned message passing,
        where spatial relationships (edge features) condition the convolution
        operations. This is particularly effective for spatial reasoning in
        collaborative perception tasks.

        Args:
            x (torch.Tensor): Node features [num_nodes, in_channels]
                             For collaborative perception: spatial coordinates,
                             occupancy probabilities, robot contributions, etc.
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
                                     Defines spatial relationships between voxels
            edge_attr (torch.Tensor, optional): Edge features [num_edges, edge_dim]
                                              If None, computed from spatial distances

        Returns:
            torch.Tensor: Updated node features [num_nodes, out_channels]
        """
        # Generate spatial edge features if not provided
        # Uses Euclidean distance between voxel centers as edge features
        if edge_attr is None:
            row, col = edge_index
            # Extract spatial coordinates (first 3 features: x, y, z)
            pos_source = x[row, :3]  # Source voxel positions
            pos_target = x[col, :3]  # Target voxel positions
            # Compute spatial distances as edge features
            edge_attr = torch.norm(pos_source - pos_target, dim=1, keepdim=True)

        # Execute message passing with edge-conditioned transformations
        aggregated_messages = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Apply self-transformation (similar to residual connections)
        self_contribution = self.self_transform(x)

        # Combine aggregated messages with self-contribution
        output = aggregated_messages + self_contribution

        return output

    def message(
        self,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate messages from source nodes conditioned on edge features.

        This function implements the core edge-conditioning mechanism where
        spatial relationships (edge features) determine how information flows
        between connected voxels in the collaborative perception graph.

        Args:
            x_j (torch.Tensor): Source node features [num_edges, in_channels]
            edge_attr (torch.Tensor): Edge features [num_edges, edge_dim]
                                     Typically spatial distances between voxels

        Returns:
            torch.Tensor: Conditioned messages [num_edges, out_channels]
        """
        # Generate transformation weights from edge features
        # The edge network learns how spatial distance affects information flow
        transformation_weights = self.edge_network(edge_attr)

        # Reshape weights to transformation matrices [num_edges, in_channels, out_channels]
        weight_matrices = transformation_weights.view(-1, self.in_channels, self.out_channels)

        # Apply edge-conditioned transformation to source node features
        # Each edge gets its own transformation matrix based on spatial distance
        conditioned_messages = torch.matmul(x_j.unsqueeze(1), weight_matrices).squeeze(1)

        return conditioned_messages


class CollaborativePerceptionGNN(nn.Module):
    """
    Graph Neural Network for Collaborative Perception Occupancy Prediction.

    This model implements a flexible GNN architecture for predicting occupancy
    in warehouse environments using collaborative perception data from multiple
    robots equipped with mmWave radar sensors.

    Key Features:
    - Multi-architecture support: GraphSAGE, GATv2, and ECC
    - Collaborative feature processing (16-dimensional input features)
    - Spatial reasoning through graph convolutions
    - Binary occupancy prediction with sigmoid activation
    - Configurable depth and regularization

    The model processes graph-structured data where nodes represent spatial voxels
    and edges encode spatial relationships. Each voxel contains collaborative
    features from multiple robots including spatial coordinates, occupancy
    probabilities, robot contribution ratios, and temporal consistency measures.
    """

    def __init__(
        self,
        input_feature_dim: int,
        hidden_feature_dim: int,
        output_dim: int = 1,
        num_gnn_layers: int = 3,
        dropout_rate: float = 0.2,
        gnn_architecture: str = "graphsage",
        enable_skip_connections: bool = True,
        enable_batch_norm: bool = True,
        enable_layer_norm: bool = False,
        graph_pooling_method: str = "mean_max",
        attention_heads: int = 4,
    ):
        """
        Initialize the Collaborative Perception GNN model.

        Args:
            input_feature_dim (int): Dimension of input node features (16 for collaborative perception)
            hidden_feature_dim (int): Dimension of hidden layer features
            output_dim (int): Dimension of output (1 for binary occupancy prediction)
            num_gnn_layers (int): Number of graph convolution layers
            dropout_rate (float): Dropout probability for regularization
            gnn_architecture (str): GNN type ('graphsage', 'gatv2', or 'ecc')
            enable_skip_connections (bool): Whether to use residual connections
            enable_batch_norm (bool): Whether to apply batch normalization
            enable_layer_norm (bool): Whether to apply layer normalization
            graph_pooling_method (str): Graph-level pooling ('mean', 'max', or 'mean_max')
            attention_heads (int): Number of attention heads for GATv2 architecture
        """
        super().__init__()

        # Store model configuration
        self.input_dim = input_feature_dim
        self.hidden_dim = hidden_feature_dim
        self.output_dim = output_dim
        self.num_layers = num_gnn_layers
        self.dropout = dropout_rate
        self.gnn_type = gnn_architecture
        self.skip_connections = enable_skip_connections
        self.batch_norm = enable_batch_norm
        self.layer_norm = enable_layer_norm
        self.pooling = graph_pooling_method
        self.attention_heads = attention_heads

        # Input feature embedding layer
        # Transforms collaborative perception features to hidden dimension
        self.feature_embedding = nn.Linear(input_feature_dim, hidden_feature_dim)

        # Graph convolution layers
        self.gnn_layers = nn.ModuleList()

        # Normalization layers (optional)
        self.batch_norm_layers = nn.ModuleList() if enable_batch_norm else None
        self.layer_norm_layers = nn.ModuleList() if enable_layer_norm else None

        # Create graph convolution layers
        for layer_idx in range(num_gnn_layers):
            input_channels = hidden_feature_dim
            output_channels = hidden_feature_dim

            # Create appropriate GNN layer based on architecture choice
            if gnn_architecture == "graphsage":
                gnn_layer = GraphSAGE(
                    in_channels=input_channels,
                    hidden_channels=hidden_feature_dim,
                    num_layers=1,  # Single layer per module
                    out_channels=output_channels,
                    dropout=dropout_rate,
                )
            elif gnn_architecture == "gatv2":
                # Graph Attention Network v2 with multi-head attention
                gnn_layer = GATv2Conv(
                    in_channels=input_channels,
                    out_channels=output_channels // attention_heads,
                    heads=attention_heads,
                    dropout=dropout_rate,
                    concat=True,  # Concatenate attention heads
                )
            elif gnn_architecture == "ecc":
                # Edge-Conditioned Convolution with spatial distance conditioning
                gnn_layer = OptimizedECCConv(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    edge_dim=1,  # Spatial distance as edge feature
                    aggregation_method='mean',
                )
            else:
                raise ValueError(f"Unsupported GNN architecture: {gnn_architecture}")

            self.gnn_layers.append(gnn_layer)

            # Add normalization layers if enabled
            if enable_batch_norm:
                self.batch_norm_layers.append(BatchNorm(hidden_feature_dim))

            # Add layer normalization if enabled
            if enable_layer_norm:
                self.layer_norm_layers.append(nn.LayerNorm(hidden_feature_dim))

        # Multi-layer perceptron classifier for final prediction
        # Handles different pooling strategies (mean, max, or concatenated mean+max)
        pooling_multiplier = 2 if graph_pooling_method == "mean_max" else 1
        classifier_input_dim = hidden_feature_dim * pooling_multiplier

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_feature_dim, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the collaborative perception GNN.

        Processes graph-structured collaborative perception data through multiple
        layers of graph convolutions, normalization, and pooling to produce
        occupancy predictions for warehouse environments.

        Args:
            x (torch.Tensor): Node features [num_nodes, input_dim]
                             Contains collaborative perception features:
                             - Spatial coordinates (x, y, z)
                             - Occupancy probabilities
                             - Robot contribution ratios
                             - Collaborative overlap indicators
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
                                     Defines spatial relationships between voxels
            batch (torch.Tensor): Batch assignment [num_nodes]
                                 Groups nodes into separate graphs

        Returns:
            torch.Tensor: Occupancy predictions [batch_size, 1]
                         Sigmoid-activated probabilities for binary classification
        """
        # Transform input features to hidden dimension
        node_features = self.feature_embedding(x)

        # Apply graph convolution layers with normalization and skip connections
        for layer_idx in range(self.num_layers):
            # Store previous features for skip connection
            previous_features = node_features

            # Apply appropriate GNN layer based on architecture
            if self.gnn_type == "ecc":
                # ECC requires edge features (spatial distances)
                # Extract spatial coordinates (first 3 features: x, y, z)
                row, col = edge_index
                spatial_distances = torch.norm(
                    x[row, :3] - x[col, :3], dim=1, keepdim=True
                )
                updated_features = self.gnn_layers[layer_idx](
                    node_features, edge_index, spatial_distances
                )
            else:
                # GraphSAGE and GATv2 don't require explicit edge features
                updated_features = self.gnn_layers[layer_idx](node_features, edge_index)

            # Apply skip connection for improved gradient flow
            if self.skip_connections and layer_idx > 0:
                node_features = updated_features + previous_features
            else:
                node_features = updated_features

            # Apply batch normalization if enabled
            if self.batch_norm:
                node_features = self.batch_norm_layers[layer_idx](node_features)

            # Apply layer normalization if enabled
            if self.layer_norm:
                node_features = self.layer_norm_layers[layer_idx](node_features)

            # Apply non-linear activation and dropout regularization
            node_features = F.relu(node_features)
            node_features = F.dropout(
                node_features, p=self.dropout, training=self.training
            )

        # Global graph-level pooling to create graph representations
        # Aggregates node features into fixed-size graph representations
        if self.pooling == "mean":
            graph_representation = global_mean_pool(node_features, batch)
        elif self.pooling == "max":
            graph_representation = global_max_pool(node_features, batch)
        elif self.pooling == "mean_max":
            # Concatenate mean and max pooling for richer representation
            mean_pooled = global_mean_pool(node_features, batch)
            max_pooled = global_max_pool(node_features, batch)
            graph_representation = torch.cat([mean_pooled, max_pooled], dim=1)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")

        # Apply classifier to produce final occupancy predictions
        occupancy_predictions = self.classifier(graph_representation)

        return occupancy_predictions


class OptimizedECCModel(nn.Module):
    """
    Optimized ECC Model following thesis architecture specifications.

    Architecture:
    - Input Embedding: 15/16 -> 64 features
    - 3 ECC Layers: 64 -> 64 with BatchNorm and Dropout
    - Global Pooling: Mean + Max -> 128
    - Classifier: 128 -> 1 with Sigmoid

    Total Parameters: ~2.5M
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        edge_dim: int = 1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Input embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # ECC layers with batch normalization
        self.ecc_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            self.ecc_layers.append(
                OptimizedECCConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    edge_dim=edge_dim,
                    aggregation_method='mean'
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Global pooling (Mean + Max = 128 features)
        pooled_dim = hidden_dim * 2  # Mean + Max

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, 1),
            nn.Sigmoid()
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.embedding.weight)
        for layer in self.classifier:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass following thesis architecture.

        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            batch: Batch indices [N]

        Returns:
            Occupancy predictions [batch_size, 1]
        """
        # Input embedding: input_dim -> 64
        h = self.embedding(x)  # [N, 64]

        # Apply 3 ECC layers with BatchNorm and Dropout
        for i in range(self.num_layers):
            # ECC layer
            h = self.ecc_layers[i](h, edge_index)  # [N, 64]

            # Batch normalization
            h = self.batch_norms[i](h)  # [N, 64]

            # ReLU activation
            h = F.relu(h)  # [N, 64]

            # Dropout (0.2)
            h = F.dropout(h, p=self.dropout, training=self.training)  # [N, 64]

        # Global pooling: Mean + Max
        h_mean = global_mean_pool(h, batch)  # [batch_size, 64]
        h_max = global_max_pool(h, batch)    # [batch_size, 64]
        h_pooled = torch.cat([h_mean, h_max], dim=1)  # [batch_size, 128]

        # Final classifier with Sigmoid
        output = self.classifier(h_pooled)  # [batch_size, 1]

        return output

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: Dict):
    """
    Create a model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Model (OccupancyGNN or OptimizedECCModel)
    """
    # Determine input dimension based on temporal window
    input_dim = config["model"]["input_dim"]
    gnn_type = config["model"]["gnn_type"]

    # Use optimized ECC model for thesis architecture
    if gnn_type == "ecc":
        model = OptimizedECCModel(
            input_dim=input_dim,
            hidden_dim=64,  # Fixed to 64 as per thesis specs
            num_layers=3,   # Fixed to 3 layers as per thesis specs
            dropout=0.2,    # Fixed to 0.2 as per thesis specs
            edge_dim=1,     # Spatial distance edge feature
        )
    else:
        # Use collaborative perception GNN for GATv2 and GraphSAGE models
        model = CollaborativePerceptionGNN(
            input_feature_dim=input_dim,
            hidden_feature_dim=config["model"]["hidden_dim"],
            output_dim=config["model"]["output_dim"],
            num_gnn_layers=config["model"]["num_layers"],
            dropout_rate=config["model"]["dropout"],
            gnn_architecture=config["model"]["gnn_type"],
            enable_skip_connections=config["model"]["skip_connections"],
            enable_batch_norm=config["model"]["batch_norm"],
            enable_layer_norm=config["model"].get("layer_norm", False),
            graph_pooling_method=config["model"]["pooling"],
            attention_heads=config["model"].get("attention_heads", 4),
        )

    return model


if __name__ == "__main__":
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create model
    model = create_model(config)

    # Print model summary
    print(model)
