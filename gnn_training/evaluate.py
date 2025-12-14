#!/usr/bin/env python3

import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any

# Add safe globals for PyTorch serialization
torch.serialization.add_safe_globals([
    'torch_geometric.data.data.Data',
    'torch_geometric.data.data.DataEdgeAttr',
    'torch_geometric.data.data.DataNodeAttr',
    'torch_geometric.data.storage.EdgeStorage',
    'torch_geometric.data.storage.NodeStorage',
])
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from data import create_data_loaders
from model import create_model


class Evaluator:
    """
    Evaluator class for the occupancy prediction model.
    """

    def __init__(self, config: Dict):
        """
        Initialize the evaluator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config["training"]["device"])

        # Create visualization directory
        os.makedirs(config["evaluation"]["visualization"]["save_dir"], exist_ok=True)

    def evaluate(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        temporal_window: int,
    ) -> Dict[str, float]:
        """
        Evaluate the model on the test set with both graph-level and node-level evaluation.

        Args:
            model: Model to evaluate
            test_loader: Test data loader
            temporal_window: Size of temporal window

        Returns:
            Dictionary of metrics
        """
        # Check if model supports node-level evaluation
        supports_node_level = self._check_node_level_compatibility(model, test_loader)

        if supports_node_level:
            print(f"🔍 Performing NODE-LEVEL evaluation for temporal_{temporal_window}")
            return self.evaluate_node_level(model, test_loader, temporal_window)
        else:
            print(f"📊 Performing GRAPH-LEVEL evaluation for temporal_{temporal_window}")
            return self.evaluate_graph_level(model, test_loader, temporal_window)

    def _check_node_level_compatibility(self, model: torch.nn.Module, test_loader: torch.utils.data.DataLoader) -> bool:
        """
        Check if the model can perform node-level evaluation.

        Args:
            model: Model to check
            test_loader: Test data loader

        Returns:
            True if node-level evaluation is possible, False otherwise
        """
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                logits = model(batch.x, batch.edge_index, batch.batch)

                # Check if output matches node count (node-level) or graph count (graph-level)
                if logits.size(0) == batch.x.size(0):  # Node-level output
                    print(f"✓ Model outputs node-level predictions: {logits.shape[0]} nodes")
                    return True
                elif logits.size(0) == batch.num_graphs:  # Graph-level output
                    print(f"⚠ Model outputs graph-level predictions: {logits.shape[0]} graphs")
                    return False
                else:
                    print(f"❌ Unexpected output shape: {logits.shape}")
                    return False
        return False

    def evaluate_node_level(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        temporal_window: int,
    ) -> Dict[str, float]:
        """
        Perform comprehensive node-level evaluation with spatial metrics.

        Args:
            model: Model to evaluate
            test_loader: Test data loader
            temporal_window: Size of temporal window

        Returns:
            Dictionary of comprehensive metrics
        """
        # Move model to device
        model = model.to(self.device)
        model.eval()

        # Initialize collections for node-level data
        all_node_predictions = []
        all_node_targets = []
        all_node_probabilities = []
        all_spatial_positions = []
        all_graph_data = []

        print(f"🔍 Starting node-level evaluation...")

        # Evaluation loop
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Node-level eval T{temporal_window}")):
                batch = batch.to(self.device)

                # Get node-level predictions
                logits = model(batch.x, batch.edge_index, batch.batch)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)

                # Get node-level targets
                targets = batch.y.cpu().numpy()

                # Get spatial positions (assuming positions are in features 2-4 for X,Y,Z)
                if hasattr(batch, 'pos') and batch.pos is not None:
                    positions = batch.pos.cpu().numpy()
                else:
                    # Extract positions from node features (X, Y, Z typically at indices 2, 3, 4)
                    positions = batch.x[:, 2:5].cpu().numpy()

                # Collect all node-level data
                all_node_predictions.extend(preds)
                all_node_targets.extend(targets)
                all_node_probabilities.extend(probs)
                all_spatial_positions.extend(positions)

                # Store graph-level data for visualization
                for i in range(batch.num_graphs):
                    mask = batch.batch.cpu() == i
                    graph_data = {
                        "pos": positions[mask],
                        "y_true": targets[mask],
                        "y_pred": preds[mask],
                        "probs": probs[mask],
                        "graph_idx": i,
                        "batch_idx": batch_idx
                    }
                    all_graph_data.append(graph_data)

        # Convert to numpy arrays
        predictions = np.array(all_node_predictions)
        targets = np.array(all_node_targets)
        probabilities = np.array(all_node_probabilities)
        positions = np.array(all_spatial_positions)

        print(f"📊 Node-level evaluation completed:")
        print(f"   Total nodes evaluated: {len(targets):,}")
        print(f"   Occupied nodes (ground truth): {np.sum(targets):,}")
        print(f"   Predicted occupied nodes: {np.sum(predictions):,}")

        # Calculate comprehensive metrics
        metrics = self._compute_node_level_metrics(targets, predictions, probabilities)

        # Add spatial analysis
        spatial_metrics = self._compute_spatial_metrics(positions, targets, predictions, probabilities)
        metrics.update(spatial_metrics)

        # Print detailed metrics
        self._print_node_level_metrics(metrics, temporal_window)

        # Generate comprehensive visualizations
        self._generate_node_level_visualizations(
            targets, predictions, probabilities, positions, all_graph_data, temporal_window
        )

        return metrics

    def evaluate_graph_level(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        temporal_window: int,
    ) -> Dict[str, float]:
        """
        Perform traditional graph-level evaluation (fallback for existing models).

        Args:
            model: Model to evaluate
            test_loader: Test data loader
            temporal_window: Size of temporal window

        Returns:
            Dictionary of metrics
        """
        # Move model to device
        model = model.to(self.device)
        model.eval()

        # Initialize metrics
        all_preds = []
        all_targets = []
        all_probs = []
        all_data = []

        # Evaluation loop
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Graph-level eval T{temporal_window}"):
                batch = batch.to(self.device)

                # Forward pass
                logits = model(batch.x, batch.edge_index, batch.batch)

                # Convert logits to predictions and probabilities
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = probs > 0.5

                # Convert node-level labels to graph-level labels using majority vote
                graph_labels = []
                for i in range(batch.num_graphs):
                    mask = batch.batch == i
                    graph_label = batch.y[mask].float().mean().round()
                    graph_labels.append(graph_label)
                targets = torch.tensor(graph_labels).cpu().numpy()

                all_probs.extend(probs.flatten())
                all_preds.extend(preds.flatten())
                all_targets.extend(targets.flatten())

                # Store data for visualization
                for i in range(batch.num_graphs):
                    mask = batch.batch.cpu() == i
                    data = {
                        "pos": batch.pos[mask].cpu().numpy() if hasattr(batch, 'pos') else batch.x[mask, 2:5].cpu().numpy(),
                        "y_true": batch.y[mask].cpu().numpy(),
                        "y_pred": preds[i],
                        "prob": probs[i],
                    }
                    all_data.append(data)

        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        # Compute metrics
        metrics = self._compute_metrics(all_targets, all_preds, all_probs)

        # Print metrics
        print(f"\nGraph-level evaluation metrics for temporal_{temporal_window}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        # Generate visualizations
        self._generate_visualizations(all_targets, all_preds, all_probs, all_data, temporal_window)

        # Perform feature importance analysis
        if self.config["ablation"]["feature_importance"]:
            self._analyze_feature_importance(model, test_loader, temporal_window)

        return metrics

    def generate_visualizations(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        temporal_window: int,
    ):
        """
        Generate visualizations without computing metrics.

        Args:
            model: Model to evaluate
            test_loader: Test data loader
            temporal_window: Size of temporal window
        """
        # Move model to device
        model = model.to(self.device)

        # Set model to evaluation mode
        model.eval()

        # Initialize data collection
        all_preds = []
        all_targets = []
        all_probs = []
        all_data = []

        # Evaluation loop
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Visualizing temporal_{temporal_window}"):
                # Move batch to device
                batch = batch.to(self.device)

                # Forward pass
                logits = model(batch.x, batch.edge_index, batch.batch)

                # Convert logits to predictions and probabilities
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = probs > 0.5

                # Ensure targets match the logits shape
                if logits.size(0) != batch.y.size(0):
                    # This is a graph-level prediction, so we need one label per graph
                    # Use the majority vote of node labels as the graph label
                    graph_labels = []
                    for i in range(batch.num_graphs):
                        mask = batch.batch == i
                        graph_label = batch.y[mask].float().mean().round()
                        graph_labels.append(graph_label)
                    targets = torch.tensor(graph_labels).cpu().numpy()
                else:
                    targets = batch.y.cpu().numpy()

                all_probs.extend(probs.flatten())
                all_preds.extend(preds.flatten())
                all_targets.extend(targets.flatten())

                # Store data for visualization
                for i in range(batch.num_graphs):
                    mask = batch.batch.cpu() == i
                    data = {
                        "pos": batch.pos[mask].cpu().numpy(),
                        "y_true": batch.y[mask].cpu().numpy(),
                        "y_pred": preds[i],
                        "prob": probs[i],
                    }
                    all_data.append(data)

                # Only process enough samples for visualization
                if len(all_data) >= self.config["evaluation"]["visualization"]["num_samples"]:
                    break

        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        # Generate visualizations
        self._generate_visualizations(all_targets, all_preds, all_probs, all_data, temporal_window)

        # Perform feature importance analysis
        if self.config["ablation"]["feature_importance"]:
            self._analyze_feature_importance(model, test_loader, temporal_window)

    def _compute_metrics(
        self,
        targets: np.ndarray,
        preds: np.ndarray,
        probs: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            targets: Ground truth labels
            preds: Predicted labels
            probs: Predicted probabilities

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(targets, preds),
            "precision": precision_score(targets, preds, zero_division=0),
            "recall": recall_score(targets, preds, zero_division=0),
            "f1": f1_score(targets, preds, zero_division=0),
            "roc_auc": roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else 0.5,
        }

        return metrics

    def _compute_node_level_metrics(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive node-level classification metrics.

        Args:
            targets: Ground truth node labels
            predictions: Predicted node labels
            probabilities: Predicted node probabilities

        Returns:
            Dictionary of comprehensive metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report
        )

        # Basic classification metrics
        metrics = {
            "accuracy": accuracy_score(targets, predictions),
            "precision": precision_score(targets, predictions, zero_division=0),
            "recall": recall_score(targets, predictions, zero_division=0),
            "f1_score": f1_score(targets, predictions, zero_division=0),
            "roc_auc": roc_auc_score(targets, probabilities) if len(np.unique(targets)) > 1 else 0.5,
        }

        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        metrics["confusion_matrix"] = cm.tolist()

        # Node-level specific metrics
        metrics["total_nodes"] = len(targets)
        metrics["occupied_nodes"] = int(np.sum(targets))
        metrics["predicted_occupied"] = int(np.sum(predictions))
        metrics["correct_predictions"] = int(np.sum(targets == predictions))

        # Class-specific metrics
        if len(np.unique(targets)) > 1:
            report = classification_report(targets, predictions,
                                         target_names=['Unoccupied', 'Occupied'],
                                         output_dict=True, zero_division=0)

            metrics["unoccupied_precision"] = report['Unoccupied']['precision']
            metrics["unoccupied_recall"] = report['Unoccupied']['recall']
            metrics["unoccupied_f1"] = report['Unoccupied']['f1-score']

            metrics["occupied_precision"] = report['Occupied']['precision']
            metrics["occupied_recall"] = report['Occupied']['recall']
            metrics["occupied_f1"] = report['Occupied']['f1-score']

        # Class balance analysis
        occupied_ratio = np.sum(targets) / len(targets)
        metrics["occupied_ratio"] = occupied_ratio
        metrics["unoccupied_ratio"] = 1 - occupied_ratio

        return metrics

    def _compute_spatial_metrics(
        self,
        positions: np.ndarray,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute spatial accuracy metrics for robotics applications.

        Args:
            positions: Node spatial positions (N, 3) for X, Y, Z
            targets: Ground truth node labels
            predictions: Predicted node labels
            probabilities: Predicted node probabilities

        Returns:
            Dictionary of spatial metrics
        """
        spatial_metrics = {}

        # Extract occupied and unoccupied points
        occupied_true_mask = targets == 1
        occupied_pred_mask = predictions == 1

        occupied_true_points = positions[occupied_true_mask]
        occupied_pred_points = positions[occupied_pred_mask]

        # Distance-based spatial accuracy evaluation
        tolerance_levels = [0.15, 0.20, 0.25]  # Robotics tolerance levels

        for tolerance in tolerance_levels:
            if len(occupied_pred_points) > 0 and len(occupied_true_points) > 0:
                # Calculate distance accuracy
                from scipy.spatial.distance import cdist
                distances = cdist(occupied_pred_points, occupied_true_points)
                min_distances = np.min(distances, axis=1)

                points_within_tolerance = np.sum(min_distances <= tolerance)
                distance_accuracy = points_within_tolerance / len(occupied_pred_points)

                spatial_metrics[f"distance_accuracy_{tolerance:.2f}m"] = distance_accuracy
                spatial_metrics[f"mean_distance_error_{tolerance:.2f}m"] = np.mean(min_distances)
            else:
                spatial_metrics[f"distance_accuracy_{tolerance:.2f}m"] = 0.0
                spatial_metrics[f"mean_distance_error_{tolerance:.2f}m"] = float('inf')

        # Spatial distribution analysis
        if len(positions) > 0:
            # Calculate spatial spread
            spatial_metrics["spatial_range_x"] = np.max(positions[:, 0]) - np.min(positions[:, 0])
            spatial_metrics["spatial_range_y"] = np.max(positions[:, 1]) - np.min(positions[:, 1])
            spatial_metrics["spatial_range_z"] = np.max(positions[:, 2]) - np.min(positions[:, 2])

            # Calculate prediction density
            if len(occupied_pred_points) > 0:
                spatial_metrics["prediction_density"] = len(occupied_pred_points) / (
                    spatial_metrics["spatial_range_x"] * spatial_metrics["spatial_range_y"]
                )
            else:
                spatial_metrics["prediction_density"] = 0.0

        return spatial_metrics

    def _print_node_level_metrics(self, metrics: Dict[str, float], temporal_window: int):
        """
        Print comprehensive node-level metrics in a formatted way.

        Args:
            metrics: Dictionary of metrics
            temporal_window: Temporal window size
        """
        print(f"\n{'='*60}")
        print(f"📊 NODE-LEVEL EVALUATION RESULTS - Temporal Window {temporal_window}")
        print(f"{'='*60}")

        # Basic metrics
        print(f"\n🎯 CLASSIFICATION METRICS:")
        print(f"   Accuracy:     {metrics['accuracy']:.4f}")
        print(f"   Precision:    {metrics['precision']:.4f}")
        print(f"   Recall:       {metrics['recall']:.4f}")
        print(f"   F1-Score:     {metrics['f1_score']:.4f}")
        print(f"   ROC-AUC:      {metrics['roc_auc']:.4f}")

        # Node-level statistics
        print(f"\n📈 NODE-LEVEL STATISTICS:")
        print(f"   Total nodes:           {metrics['total_nodes']:,}")
        print(f"   Occupied nodes (GT):   {metrics['occupied_nodes']:,}")
        print(f"   Predicted occupied:    {metrics['predicted_occupied']:,}")
        print(f"   Correct predictions:   {metrics['correct_predictions']:,}")
        print(f"   Occupied ratio:        {metrics['occupied_ratio']:.3f}")

        # Class-specific metrics
        if 'occupied_precision' in metrics:
            print(f"\n🔍 CLASS-SPECIFIC METRICS:")
            print(f"   Unoccupied - Precision: {metrics['unoccupied_precision']:.4f}, "
                  f"Recall: {metrics['unoccupied_recall']:.4f}, F1: {metrics['unoccupied_f1']:.4f}")
            print(f"   Occupied   - Precision: {metrics['occupied_precision']:.4f}, "
                  f"Recall: {metrics['occupied_recall']:.4f}, F1: {metrics['occupied_f1']:.4f}")

        # Spatial metrics
        spatial_keys = [k for k in metrics.keys() if 'distance_accuracy' in k or 'spatial_range' in k]
        if spatial_keys:
            print(f"\n🗺️  SPATIAL ACCURACY METRICS:")
            for key in sorted(spatial_keys):
                if 'distance_accuracy' in key:
                    print(f"   {key}: {metrics[key]:.4f}")
                elif 'spatial_range' in key:
                    print(f"   {key}: {metrics[key]:.2f}m")

        print(f"{'='*60}")

    def _generate_node_level_visualizations(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        positions: np.ndarray,
        graph_data: List[Dict],
        temporal_window: int,
    ):
        """
        Generate comprehensive node-level visualizations.

        Args:
            targets: Ground truth node labels
            predictions: Predicted node labels
            probabilities: Predicted node probabilities
            positions: Node spatial positions
            graph_data: List of graph data for individual visualizations
            temporal_window: Temporal window size
        """
        save_dir = self.config["evaluation"]["visualization"]["save_dir"]

        # Create node-level visualization directory
        node_viz_dir = os.path.join(save_dir, f"node_level_temporal_{temporal_window}")
        os.makedirs(node_viz_dir, exist_ok=True)

        print(f"🎨 Generating node-level visualizations...")

        # 1. Comprehensive spatial overview
        self._create_spatial_overview_plot(targets, predictions, probabilities, positions,
                                         node_viz_dir, temporal_window)

        # 2. Confusion matrix
        self._create_node_confusion_matrix(targets, predictions, node_viz_dir, temporal_window)

        # 3. Prediction confidence analysis
        self._create_confidence_analysis_plot(targets, predictions, probabilities,
                                            node_viz_dir, temporal_window)

        # 4. Sample graph visualizations
        self._create_sample_graph_visualizations(graph_data, node_viz_dir, temporal_window)

        # 5. Spatial error analysis
        self._create_spatial_error_analysis(targets, predictions, positions,
                                          node_viz_dir, temporal_window)

        print(f"✅ Node-level visualizations saved to: {node_viz_dir}")

    def _create_spatial_overview_plot(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        positions: np.ndarray,
        save_dir: str,
        temporal_window: int,
    ):
        """Create comprehensive spatial overview plot."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Node-Level Spatial Analysis - Temporal Window {temporal_window}',
                     fontsize=16, fontweight='bold')

        # Plot 1: Ground Truth
        scatter1 = axes[0,0].scatter(positions[:, 0], positions[:, 1],
                                    c=targets, cmap='RdYlBu_r', s=8, alpha=0.7)
        axes[0,0].set_title('Ground Truth Occupancy', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('X Position (m)')
        axes[0,0].set_ylabel('Y Position (m)')
        cbar1 = plt.colorbar(scatter1, ax=axes[0,0])
        cbar1.set_label('Occupancy (0=Unoccupied, 1=Occupied)')

        # Plot 2: Model Predictions
        scatter2 = axes[0,1].scatter(positions[:, 0], positions[:, 1],
                                    c=predictions, cmap='RdYlBu_r', s=8, alpha=0.7)
        axes[0,1].set_title('Model Predictions', fontsize=12, fontweight='bold')
        axes[0,1].set_xlabel('X Position (m)')
        axes[0,1].set_ylabel('Y Position (m)')
        cbar2 = plt.colorbar(scatter2, ax=axes[0,1])
        cbar2.set_label('Predicted Occupancy')

        # Plot 3: Prediction Confidence
        scatter3 = axes[1,0].scatter(positions[:, 0], positions[:, 1],
                                    c=probabilities, cmap='viridis', s=8, alpha=0.7)
        axes[1,0].set_title('Prediction Confidence', fontsize=12, fontweight='bold')
        axes[1,0].set_xlabel('X Position (m)')
        axes[1,0].set_ylabel('Y Position (m)')
        cbar3 = plt.colorbar(scatter3, ax=axes[1,0])
        cbar3.set_label('Confidence Score')

        # Plot 4: Error Analysis
        errors = (targets != predictions).astype(int)
        scatter4 = axes[1,1].scatter(positions[:, 0], positions[:, 1],
                                    c=errors, cmap='Reds', s=8, alpha=0.7)
        axes[1,1].set_title('Prediction Errors', fontsize=12, fontweight='bold')
        axes[1,1].set_xlabel('X Position (m)')
        axes[1,1].set_ylabel('Y Position (m)')
        cbar4 = plt.colorbar(scatter4, ax=axes[1,1])
        cbar4.set_label('Error (0=Correct, 1=Wrong)')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'spatial_overview.pdf'),
                   dpi=300, bbox_inches='tight', format='pdf')
        plt.close()

    def _create_node_confusion_matrix(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        save_dir: str,
        temporal_window: int,
    ):
        """Create confusion matrix for node-level predictions."""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(targets, predictions)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Unoccupied', 'Occupied'],
                   yticklabels=['Unoccupied', 'Occupied'])
        plt.title(f'Node-Level Confusion Matrix - Temporal Window {temporal_window}',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.pdf'),
                   dpi=300, bbox_inches='tight', format='pdf')
        plt.close()

    def _create_confidence_analysis_plot(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        save_dir: str,
        temporal_window: int,
    ):
        """Create prediction confidence analysis plot."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Prediction Confidence Analysis - Temporal Window {temporal_window}',
                     fontsize=14, fontweight='bold')

        # Confidence histogram by class
        occupied_probs = probabilities[targets == 1]
        unoccupied_probs = probabilities[targets == 0]

        axes[0].hist(unoccupied_probs, bins=50, alpha=0.7, label='Unoccupied (GT)', color='blue')
        axes[0].hist(occupied_probs, bins=50, alpha=0.7, label='Occupied (GT)', color='red')
        axes[0].set_xlabel('Prediction Confidence')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Confidence Distribution by Ground Truth')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Confidence vs Accuracy
        confidence_bins = np.linspace(0, 1, 11)
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        bin_accuracies = []

        for i in range(len(confidence_bins) - 1):
            mask = (probabilities >= confidence_bins[i]) & (probabilities < confidence_bins[i+1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(targets[mask] == predictions[mask])
                bin_accuracies.append(bin_accuracy)
            else:
                bin_accuracies.append(0)

        axes[1].plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=6)
        axes[1].plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, label='Perfect Calibration')
        axes[1].set_xlabel('Prediction Confidence')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Confidence vs Accuracy (Calibration)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confidence_analysis.pdf'),
                   dpi=300, bbox_inches='tight', format='pdf')
        plt.close()

    def _create_sample_graph_visualizations(
        self,
        graph_data: List[Dict],
        save_dir: str,
        temporal_window: int,
    ):
        """Create visualizations for sample graphs."""
        num_samples = min(6, len(graph_data))  # Show up to 6 sample graphs

        if num_samples == 0:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Sample Graph Predictions - Temporal Window {temporal_window}',
                     fontsize=16, fontweight='bold')

        axes = axes.flatten()

        for i in range(num_samples):
            sample = graph_data[i]
            ax = axes[i]

            # Create color map for nodes
            colors = ['red' if label == 1 else 'lightblue' for label in sample['y_true']]

            # Plot nodes
            scatter = ax.scatter(sample['pos'][:, 0], sample['pos'][:, 1],
                               c=colors, s=20, alpha=0.8, edgecolors='black', linewidth=0.5)

            # Calculate accuracy for this graph
            graph_accuracy = np.mean(sample['y_true'] == sample['y_pred'])

            ax.set_title(f'Graph {i+1}\nNodes: {len(sample["y_true"])}, '
                        f'Accuracy: {graph_accuracy:.3f}', fontsize=10)
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sample_graphs.pdf'),
                   dpi=300, bbox_inches='tight', format='pdf')
        plt.close()

    def _create_spatial_error_analysis(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        positions: np.ndarray,
        save_dir: str,
        temporal_window: int,
    ):
        """Create spatial error analysis visualization."""
        # Calculate different types of errors
        true_positives = (targets == 1) & (predictions == 1)
        false_positives = (targets == 0) & (predictions == 1)
        true_negatives = (targets == 0) & (predictions == 0)
        false_negatives = (targets == 1) & (predictions == 0)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Spatial Error Analysis - Temporal Window {temporal_window}',
                     fontsize=16, fontweight='bold')

        # True Positives
        tp_positions = positions[true_positives]
        if len(tp_positions) > 0:
            axes[0,0].scatter(tp_positions[:, 0], tp_positions[:, 1],
                            c='green', s=8, alpha=0.7, label=f'TP: {len(tp_positions)}')
        axes[0,0].set_title('True Positives (Correctly Predicted Occupied)')
        axes[0,0].set_xlabel('X Position (m)')
        axes[0,0].set_ylabel('Y Position (m)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # False Positives
        fp_positions = positions[false_positives]
        if len(fp_positions) > 0:
            axes[0,1].scatter(fp_positions[:, 0], fp_positions[:, 1],
                            c='orange', s=8, alpha=0.7, label=f'FP: {len(fp_positions)}')
        axes[0,1].set_title('False Positives (Incorrectly Predicted Occupied)')
        axes[0,1].set_xlabel('X Position (m)')
        axes[0,1].set_ylabel('Y Position (m)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # True Negatives
        tn_positions = positions[true_negatives]
        if len(tn_positions) > 0:
            axes[1,0].scatter(tn_positions[:, 0], tn_positions[:, 1],
                            c='lightblue', s=8, alpha=0.7, label=f'TN: {len(tn_positions)}')
        axes[1,0].set_title('True Negatives (Correctly Predicted Unoccupied)')
        axes[1,0].set_xlabel('X Position (m)')
        axes[1,0].set_ylabel('Y Position (m)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # False Negatives
        fn_positions = positions[false_negatives]
        if len(fn_positions) > 0:
            axes[1,1].scatter(fn_positions[:, 0], fn_positions[:, 1],
                            c='red', s=8, alpha=0.7, label=f'FN: {len(fn_positions)}')
        axes[1,1].set_title('False Negatives (Missed Occupied)')
        axes[1,1].set_xlabel('X Position (m)')
        axes[1,1].set_ylabel('Y Position (m)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'spatial_error_analysis.pdf'),
                   dpi=300, bbox_inches='tight', format='pdf')
        plt.close()

    def _generate_visualizations(
        self,
        targets: np.ndarray,
        preds: np.ndarray,
        probs: np.ndarray,
        data: List[Dict],
        temporal_window: int,
    ):
        """
        Generate visualizations.

        Args:
            targets: Ground truth labels
            preds: Predicted labels
            probs: Predicted probabilities
            data: List of data samples
            temporal_window: Size of temporal window
        """
        # Create confusion matrix
        cm = confusion_matrix(targets, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Unoccupied", "Occupied"],
            yticklabels=["Unoccupied", "Occupied"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix (Temporal Window: {temporal_window})")
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.config["evaluation"]["visualization"]["save_dir"],
            f"confusion_matrix_temporal_{temporal_window}.png"
        ))
        plt.close()

        # Create ROC curve
        fpr, tpr, _ = roc_curve(targets, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (Temporal Window: {temporal_window})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.config["evaluation"]["visualization"]["save_dir"],
            f"roc_curve_temporal_{temporal_window}.png"
        ))
        plt.close()

        # Visualize sample point clouds with predictions
        num_samples = min(self.config["evaluation"]["visualization"]["num_samples"], len(data))
        for i in range(num_samples):
            sample = data[i]

            # Create 2D scatter plot (only X and Y)
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            # Plot points
            colors = ["blue" if label == 0 else "red" for label in sample["y_true"]]
            ax.scatter(
                sample["pos"][:, 0],  # X coordinate
                sample["pos"][:, 1],  # Y coordinate
                c=colors,
                alpha=0.5,
            )

            # Set labels
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"Point Cloud Visualization (Sample {i+1})\n"
                         f"True: {'Occupied' if sample['y_true'][0] == 1 else 'Unoccupied'}, "
                         f"Predicted: {'Occupied' if sample['y_pred'] else 'Unoccupied'} "
                         f"(Prob: {sample['prob'][0]:.2f})")

            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=10, label="Unoccupied"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Occupied"),
            ]
            ax.legend(handles=legend_elements, loc="upper right")

            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(
                self.config["evaluation"]["visualization"]["save_dir"],
                f"point_cloud_temporal_{temporal_window}_sample_{i+1}.png"
            ))
            plt.close()

    def _analyze_feature_importance(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        temporal_window: int,
    ):
        """
        Analyze feature importance by feature perturbation.

        Args:
            model: Model to evaluate
            test_loader: Test data loader
            temporal_window: Size of temporal window
        """
        # Set model to evaluation mode
        model.eval()

        # Get a batch of data
        for batch in test_loader:
            batch = batch.to(self.device)
            break

        # Get baseline predictions
        with torch.no_grad():
            baseline_logits = model(batch.x, batch.edge_index, batch.batch)
            baseline_probs = torch.sigmoid(baseline_logits).cpu().numpy()

        # Dynamic feature names based on actual feature count
        num_features = batch.x.size(1)

        if num_features == 16:  # Collaborative dataset
            feature_names = [
                "Normalized X", "Normalized Y",
                "Raw X", "Raw Y",
                "Relative X", "Relative Y",
                "Origin X", "Origin Y",
                "Distance to Center",
                "Temporal Offset",
                "Robot1 Points", "Robot2 Points",
                "Collaboration Score", "Robot1 Ratio",
                "Robot2 Ratio", "Collaborative Feature"
            ]
        elif num_features == 10:  # Temporal dataset
            feature_names = [
                "Normalized X", "Normalized Y",
                "Raw X", "Raw Y",
                "Relative X", "Relative Y",
                "Origin X", "Origin Y",
                "Distance to Center",
                "Temporal Offset"
            ]
        elif num_features == 9:  # Basic dataset
            feature_names = [
                "Normalized X", "Normalized Y",
                "Raw X", "Raw Y",
                "Relative X", "Relative Y",
                "Origin X", "Origin Y",
                "Distance to Center"
            ]
        else:
            # Fallback for unknown feature counts
            feature_names = [f"Feature_{i}" for i in range(num_features)]

        importance_scores = []

        # Perturb each feature and measure the impact
        for i in range(num_features):
            # Create a copy of the batch
            perturbed_x = batch.x.clone()

            # Perturb the feature
            perturbed_x[:, i] = torch.randn_like(perturbed_x[:, i])

            # Get predictions with perturbed feature
            with torch.no_grad():
                perturbed_logits = model(perturbed_x, batch.edge_index, batch.batch)
                perturbed_probs = torch.sigmoid(perturbed_logits).cpu().numpy()

            # Compute the mean absolute difference in probabilities
            importance = np.mean(np.abs(perturbed_probs - baseline_probs))
            importance_scores.append(importance)

        # Normalize importance scores
        importance_scores = np.array(importance_scores)
        if np.sum(importance_scores) > 0:
            importance_scores = importance_scores / np.sum(importance_scores)

        # Plot feature importance
        plt.figure(figsize=(15, 8))
        plt.bar(feature_names, importance_scores)
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.title(f"Feature Importance (Temporal Window: {temporal_window}, {num_features} features)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.config["evaluation"]["visualization"]["save_dir"],
            f"feature_importance_temporal_{temporal_window}.png"
        ), dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create data loaders
    data_loaders = create_data_loaders(config)

    # Create evaluator
    evaluator = Evaluator(config)

    # Evaluate models for each temporal window
    for temporal_window in config["data"]["temporal_windows"]:
        print(f"\nEvaluating model for temporal window {temporal_window}")

        # Keep the original input dimension from config 
        # (ECC T5 model was trained with specific input_dim)

        # Create model
        model = create_model(config)

        # Load checkpoint
        checkpoint_path = os.path.join(
            config["training"]["checkpoint_dir"],
            f"model_temporal_{temporal_window}_best.pt"
        )

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint from {checkpoint_path}")

            # Evaluate model
            metrics = evaluator.evaluate(
                model,
                data_loaders[f"temporal_{temporal_window}"]["test"],
                temporal_window,
            )
        else:
            print(f"No checkpoint found at {checkpoint_path}. Skipping evaluation.")
