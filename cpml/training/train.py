#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from data import create_data_loaders
from model import create_model
from class_weight_config import ClassWeightConfig, create_loss_function


class WeightedBCELoss(torch.nn.Module):
    """
    Custom weighted BCE loss for imbalanced binary classification.
    Based on raw CSV data analysis: 61.8% occupied, 38.2% unoccupied.
    Gives higher weight to minority class (unoccupied).
    """
    def __init__(self, weight_unoccupied: float = 0.618, weight_occupied: float = 0.382) -> None:
        super().__init__()
        # Minority class (unoccupied) gets higher weight
        self.weight_unoccupied = weight_unoccupied
        # Majority class (occupied) gets lower weight
        self.weight_occupied = weight_occupied

        print(f"Custom WeightedBCELoss initialized:")
        print(f"  Unoccupied weight (minority): {self.weight_unoccupied}")
        print(f"  Occupied weight (majority): {self.weight_occupied}")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions, targets.float(), reduction='none'
        )
        # targets == 1 means occupied, targets == 0 means unoccupied
        weights = torch.where(targets == 1, self.weight_occupied, self.weight_unoccupied)
        return (bce_loss * weights).mean()


def calculate_class_weights(train_loader: torch.utils.data.DataLoader, device: torch.device) -> torch.Tensor:
    """
    Calculate pos_weight for BCEWithLogitsLoss based on VOXEL-LEVEL class distribution.

    Args:
        train_loader: Training data loader
        device: Device to use for calculations

    Returns:
        pos_weight tensor for BCEWithLogitsLoss
    """
    total_occupied_voxels = 0
    total_unoccupied_voxels = 0

    print("Calculating class weights from training data (voxel-level)...")
    for batch in tqdm(train_loader, desc="Analyzing voxel class distribution"):
        # Move batch to device
        batch = batch.to(device)

        # Count INDIVIDUAL voxel labels in each batch
        occupied_voxels = torch.sum(batch.y == 1).item()
        unoccupied_voxels = torch.sum(batch.y == 0).item()

        total_occupied_voxels += occupied_voxels
        total_unoccupied_voxels += unoccupied_voxels

    # Calculate pos_weight (ratio of negative to positive samples)
    pos_weight = total_unoccupied_voxels / total_occupied_voxels if total_occupied_voxels > 0 else 1.0

    print(f"Voxel-level dataset statistics:")
    print(f"  Total occupied voxels: {total_occupied_voxels:,}")
    print(f"  Total unoccupied voxels: {total_unoccupied_voxels:,}")
    print(f"  Voxel imbalance ratio: {total_unoccupied_voxels/total_occupied_voxels:.2f}:1")
    print(f"  Calculated pos_weight: {pos_weight:.3f}")

    return torch.tensor([pos_weight]).to(device)


class Trainer:
    """
    Trainer class for the occupancy prediction model with class weighting for imbalanced data.
    """

    def __init__(self, config: Dict):
        """
        Initialize the trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config["training"]["device"])

        # Set random seed for reproducibility
        torch.manual_seed(config["training"]["seed"])
        np.random.seed(config["training"]["seed"])

        # Create checkpoint directory
        os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)

        # Initialize metrics history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_f1": [],
            "val_f1": [],
        }

    def train(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        temporal_window: int,
    ) -> nn.Module:
        """
        Train the model with class weighting for imbalanced data.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            temporal_window: Temporal window size

        Returns:
            Trained model
        """
        # Move model to device
        model = model.to(self.device)

        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )

        # Create learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.config["training"]["lr_scheduler"]["patience"],
            factor=self.config["training"]["lr_scheduler"]["factor"],
            min_lr=self.config["training"]["lr_scheduler"]["min_lr"],
        )

        # Print dataset statistics
        print("\n📊 DATASET STATISTICS:")
        ClassWeightConfig.print_statistics()

        # Create loss function with correct class weights
        # Method 1: BCEWithLogitsLoss with pos_weight (RECOMMENDED)
        criterion = create_loss_function('bce_with_logits', device=self.device)

        # Alternative methods (uncomment to use):
        # Method 2: Custom weighted loss
        # criterion = create_loss_function('custom_weighted', device=self.device)

        # Method 3: Calculate from current processed data (may be corrupted)
        # pos_weight = calculate_class_weights(train_loader, self.device)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # print(f"Using processed data weights: {pos_weight.item():.3f}")

        print(f"✅ Loss function configured for correct class weighting")

        # Early stopping variables
        best_val_loss = float("inf")
        early_stopping_counter = 0

        # Training loop
        for epoch in range(self.config["training"]["epochs"]):
            print(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")

            # Train
            train_metrics = self._train_epoch(model, train_loader, optimizer, criterion)

            # Validate
            val_metrics = self._validate_epoch(model, val_loader, criterion)

            # Update learning rate
            scheduler.step(val_metrics["loss"])

            print(f"  Train Loss: {train_metrics['loss']:.4f}, "
                  f"Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                  f"Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_accuracy"].append(train_metrics["accuracy"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["val_f1"].append(val_metrics["f1"])

            # Check for improvement
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                early_stopping_counter = 0

                # Save best model
                checkpoint_path = os.path.join(
                    self.config["training"]["checkpoint_dir"],
                    f"model_temporal_{temporal_window}_best.pt",
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_metrics["loss"],
                        "config": self.config,
                    },
                    checkpoint_path,
                )
                print(f"  New best model saved to {checkpoint_path}")
            else:
                early_stopping_counter += 1

            # Early stopping
            if (
                early_stopping_counter
                >= self.config["training"]["early_stopping"]["patience"]
            ):
                print(f"Early stopping after {epoch + 1} epochs")
                break

        return model

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function

        Returns:
            Dictionary of metrics
        """
        # Set model to training mode
        model.train()

        # Initialize metrics
        total_loss = 0.0
        all_preds = []
        all_targets = []

        # Training loop
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to device
            batch = batch.to(self.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(batch.x, batch.edge_index, batch.batch)

            # Convert node-level labels to graph-level labels using majority vote
            graph_labels = []
            for i in range(batch.num_graphs):
                mask = batch.batch == i
                # Use mean of node labels and round to get binary classification
                graph_label = batch.y[mask].float().mean().round()
                graph_labels.append(graph_label)

            targets = torch.tensor(graph_labels, device=batch.y.device).float().view(-1, 1)

            # Compute loss
            loss = criterion(logits.squeeze(), targets.squeeze())

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Update metrics
            total_loss += loss.item() * batch.num_graphs

            # Convert logits to predictions
            preds = torch.sigmoid(logits).detach().cpu().numpy() > 0.5
            targets_np = targets.detach().cpu().numpy()

            all_preds.extend(preds.flatten())
            all_targets.extend(targets_np.flatten())

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, zero_division=0)

        return {
            "loss": total_loss / len(train_loader.dataset),
            "accuracy": accuracy,
            "f1": f1,
        }

    def _validate_epoch(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            model: Model to validate
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Dictionary of metrics
        """
        # Set model to evaluation mode
        model.eval()

        # Initialize metrics
        total_loss = 0.0
        all_preds = []
        all_targets = []

        # Validation loop
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = batch.to(self.device)

                # Forward pass
                logits = model(batch.x, batch.edge_index, batch.batch)

                # Convert node-level labels to graph-level labels
                graph_labels = []
                for i in range(batch.num_graphs):
                    mask = batch.batch == i
                    graph_label = batch.y[mask].float().mean().round()
                    graph_labels.append(graph_label)

                targets = torch.tensor(graph_labels, device=batch.y.device).float().view(-1, 1)

                # Compute loss
                loss = criterion(logits.squeeze(), targets.squeeze())

                # Update metrics
                total_loss += loss.item() * batch.num_graphs

                # Convert logits to predictions
                preds = torch.sigmoid(logits).detach().cpu().numpy() > 0.5
                targets_np = targets.detach().cpu().numpy()

                all_preds.extend(preds.flatten())
                all_targets.extend(targets_np.flatten())

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, zero_division=0)

        return {
            "loss": total_loss / len(val_loader.dataset),
            "accuracy": accuracy,
            "f1": f1,
        }


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train GNN Occupancy Model with Class Weighting")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Force input dimension to 16
    config["model"]["input_dim"] = 16

    print(f"Using config: {args.config}")
    print(f"Model input dimension: {config['model']['input_dim']}")
    print(f"Temporal windows: {config['data']['temporal_windows']}")

    # Create data loaders
    data_loaders = create_data_loaders(config)

    # Create trainer
    trainer = Trainer(config)

    # Train models for each temporal window
    for temporal_window in config["data"]["temporal_windows"]:
        print(f"\nTraining model for temporal window {temporal_window}")

        # Create model
        model = create_model(config)

        # Train model
        model = trainer.train(
            model,
            data_loaders[f"temporal_{temporal_window}"]["train"],
            data_loaders[f"temporal_{temporal_window}"]["val"],
            temporal_window,
        )

        print(f"Training completed for temporal window {temporal_window}")
