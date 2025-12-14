#!/usr/bin/env python3
"""
Collaborative Perception GNN Training System - Main Entry Point

This module serves as the main entry point for the collaborative perception framework
that trains Graph Neural Networks (GNNs) for occupancy prediction using mmWave radar
data from multiple robots in warehouse environments.

The system supports multiple GNN architectures (GATv2, ECC, GraphSAGE) and temporal
configurations for collaborative robot perception tasks.

Author: Thiyanayugi Mariraj
Project: Development of a Framework for Collaborative Perception Management Layer
         for Future 6G-Enabled Robotic Systems
"""

import os
import argparse
import yaml
import torch
from typing import Dict, List, Tuple, Optional, Union, Any

# Import custom modules for collaborative perception framework
from data import create_data_loaders
from model import create_model
from train import Trainer
from evaluate import Evaluator
from ablation import AblationStudy
from utils import set_seed, setup_logging, load_config, save_config, count_parameters


def parse_command_line_arguments():
    """
    Parse and validate command line arguments for the GNN training system.

    This function defines all available command line options for configuring
    the collaborative perception training pipeline, including model selection,
    temporal window configuration, and execution modes.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - config: Path to YAML configuration file
            - mode: Execution mode (train/evaluate/ablation/all)
            - temporal_window: Temporal window size for multi-frame processing
            - gnn_type: Graph Neural Network architecture type
            - seed: Random seed for reproducible experiments
            - visualize_only: Flag for visualization-only mode
    """
    parser = argparse.ArgumentParser(
        description="Collaborative Perception GNN Training System for Warehouse Robotics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config config.yaml --mode train --gnn_type gatv2
  %(prog)s --config config.yaml --mode evaluate --temporal_window 3
  %(prog)s --config config.yaml --mode all --seed 42
        """
    )

    # Configuration file specification
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file containing model and training parameters",
    )

    # Execution mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "ablation", "all"],
        default="all",
        help="Execution mode: train (training only), evaluate (evaluation only), "
             "ablation (ablation study), all (complete pipeline)",
    )

    # Temporal window configuration for multi-frame processing
    parser.add_argument(
        "--temporal_window",
        type=int,
        choices=[1, 3, 5],
        default=None,
        help="Temporal window size for multi-frame collaborative perception "
             "(1=single frame, 3/5=multi-frame). If not specified, uses all configured windows",
    )

    # Graph Neural Network architecture selection
    parser.add_argument(
        "--gnn_type",
        type=str,
        choices=["graphsage", "gatv2", "ecc"],
        default=None,
        help="GNN architecture type: graphsage (GraphSAGE), gatv2 (Graph Attention v2), "
             "ecc (Edge-Conditioned Convolution). Overrides config file setting",
    )

    # Random seed for reproducible experiments
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible experiments. Overrides config file setting",
    )

    # Visualization-only mode for analysis
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Generate visualizations only without running full training/evaluation pipeline",
    )

    return parser.parse_args()


def main():
    """
    Main execution function for the collaborative perception GNN training system.

    This function orchestrates the complete training and evaluation pipeline:
    1. Parses command line arguments and loads configuration
    2. Sets up logging and creates necessary directories
    3. Initializes data loaders for collaborative perception datasets
    4. Executes training, evaluation, or ablation studies based on mode
    5. Handles multiple temporal windows and GNN architectures

    The function supports collaborative perception with mmWave radar data from
    multiple robots, processing both single-frame and multi-frame temporal windows.
    """
    # Parse command line arguments
    args = parse_command_line_arguments()

    # Load base configuration from YAML file
    config = load_config(args.config)

    # Override configuration parameters with command line arguments
    # This allows for flexible experimentation without modifying config files
    if args.temporal_window is not None:
        config["data"]["temporal_windows"] = [args.temporal_window]

    if args.gnn_type is not None:
        config["model"]["gnn_type"] = args.gnn_type

    if args.seed is not None:
        config["training"]["seed"] = args.seed

    # Initialize random seed for reproducible experiments across all libraries
    set_seed(config["training"]["seed"])

    # Set up logging system for tracking training progress and debugging
    logger = setup_logging()
    logger.info("Starting Collaborative Perception GNN Training System")
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Execution mode: {args.mode}")
    logger.info(f"GNN architecture: {config['model']['gnn_type']}")
    logger.info(f"Temporal windows: {config['data']['temporal_windows']}")

    # Create necessary directories for checkpoints and visualizations
    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["evaluation"]["visualization"]["save_dir"], exist_ok=True)
    logger.info(f"Checkpoint directory: {config['training']['checkpoint_dir']}")
    logger.info(f"Visualization directory: {config['evaluation']['visualization']['save_dir']}")

    # Save the final configuration (including command line overrides) for reproducibility
    config_save_path = os.path.join(config["training"]["checkpoint_dir"], "config.yaml")
    save_config(config, config_save_path)
    logger.info(f"Configuration saved to: {config_save_path}")

    # Create data loaders for collaborative perception datasets
    # This handles loading of graph-structured data from multiple robots
    logger.info("Initializing collaborative perception data loaders...")
    data_loaders = create_data_loaders(config)
    logger.info("Data loaders created successfully")

    # Execute training pipeline if requested
    if args.mode in ["train", "all"]:
        logger.info("=" * 60)
        logger.info("STARTING TRAINING PHASE")
        logger.info("=" * 60)

        # Train models for each configured temporal window
        for temporal_window in config["data"]["temporal_windows"]:
            logger.info(f"\nTraining GNN model for temporal window: {temporal_window} frames")

            # Log collaborative perception feature configuration
            # The system uses 16 features for full collaborative perception
            input_dim = config['model']['input_dim']
            logger.info(f"Input feature dimension: {input_dim} (collaborative perception features)")

            if input_dim == 16:
                logger.info("Using full collaborative perception features:")
                logger.info("  - Spatial coordinates (x, y, z)")
                logger.info("  - Occupancy probability")
                logger.info("  - Robot contribution ratios")
                logger.info("  - Collaborative overlap indicators")
                logger.info("  - Temporal consistency features")

            # Create and initialize the GNN model
            model = create_model(config)
            param_count = count_parameters(model)
            logger.info(f"Model architecture: {config['model']['gnn_type'].upper()}")
            logger.info(f"Total trainable parameters: {param_count:,}")

            # Initialize trainer with configuration
            trainer = Trainer(config)

            # Execute training process with collaborative perception data
            logger.info("Starting training process...")
            trained_model = trainer.train(
                model,
                data_loaders[f"temporal_{temporal_window}"]["train"],
                data_loaders[f"temporal_{temporal_window}"]["val"],
                temporal_window,
            )
            logger.info(f"Training completed for temporal window {temporal_window}")

        logger.info("Training phase completed successfully")

    # Execute evaluation pipeline if requested
    if args.mode in ["evaluate", "all"]:
        logger.info("=" * 60)
        logger.info("STARTING EVALUATION PHASE")
        logger.info("=" * 60)

        # Evaluate trained models for each temporal window
        for temporal_window in config["data"]["temporal_windows"]:
            if args.visualize_only:
                logger.info(f"\nGenerating visualizations for temporal window: {temporal_window} frames")
            else:
                logger.info(f"\nEvaluating trained model for temporal window: {temporal_window} frames")

            # Log collaborative perception feature configuration for evaluation
            input_dim = config['model']['input_dim']
            logger.info(f"Model input dimension: {input_dim} features")

            # Create model architecture (same as used during training)
            model = create_model(config)
            logger.info(f"Created {config['model']['gnn_type'].upper()} model for evaluation")

            # Construct path to trained model checkpoint
            checkpoint_path = os.path.join(
                config["training"]["checkpoint_dir"],
                f"model_temporal_{temporal_window}_best.pt"
            )

            # Load trained model weights if checkpoint exists
            if os.path.exists(checkpoint_path):
                logger.info(f"Loading trained model from: {checkpoint_path}")

                # Load checkpoint with appropriate device handling
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(device)

                logger.info("Model weights loaded successfully")

                # Initialize evaluator for performance assessment
                evaluator = Evaluator(config)

                # Execute evaluation based on mode
                if args.visualize_only:
                    # Generate visualizations only (for analysis and debugging)
                    logger.info("Generating visualization outputs...")
                    evaluator.generate_visualizations(
                        model,
                        data_loaders[f"temporal_{temporal_window}"]["test"],
                        temporal_window,
                    )
                    viz_dir = config['evaluation']['visualization']['save_dir']
                    logger.info(f"Visualizations saved to: {viz_dir}")
                else:
                    # Perform comprehensive evaluation with metrics computation
                    logger.info("Performing comprehensive model evaluation...")
                    evaluation_metrics = evaluator.evaluate(
                        model,
                        data_loaders[f"temporal_{temporal_window}"]["test"],
                        temporal_window,
                    )
                    logger.info("Evaluation completed successfully")

                    # Log key performance metrics
                    if evaluation_metrics:
                        logger.info("Key Performance Metrics:")
                        for metric_name, metric_value in evaluation_metrics.items():
                            if isinstance(metric_value, float):
                                logger.info(f"  {metric_name}: {metric_value:.4f}")
                            else:
                                logger.info(f"  {metric_name}: {metric_value}")
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                logger.warning("Skipping evaluation for this temporal window")

        logger.info("Evaluation phase completed")

    # Execute ablation studies if requested
    if args.mode in ["ablation", "all"]:
        logger.info("=" * 60)
        logger.info("STARTING ABLATION STUDIES")
        logger.info("=" * 60)

        logger.info("Initializing ablation study framework...")

        # Create ablation study manager
        ablation_study = AblationStudy(config)

        # Execute systematic ablation studies
        logger.info("Running GNN architecture ablation study...")
        ablation_study.run_gnn_type_ablation()

        logger.info("Running temporal window ablation study...")
        ablation_study.run_temporal_window_ablation()

        logger.info("Ablation studies completed")

    logger.info("=" * 60)
    logger.info("COLLABORATIVE PERCEPTION SYSTEM EXECUTION COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
