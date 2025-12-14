#!/usr/bin/env python3
"""
Ablation Study Framework for Collaborative Perception GNN Models

This module implements comprehensive ablation studies for evaluating different
GNN architectures and configurations in collaborative perception tasks.
The framework systematically tests various model components to understand
their individual contributions to performance.

Author: Thiyanayugi Mariraj
Project: Development of a Framework for Collaborative Perception Management Layer
         for Future 6G-Enabled Robotic Systems
"""

import os
import yaml
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from data import create_data_loaders
from model import create_model
from train import Trainer
from evaluate import Evaluator
from utils import set_seed, setup_logging, count_parameters


class AblationStudy:
    """
    Comprehensive Ablation Study Framework for Collaborative Perception GNNs.
    
    This class orchestrates systematic ablation studies to evaluate the impact
    of different model architectures, temporal configurations, and feature
    combinations on collaborative perception performance.
    
    Key Studies:
    - GNN Architecture Comparison (GATv2, ECC, GraphSAGE)
    - Temporal Window Analysis (1, 3, 5 frames)
    - Feature Ablation (collaborative vs. single-robot features)
    - Hyperparameter Sensitivity Analysis
    """
    
    def __init__(self, base_config: Dict[str, Any]):
        """
        Initialize the ablation study framework.
        
        Args:
            base_config (dict): Base configuration for experiments
        """
        self.base_config = base_config.copy()
        self.logger = setup_logging()
        self.results = []
        
        # Create results directory
        self.results_dir = os.path.join(
            self.base_config["training"]["checkpoint_dir"], 
            "ablation_studies"
        )
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger.info("Initialized Ablation Study Framework")
        self.logger.info(f"Results will be saved to: {self.results_dir}")
    
    def run_gnn_type_ablation(self):
        """
        Run ablation study comparing different GNN architectures.
        
        Tests GATv2, ECC, and GraphSAGE architectures with identical
        configurations to isolate the impact of the graph convolution method.
        """
        self.logger.info("Starting GNN Architecture Ablation Study")
        
        gnn_types = ["gatv2", "ecc", "graphsage"]
        
        for gnn_type in gnn_types:
            self.logger.info(f"Testing GNN architecture: {gnn_type.upper()}")
            
            # Create modified configuration
            config = self.base_config.copy()
            config["model"]["gnn_type"] = gnn_type
            
            # Run experiment
            results = self._run_single_experiment(
                config, 
                experiment_name=f"gnn_ablation_{gnn_type}"
            )
            
            if results:
                results["ablation_type"] = "gnn_architecture"
                results["gnn_type"] = gnn_type
                self.results.append(results)
        
        self.logger.info("Completed GNN Architecture Ablation Study")
    
    def run_temporal_window_ablation(self):
        """
        Run ablation study comparing different temporal window sizes.
        
        Tests single-frame (1) vs. multi-frame (3, 5) processing to evaluate
        the benefit of temporal information in collaborative perception.
        """
        self.logger.info("Starting Temporal Window Ablation Study")
        
        temporal_windows = [1, 3, 5]
        
        for window_size in temporal_windows:
            self.logger.info(f"Testing temporal window: {window_size} frames")
            
            # Create modified configuration
            config = self.base_config.copy()
            config["data"]["temporal_windows"] = [window_size]
            
            # Run experiment
            results = self._run_single_experiment(
                config,
                experiment_name=f"temporal_ablation_{window_size}frames"
            )
            
            if results:
                results["ablation_type"] = "temporal_window"
                results["temporal_window"] = window_size
                self.results.append(results)
        
        self.logger.info("Completed Temporal Window Ablation Study")
    
    def _run_single_experiment(
        self, 
        config: Dict[str, Any], 
        experiment_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Run a single ablation experiment.
        
        Args:
            config (dict): Experiment configuration
            experiment_name (str): Name for this experiment
            
        Returns:
            dict: Experiment results or None if failed
        """
        try:
            self.logger.info(f"Running experiment: {experiment_name}")
            
            # Set random seed for reproducibility
            set_seed(config["training"]["seed"])
            
            # Create data loaders
            data_loaders = create_data_loaders(config)
            
            # Get temporal window for this experiment
            temporal_window = config["data"]["temporal_windows"][0]
            
            # Create model
            model = create_model(config)
            param_count = count_parameters(model)
            
            self.logger.info(f"Model parameters: {param_count:,}")
            
            # Create trainer
            trainer = Trainer(config)
            
            # Train model
            trained_model = trainer.train(
                model,
                data_loaders[f"temporal_{temporal_window}"]["train"],
                data_loaders[f"temporal_{temporal_window}"]["val"],
                temporal_window,
            )
            
            # Evaluate model
            evaluator = Evaluator(config)
            metrics = evaluator.evaluate(
                trained_model,
                data_loaders[f"temporal_{temporal_window}"]["test"],
                temporal_window,
            )
            
            # Compile results
            results = {
                "experiment_name": experiment_name,
                "timestamp": datetime.now().isoformat(),
                "model_parameters": param_count,
                "temporal_window": temporal_window,
                "gnn_type": config["model"]["gnn_type"],
                "metrics": metrics,
                "config": config
            }
            
            self.logger.info(f"Experiment {experiment_name} completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Experiment {experiment_name} failed: {e}")
            return None
    
    def save_results(self):
        """Save ablation study results to files."""
        if not self.results:
            self.logger.warning("No results to save")
            return
        
        # Save detailed results as JSON
        results_file = os.path.join(self.results_dir, "ablation_results.json")
        import json
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create summary CSV
        summary_data = []
        for result in self.results:
            summary_row = {
                "experiment_name": result["experiment_name"],
                "ablation_type": result.get("ablation_type", "unknown"),
                "gnn_type": result["gnn_type"],
                "temporal_window": result["temporal_window"],
                "model_parameters": result["model_parameters"],
            }
            
            # Add metrics if available
            if result.get("metrics"):
                for metric_name, metric_value in result["metrics"].items():
                    if isinstance(metric_value, (int, float)):
                        summary_row[f"metric_{metric_name}"] = metric_value
            
            summary_data.append(summary_row)
        
        # Save summary CSV
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.results_dir, "ablation_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        
        self.logger.info(f"Results saved to {results_file}")
        self.logger.info(f"Summary saved to {summary_file}")
    
    def generate_report(self):
        """Generate a comprehensive ablation study report."""
        if not self.results:
            self.logger.warning("No results available for report generation")
            return
        
        report_file = os.path.join(self.results_dir, "ablation_report.md")
        
        with open(report_file, 'w') as f:
            f.write("# Collaborative Perception GNN Ablation Study Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # GNN Architecture Results
            gnn_results = [r for r in self.results if r.get("ablation_type") == "gnn_architecture"]
            if gnn_results:
                f.write("## GNN Architecture Comparison\n\n")
                f.write("| Architecture | Parameters | Performance |\n")
                f.write("|--------------|------------|-------------|\n")
                for result in gnn_results:
                    f.write(f"| {result['gnn_type'].upper()} | {result['model_parameters']:,} | - |\n")
                f.write("\n")
            
            # Temporal Window Results
            temporal_results = [r for r in self.results if r.get("ablation_type") == "temporal_window"]
            if temporal_results:
                f.write("## Temporal Window Analysis\n\n")
                f.write("| Window Size | Parameters | Performance |\n")
                f.write("|-------------|------------|-------------|\n")
                for result in temporal_results:
                    f.write(f"| {result['temporal_window']} frames | {result['model_parameters']:,} | - |\n")
                f.write("\n")
        
        self.logger.info(f"Report generated: {report_file}")


if __name__ == "__main__":
    # Example usage
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    ablation = AblationStudy(config)
    ablation.run_gnn_type_ablation()
    ablation.run_temporal_window_ablation()
    ablation.save_results()
    ablation.generate_report()
