#!/usr/bin/env python3
"""
Class weight configuration based on raw CSV data analysis.
This module provides correct class weights for the GNN occupancy prediction task.
"""

import torch


class ClassWeightConfig:
    """
    Configuration for class weights based on comprehensive data analysis.
    """
    
    # Raw CSV data analysis results (ground truth)
    RAW_DATA_STATS = {
        'total_points': 473911,
        'workstation': 247847,  # 52.30%
        'unknown': 180981,      # 38.19% 
        'boundary': 35667,      # 7.53%
        'robot': 9305,          # 1.96%
        'KLT': 111,             # 0.02%
    }
    
    # Binary classification mapping
    OCCUPIED_LABELS = ['workstation', 'robot', 'boundary', 'KLT']
    UNOCCUPIED_LABELS = ['unknown']
    
    # Calculated totals
    TOTAL_OCCUPIED = 247847 + 9305 + 35667 + 111  # 292,930 (61.8%)
    TOTAL_UNOCCUPIED = 180981                       # 180,981 (38.2%)
    
    # Class ratios
    OCCUPIED_PERCENTAGE = 61.8
    UNOCCUPIED_PERCENTAGE = 38.2
    UNOCCUPIED_TO_OCCUPIED_RATIO = TOTAL_UNOCCUPIED / TOTAL_OCCUPIED  # 0.618
    
    @classmethod
    def get_pos_weight(cls, device='cpu'):
        """
        Get pos_weight tensor for BCEWithLogitsLoss.
        
        Args:
            device: Device to place tensor on
            
        Returns:
            torch.Tensor: pos_weight for BCEWithLogitsLoss
        """
        pos_weight = cls.UNOCCUPIED_TO_OCCUPIED_RATIO
        return torch.tensor([pos_weight]).to(device)
    
    @classmethod
    def get_bce_with_logits_loss(cls, device='cpu'):
        """
        Get BCEWithLogitsLoss with correct pos_weight.
        
        Args:
            device: Device for computation
            
        Returns:
            torch.nn.BCEWithLogitsLoss: Configured loss function
        """
        pos_weight = cls.get_pos_weight(device)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    @classmethod
    def get_custom_weighted_loss(cls, weight_unoccupied=0.65, weight_occupied=0.35):
        """
        Get custom weighted BCE loss.
        
        Args:
            weight_unoccupied: Weight for minority class (unoccupied)
            weight_occupied: Weight for majority class (occupied)
            
        Returns:
            WeightedBCELoss: Custom weighted loss function
        """
        from train import WeightedBCELoss
        return WeightedBCELoss(weight_unoccupied=weight_unoccupied, 
                              weight_occupied=weight_occupied)
    
    @classmethod
    def print_statistics(cls):
        """Print comprehensive statistics about the dataset."""
        print("📊 RAW CSV DATA ANALYSIS (Ground Truth)")
        print("=" * 60)
        print(f"Total data points: {cls.RAW_DATA_STATS['total_points']:,}")
        print()
        
        print("🏷️  Label Distribution:")
        for label, count in cls.RAW_DATA_STATS.items():
            if label != 'total_points':
                percentage = (count / cls.RAW_DATA_STATS['total_points']) * 100
                print(f"  {label}: {count:,} ({percentage:.2f}%)")
        print()
        
        print("⚖️  Binary Classification:")
        print(f"  Occupied: {cls.TOTAL_OCCUPIED:,} ({cls.OCCUPIED_PERCENTAGE:.1f}%)")
        print(f"  Unoccupied: {cls.TOTAL_UNOCCUPIED:,} ({cls.UNOCCUPIED_PERCENTAGE:.1f}%)")
        print(f"  Ratio (unoccupied:occupied): {cls.UNOCCUPIED_TO_OCCUPIED_RATIO:.3f}:1")
        print()
        
        print("🎯 Recommended Class Weights:")
        print(f"  pos_weight for BCEWithLogitsLoss: {cls.UNOCCUPIED_TO_OCCUPIED_RATIO:.3f}")
        print(f"  Custom weights: unoccupied=0.65, occupied=0.35")
        print()
        
        print("⚠️  Data Processing Issues Detected:")
        print("  • Robot detections missing in processed graphs")
        print("  • Boundary classifications inflated 6x")
        print("  • Fix data pipeline before training for best results")


def create_loss_function(method='bce_with_logits', device='cpu', **kwargs):
    """
    Factory function to create loss functions with correct class weights.
    
    Args:
        method: 'bce_with_logits' or 'custom_weighted'
        device: Device for computation
        **kwargs: Additional arguments for custom weighted loss
        
    Returns:
        Loss function with correct weights
    """
    if method == 'bce_with_logits':
        loss_fn = ClassWeightConfig.get_bce_with_logits_loss(device)
        print(f"✅ Created BCEWithLogitsLoss with pos_weight: {ClassWeightConfig.UNOCCUPIED_TO_OCCUPIED_RATIO:.3f}")
        
    elif method == 'custom_weighted':
        weight_unoccupied = kwargs.get('weight_unoccupied', 0.65)
        weight_occupied = kwargs.get('weight_occupied', 0.35)
        loss_fn = ClassWeightConfig.get_custom_weighted_loss(weight_unoccupied, weight_occupied)
        print(f"✅ Created Custom WeightedBCELoss (unoccupied: {weight_unoccupied}, occupied: {weight_occupied})")
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bce_with_logits' or 'custom_weighted'")
    
    return loss_fn


def compare_weight_methods():
    """Compare different weighting approaches."""
    print("🔍 COMPARING CLASS WEIGHT METHODS")
    print("=" * 60)
    
    # Method 1: No weighting
    print("1️⃣  No Class Weighting:")
    print("   Loss: nn.BCEWithLogitsLoss()")
    print("   Effect: Model biased toward majority class (occupied)")
    print("   Expected: Poor recall for unoccupied voxels")
    print()
    
    # Method 2: BCEWithLogitsLoss + pos_weight
    pos_weight = ClassWeightConfig.UNOCCUPIED_TO_OCCUPIED_RATIO
    print("2️⃣  BCEWithLogitsLoss + pos_weight (RECOMMENDED):")
    print(f"   Loss: nn.BCEWithLogitsLoss(pos_weight={pos_weight:.3f})")
    print("   Effect: Balances loss contribution from both classes")
    print("   Expected: Improved recall for unoccupied voxels")
    print()
    
    # Method 3: Custom weighted loss
    print("3️⃣  Custom Weighted BCE Loss:")
    print("   Loss: WeightedBCELoss(weight_unoccupied=0.65, weight_occupied=0.35)")
    print("   Effect: Manual control over class importance")
    print("   Expected: Similar to pos_weight but more flexible")
    print()
    
    print("🎯 Recommendation: Use Method 2 (BCEWithLogitsLoss + pos_weight)")
    print("   It's the PyTorch standard and most efficient implementation.")


if __name__ == "__main__":
    print("🎯 GNN OCCUPANCY CLASS WEIGHT CONFIGURATION")
    print("=" * 80)
    print()
    
    # Print statistics
    ClassWeightConfig.print_statistics()
    
    # Compare methods
    compare_weight_methods()
    
    # Test loss function creation
    print("\n🧪 TESTING LOSS FUNCTION CREATION")
    print("=" * 60)
    
    try:
        # Test BCEWithLogitsLoss
        loss1 = create_loss_function('bce_with_logits', device='cpu')
        
        # Test custom weighted loss
        loss2 = create_loss_function('custom_weighted', device='cpu')
        
        print("\n✅ All loss functions created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating loss functions: {e}")
    
    print("\n📋 Usage in training script:")
    print("from class_weight_config import create_loss_function")
    print("criterion = create_loss_function('bce_with_logits', device=device)")
