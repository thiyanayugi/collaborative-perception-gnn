# Configuration Files

This directory contains configuration files for model training and evaluation.

## Structure

- `models/`: Model architecture configurations
  - GATv2 configurations
  - ECC configurations  
  - GraphSAGE configurations

## Usage

Use these YAML files to configure model training parameters, architecture settings, and hyperparameters.

## Example

```bash
python -m cpml.training.main --config configs/models/config_standard_gatv2_t3.yaml
```
