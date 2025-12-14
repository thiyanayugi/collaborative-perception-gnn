# Quick Reference: CPML Repository

## Repository Location

**Path**: `/home/yugi/cpml_gnn`  
**Recommended GitHub Name**: `collaborative-perception-gnn`

## Package Structure

```
cpml/
├── preprocessing/    # 9 modules for data processing
├── training/         # 8 modules for GNN training
└── visualization/    # 5 modules for visualization
```

## Key Commands

### Installation

```bash
# Using pip
pip install -e .

# Using conda
conda env create -f environment.yml
conda activate cpml

# Using Docker
docker-compose up -d
```

### Training

```bash
# Command line
cpml-train --config configs/models/config_standard_gatv2_t3.yaml

# Python
python examples/train_model.py --config configs/models/config_standard_gatv2_t3.yaml
```

### Git Commands

```bash
# View status
git status

# View commit history
git log --oneline

# Add remote (after creating GitHub repo)
git remote add origin https://github.com/yourusername/collaborative-perception-gnn.git

# Push to GitHub
git push -u origin main
```

## Files to Update Before GitHub Push

1. **setup.py** (Line 18, 21): Update email and GitHub URL
2. **CITATION.bib** (Lines 5, 8, 9): Update university, location, month
3. **README.md** (Line 258): Update email
4. **cpml/**init**.py** (Line 11): Update email

## Repository Statistics

- **Python Modules**: 25 in cpml/
- **Documentation Files**: 6
- **Data Frames**: 59,541
- **Model Checkpoints**: 6
- **Git Commits**: 1 (initial)

## Next Steps

1. Update personal information in files above
2. Create GitHub repository
3. Push code: `git push -u origin main`
4. Configure repository settings
5. Add topics and description
6. Optional: Create v1.0.0 release
