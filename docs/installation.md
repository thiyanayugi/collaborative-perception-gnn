# Installation Guide

Complete installation instructions for CPML on different platforms.

## System Requirements

### Minimum Requirements

- **OS**: Ubuntu 20.04+ / macOS 11+ / Windows 10+ (WSL2)
- **Python**: 3.9 or higher
- **RAM**: 8GB
- **Storage**: 50GB (for dataset)

### Recommended Requirements

- **OS**: Ubuntu 22.04
- **Python**: 3.10
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: 100GB SSD

## Installation Methods

### Method 1: pip (Recommended for Development)

1. **Install Python 3.9+**

   ```bash
   python3 --version  # Should be 3.9 or higher
   ```

2. **Clone Repository**

   ```bash
   git clone https://github.com/thiyanayugi/collaborative-perception-gnn.git
   cd collaborative-perception-gnn
   ```

3. **Create Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install Package**

   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

5. **Verify Installation**
   ```bash
   python -c "import cpml; print(cpml.__version__)"
   ```

### Method 2: Conda (Recommended for Research)

1. **Install Miniconda/Anaconda**

   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

2. **Clone Repository**

   ```bash
   git clone https://github.com/thiyanayugi/collaborative-perception-gnn.git
   cd collaborative-perception-gnn
   ```

3. **Create Conda Environment**

   ```bash
   conda env create -f environment.yml
   conda activate cpml
   ```

4. **Verify Installation**
   ```bash
   python -c "import cpml; print(cpml.__version__)"
   ```

### Method 3: Docker (Recommended for Production)

1. **Install Docker**

   ```bash
   # Ubuntu
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

   # Add user to docker group
   sudo usermod -aG docker $USER
   newgrp docker
   ```

2. **Install Docker Compose**

   ```bash
   sudo apt-get install docker-compose-plugin
   ```

3. **Clone Repository**

   ```bash
   git clone https://github.com/thiyanayugi/collaborative-perception-gnn.git
   cd collaborative-perception-gnn
   ```

4. **Build and Run**

   ```bash
   docker-compose up -d
   docker exec -it cpml_container bash
   ```

5. **Verify Installation**
   ```bash
   python -c "import cpml; print(cpml.__version__)"
   ```

## Installing ROS2 Humble (Optional)

Required only if you need to extract data from ROS2 bag files.

### Ubuntu 22.04

```bash
# Set locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS2 repository
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Humble
sudo apt update
sudo apt install ros-humble-ros-base python3-argcomplete
sudo apt install ros-dev-tools

# Source ROS2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## GPU Support (Optional)

### NVIDIA GPU with CUDA

1. **Install NVIDIA Drivers**

   ```bash
   ubuntu-drivers devices
   sudo ubuntu-drivers autoinstall
   ```

2. **Install CUDA Toolkit**

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
   sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda
   ```

3. **Install PyTorch with CUDA**

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Verify GPU Support**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'cpml'`**

- Solution: Install package with `pip install -e .`
- Ensure virtual environment is activated

**Issue: PyTorch Geometric installation fails**

- Solution: Install dependencies first
  ```bash
  pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
  pip install torch-geometric
  ```

**Issue: ROS2 not found**

- Solution: Source ROS2 setup
  ```bash
  source /opt/ros/humble/setup.bash
  ```

**Issue: CUDA out of memory**

- Solution: Use CPU or reduce batch size
  ```yaml
  # In config file
  training:
    device: "cpu"
    batch_size: 16 # Reduce from 32
  ```

## Uninstallation

### pip Installation

```bash
pip uninstall cpml
rm -rf venv/
```

### Conda Installation

```bash
conda env remove -n cpml
```

### Docker Installation

```bash
docker-compose down
docker rmi cpml:latest
```

## Next Steps

- Follow the [Quick Start Guide](quickstart.md)
- Read the [Architecture Documentation](architecture.md)
- Explore [examples/](../examples/)
