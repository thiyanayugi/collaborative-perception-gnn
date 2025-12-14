# Multi-stage Dockerfile for CPML
# Optimized for production deployment with ROS2 and PyTorch

# Stage 1: Base image with ROS2 Humble
FROM ros:humble-ros-base AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ROS_DISTRO=humble

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python dependencies
FROM base AS dependencies

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies AS application

# Copy application code
COPY cpml/ /app/cpml/
COPY configs/ /app/configs/
COPY setup.py /app/
COPY README.md /app/
COPY LICENSE /app/

# Install package in editable mode
RUN pip3 install -e .

# Create non-root user for security
RUN useradd -m -u 1000 cpml_user && \
    chown -R cpml_user:cpml_user /app

USER cpml_user

# Set working directory
WORKDIR /app

# Expose ports (if needed for visualization)
EXPOSE 8050

# Default command
CMD ["bash"]
