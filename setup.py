"""
Setup configuration for Collaborative Perception Management Layer (CPML)
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_file(filename):
    with open(filename, encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements(filename):
    with open(filename, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="cpml",
    version="1.0.0",
    author="Thiyanayugi Mariraj",
    author_email="your.email@example.com",  # Update with your email
    description="Collaborative Perception Management Layer for Multi-Robot Systems",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/thiyanayugi/collaborative-perception-gnn",  # Update with your GitHub URL
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cpml-train=cpml.training.main:main",
            "cpml-evaluate=cpml.training.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cpml": ["configs/*.yaml"],
    },
    zip_safe=False,
    keywords="graph-neural-networks collaborative-perception multi-robot robotics warehouse-automation 6g",
)
