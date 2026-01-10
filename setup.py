#!/usr/bin/env python3
"""
Setup script for Collaborative Perception Management Layer (CPML).

This setup script configures the CPML package for installation, defining
dependencies, package structure, and metadata for the collaborative perception
framework.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    """Read README.md file for package long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements from requirements.txt
def read_requirements():
    """Read and parse requirements.txt file."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name='cpml',
    version='1.0.0',
    author='Thiyanayugi Mariraj',
    author_email='yugimariraj01@gmail.com',
    description='Collaborative Perception Management Layer for Multi-Robot Systems',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/thiyanayugi/collaborative-perception-gnn',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.9',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
        'vis': [
            'plotly>=5.0.0',
            'dash>=2.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'cpml-train=cpml.training.main:main',
            'cpml-evaluate=cpml.training.evaluate:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
