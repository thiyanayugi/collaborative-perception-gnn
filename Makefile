# Makefile for CPML project

.PHONY: help install install-dev test lint format clean docs

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package and dependencies
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e .
	pip install -r requirements-dev.txt

test:  ## Run tests
	pytest tests/ -v --cov=cpml --cov-report=html

lint:  ## Run linting checks
	flake8 cpml/
	pylint cpml/
	mypy cpml/

format:  ## Format code with black and isort
	black cpml/
	isort cpml/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

docs:  ## Build documentation
	cd docs && make html
