# Contributing to CPML

Thank you for your interest in contributing to the Collaborative Perception Management Layer (CPML) project! This document provides guidelines for contributing to the project.

## Code of Conduct

This project follows a professional and respectful code of conduct. Please be considerate and constructive in all interactions.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in the [Issues](https://github.com/thiyanayugi/collaborative-perception-gnn/issues) section
2. If not, create a new issue with a clear title and description
3. Include relevant details:
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (OS, Python version, etc.)
   - Error messages or logs

### Submitting Changes

1. **Fork the repository** and create a new branch from `main`
2. **Make your changes** following the code style guidelines below
3. **Test your changes** thoroughly
4. **Commit your changes** with clear, descriptive commit messages
5. **Push to your fork** and submit a pull request

### Pull Request Process

1. Update documentation if you're changing functionality
2. Add tests for new features
3. Ensure all tests pass
4. Update the README.md if needed
5. Reference any related issues in your PR description

## Code Style Guidelines

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Maximum line length: 100 characters

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Brief description of the function.

    Longer description if needed, explaining the purpose and behavior.

    Args:
        param1 (int): Description of param1
        param2 (str): Description of param2

    Returns:
        bool: Description of return value

    Raises:
        ValueError: When invalid input is provided
    """
    pass
```

### Commit Message Format

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line should be concise (50 chars or less)
- Reference issues and pull requests when relevant

Example:

```
Add multi-scale temporal window support

- Implement temporal window aggregation
- Add configuration options for window sizes
- Update documentation

Fixes #123
```

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/thiyanayugi/collaborative-perception-gnn.git
cd collaborative-perception-gnn
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:

```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks (optional):

```bash
pip install pre-commit
pre-commit install
```

## Testing

Before submitting a pull request, ensure your code passes all tests:

```bash
pytest tests/
```

## Documentation

When adding new features:

1. Update relevant documentation in `docs/`
2. Add docstrings to all new functions/classes
3. Update README.md if the feature affects usage
4. Add examples if applicable

## Questions?

If you have questions about contributing, feel free to:

- Open an issue with the "question" label
- Contact the maintainer at [your.email@example.com]

Thank you for contributing to CPML! 🚀
