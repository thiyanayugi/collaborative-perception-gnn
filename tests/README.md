# Tests

Unit tests for the CPML framework.

## Running Tests

Run all tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=cpml --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_preprocessing.py
```

## Test Structure

- `test_preprocessing/`: Tests for preprocessing modules
- `test_training/`: Tests for training modules
- `test_visualization/`: Tests for visualization modules

## Writing Tests

Follow these guidelines when writing tests:
- Use descriptive test names
- Test both success and failure cases
- Use fixtures for common setup
- Aim for high code coverage
