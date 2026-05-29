# Contributing to AutoProj

Welcome to AutoProj! We appreciate your interest in contributing to this project.

## How to Contribute

### 1. Fork the Repository

Fork the repository to your GitHub account.

### 2. Clone Your Fork

```bash
git clone git@github.com:your-username/autoproj.git
cd autoproj
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Changes

- Follow the code style guidelines
- Write tests for new functionality
- Update documentation as needed

### 5. Run Tests

```bash
pip install -e .[dev]
pytest tests/ -v
```

### 6. Commit Changes

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```bash
git add .
git commit -m "feat: add new feature"
```

### 7. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 8. Create a Pull Request

Create a pull request from your feature branch to the `develop` branch of the main repository.

## Code Style Guidelines

- Follow PEP 8 style guide for Python code
- Use type hints for all functions and methods
- Write docstrings for all public functions and classes
- Keep lines under 120 characters
- Use 4 spaces for indentation (not tabs)

## Testing Guidelines

- All new features must have corresponding unit tests
- Tests should be placed in the `tests/` directory
- Use pytest for testing
- Ensure all tests pass before submitting a pull request

## Documentation Guidelines

- Update README.md when adding new features
- Add docstrings to all public APIs
- Update examples when appropriate

## Pull Request Checklist

- [ ] All tests pass
- [ ] Code follows PEP 8 style guidelines
- [ ] Documentation is updated
- [ ] Commit messages follow Conventional Commits format
- [ ] Pull request title is descriptive

## Issue Reporting

When reporting issues, please include:

1. A clear description of the problem
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Version information

## License

By contributing to AutoProj, you agree that your contributions will be licensed under the MIT License.
