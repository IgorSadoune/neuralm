# Contributing to neuralm

Thank you for considering contributing to `neuralm`! This document outlines the process for contributing to the project. After you have read this document, please contact me at [igor.sadoune@pm.me](mailto:igor.sadoune@pm.me). Write a short description of yourself and explain your motivation to contribute.

## Project Structure

```
├── CONTRIBUTING.md           # Contribution guidelines
├── LICENSE                   # Project license
├── MANIFEST.in               # Package manifest file
├── Makefile                  # Build automation
├── README.md                 # Project overview
├── TUTORIAL.md               # Usage tutorial
├── api_reference.md          # API documentation
├── examples/                 # Example scripts
│   ├── configs/              # Configuration files for examples
│   ├── gan_example.py        # GAN implementation example
│   ├── mnist_cnn_example.py  # CNN with MNIST example
│   ├── simple_example.py     # Basic usage example
│   └── vae_example.py        # VAE implementation example
├── neuralpy/                 # Main package
│   ├── core/                 # Core functionality
│   │   └── model_builder.py  # Model construction utilities
│   ├── layers/               # Neural network layers
│   │   └── layer_factory.py  # Layer creation utilities
│   ├── models/               # Pre-defined model architectures
│   │   ├── attention.py      # Attention-based models
│   │   ├── autoencoder.py    # Autoencoder implementations
│   │   ├── cnn.py            # Convolutional neural networks
│   │   ├── gan.py            # Generative adversarial networks
│   │   ├── mlp.py            # Multi-layer perceptrons
│   │   ├── rnn.py            # Recurrent neural networks
│   │   ├── sequential.py     # Sequential model implementation
│   │   └── siamese.py        # Siamese networks
│   └── utils/                # Utility functions
│       └── yaml_parser.py    # YAML configuration parser
├── setup.py                  # Package setup script
└── tests/                    # Test suite
    └── test_model_builder.py # Tests for model builder
```

## Development Workflow

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/IgorSadoune/neuralm.git
   cd neuralm
   ```
3. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install the package in development mode:
   ```bash
   make dev
   ```

### Making Changes

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
   or
   ```bash
   git checkout -b fix/issue-description
   ```

2. Make your changes and ensure they follow the project's coding standards

3. Run tests to ensure your changes don't break existing functionality:
   ```bash
   make test
   ```

4. Run linting to ensure code quality:
   ```bash
   make lint
   ```

### Submitting Changes

1. Commit your changes with a descriptive commit message:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

2. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Create a Pull Request (PR) from your fork to the main repository

4. Respond to any code review feedback

## Git Best Practices

1. **Keep commits focused**: Each commit should represent a single logical change

2. **Write meaningful commit messages**: Use the format:
   ```
   Short summary (50 chars or less)
   
   More detailed explanation, if necessary. Wrap lines at 72 characters.
   Explain the problem that this commit is solving and why this change
   is the right solution.
   ```

3. **Keep your branch up to date**:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

4. **Use feature branches**: Never work directly on the main branch

5. **Squash commits**: Before submitting a PR, consider squashing multiple commits into logical units

## Python Best Practices

1. **Follow PEP 8**: Adhere to Python's style guide

2. **Write docstrings**: Document all functions, classes, and modules using Google-style docstrings

3. **Type hints**: Use type annotations where appropriate

4. **Write tests**: All new features should include tests

5. **Keep functions small**: Functions should do one thing well

6. **Use meaningful names**: Choose descriptive names for variables, functions, and classes

7. **Handle errors gracefully**: Use appropriate exception handling

## Testing

- Write unit tests for all new functionality
- Ensure all tests pass before submitting a PR
- Aim for high test coverage

## Documentation

- Update documentation for any new features or changes
- Include docstrings for all public functions, classes, and methods
- Add examples for new features

## Release Process

The maintainers will handle the release process, which typically involves:

1. Updating the version number
2. Creating a release branch
3. Building and testing the package
4. Uploading to PyPI
5. Creating a GitHub release

## Questions?

If you have any questions about contributing, please open an issue or reach out to me ([igor.sadoune@pm.me](mailto:igor.sadoune@pm.me)).

## Citing neuralm

If you use this software in your work, please cite neuralm: 

```bibtex
@software{neuralm,
  author = {Sadoune, Igor},
  title = {NeuralM: A Neural Network Model Builder},
  year = {2023},
  url = {https://github.com/IgorSadoune/neuralm},
  version = {0.1.0},
  publisher = {GitHub},
  description = {A flexible framework for building and training neural network models with YAML configuration.}
}
```