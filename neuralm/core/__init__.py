"""Neural Machines (neuralm): A PyTorch model generator from YAML configurations

This package provides tools to build neural network models from YAML configurations
or Python dictionaries. It simplifies the process of creating complex neural network
architectures by allowing users to define models declaratively.
"""

__version__ = '0.1.0'

from neuralm.core.model_builder import build_model_from_yaml, build_model_from_config

__all__ = ['build_model_from_yaml', 'build_model_from_config']