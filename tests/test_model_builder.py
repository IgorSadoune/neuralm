"""
Unit tests for the model builder functionality in the neuralm package.

This module tests the model building capabilities, including:
- Configuration validation
- Building sequential models from YAML
- Building MLP (Multi-Layer Perceptron) models from YAML
- Building CNN (Convolutional Neural Network) models from YAML

Each test verifies that models are correctly constructed according to their
configuration and can perform forward passes with appropriate input/output shapes.
"""

import unittest
import torch
import os
import yaml
import tempfile

from neuralm import build_model_from_yaml
from neuralm.utils.yaml_parser import validate_config

class TestModelBuilder(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test YAML files
        self.test_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()
    
    def test_validate_config(self):
        # Valid config
        valid_config = {
            'model_type': 'sequential',
            'layers': [
                {'type': 'linear', 'in_features': 10, 'out_features': 5},
                {'type': 'relu'}
            ]
        }
        self.assertTrue(validate_config(valid_config))
        
        # Invalid config - missing model_type
        invalid_config1 = {
            'layers': [
                {'type': 'linear', 'in_features': 10, 'out_features': 5}
            ]
        }
        with self.assertRaises(ValueError):
            validate_config(invalid_config1)
        
        # Invalid config - missing layers
        invalid_config2 = {
            'model_type': 'sequential'
        }
        with self.assertRaises(ValueError):
            validate_config(invalid_config2)
        
        # Invalid config - unsupported model type
        invalid_config3 = {
            'model_type': 'unsupported_type',
            'layers': [
                {'type': 'linear', 'in_features': 10, 'out_features': 5}
            ]
        }
        with self.assertRaises(ValueError):
            validate_config(invalid_config3)
    
    def test_build_sequential_model(self):
        # Create a test YAML file
        config = {
            'model_type': 'sequential',
            'name': 'TestSequential',
            'layers': [
                {'type': 'linear', 'in_features': 10, 'out_features': 5},
                {'type': 'relu'},
                {'type': 'linear', 'in_features': 5, 'out_features': 2}
            ]
        }
        
        yaml_path = os.path.join(self.test_dir.name, 'sequential.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)
        
        # Build model from YAML
        model = build_model_from_yaml(yaml_path)
        
        # Check model properties
        self.assertEqual(model.name, 'TestSequential')
        self.assertIsInstance(model.model, torch.nn.Sequential)
        self.assertEqual(len(model.model), 3)
        
        # Test forward pass
        x = torch.randn(2, 10)
        output = model(x)
        self.assertEqual(output.shape, (2, 2))
    
    def test_build_mlp_model(self):
        # Create a test YAML file
        config = {
            'model_type': 'mlp',
            'name': 'TestMLP',
            'input_size': 10,
            'hidden_sizes': [20, 10],
            'output_size': 5,
            'activation': 'relu',
            'dropout': 0.2,
            'layers': [
                {'type': 'linear', 'in_features': 10, 'out_features': 20},
                {'type': 'relu'},
                {'type': 'dropout', 'p': 0.2},
                {'type': 'linear', 'in_features': 20, 'out_features': 10},
                {'type': 'relu'},
                {'type': 'dropout', 'p': 0.2},
                {'type': 'linear', 'in_features': 10, 'out_features': 5}
            ]
        }
        
        yaml_path = os.path.join(self.test_dir.name, 'mlp.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)
        
        # Build model from YAML
        model = build_model_from_yaml(yaml_path)
        
        # Check model properties
        self.assertEqual(model.name, 'TestMLP')
        self.assertEqual(model.input_size, 10)
        self.assertEqual(model.output_size, 5)
        self.assertEqual(model.hidden_sizes, [20, 10])
        
        # Test forward pass
        x = torch.randn(2, 10)
        output = model(x)
        self.assertEqual(output.shape, (2, 5))
    
    def test_build_cnn_model(self):
        # Create a test YAML file
        config = {
            'model_type': 'cnn2d',
            'name': 'TestCNN',
            'in_channels': 3,
            'layers': [
                {'type': 'conv2d', 'in_channels': 3, 'out_channels': 16, 'kernel_size': 3, 'padding': 1},
                {'type': 'relu'},
                {'type': 'maxpool2d', 'kernel_size': 2},
                {'type': 'conv2d', 'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'padding': 1},
                {'type': 'relu'},
                {'type': 'maxpool2d', 'kernel_size': 2},
                {'type': 'flatten'},
                {'type': 'linear', 'in_features': 32 * 7 * 7, 'out_features': 10}
            ]
        }
        
        yaml_path = os.path.join(self.test_dir.name, 'cnn.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)
        
        # Build model from YAML
        model = build_model_from_yaml(yaml_path)
        
        # Check model properties
        self.assertEqual(model.name, 'TestCNN')
        self.assertEqual(model.in_channels, 3)
        
        # Test forward pass
        x = torch.randn(2, 3, 28, 28)  # Batch size, channels, height, width
        output = model(x)
        self.assertEqual(output.shape, (2, 10))

if __name__ == '__main__':
    unittest.main()