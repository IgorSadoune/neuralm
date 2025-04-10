"""Simple Examples for neuralm Library

This script demonstrates various ways to build and use neural network models with the neuralm library.
It includes examples of:
- Building models from YAML configuration files
- Building models from Python dictionaries
- Working with different model architectures (CNN, VAE, GAN)
- Performing forward passes with random input data
"""

import torch
from neuralm import build_model_from_yaml, build_model_from_config

def example_from_yaml():
    print("\nBuilding model from YAML...")
    # Build a model from a YAML file
    model = build_model_from_yaml('configs/cnn_mnist.yaml')
    print(model)
    
    # Create a random input tensor
    x = torch.randn(2, 1, 28, 28)  # Batch size, channels, height, width
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

def example_from_config():
    print("\nBuilding model from config dictionary...")
    # Build a model from a configuration dictionary
    config = {
        'model_type': 'sequential',
        'name': 'SimpleModel',
        'layers': [
            {'type': 'linear', 'in_features': 784, 'out_features': 128},
            {'type': 'relu'},
            {'type': 'linear', 'in_features': 128, 'out_features': 10}
        ]
    }
    
    model = build_model_from_config(config)
    print(model)
    
    # Create a random input tensor
    x = torch.randn(2, 784)  # Batch size, input features
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

def example_vae():
    print("\nBuilding VAE model from YAML...")
    # Build a VAE model from a YAML file
    model = build_model_from_yaml('configs/vae.yaml')
    print(model)
    
    # Create a random input tensor
    x = torch.randn(2, 784)  # Batch size, input features
    
    # Forward pass
    recon_x, mu, log_var = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {recon_x.shape}")
    print(f"Mu shape: {mu.shape if mu is not None else None}")
    print(f"Log Var shape: {log_var.shape if log_var is not None else None}")

def example_gan():
    print("\nBuilding GAN model from YAML...")
    # Build a GAN model from a YAML file
    model = build_model_from_yaml('configs/gan.yaml')
    print(model)
    
    # Create random input tensors
    real_samples = torch.randn(2, 784)  # Batch size, input features
    latent_vectors = torch.randn(2, 100)  # Batch size, latent size
    
    # Forward pass
    disc_output, gen_output = model(real_samples, latent_vectors)
    print(f"Real samples shape: {real_samples.shape}")
    print(f"Latent vectors shape: {latent_vectors.shape}")
    print(f"Discriminator output shape: {disc_output.shape if disc_output is not None else None}")
    print(f"Generator output shape: {gen_output.shape if gen_output is not None else None}")

def main():
    print("NeuralPy Simple Examples")
    print("=======================")
    
    example_from_yaml()
    example_from_config()
    example_vae()
    example_gan()

if __name__ == '__main__':
    main()