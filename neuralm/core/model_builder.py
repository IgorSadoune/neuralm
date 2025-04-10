"""
Model Builder Module

This module provides functionality to build PyTorch neural network models from YAML
configuration files or dictionaries. It serves as the core component of the neuralm
framework for creating various types of neural network architectures.

The module supports building different types of models including:
- Sequential models
- Recurrent Neural Networks (RNN, LSTM, GRU)
- Convolutional Neural Networks (CNN)
- Multi-Layer Perceptrons (MLP)
- Attention-based models and Transformers
- Autoencoders and Variational Autoencoders (VAE)
- Generative Adversarial Networks (GAN)
- Siamese Networks
- Graph Neural Networks (GNN) - placeholder
- Restricted Boltzmann Machines (RBM) - placeholder

Each model type has a dedicated builder function that extracts the relevant
parameters from the configuration and constructs the appropriate model.
"""

import torch.nn as nn
from typing import Dict, Any

from neuralm.utils.yaml_parser import load_yaml, validate_config
from neuralm.layers.layer_factory import LayerFactory
from neuralm.models.sequential import SequentialModel
from neuralm.models.rnn import RNNModel
from neuralm.models.cnn import CNNModel
from neuralm.models.mlp import MLPModel
from neuralm.models.attention import AttentionModel
from neuralm.models.autoencoder import AutoencoderModel
from neuralm.models.gan import GANModel
from neuralm.models.siamese import SiameseModel

def build_model_from_yaml(yaml_path: str) -> nn.Module:
    """
    Build a PyTorch model from a YAML configuration file.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        PyTorch model
        
    Raises:
        FileNotFoundError: If the YAML file does not exist
        ValueError: If the configuration is invalid
    """
    config = load_yaml(yaml_path)
    validate_config(config)
    return build_model_from_config(config)

def build_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Build a PyTorch model from a configuration dictionary.
    
    Args:
        config: Dictionary containing the model configuration
        
    Returns:
        PyTorch model
        
    Raises:
        ValueError: If the configuration is invalid or the model type is not supported
    """
    model_type = config['model_type'].lower()
    
    # Create the model based on the type
    if model_type == 'sequential':
        return _build_sequential_model(config)
    elif model_type in ['rnn', 'lstm', 'gru']:
        return _build_rnn_model(config)
    elif model_type in ['cnn1d', 'cnn2d', 'cnn3d']:
        return _build_cnn_model(config)
    elif model_type == 'mlp':
        return _build_mlp_model(config)
    elif model_type in ['attention', 'transformer']:
        return _build_attention_model(config)
    elif model_type in ['autoencoder', 'vae']:
        return _build_autoencoder_model(config)
    elif model_type == 'gan':
        return _build_gan_model(config)
    elif model_type == 'siamese':
        return _build_siamese_model(config)
    elif model_type == 'gnn':
        return _build_gnn_model(config)
    elif model_type == 'rbm':
        return _build_rbm_model(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def _build_sequential_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build a sequential model.
    
    Args:
        config: Dictionary containing the model configuration
        
    Returns:
        Sequential PyTorch model
    """
    layers = []
    for layer_config in config['layers']:
        layers.append(LayerFactory.create_layer(layer_config))
    
    return SequentialModel(layers, config.get('name', 'SequentialModel'))

def _build_rnn_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build an RNN model.
    
    Args:
        config: Dictionary containing the model configuration
        
    Returns:
        RNN PyTorch model
    """
    rnn_type = config['model_type'].lower()
    input_size = config.get('input_size', None)
    hidden_size = config.get('hidden_size', None)
    num_layers = config.get('num_layers', 1)
    bidirectional = config.get('bidirectional', False)
    dropout = config.get('dropout', 0.0)
    batch_first = config.get('batch_first', True)
    
    # If input_size and hidden_size are not provided, try to infer from layers
    if input_size is None or hidden_size is None:
        for layer in config['layers']:
            if layer['type'].lower() in ['rnn', 'lstm', 'gru']:
                input_size = layer.get('input_size', input_size)
                hidden_size = layer.get('hidden_size', hidden_size)
                break
    
    if input_size is None or hidden_size is None:
        raise ValueError("input_size and hidden_size must be provided for RNN models")
    
    layers = []
    for layer_config in config['layers']:
        layers.append(LayerFactory.create_layer(layer_config))
    
    return RNNModel(
        layers, rnn_type, input_size, hidden_size, num_layers,
        bidirectional, dropout, batch_first, config.get('name', 'RNNModel')
    )

def _build_cnn_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build a CNN model.
    
    Args:
        config: Dictionary containing the model configuration
        
    Returns:
        CNN PyTorch model
    """
    cnn_type = config['model_type'].lower()
    in_channels = config.get('in_channels', None)
    
    # If in_channels is not provided, try to infer from layers
    if in_channels is None:
        for layer in config['layers']:
            if layer['type'].lower() in ['conv1d', 'conv2d', 'conv3d']:
                in_channels = layer.get('in_channels', in_channels)
                break
    
    if in_channels is None:
        raise ValueError("in_channels must be provided for CNN models")
    
    layers = []
    for layer_config in config['layers']:
        layers.append(LayerFactory.create_layer(layer_config))
    
    return CNNModel(layers, cnn_type, in_channels, config.get('name', 'CNNModel'))

def _build_mlp_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build an MLP model.
    
    Args:
        config: Dictionary containing the model configuration
        
    Returns:
        MLP PyTorch model
    """
    input_size = config.get('input_size', None)
    hidden_sizes = config.get('hidden_sizes', [])
    output_size = config.get('output_size', None)
    activation = config.get('activation', 'relu')
    dropout = config.get('dropout', 0.0)
    
    # If input_size and output_size are not provided, try to infer from layers
    if input_size is None or output_size is None:
        for i, layer in enumerate(config['layers']):
            if layer['type'].lower() == 'linear':
                if i == 0:
                    input_size = layer.get('in_features', input_size)
                if i == len(config['layers']) - 1:
                    output_size = layer.get('out_features', output_size)
    
    if input_size is None or output_size is None:
        raise ValueError("input_size and output_size must be provided for MLP models")
    
    layers = []
    for layer_config in config['layers']:
        layers.append(LayerFactory.create_layer(layer_config))
    
    return MLPModel(
        layers, input_size, hidden_sizes, output_size,
        activation, dropout, config.get('name', 'MLPModel')
    )

def _build_attention_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build an attention model.
    
    Args:
        config: Dictionary containing the model configuration
        
    Returns:
        Attention PyTorch model
    """
    embed_dim = config.get('embed_dim', None)
    num_heads = config.get('num_heads', None)
    
    # If embed_dim and num_heads are not provided, try to infer from layers
    if embed_dim is None or num_heads is None:
        for layer in config['layers']:
            if layer['type'].lower() in ['multiheadattention', 'selfattention']:
                embed_dim = layer.get('embed_dim', embed_dim)
                num_heads = layer.get('num_heads', num_heads)
                break
    
    if embed_dim is None or num_heads is None:
        raise ValueError("embed_dim and num_heads must be provided for attention models")
    
    layers = []
    for layer_config in config['layers']:
        layers.append(LayerFactory.create_layer(layer_config))
    
    return AttentionModel(
        layers, embed_dim, num_heads, config.get('name', 'AttentionModel')
    )

def _build_autoencoder_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build an autoencoder model.
    
    Args:
        config: Dictionary containing the model configuration
        
    Returns:
        Autoencoder PyTorch model
    """
    model_type = config['model_type'].lower()
    input_size = config.get('input_size', None)
    latent_size = config.get('latent_size', None)
    
    # If input_size and latent_size are not provided, try to infer from layers
    if input_size is None or latent_size is None:
        encoder_layers = []
        decoder_layers = []
        
        # Split layers into encoder and decoder
        if 'encoder_layers' in config and 'decoder_layers' in config:
            encoder_layers = config['encoder_layers']
            decoder_layers = config['decoder_layers']
        else:
            # Assume the first half of layers are encoder and the second half are decoder
            middle = len(config['layers']) // 2
            encoder_layers = config['layers'][:middle]
            decoder_layers = config['layers'][middle:]
        
        # Try to infer input_size and latent_size
        for layer in encoder_layers:
            if layer['type'].lower() == 'linear':
                if input_size is None:
                    input_size = layer.get('in_features', input_size)
                latent_size = layer.get('out_features', latent_size)
    
    if input_size is None or latent_size is None:
        raise ValueError("input_size and latent_size must be provided for autoencoder models")
    
    # Create encoder and decoder layers
    encoder_layers = []
    decoder_layers = []
    
    if 'encoder_layers' in config and 'decoder_layers' in config:
        for layer_config in config['encoder_layers']:
            encoder_layers.append(LayerFactory.create_layer(layer_config))
        
        for layer_config in config['decoder_layers']:
            decoder_layers.append(LayerFactory.create_layer(layer_config))
    else:
        # Assume the first half of layers are encoder and the second half are decoder
        middle = len(config['layers']) // 2
        
        for layer_config in config['layers'][:middle]:
            encoder_layers.append(LayerFactory.create_layer(layer_config))
        
        for layer_config in config['layers'][middle:]:
            decoder_layers.append(LayerFactory.create_layer(layer_config))
    
    return AutoencoderModel(
        encoder_layers, decoder_layers, input_size, latent_size,
        model_type == 'vae', config.get('name', 'AutoencoderModel')
    )

def _build_gan_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build a GAN model.
    
    Args:
        config: Dictionary containing the model configuration
        
    Returns:
        GAN PyTorch model
    """
    latent_size = config.get('latent_size', None)
    
    # If latent_size is not provided, try to infer from layers
    if latent_size is None:
        for layer in config.get('generator_layers', []):
            if layer['type'].lower() == 'linear' and layer.get('is_first', False):
                latent_size = layer.get('in_features', latent_size)
                break
    
    if latent_size is None:
        raise ValueError("latent_size must be provided for GAN models")
    
    # Create generator and discriminator layers
    generator_layers = []
    discriminator_layers = []
    
    if 'generator_layers' in config and 'discriminator_layers' in config:
        for layer_config in config['generator_layers']:
            generator_layers.append(LayerFactory.create_layer(layer_config))
        
        for layer_config in config['discriminator_layers']:
            discriminator_layers.append(LayerFactory.create_layer(layer_config))
    else:
        raise ValueError("generator_layers and discriminator_layers must be provided for GAN models")
    
    return GANModel(
        generator_layers, discriminator_layers, latent_size,
        config.get('name', 'GANModel')
    )

def _build_siamese_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build a siamese model.
    
    Args:
        config: Dictionary containing the model configuration
        
    Returns:
        Siamese PyTorch model
    """
    # Create shared network layers
    shared_layers = []
    
    if 'shared_layers' in config:
        for layer_config in config['shared_layers']:
            shared_layers.append(LayerFactory.create_layer(layer_config))
    else:
        # Assume all layers are shared
        for layer_config in config['layers']:
            shared_layers.append(LayerFactory.create_layer(layer_config))
    
    # Create head layers if provided
    head_layers = []
    
    if 'head_layers' in config:
        for layer_config in config['head_layers']:
            head_layers.append(LayerFactory.create_layer(layer_config))
    
    return SiameseModel(
        shared_layers, head_layers, config.get('name', 'SiameseModel')
    )

def _build_gnn_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build a GNN model.
    
    Args:
        config: Dictionary containing the model configuration
        
    Returns:
        GNN PyTorch model
    """
    # This is a placeholder for GNN model implementation
    # In a real implementation, this would create a GNN model based on the configuration
    raise NotImplementedError("GNN model building is not implemented yet")

def _build_rbm_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build an RBM model.
    
    Args:
        config: Dictionary containing the model configuration
        
    Returns:
        RBM PyTorch model
    """
    # This is a placeholder for RBM model implementation
    # In a real implementation, this would create an RBM model based on the configuration
    raise NotImplementedError("RBM model building is not implemented yet")