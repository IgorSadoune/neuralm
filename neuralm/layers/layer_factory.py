"""
Layer Factory Module

This module provides a factory class for creating PyTorch neural network layers
based on configuration dictionaries. It simplifies the process of dynamically
constructing neural networks by providing a consistent interface for instantiating
various types of layers including:

- Linear layers
- Convolutional layers
- Pooling layers
- Recurrent layers (RNN, LSTM, GRU)
- Normalization layers
- Dropout layers
- Activation functions
- Attention mechanisms
- Embedding layers
- Reshape/flatten operations

Each layer can be created by providing a configuration dictionary with the appropriate
parameters for the desired layer type.
"""

import torch.nn as nn
from typing import Dict, Any, Union

class LayerFactory:
    """
    Factory class for creating PyTorch layers based on configuration.
    """
    
    @staticmethod
    def create_layer(layer_config: Dict[str, Any]) -> nn.Module:
        """
        Create a PyTorch layer based on the configuration.
        
        Args:
            layer_config: Dictionary containing the layer configuration
            
        Returns:
            PyTorch layer
            
        Raises:
            ValueError: If the layer type is not supported
        """
        layer_type = layer_config['type'].lower()
        
        # Linear layers
        if layer_type == 'linear':
            return LayerFactory._create_linear_layer(layer_config)
        
        # Convolutional layers
        elif layer_type in ['conv1d', 'conv2d', 'conv3d']:
            return LayerFactory._create_conv_layer(layer_config)
        
        # Pooling layers
        elif layer_type in ['maxpool1d', 'maxpool2d', 'maxpool3d',
                           'avgpool1d', 'avgpool2d', 'avgpool3d']:
            return LayerFactory._create_pool_layer(layer_config)
        
        # Recurrent layers
        elif layer_type in ['rnn', 'lstm', 'gru']:
            return LayerFactory._create_recurrent_layer(layer_config)
        
        # Normalization layers
        elif layer_type in ['batchnorm1d', 'batchnorm2d', 'batchnorm3d',
                           'layernorm', 'instancenorm1d', 'instancenorm2d', 'instancenorm3d']:
            return LayerFactory._create_norm_layer(layer_config)
        
        # Dropout layers
        elif layer_type in ['dropout', 'dropout2d', 'dropout3d']:
            return LayerFactory._create_dropout_layer(layer_config)
        
        # Activation layers
        elif layer_type in ['relu', 'leakyrelu', 'prelu', 'elu', 'selu', 'celu',
                           'gelu', 'sigmoid', 'tanh', 'softmax', 'logsoftmax']:
            return LayerFactory._create_activation_layer(layer_config)
        
        # Attention layers
        elif layer_type in ['multiheadattention', 'selfattention']:
            return LayerFactory._create_attention_layer(layer_config)
        
        # Embedding layers
        elif layer_type == 'embedding':
            return LayerFactory._create_embedding_layer(layer_config)
        
        # Flatten layer
        elif layer_type == 'flatten':
            return nn.Flatten(**LayerFactory._get_flatten_params(layer_config))
        
        # Reshape layer
        elif layer_type == 'reshape':
            return LayerFactory._create_reshape_layer(layer_config)
        
        # Custom layer
        elif layer_type == 'custom':
            return LayerFactory._create_custom_layer(layer_config)
        
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
    
    @staticmethod
    def _create_linear_layer(config: Dict[str, Any]) -> nn.Linear:
        """
        Create a linear layer.
        
        Args:
            config: Dictionary containing the layer configuration
            
        Returns:
            PyTorch linear layer
        """
        in_features = config['in_features']
        out_features = config['out_features']
        bias = config.get('bias', True)
        
        return nn.Linear(in_features, out_features, bias=bias)
    
    @staticmethod
    def _create_conv_layer(config: Dict[str, Any]) -> Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]:
        """
        Create a convolutional layer.
        
        Args:
            config: Dictionary containing the layer configuration
            
        Returns:
            PyTorch convolutional layer
        """
        layer_type = config['type'].lower()
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        kernel_size = config['kernel_size']
        stride = config.get('stride', 1)
        padding = config.get('padding', 0)
        dilation = config.get('dilation', 1)
        groups = config.get('groups', 1)
        bias = config.get('bias', True)
        padding_mode = config.get('padding_mode', 'zeros')
        
        if layer_type == 'conv1d':
            return nn.Conv1d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups,
                bias=bias, padding_mode=padding_mode
            )
        elif layer_type == 'conv2d':
            return nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups,
                bias=bias, padding_mode=padding_mode
            )
        elif layer_type == 'conv3d':
            return nn.Conv3d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups,
                bias=bias, padding_mode=padding_mode
            )
    
    @staticmethod
    def _create_pool_layer(config: Dict[str, Any]) -> nn.Module:
        """
        Create a pooling layer.
        
        Args:
            config: Dictionary containing the layer configuration
            
        Returns:
            PyTorch pooling layer
        """
        layer_type = config['type'].lower()
        kernel_size = config['kernel_size']
        stride = config.get('stride', None)
        padding = config.get('padding', 0)
        dilation = config.get('dilation', 1)
        ceil_mode = config.get('ceil_mode', False)
        
        if stride is None:
            stride = kernel_size
        
        if layer_type == 'maxpool1d':
            return nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding,
                dilation=dilation, ceil_mode=ceil_mode
            )
        elif layer_type == 'maxpool2d':
            return nn.MaxPool2d(
                kernel_size, stride=stride, padding=padding,
                dilation=dilation, ceil_mode=ceil_mode
            )
        elif layer_type == 'maxpool3d':
            return nn.MaxPool3d(
                kernel_size, stride=stride, padding=padding,
                dilation=dilation, ceil_mode=ceil_mode
            )
        elif layer_type == 'avgpool1d':
            return nn.AvgPool1d(
                kernel_size, stride=stride, padding=padding,
                ceil_mode=ceil_mode
            )
        elif layer_type == 'avgpool2d':
            return nn.AvgPool2d(
                kernel_size, stride=stride, padding=padding,
                ceil_mode=ceil_mode
            )
        elif layer_type == 'avgpool3d':
            return nn.AvgPool3d(
                kernel_size, stride=stride, padding=padding,
                ceil_mode=ceil_mode
            )
    
    @staticmethod
    def _create_recurrent_layer(config: Dict[str, Any]) -> Union[nn.RNN, nn.LSTM, nn.GRU]:
        """
        Create a recurrent layer.
        
        Args:
            config: Dictionary containing the layer configuration
            
        Returns:
            PyTorch recurrent layer
        """
        layer_type = config['type'].lower()
        input_size = config['input_size']
        hidden_size = config['hidden_size']
        num_layers = config.get('num_layers', 1)
        bias = config.get('bias', True)
        batch_first = config.get('batch_first', False)
        dropout = config.get('dropout', 0)
        bidirectional = config.get('bidirectional', False)
        
        if layer_type == 'rnn':
            nonlinearity = config.get('nonlinearity', 'tanh')
            return nn.RNN(
                input_size, hidden_size, num_layers=num_layers,
                nonlinearity=nonlinearity, bias=bias, batch_first=batch_first,
                dropout=dropout, bidirectional=bidirectional
            )
        elif layer_type == 'lstm':
            return nn.LSTM(
                input_size, hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout,
                bidirectional=bidirectional
            )
        elif layer_type == 'gru':
            return nn.GRU(
                input_size, hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout,
                bidirectional=bidirectional
            )
    
    @staticmethod
    def _create_norm_layer(config: Dict[str, Any]) -> nn.Module:
        """
        Create a normalization layer.
        
        Args:
            config: Dictionary containing the layer configuration
            
        Returns:
            PyTorch normalization layer
        """
        layer_type = config['type'].lower()
        
        if layer_type in ['batchnorm1d', 'batchnorm2d', 'batchnorm3d']:
            num_features = config['num_features']
            eps = config.get('eps', 1e-5)
            momentum = config.get('momentum', 0.1)
            affine = config.get('affine', True)
            track_running_stats = config.get('track_running_stats', True)
            
            if layer_type == 'batchnorm1d':
                return nn.BatchNorm1d(
                    num_features, eps=eps, momentum=momentum,
                    affine=affine, track_running_stats=track_running_stats
                )
            elif layer_type == 'batchnorm2d':
                return nn.BatchNorm2d(
                    num_features, eps=eps, momentum=momentum,
                    affine=affine, track_running_stats=track_running_stats
                )
            elif layer_type == 'batchnorm3d':
                return nn.BatchNorm3d(
                    num_features, eps=eps, momentum=momentum,
                    affine=affine, track_running_stats=track_running_stats
                )
        
        elif layer_type == 'layernorm':
            normalized_shape = config['normalized_shape']
            eps = config.get('eps', 1e-5)
            elementwise_affine = config.get('elementwise_affine', True)
            
            return nn.LayerNorm(
                normalized_shape, eps=eps, elementwise_affine=elementwise_affine
            )
        
        elif layer_type in ['instancenorm1d', 'instancenorm2d', 'instancenorm3d']:
            num_features = config['num_features']
            eps = config.get('eps', 1e-5)
            momentum = config.get('momentum', 0.1)
            affine = config.get('affine', False)
            track_running_stats = config.get('track_running_stats', False)
            
            if layer_type == 'instancenorm1d':
                return nn.InstanceNorm1d(
                    num_features, eps=eps, momentum=momentum,
                    affine=affine, track_running_stats=track_running_stats
                )
            elif layer_type == 'instancenorm2d':
                return nn.InstanceNorm2d(
                    num_features, eps=eps, momentum=momentum,
                    affine=affine, track_running_stats=track_running_stats
                )
            elif layer_type == 'instancenorm3d':
                return nn.InstanceNorm3d(
                    num_features, eps=eps, momentum=momentum,
                    affine=affine, track_running_stats=track_running_stats
                )
    
    @staticmethod
    def _create_dropout_layer(config: Dict[str, Any]) -> Union[nn.Dropout, nn.Dropout2d, nn.Dropout3d]:
        """
        Create a dropout layer.
        
        Args:
            config: Dictionary containing the layer configuration
            
        Returns:
            PyTorch dropout layer
        """
        layer_type = config['type'].lower()
        p = config.get('p', 0.5)
        inplace = config.get('inplace', False)
        
        if layer_type == 'dropout':
            return nn.Dropout(p=p, inplace=inplace)
        elif layer_type == 'dropout2d':
            return nn.Dropout2d(p=p, inplace=inplace)
        elif layer_type == 'dropout3d':
            return nn.Dropout3d(p=p, inplace=inplace)
    
    @staticmethod
    def _create_activation_layer(config: Dict[str, Any]) -> nn.Module:
        """
        Create an activation layer.
        
        Args:
            config: Dictionary containing the layer configuration
            
        Returns:
            PyTorch activation layer
        """
        layer_type = config['type'].lower()
        inplace = config.get('inplace', False)
        
        if layer_type == 'relu':
            return nn.ReLU(inplace=inplace)
        elif layer_type == 'leakyrelu':
            negative_slope = config.get('negative_slope', 0.01)
            return nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
        elif layer_type == 'prelu':
            num_parameters = config.get('num_parameters', 1)
            init = config.get('init', 0.25)
            return nn.PReLU(num_parameters=num_parameters, init=init)
        elif layer_type == 'elu':
            alpha = config.get('alpha', 1.0)
            return nn.ELU(alpha=alpha, inplace=inplace)
        elif layer_type == 'selu':
            return nn.SELU(inplace=inplace)
        elif layer_type == 'celu':
            alpha = config.get('alpha', 1.0)
            return nn.CELU(alpha=alpha, inplace=inplace)
        elif layer_type == 'gelu':
            return nn.GELU()
        elif layer_type == 'sigmoid':
            return nn.Sigmoid()
        elif layer_type == 'tanh':
            return nn.Tanh()
        elif layer_type == 'softmax':
            dim = config.get('dim', -1)
            return nn.Softmax(dim=dim)
        elif layer_type == 'logsoftmax':
            dim = config.get('dim', -1)
            return nn.LogSoftmax(dim=dim)
    
    @staticmethod
    def _create_attention_layer(config: Dict[str, Any]) -> nn.Module:
        """
        Create an attention layer.
        
        Args:
            config: Dictionary containing the layer configuration
            
        Returns:
            PyTorch attention layer
        """
        layer_type = config['type'].lower()
        
        if layer_type == 'multiheadattention':
            embed_dim = config['embed_dim']
            num_heads = config['num_heads']
            dropout = config.get('dropout', 0.0)
            bias = config.get('bias', True)
            add_bias_kv = config.get('add_bias_kv', False)
            add_zero_attn = config.get('add_zero_attn', False)
            kdim = config.get('kdim', None)
            vdim = config.get('vdim', None)
            batch_first = config.get('batch_first', False)
            
            return nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, bias=bias,
                add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
                kdim=kdim, vdim=vdim, batch_first=batch_first
            )
        elif layer_type == 'selfattention':
            # Implement a custom self-attention layer
            class SelfAttention(nn.Module):
                def __init__(self, embed_dim, num_heads=1, dropout=0.0):
                    super().__init__()
                    self.attention = nn.MultiheadAttention(
                        embed_dim, num_heads, dropout=dropout, batch_first=True
                    )
                
                def forward(self, x):
                    return self.attention(x, x, x)[0]
            
            embed_dim = config['embed_dim']
            num_heads = config.get('num_heads', 1)
            dropout = config.get('dropout', 0.0)
            
            return SelfAttention(embed_dim, num_heads, dropout)
    
    @staticmethod
    def _create_embedding_layer(config: Dict[str, Any]) -> nn.Embedding:
        """
        Create an embedding layer.
        
        Args:
            config: Dictionary containing the layer configuration
            
        Returns:
            PyTorch embedding layer
        """
        num_embeddings = config['num_embeddings']
        embedding_dim = config['embedding_dim']
        padding_idx = config.get('padding_idx', None)
        max_norm = config.get('max_norm', None)
        norm_type = config.get('norm_type', 2.0)
        scale_grad_by_freq = config.get('scale_grad_by_freq', False)
        sparse = config.get('sparse', False)
        
        return nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx,
            max_norm=max_norm, norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq, sparse=sparse
        )
    
    @staticmethod
    def _get_flatten_params(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get parameters for a flatten layer.
        
        Args:
            config: Dictionary containing the layer configuration
            
        Returns:
            Dictionary of parameters for nn.Flatten
        """
        params = {}
        if 'start_dim' in config:
            params['start_dim'] = config['start_dim']
        if 'end_dim' in config:
            params['end_dim'] = config['end_dim']
        return params
    
    @staticmethod
    def _create_reshape_layer(config: Dict[str, Any]) -> nn.Module:
        """
        Create a reshape layer.
        
        Args:
            config: Dictionary containing the layer configuration
            
        Returns:
            PyTorch reshape layer
        """
        shape = config['shape']
        
        class Reshape(nn.Module):
            def __init__(self, shape):
                super().__init__()
                self.shape = shape
            
            def forward(self, x):
                return x.view(self.shape)
        
        return Reshape(shape)
    
    @staticmethod
    def _create_custom_layer(config: Dict[str, Any]) -> nn.Module:
        """
        Create a custom layer.
        
        Args:
            config: Dictionary containing the layer configuration
            
        Returns:
            PyTorch custom layer
        """
        # This is a placeholder for custom layer implementation
        # In a real implementation, this would dynamically import and instantiate
        # a custom layer class based on the configuration
        raise NotImplementedError("Custom layer creation is not implemented yet")