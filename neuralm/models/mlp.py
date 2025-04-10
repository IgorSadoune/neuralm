"""
This module implements a Multi-Layer Perceptron (MLP) neural network model.

The MLPModel class provides a flexible implementation of an MLP that can be
configured with different layer sizes, activation functions, and dropout rates.
It serves as a building block for creating feed-forward neural networks for
various machine learning tasks such as classification and regression.
"""

import torch
import torch.nn as nn
from typing import List, Optional

class MLPModel(nn.Module):
    """
    A multi-layer perceptron model.
    """
    
    def __init__(
        self,
        layers: List[nn.Module],
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: str = 'relu',
        dropout: float = 0.0,
        name: Optional[str] = None
    ):
        """
        Initialize the MLP model.
        
        Args:
            layers: List of PyTorch layers
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Size of output features
            activation: Activation function to use
            dropout: Dropout probability
            name: Name of the model
        """
        super().__init__()
        self.name = name or 'MLPModel'
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.dropout = dropout
        
        # Create the model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.model(x)
    
    def __str__(self) -> str:
        return f"{self.name}\n{super().__str__()}"