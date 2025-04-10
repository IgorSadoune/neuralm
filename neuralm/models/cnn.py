"""
This module defines a Convolutional Neural Network (CNN) model.

It provides a flexible CNN implementation that supports 1D, 2D, and 3D convolutions
through a unified interface. The model is constructed by passing a list of PyTorch layers
which are then wrapped in a sequential container.
"""

import torch
import torch.nn as nn
from typing import List, Optional

class CNNModel(nn.Module):
    """
    A convolutional neural network model.
    """
    
    def __init__(
        self,
        layers: List[nn.Module],
        cnn_type: str,
        in_channels: int,
        name: Optional[str] = None
    ):
        """
        Initialize the CNN model.
        
        Args:
            layers: List of PyTorch layers
            cnn_type: Type of CNN ('cnn1d', 'cnn2d', 'cnn3d')
            in_channels: Number of input channels
            name: Name of the model
        """
        super().__init__()
        self.name = name or f"{cnn_type.upper()}Model"
        self.cnn_type = cnn_type.lower()
        self.in_channels = in_channels
        
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