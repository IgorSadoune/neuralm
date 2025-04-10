"""
This module implements a sequential neural network model.

It provides a wrapper around PyTorch's nn.Sequential that adds naming
and better string representation. The SequentialModel stacks layers
in order and executes them sequentially during the forward pass.
"""

import torch
import torch.nn as nn
from typing import List, Optional

class SequentialModel(nn.Module):
    """
    A sequential model that stacks layers in order.
    """
    
    def __init__(self, layers: List[nn.Module], name: Optional[str] = None):
        """
        Initialize the sequential model.
        
        Args:
            layers: List of PyTorch layers
            name: Name of the model
        """
        super().__init__()
        self.name = name or 'SequentialModel'
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