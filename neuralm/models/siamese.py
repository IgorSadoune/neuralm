"""
This module implements a Siamese Neural Network architecture.

Siamese networks are a class of neural network architectures that contain two or more
identical subnetworks. These subnetworks share the same parameters and weights,
making them particularly useful for tasks like similarity learning, one-shot learning,
and verification tasks.

The implementation consists of:
- A shared network that processes both inputs identically
- Optional head layers for further processing
- Distance calculation between the outputs of the two branches
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

class SiameseModel(nn.Module):
    """
    A siamese network model.
    """
    
    def __init__(
        self,
        shared_layers: List[nn.Module],
        head_layers: List[nn.Module],
        name: Optional[str] = None
    ):
        """
        Initialize the siamese model.
        
        Args:
            shared_layers: List of shared network layers
            head_layers: List of head layers
            name: Name of the model
        """
        super().__init__()
        self.name = name or 'SiameseModel'
        
        # Create the shared network and head
        self.shared_network = nn.Sequential(*shared_layers)
        
        if head_layers:
            self.head = nn.Sequential(*head_layers)
        else:
            self.head = None
    
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through one branch of the siamese network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.shared_network(x)
        
        if self.head is not None:
            x = self.head(x)
            
        return x
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the siamese network.
        
        Args:
            x1: First input tensor
            x2: Second input tensor
            
        Returns:
            Tuple of (output1, output2, distance)
        """
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)
        
        # Compute Euclidean distance
        distance = torch.nn.functional.pairwise_distance(output1, output2)
        
        return output1, output2, distance
    
    def __str__(self) -> str:
        return f"{self.name}\n{super().__str__()}"