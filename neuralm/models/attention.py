"""
This module implements an attention-based neural network model.

The attention mechanism allows the model to focus on different parts of the input
when producing the output. This is particularly useful for sequence-to-sequence tasks
where the relevance of input elements varies for different output elements.

The implementation provides a flexible AttentionModel class that can be configured
with different layers, embedding dimensions, and number of attention heads.
"""

import torch
import torch.nn as nn
from typing import List, Optional

class AttentionModel(nn.Module):
    """
    An attention-based model.
    """
    
    def __init__(
        self,
        layers: List[nn.Module],
        embed_dim: int,
        num_heads: int,
        name: Optional[str] = None
    ):
        """
        Initialize the attention model.
        
        Args:
            layers: List of PyTorch layers
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            name: Name of the model
        """
        super().__init__()
        self.name = name or 'AttentionModel'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
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