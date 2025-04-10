"""
RNN Model Implementation

This module provides a flexible implementation of Recurrent Neural Networks (RNNs)
for sequence modeling tasks. It supports different RNN variants including vanilla RNN,
LSTM (Long Short-Term Memory), and GRU (Gated Recurrent Unit).

The implementation allows for customization of network architecture through
configurable parameters such as number of layers, hidden size, bidirectionality,
and dropout for regularization.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

class RNNModel(nn.Module):
    """
    A recurrent neural network model.
    """
    
    def __init__(
        self,
        layers: List[nn.Module],
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        batch_first: bool = True,
        name: Optional[str] = None
    ):
        """
        Initialize the RNN model.
        
        Args:
            layers: List of PyTorch layers
            rnn_type: Type of RNN ('rnn', 'lstm', 'gru')
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of recurrent layers
            bidirectional: Whether to use bidirectional RNN
            dropout: Dropout probability
            batch_first: Whether input/output has batch size as first dimension
            name: Name of the model
        """
        super().__init__()
        self.name = name or f"{rnn_type.upper()}Model"
        self.rnn_type = rnn_type.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.batch_first = batch_first
        
        # Create the RNN layer
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(
                input_size, hidden_size, num_layers,
                batch_first=batch_first, bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=batch_first, bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers,
                batch_first=batch_first, bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
        
        # Create the rest of the model
        self.model = nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            hidden: Initial hidden state
            
        Returns:
            Tuple of (output, hidden_state)
        """
        # Pass through RNN
        output, hidden = self.rnn(x, hidden)
        
        # Pass through the rest of the model
        output = self.model(output)
        
        return output, hidden
    
    def __str__(self) -> str:
        return f"{self.name}\n{super().__str__()}"