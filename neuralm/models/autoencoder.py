"""
This module implements an Autoencoder neural network architecture.

The module provides a flexible implementation of both standard autoencoders and
variational autoencoders (VAEs). Autoencoders are neural networks that learn to
compress data into a lower-dimensional latent space and then reconstruct the
original data from this compressed representation.

The implementation allows for customizable encoder and decoder architectures
through layer lists, and handles the specific requirements of variational
autoencoders including the reparameterization trick.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

class AutoencoderModel(nn.Module):
    """
    An autoencoder model.
    """
    
    def __init__(
        self,
        encoder_layers: List[nn.Module],
        decoder_layers: List[nn.Module],
        input_size: int,
        latent_size: int,
        is_variational: bool = False,
        name: Optional[str] = None
    ):
        """
        Initialize the autoencoder model.
        
        Args:
            encoder_layers: List of encoder layers
            decoder_layers: List of decoder layers
            input_size: Size of input features
            latent_size: Size of latent space
            is_variational: Whether this is a variational autoencoder
            name: Name of the model
        """
        super().__init__()
        self.name = name or ('VAE' if is_variational else 'Autoencoder')
        self.input_size = input_size
        self.latent_size = latent_size
        self.is_variational = is_variational
        
        # Create the encoder and decoder
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        
        # For variational autoencoder
        if is_variational:
            self.fc_mu = nn.Linear(latent_size, latent_size)
            self.fc_var = nn.Linear(latent_size, latent_size)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent space.
        
        Args:
            z: Latent tensor
            
        Returns:
            Reconstructed tensor
        """
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean tensor
            log_var: Log variance tensor
            
        Returns:
            Sampled latent tensor
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstructed_x, mu, log_var) for VAE or (reconstructed_x, None, None) for standard autoencoder
        """
        if self.is_variational:
            # Encode
            encoded = self.encode(x)
            
            # Get mu and log_var
            mu = self.fc_mu(encoded)
            log_var = self.fc_var(encoded)
            
            # Sample from latent space
            z = self.reparameterize(mu, log_var)
            
            # Decode
            reconstructed = self.decode(z)
            
            return reconstructed, mu, log_var
        else:
            # Standard autoencoder
            encoded = self.encode(x)
            reconstructed = self.decode(encoded)
            
            return reconstructed, None, None
    
    def __str__(self) -> str:
        return f"{self.name}\n{super().__str__()}"