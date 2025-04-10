"""
GAN (Generative Adversarial Network) Model Implementation

This module provides a flexible implementation of a Generative Adversarial Network (GAN).
It contains a GANModel class that encapsulates both the generator and discriminator networks,
allowing for easy training and generation of samples. The implementation supports customizable
network architectures through layer lists and provides utility methods for sampling and inference.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

class GANModel(nn.Module):
    """
    A generative adversarial network model.
    """
    
    def __init__(
        self,
        generator_layers: List[nn.Module],
        discriminator_layers: List[nn.Module],
        latent_size: int,
        name: Optional[str] = None
    ):
        """
        Initialize the GAN model.
        
        Args:
            generator_layers: List of generator layers
            discriminator_layers: List of discriminator layers
            latent_size: Size of latent space
            name: Name of the model
        """
        super().__init__()
        self.name = name or 'GANModel'
        self.latent_size = latent_size
        
        # Create the generator and discriminator
        self.generator = nn.Sequential(*generator_layers)
        self.discriminator = nn.Sequential(*discriminator_layers)
    
    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate samples from latent vectors.
        
        Args:
            z: Latent vectors
            
        Returns:
            Generated samples
        """
        return self.generator(z)
    
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Discriminate real from fake samples.
        
        Args:
            x: Input samples
            
        Returns:
            Discrimination scores
        """
        return self.discriminator(x)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from the generator.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to use
            
        Returns:
            Generated samples
        """
        z = torch.randn(num_samples, self.latent_size, device=device)
        return self.generate(z)
    
    def forward(self, x: Optional[torch.Tensor] = None, z: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Real samples (optional)
            z: Latent vectors (optional)
            
        Returns:
            Tuple of (discriminator_output, generator_output)
        """
        discriminator_output = None
        generator_output = None
        
        if x is not None:
            discriminator_output = self.discriminate(x)
        
        if z is not None:
            generator_output = self.generate(z)
            
            if x is not None:
                # Also discriminate the generated samples
                discriminator_output = self.discriminate(torch.cat([x, generator_output.detach()], dim=0))
        
        return discriminator_output, generator_output
    
    def __str__(self) -> str:
        return f"{self.name}\n{super().__str__()}"