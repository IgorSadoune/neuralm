"""Variational Autoencoder (VAE) Example

This script demonstrates how to train a Variational Autoencoder on the MNIST dataset
using the neuralm library. It loads a VAE model configuration from a YAML file,
trains the model for 10 epochs, and generates sample images from the latent space
every 2 epochs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from neuralm import build_model_from_yaml

def vae_loss(recon_x, x, mu, log_var):
    """
    VAE loss function.
    """
    # Reconstruction loss (binary cross entropy)
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return BCE + KLD

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    # Build model from YAML
    model = build_model_from_yaml('configs/vae.yaml')
    model = model.to(device)
    print(model)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data.view(-1, 784))
            loss = vae_loss(recon_batch, data, mu, log_var)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()/len(data):.4f}')
        
        print(f'Epoch: {epoch+1}/{num_epochs}, Average Loss: {running_loss/len(train_loader.dataset):.4f}')
        
        # Generate some samples
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(64, 20).to(device)
            sample = model.decode(z).cpu()
            
            # Display samples
            if epoch % 2 == 0:
                plt.figure(figsize=(8, 8))
                for i in range(16):
                    plt.subplot(4, 4, i+1)
                    plt.imshow(sample[i].view(28, 28).numpy(), cmap='gray')
                    plt.axis('off')
                plt.savefig(f'vae_samples_epoch_{epoch+1}.png')
                plt.close()
    
    print('Finished Training')

if __name__ == '__main__':
    main()