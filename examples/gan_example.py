"""Generative Adversarial Network (GAN) Example

This script demonstrates how to train a GAN on the MNIST dataset using the neuralm library.
It loads a GAN model configuration from a YAML file, trains the generator and discriminator
for 20 epochs, and generates sample images from random noise after each epoch. The script
also plots the loss curves for both the generator and discriminator.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from neuralm import build_model_from_yaml

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Build model from YAML
    model = build_model_from_yaml('configs/gan.yaml')
    model = model.to(device)
    print(model)
    
    # Define loss function and optimizers
    criterion = nn.BCELoss()
    
    # Optimizers
    optimizer_G = optim.Adam(model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Train the model
    num_epochs = 20
    latent_size = model.latent_size
    
    # Create fixed noise for visualization
    fixed_noise = torch.randn(64, latent_size, device=device)
    
    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    
    for epoch in range(num_epochs):
        for batch_idx, (real_data, _) in enumerate(train_loader):
            batch_size = real_data.size(0)
            real_data = real_data.view(batch_size, -1).to(device)
            
            # Labels
            real_label = torch.ones(batch_size, 1, device=device)
            fake_label = torch.zeros(batch_size, 1, device=device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Real data
            output_real = model.discriminate(real_data)
            loss_real = criterion(output_real, real_label)
            
            # Fake data
            noise = torch.randn(batch_size, latent_size, device=device)
            fake_data = model.generate(noise)
            output_fake = model.discriminate(fake_data.detach())
            loss_fake = criterion(output_fake, fake_label)
            
            # Combined loss
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            
            # Generate fake data again
            output_fake = model.discriminate(fake_data)
            loss_G = criterion(output_fake, real_label)
            
            loss_G.backward()
            optimizer_G.step()
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}')
            
            # Save losses for plotting
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())
        
        # Generate samples with fixed noise
        with torch.no_grad():
            fake_samples = model.generate(fixed_noise).cpu()
            
            # Display samples
            plt.figure(figsize=(8, 8))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(fake_samples[i].view(28, 28).numpy(), cmap='gray')
                plt.axis('off')
            plt.savefig(f'gan_samples_epoch_{epoch+1}.png')
            plt.close()
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('gan_losses.png')
    plt.close()
    
    print('Finished Training')

if __name__ == '__main__':
    main()