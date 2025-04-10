# Tutorial

This tutorial will guide you through the process of creating and using yaml inputs with `neuralm` in order to create custom network topologies. This is also a good place to start for beginners as this document provides general insgiths on neural networks and their architectures. 

## Table of Contents

- [Installation](#installation)
- [Exemple 1: Creating a simple CNN for the MNIST dataset](#exemple-1-creating-a-simple-cnn-for-the-mnist-dataset)
- [Example 2: Creating a Variational Autoencoder (VAE)](#example-2-creating-a-variational-autoencoder-vae)
- [Example 3: Creating a Transformer](#example-3-creating-a-transformer)
- [Example 4: Creating a Custom Architecture](#example-4-creating-a-custom-architecture)
- [Extending neuralm](#extending-neuralm)
- [Citing neuralm](#citing-neuralm)


## Installation

First, install `neuralm`:

```bash
pip install neuralm
```

## Exemple 1: Creating a simple CNN for the MNIST dataset.
The first step is to create a YAML file to configure the topology, the activation functions, and specialized layers of your network. For instance, your `mnist_cnn.yaml` could look like this:

```yaml
model_type: cnn2d # Specifies a 2D convolutional neural network architecture for image processing
name: MNIST_CNN  # Name of the model for reference
in_channels: 1   # MNIST images are grayscale, so only 1 input channel

layers:
  # First convolutional block
  - type: conv2d    # 2D convolution layer to extract features from the input image
    in_channels: 1  # Match the input channels of the model
    out_channels: 32  # Number of filters/feature maps to learn
    kernel_size: 3    # 3x3 filter size
    stride: 1         # Move the filter 1 pixel at a time
    padding: 1        # Add 1 pixel padding to preserve spatial dimensions
  - type: relu      # Activation function to introduce non-linearity
    inplace: true   # Modify the input directly to save memory
  - type: maxpool2d # Reduce spatial dimensions by taking maximum value in each 2x2 region
    kernel_size: 2  # This halves the width and height (28x28 -> 14x14)

  # Second convolutional block
  - type: conv2d    # Second convolutional layer to extract more complex features
    in_channels: 32 # Must match the out_channels of previous conv layer
    out_channels: 64  # Increase the number of feature maps
    kernel_size: 3
    stride: 1
    padding: 1
  - type: relu
    inplace: true
  - type: maxpool2d # Further reduce dimensions (14x14 -> 7x7)
    kernel_size: 2

  # Transition to fully connected layers
  - type: flatten   # Convert 3D feature maps (64 channels of 7x7) to 1D vector

  # Fully connected layers
  - type: linear    # First fully connected layer
    in_features: 3136  # 64 channels * 7 * 7 spatial dimensions
    out_features: 128  # Reduce to 128 neurons
  - type: relu      # Apply non-linearity

  # Output layer
  - type: linear    # Final classification layer
    in_features: 128
    out_features: 10  # 10 output classes for digits 0-9
    # Note: No activation after final layer
    # During training: Use with CrossEntropyLoss which applies softmax internally
    # During inference: Apply softmax to get class probabilities
```

Then, create a python file (e.g.,`train_mnist.py`) to train your Pytorch model created with neuralm:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neuralm import build_model_from_yaml

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Build model from YAML
model = build_model_from_yaml('mnist_cnn.yaml')
model = model.to(device)
print(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    print(f'Epoch: {epoch+1}/{num_epochs}, Accuracy: {100 * correct / total:.2f}%')

print('Finished Training')

# Save the model
torch.save(model.state_dict(), 'mnist_cnn.pth')
```

## Example 2: Creating a Variational Autoencoder (VAE)

Let's create a VAE for the MNIST dataset. Create a file named `mnist_vae.yaml`:

```yaml
model_type: vae # variational autoencoder
name: MNIST_VAE
input_size: 784  # 28x28 flattened MNIST images
latent_size: 20  # dimension of the latent space representation
encoder_layers:
  - type: linear # no convolutional layers here since we are working with fully connected layers
    in_features: 784
    out_features: 400  # intermediate representation
  - type: relu  # activation function to introduce non-linearity
  - type: linear
    in_features: 400
    out_features: 20  # outputs mu and log_var for the latent space
decoder_layers:
  - type: linear
    in_features: 20  # takes samples from the latent distribution
    out_features: 400
  - type: relu
  - type: linear
    in_features: 400
    out_features: 784  # reconstructs the original image dimensions
  - type: sigmoid  # ensures output values are between 0 and 1, appropriate for image pixel values 
```

Then, create a Python file named `train_vae.py` for training. Maybe something like:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from neuralm import build_model_from_yaml

# VAE loss function
def vae_loss(recon_x, x, mu, log_var):
    # Reconstruction loss (binary cross entropy)
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return BCE + KLD

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Build model from YAML
model = build_model_from_yaml('mnist_vae.yaml')
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
        plt.figure(figsize=(8, 8))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(sample[i].view(28, 28).numpy(), cmap='gray')
            plt.axis('off')
        plt.savefig(f'vae_samples_epoch_{epoch+1}.png')
        plt.close()

print('Finished Training')

# Save the model
torch.save(model.state_dict(), 'mnist_vae.pth')
```

## Example 3: Creating a Transformer

Let's create a simple transformer model for Natural Language Processing (NLP). Create a file named `transformer_model.yaml`:

```yaml
model_type: attention
name: MyTransformer
embed_dim: 512
num_heads: 8
layers:
  # Embedding layer
  - type: embedding
    num_embeddings: 10000  # Vocabulary size
    embedding_dim: 512
  
  # Positional encoding (implemented as a custom layer)
  - type: custom
    name: PositionalEncoding
    max_seq_length: 512
    dropout: 0.1
  
  # Transformer encoder blocks (repeat as needed)
  # Block 1
  - type: layernorm
    normalized_shape: 512
  - type: multiheadattention
    embed_dim: 512
    num_heads: 8
    dropout: 0.1
    batch_first: true
  - type: dropout
    p: 0.1
  
  # Feed-forward network
  - type: layernorm
    normalized_shape: 512
  - type: linear
    in_features: 512
    out_features: 2048
  - type: relu
  - type: linear
    in_features: 2048
    out_features: 512
  - type: dropout
    p: 0.1
  
  # Block 2 (repeat the pattern for more blocks)
  - type: layernorm
    normalized_shape: 512
  - type: multiheadattention
    embed_dim: 512
    num_heads: 8
    dropout: 0.1
    batch_first: true
  - type: dropout
    p: 0.1
  
  # Feed-forward network
  - type: layernorm
    normalized_shape: 512
  - type: linear
    in_features: 512
    out_features: 2048
  - type: relu
  - type: linear
    in_features: 2048
    out_features: 512
  - type: dropout
    p: 0.1
  
  # Output projection
  - type: layernorm
    normalized_shape: 512
  - type: linear
    in_features: 512
    out_features: 10000  # Vocabulary size for language modeling
```

Now create a Python file named `train_transformer.py` for training:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from neuralm import build_model_from_yaml

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model from YAML
model = build_model_from_yaml('transformer_model.yaml')
model = model.to(device)

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset (example with MNIST)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Reshape data for transformer: [batch_size, seq_len, features]
        # For MNIST, we'll treat each row as a sequence element
        data = data.view(data.size(0), 28, 28).to(device)
        target = target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), 28, 28).to(device)
            target = target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'Epoch: {epoch+1}, Accuracy: {100 * correct / total:.2f}%')

print('Finished Training')
torch.save(model.state_dict(), 'transformer_mnist.pth')
```

## Example 4: Creating a Custom Architecture

You can also create custom models by combining different components. For instance:

```yaml
model_type: sequential  # Sequential model that processes layers in order
name: CustomModel       # Name identifier for the model
layers:
  # CNN part - for feature extraction from images
  - type: conv2d        # 2D convolution layer
    in_channels: 3      # RGB input (3 channels)
    out_channels: 16    # Number of filters/feature maps
    kernel_size: 3      # 3x3 convolution kernel
    padding: 1          # Padding to maintain spatial dimensions
  - type: relu          # ReLU activation function for non-linearity
  - type: maxpool2d     # Max pooling to reduce spatial dimensions
    kernel_size: 2      # 2x2 pooling window (reduces dimensions by half)

  # RNN part - for sequential processing of features
  - type: reshape       # Reshape tensor for RNN input
    shape: [-1, 16, 16 * 16]  # [batch_size, sequence_length, features_per_step]
  - type: lstm          # Long Short-Term Memory layer
    input_size: 16 * 16 # Number of features per time step
    hidden_size: 128    # Size of hidden state in LSTM
    batch_first: true   # Batch dimension is first in the input tensor

  # MLP part - for final classification
  - type: flatten       # Flatten the output for the fully connected layers
  - type: linear        # Fully connected layer
    in_features: 128 * 16  # Input size (hidden_size * sequence_length)
    out_features: 256   # Output size (intermediate representation)
  - type: relu          # ReLU activation
  - type: dropout       # Dropout for regularization
    p: 0.5              # 50% dropout probability to prevent overfitting
  - type: linear        # Final classification layer
    in_features: 256    # Input from previous layer
    out_features: 10    # 10 output classes (e.g., for MNIST or CIFAR-10)
```
Then train your custom model as before.


## Extending neuralm

You can extend neuralm by adding custom layers or models using neuralm's LayerFactory. Here's an example: 

```python
import torch.nn as nn
from neuralm.layers.layer_factory import LayerFactory

# Define a custom layer
class MyCustomLayer(nn.Module):
    def __init__(self, in_features, out_features, activation='relu'):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x):
        return self.activation(self.linear(x))

# Add the custom layer to the LayerFactory
def create_custom_layer(config):
    in_features = config['in_features']
    out_features = config['out_features']
    activation = config.get('activation', 'relu')
    return MyCustomLayer(in_features, out_features, activation)

# Monkey patch the LayerFactory
old_create_layer = LayerFactory.create_layer

def new_create_layer(layer_config):
    layer_type = layer_config['type'].lower()
    if layer_type == 'mycustomlayer':
        return create_custom_layer(layer_config)
    else:
        return old_create_layer(layer_config)

LayerFactory.create_layer = staticmethod(new_create_layer)
```
Now you can use your custom layer in YAML configurations:

```yaml
model_type: sequential
name: ModelWithCustomLayer
layers:
  - type: mycustomlayer
    in_features: 784
    out_features: 128
    activation: relu
  - type: mycustomlayer
    in_features: 128
    out_features: 10
    activation: sigmoid
```

## Citing neuralm

If you use this software in your work, please cite neuralm: 

```bibtex
@software{neuralm,
  author = {Sadoune, Igor},
  title = {NeuralM: A Neural Network Model Builder},
  year = {2023},
  url = {https://github.com/IgorSadoune/neuralm},
  version = {0.1.0},
  publisher = {GitHub},
  description = {A flexible framework for building and training neural network models with YAML configuration.}
}
```