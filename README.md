# Neural Machines (neuralm)

`neuralm` is a Python/PyTorch package that generates PyTorch models from YAML configuration files. It focuses on neural architecture modeling, allowing users to create various types of PyTorch models without writing boilerplate code. The resulting Pytorch models can be translated into ONNX format for deployment or interoperability with other frameworks. 

## Table of Contents
- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Example YAML Configuration: Creating GANs](#example-yaml-configuration-creating-gans)
  - [Programmatic Model Building](#programmatic-model-building)
  - [Error Handling](#error-handling)
- [Supported Model Types](#supported-model-types)
- [Supported Layer Types](#supported-layer-types)
  - [Linear Layers](#linear-layers)
  - [Convolutional Layers](#convolutional-layers)
  - [Pooling Layers](#pooling-layers)
  - [Recurrent Layers](#recurrent-layers)
  - [Normalization Layers](#normalization-layers)
  - [Dropout Layers](#dropout-layers)
  - [Activation Layers](#activation-layers)
  - [Attention Layers](#attention-layers)
  - [Other Layers](#other-layers)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [License](#license)
- [Citing neuralm](#citing-neuralm)

## Installation

```bash
pip install neuralm
```

Or install from source:

```bash
git clone https://github.com/IgorSadoune/neuralm.git
cd neuralm
pip install -e .
```

## Features

- Generate PyTorch models from YAML configuration files
- Support for various model architectures (see below)
- Support custom model 
- Allow for building Transformers via attention type model 
- Flexible layer configuration
- Allow for saving and reusing custom layers via neuralm LayerFactory 
- Provide extensive documentation and use case to get started
- Provide extensive documentation for learning neural network model arhcitecture

## Usage

### Basic Usage

```python
from neuralm import build_model_from_yaml

# Build a model from a YAML file
model = build_model_from_yaml('model_config.yaml')

# Use the model like any PyTorch model
output = model(input_data)
```

### Example YAML Configuration: Creating GANs

```yaml
model_type: gan
name: MyGAN
latent_size: 100
generator_layers:
  - type: linear
    in_features: 100
    out_features: 256
    is_first: true
  - type: relu
  - type: linear
    in_features: 256
    out_features: 512
  - type: relu
  - type: linear
    in_features: 512
    out_features: 784
  - type: sigmoid
discriminator_layers:
  - type: linear
    in_features: 784
    out_features: 512
  - type: leakyrelu
    negative_slope: 0.2
  - type: linear
    in_features: 512
    out_features: 256
  - type: leakyrelu
    negative_slope: 0.2
  - type: linear
    in_features: 256
    out_features: 1
  - type: sigmoid
```

Note that `model_type` and `layers` are required fields.

### Programmatic Model Building

You can also build models programmatically using a configuration dictionary:

```python
from neuralpy import build_model_from_config

config = {
    'model_type': 'sequential',
    'layers': [
        {'type': 'linear', 'in_features': 784, 'out_features': 128},
        {'type': 'relu'},
        {'type': 'linear', 'in_features': 128, 'out_features': 10}
    ]
}

model = build_model_from_config(config)
```

### Error Handling

`neuralm` provides detailed error messages when there are issues with your configuration:

- Missing required keys
- Unsupported model types or layer types
- Invalid parameter values
- Inconsistent layer dimensions

Make sure to check the error messages if you encounter any issues.

## Supported Model Types

- `sequential`: The most general model type to build anything  
- `mlp`: The building block, Multi-layer perceptron are simple feedforward networks with fully connected layers that can process any type of data, but not as efficiently as specialized architectures 
- `rnn`, `lstm`, `gru`: Recurrent neural networks for processing sequential data like text or time series
- `cnn1d`, `cnn2d`, `cnn3d`: Convolutional neural networks for processing data with spatial patterns (1D signals, images, videos)
- `attention`: Attention-based models (e.g., Transformers) that focus on relevant parts of the input data
- `autoencoder`, `vae`: Models that learn to compress data and reconstruct it, useful for dimensionality reduction or data encoding
- `gan`: Generative adversarial networks that can create new data similar to training examples
- `siamese`: Networks with two identical subnetworks used to compare two inputs, great for similarity tasks
- `gnn`: Graph neural networks for processing data represented as graphs (coming soon)
- `rbm`: Restricted Boltzmann machines, generative stochastic networks for unsupervised learning (coming soon)

## Supported Layer Types

### Linear Layers
- `linear`: Fully connected layer

```yaml
- type: linear
  in_features: 784
  out_features: 128
  bias: true  # Optional, default is true
```

### Convolutional Layers
- `conv1d`, `conv2d`, `conv3d`: Convolutional layers

Example: 

```yaml
- type: conv2d
  in_channels: 3
  out_channels: 16
  kernel_size: 3
  stride: 1  # Optional, default is 1
  padding: 1  # Optional, default is 0
  dilation: 1  # Optional, default is 1
  groups: 1  # Optional, default is 1
  bias: true  # Optional, default is true
  padding_mode: zeros  # Optional, default is 'zeros'
```

### Pooling Layers
- `maxpool1d`, `maxpool2d`, `maxpool3d`: Max pooling layers
- `avgpool1d`, `avgpool2d`, `avgpool3d`: Average pooling layers

Example: 

```yaml
- type: maxpool2d
  kernel_size: 2
  stride: 2  # Optional, default is kernel_size
  padding: 0  # Optional, default is 0
  dilation: 1  # Optional, default is 1
  ceil_mode: false  # Optional, default is false
```

### Recurrent Layers
- `rnn`: Simple RNN layer
- `lstm`: LSTM layer
- `gru`: GRU layer

Example: 

```yaml
- type: lstm
  input_size: 300
  hidden_size: 256
  num_layers: 2  # Optional, default is 1
  bias: true  # Optional, default is true
  batch_first: true  # Optional, default is false
  dropout: 0.2  # Optional, default is 0
  bidirectional: true  # Optional, default is false
```

### Normalization Layers
- `batchnorm1d`, `batchnorm2d`, `batchnorm3d`: Batch normalization layers
- `layernorm`: Layer normalization
- `instancenorm1d`, `instancenorm2d`, `instancenorm3d`: Instance normalization layers

Example: 

```yaml
- type: batchnorm2d
  num_features: 16
  eps: 1e-5  # Optional, default is 1e-5
  momentum: 0.1  # Optional, default is 0.1
  affine: true  # Optional, default is true
  track_running_stats: true  # Optional, default is true
```

### Dropout Layers
- `dropout`, `dropout2d`, `dropout3d`: Dropout layers

Example: 

```yaml
- type: dropout
  p: 0.5  # Optional, default is 0.5
  inplace: false  # Optional, default is false
```

### Activation Layers
- `relu`, `leakyrelu`, `prelu`, `elu`, `selu`, `celu`, `gelu`: ReLU and variants
- `sigmoid`: Sigmoid activation
- `tanh`: Tanh activation
- `softmax`: Softmax activation
- `logsoftmax`: LogSoftmax activation

Example: 

```yaml
- type: relu
  inplace: false  # Optional, default is false
```

### Attention Layers
- `multiheadattention`: Multi-head attention layer
- `selfattention`: Self-attention layer

Example: 

```yaml
- type: multiheadattention
  embed_dim: 512
  num_heads: 8
  dropout: 0.1  # Optional, default is 0
  bias: true  # Optional, default is true
  batch_first: true  # Optional, default is false
```

### Other Layers
- `embedding`: Embedding layer
- `flatten`: Flatten layer
- `reshape`: Reshape layer

## Advanced Usage
For advanced user case please refer to the [tutorial](TUTORIAL.md). 

## Contributing

Contributions are welcome! If you wish to contribute to the project read the [the contributing guidelines](CONTRIBUTING.md) and contact me ([igor.sadoune@pm.me](mailto:igor.sadoune@pm.me)).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

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