import yaml
import os
from typing import Dict, Any, Union

def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dictionary containing the YAML configuration
        
    Raises:
        FileNotFoundError: If the file does not exist
        yaml.YAMLError: If the YAML file is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found: {file_path}")
        
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate the configuration dictionary.
    
    Args:
        config: Dictionary containing the model configuration
        
    Returns:
        True if the configuration is valid, False otherwise
        
    Raises:
        ValueError: If the configuration is invalid
    """
    # Check if the config has the required keys
    required_keys = ['model_type', 'layers']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in configuration: {key}")
    
    # Check if the model type is supported
    supported_model_types = [
        'sequential', 'rnn', 'lstm', 'gru', 'cnn1d', 'cnn2d', 'cnn3d',
        'mlp', 'attention', 'gnn', 'autoencoder', 'vae', 'gan', 'siamese',
        'rbm'
    ]
    
    if config['model_type'] not in supported_model_types:
        raise ValueError(f"Unsupported model type: {config['model_type']}. "
                       f"Supported types are: {supported_model_types}")
    
    # Check if layers is a list
    if not isinstance(config['layers'], list):
        raise ValueError("'layers' must be a list")
    
    # Check each layer
    for i, layer in enumerate(config['layers']):
        if not isinstance(layer, dict):
            raise ValueError(f"Layer {i} must be a dictionary")
        
        if 'type' not in layer:
            raise ValueError(f"Layer {i} is missing 'type' key")
    
    return True