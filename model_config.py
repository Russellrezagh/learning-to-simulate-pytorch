import torch.nn as nn
import functools

class ModelConfig:
    def __init__(self, latent_size, num_layers, output_size):
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.output_size = output_size

def build_mlp(model_config, layer_norm=False, name=None):
    """Builds an MLP with layer normalization."""
    layers = []
    for _ in range(model_config.num_layers):
        layers.append(nn.Linear(model_config.latent_size, model_config.latent_size))
        if layer_norm:
            layers.append(nn.LayerNorm(model_config.latent_size))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(model_config.latent_size, model_config.output_size))
    return nn.Sequential(*layers)

def get_model_config(model_config_name):
    """Gets the model config specified by the name."""
    # Hyperparameters for models.
    if model_config_name == 'keypoint_detector':
        return ModelConfig(latent_size=32, num_layers=3, output_size=28)
    elif model_config_name in ['cloth_model', 'flag_model', 'water_model', 'gns_model']:
        return ModelConfig(latent_size=128, num_layers=2, output_size=3)
    else:
        raise ValueError('Unknown model config name {}'.format(model_config_name))

def get_output_size(model_config_name, num_dimensions):
    model_config = get_model_config(model_config_name)
    if model_config_name == 'keypoint_detector':
        return model_config.output_size
    else:
        return num_dimensions

def get_edge_size(num_particle_types, particle_type_embedding_size, num_dimensions):
    # + 3 for one-hot encoding of from/to type, and dist.
    return (num_particle_types * particle_type_embedding_size) * 2 + num_dimensions + 3

def get_node_size(num_particle_types, particle_type_embedding_size, num_dimensions):
    # + 2 for x,y-velocity, and + num_particle_types for one-hot particle type.
    return num_particle_types * particle_type_embedding_size + 2 + num_dimensions

def get_global_size(num_particle_types, particle_type_embedding_size, num_dimensions):
    del num_particle_types, particle_type_embedding_size, num_dimensions
    # The global feature is a placeholder for shape stats in this demo.
    return 3