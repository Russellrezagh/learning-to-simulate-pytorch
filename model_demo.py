import argparse
import os
import pickle

import numpy as np
import torch

import graph_network
import learned_simulator
import model_config
import normalization
import reading_utils

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_DIMENSIONS = 3
NUM_PARTICLE_TYPES = 9 # Changed from 6 to 9
BATCH_SIZE = 5
GLOBAL_CONTEXT_SIZE = 3 # Changed from 6 to 3

def get_random_walk_noise_for_position_sequence(
        position_sequence_noise, noise_std_last_step):
    """Returns random-walk noise in the velocity applied to the position."""
    velocity_sequence_noise = torch.diff(position_sequence_noise, dim=1)
    num_velocities = velocity_sequence_noise.shape[1]
    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)
    velocity_sequence_noise /= torch.linspace(1, num_velocities, num_velocities, device=velocity_sequence_noise.device)[
        None, :, None
    ]
    velocity_sequence_noise *= noise_std_last_step
    position_sequence_noise = torch.cat(
        [
            torch.zeros_like(velocity_sequence_noise[:, 0:1]),
            torch.cumsum(velocity_sequence_noise, dim=1),
        ],
        dim=1,
    )
    return position_sequence_noise

def sample_random_position_sequence(
        boundaries, num_particles_min, num_particles_max,
        num_dimensions, sequence_length, particle_types, device):
    """Samples a random position sequence for the given number of particles."""
    num_particles = torch.randint(num_particles_min, num_particles_max + 1, (1,), device=device).item()
    # Sample initial velocities for particles as random-walk-noise.
    position_sequence = torch.rand(
        (1, num_particles, num_dimensions), dtype=torch.float32, device=device)
    velocity_sequence_noise = get_random_walk_noise_for_position_sequence(
        torch.zeros((1, sequence_length, num_dimensions), device=device, dtype=torch.float32),
        noise_std_last_step=0.1
    )
    sampled_velocity_sequence = torch.zeros((num_particles, sequence_length, num_dimensions), dtype=torch.float32, device=device)
    for i in range(num_particles):
        particle_type = particle_types[i]
        sampled_velocity_sequence[i] = velocity_sequence_noise * 0.1
    position_sequence = torch.cumsum(sampled_velocity_sequence, dim=1)
    # Sample types for particles as integers.
    position_sequence += torch.rand((1, 1, num_dimensions), dtype=torch.float32, device=device)
    return position_sequence, num_particles

def get_simulator(config, metadata, noise_std, device):
    """Instantiates the simulator."""
    # Cast statistics to numpy so they are arrays when entering the model.
    cast = lambda v: np.array(v, dtype=np.float32)
    acceleration_stats = normalization.Stats(
        cast(metadata['acc_mean']),
        cast(np.sqrt(metadata['acc_std']**2 + noise_std**2)))
    velocity_stats = normalization.Stats(
        cast(metadata['vel_mean']),
        cast(np.sqrt(metadata['vel_std']**2 + noise_std**2)))
    normalization_stats = {'acceleration': acceleration_stats,
                            'velocity': velocity_stats}
    if 'context_mean' in metadata:
        context_stats = normalization.Stats(
            cast(metadata['context_mean']),
            cast(metadata['context_std']))
        normalization_stats['context'] = context_stats

    model_config = graph_network.get_model_config('gns_model')
    model_config = model_config._replace(output_size=3) # No other option in the original code?
    graph_network_kwargs = dict(model_config._asdict())

    simulator = learned_simulator.LearnedSimulator(
        num_dimensions=NUM_DIMENSIONS,
        connectivity_radius=metadata['default_connectivity_radius'],
        graph_network_kwargs=graph_network_kwargs,
        boundaries=metadata['bounds'],
        normalization_stats=normalization_stats,
        num_particle_types=NUM_PARTICLE_TYPES,
        particle_type_embedding_size=16, #TODO: add to config file
    )
    simulator.to(device)
    return simulator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Directory to load data from.")
    parser.add_argument("--model_path", type=str, required=True, help="Directory to load model from.")
    args = parser.parse_args()

    # set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model config.
    with open(os.path.join(args.model_path, "config.json"), "rt") as f:
        loaded_config = json.loads(f.read())
    
    # Load the model metadata.
    with open(os.path.join(args.data_path, "metadata.json"), "rt") as f:
        loaded_metadata = json.loads(f.read())

    # Create an instance of the simulator.
    simulator = get_simulator(loaded_config, loaded_metadata, loaded_config['noise_std'], device)

    # Load the checkpoint.
    checkpoint = torch.load(os.path.join(args.model_path, "checkpoint.pth"))
    simulator.load_state_dict(checkpoint['model_state_dict'])
    simulator.eval()

    # Sample a batch of particle sequences.
    sampled_positions = []
    n_particles_per_example = []
    particle_types = []
    for i in range(BATCH_SIZE):
        sampled_position_sequence, num_particles = sample_random_position_sequence(
            loaded_metadata['bounds'], 50, 100, NUM_DIMENSIONS, INPUT_SEQUENCE_LENGTH + 1, device=device)
        sampled_positions.append(sampled_position_sequence)
        n_particles_per_example.append(num_particles)
        particle_types.append(torch.randint(0, NUM_PARTICLE_TYPES, (num_particles,), device=device))

    sampled_positions = torch.cat(sampled_positions, dim=0)
    n_particles_per_example = torch.tensor(n_particles_per_example, dtype=torch.int32, device=device)
    particle_types = torch.cat(particle_types, dim=0)

    # Sample global context.
    global_context = torch.rand((BATCH_SIZE, GLOBAL_CONTEXT_SIZE), dtype=torch.float32, device=device) * 2.0 - 1.0

    # Separate input sequence from target sequence.
    input_position_sequence = sampled_positions[:, :-1]
    target_next_position = sampled_positions[:, -1]

    # Single step of inference with the model to predict the next position for each particle.
    with torch.no_grad():
        predicted_next_position = simulator(
            input_position_sequence,
            n_particles_per_example,
            global_context,
            particle_types
        )

    print(f"Per-particle output tensor: {predicted_next_position}")

    # Obtaining predicted and target normalized accelerations for training.
    sampled_noise = get_random_walk_noise_for_position_sequence(
        input_position_sequence, noise_std_last_step=loaded_config['noise_std'])
    with torch.no_grad():
      predicted_normalized_acceleration, target_normalized_acceleration = simulator.get_predicted_and_target_normalized_accelerations(
          target_next_position, sampled_noise, input_position_sequence, n_particles_per_example, global_context, particle_types)

    print(f"Predicted normalized acceleration: {predicted_normalized_acceleration}")
    print(f"Target normalized acceleration: {target_normalized_acceleration}")

if __name__ == "__main__":
    main()