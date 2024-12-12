import torch
import torch.nn as nn
from torch_scatter import scatter_add
import functools

import connectivity_utils
import graph_network
import normalization

STD_EPSILON = 1e-8

class LearnedSimulator(nn.Module):
    """Learned simulator for complex physics systems."""

    def __init__(
            self,
            num_dimensions: int,
            connectivity_radius: float,
            graph_network_kwargs: dict,
            boundaries: list,
            normalization_stats: dict,
            num_particle_types: int,
            particle_type_embedding_size: int,
            name: str = "LearnedSimulator"):
        """Initializes the model.

        Args:
            num_dimensions: Dimensionality of the problem.
            connectivity_radius: Scalar with the radius of connectivity.
            graph_network_kwargs: Keyword arguments to pass to the graph network.
            boundaries: List of 2-tuples, containing the lower and upper
                boundaries of the cuboid containing the particles along each dimension.
            normalization_stats: Dictionary with statistics for normalizing
                acceleration and velocity.
            num_particle_types: Number of different particle types.
            particle_type_embedding_size: Embedding size for the particle type.
            name: Name of the model.
        """
        super().__init__()
        self._connectivity_radius = connectivity_radius
        self._num_particle_types = num_particle_types
        self._boundaries = boundaries
        self._normalization_stats = normalization_stats
        self._num_dimensions = num_dimensions
        self._name = name

        with torch.no_grad():
          self._graph_network = graph_network.EncodeProcessDecode(output_size=num_dimensions, **graph_network_kwargs)

        if self._num_particle_types > 1:
            self._particle_type_embedding = nn.Embedding(num_embeddings=self._num_particle_types, embedding_dim=particle_type_embedding_size)

    def forward(self, position_sequence, n_particles_per_example, global_context=None, particle_types=None):
        """Produces a model step, outputting the next position for each particle.

        Args:
            position_sequence: Sequence of positions for each node in the batch.
                Shape: [num_particles_in_batch, sequence_length, num_dimensions]
            n_particles_per_example: Number of particles for each graph in the batch.
                Shape: [batch_size]
            global_context: Tensor of shape [batch_size, context_size], global context.
            particle_types: Integer tensor of shape [num_particles_in_batch]
                with the types of the particles from 0 to `num_particle_types - 1`.

        Returns:
            Next position with shape [num_particles_in_batch, num_dimensions] for one
            step into the future from the input sequence.
        """
        input_graphs_tuple = self._encoder_preprocessor(
            position_sequence, n_particles_per_example, global_context, particle_types)

        normalized_acceleration = self._graph_network(input_graphs_tuple)

        next_position = self._decoder_postprocessor(
            normalized_acceleration, position_sequence)

        return next_position

    def _encoder_preprocessor(
            self, position_sequence, n_node, global_context, particle_types):
        # Extract important features from the position_sequence.
        most_recent_position = position_sequence[:, -1]
        velocity_sequence = time_diff(position_sequence)

        # Get connectivity of the graph.
        senders, receivers, n_edge = connectivity_utils.compute_connectivity_for_batch_pyfunc(
            most_recent_position, n_node, self._connectivity_radius)

        # Collect node features.
        node_features = []

        # Normalized velocity sequence, merging spatial and time axis.
        velocity_stats = self._normalization_stats["velocity"]
        normalized_velocity_sequence = (
            velocity_sequence - torch.tensor(velocity_stats.mean, device=velocity_sequence.device, dtype=torch.float32)) / torch.tensor(velocity_stats.std, device=velocity_sequence.device, dtype=torch.float32)

        flat_velocity_sequence = normalized_velocity_sequence.view(
            normalized_velocity_sequence.shape[0], -1)
        node_features.append(flat_velocity_sequence)

        # Normalized clipped distances to lower and upper boundaries.
        boundaries = torch.tensor(self._boundaries, device=most_recent_position.device, dtype=torch.float32)
        distance_to_lower_boundary = most_recent_position - boundaries[:, 0]
        distance_to_upper_boundary = boundaries[:, 1] - most_recent_position
        distance_to_boundaries = torch.cat(
            [distance_to_lower_boundary, distance_to_upper_boundary], dim=1)
        normalized_clipped_distance_to_boundaries = torch.clamp(
            distance_to_boundaries / self._connectivity_radius, -1., 1.)
        node_features.append(normalized_clipped_distance_to_boundaries)

        # Particle type.
        if self._num_particle_types > 1:
            particle_type_embeddings = self._particle_type_embedding(particle_types)
            node_features.append(particle_type_embeddings)

        # Collect edge features.
        edge_features = []

        # Relative displacement and distances normalized to radius.
        normalized_relative_displacements = (
            most_recent_position[senders] -
            most_recent_position[receivers]) / self._connectivity_radius
        edge_features.append(normalized_relative_displacements)

        normalized_relative_distances = torch.norm(
            normalized_relative_displacements, dim=-1, keepdim=True)
        edge_features.append(normalized_relative_distances)

        # Normalize the global context.
        if global_context is not None:
            context_stats = self._normalization_stats["context"]
            global_context = (global_context - torch.tensor(context_stats.mean, device=global_context.device, dtype=torch.float32)) / torch.max(
                torch.tensor(context_stats.std, device=global_context.device, dtype=torch.float32), torch.tensor(STD_EPSILON, device=global_context.device, dtype=torch.float32))

        return {
            "nodes": torch.cat(node_features, dim=-1),
            "edges": torch.cat(edge_features, dim=-1),
            "globals": global_context,
            "n_node": n_node,
            "n_edge": n_edge,
            "senders": senders,
            "receivers": receivers
        }

    def _decoder_postprocessor(self, normalized_acceleration, position_sequence):
        # The model produces the output in normalized space so we apply inverse
        # normalization.
        acceleration_stats = self._normalization_stats["acceleration"]
        acceleration = (
            normalized_acceleration * torch.tensor(acceleration_stats.std, device=normalized_acceleration.device, dtype=torch.float32)
            + torch.tensor(acceleration_stats.mean, device=normalized_acceleration.device, dtype=torch.float32)
        )

        # Use an Euler integrator to go from acceleration to position, assuming
        # a dt=1 corresponding to the size of the finite difference.
        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = most_recent_position - position_sequence[:, -2]

        new_velocity = most_recent_velocity + acceleration
        new_position = most_recent_position + new_velocity
        return new_position

    def get_predicted_and_target_normalized_accelerations(
            self, next_position, position_sequence_noise, position_sequence,
            n_particles_per_example, global_context=None, particle_types=None):
        """Produces normalized and predicted acceleration targets.

        Args:
            next_position: Tensor of shape [num_particles_in_batch, num_dimensions]
                with the positions the model should output given the inputs.
            position_sequence_noise: Tensor of the same shape as `position_sequence`
                with the noise to apply to each particle.
            position_sequence, n_particles_per_example, global_context, particle_types:
                Inputs to the model.

        Returns:
            Tensors of shape [num_particles_in_batch, num_dimensions] with the
            predicted and target normalized accelerations.
        """
        # Add noise to the input position sequence.
        noisy_position_sequence = position_sequence + position_sequence_noise

        # Perform the forward pass with the noisy position sequence.
        input_graphs_tuple = self._encoder_preprocessor(
            noisy_position_sequence, n_particles_per_example, global_context, particle_types)
        predicted_normalized_acceleration = self._graph_network(input_graphs_tuple)

        # Calculate the target acceleration, using an `adjusted_next_position `that
        # is shifted by the noise in the last input position.
        next_position_adjusted = next_position + position_sequence_noise[:, -1]
        target_normalized_acceleration = self._inverse_decoder_postprocessor(
            next_position_adjusted, noisy_position_sequence)

        return predicted_normalized_acceleration, target_normalized_acceleration

    def _inverse_decoder_postprocessor(self, next_position, position_sequence):
        """Inverse of `_decoder_postprocessor`."""

        previous_position = position_sequence[:, -1]
        previous_velocity = previous_position - position_sequence[:, -2]
        next_velocity = next_position - previous_position
        acceleration = next_velocity - previous_velocity

        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration = (
            acceleration - torch.tensor(acceleration_stats.mean, device=acceleration.device, dtype=torch.float32)) / torch.tensor(acceleration_stats.std, device=acceleration.device, dtype=torch.float32)
        return normalized_acceleration

def time_diff(input_sequence, axis=1):
    """Calculates the time derivative of the input sequence."""
    return torch.diff(input_sequence, dim=axis)