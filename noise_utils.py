import torch

def get_random_walk_noise_for_position_sequence(
        position_sequence: torch.Tensor, noise_std_last_step: float) -> torch.Tensor:
    """Returns random-walk noise in the velocity applied to the position.

    Args:
        position_sequence: A sequence of positions in a torch.Tensor with shape
            [num_particles, sequence_length, num_dimensions].
        noise_std_last_step: Standard deviation of the noise at the last step.
            Each intermediate step will have a standard deviation between 0 and
            `noise_std_last_step`.

    Returns:
        A tensor of shape [num_particles, sequence_length, num_dimensions]
        containing random-walk noise which can be added to each position in
        `position_sequence`.
    """
    # Calculate the standard deviation of the noise for each step.
    sequence_length = position_sequence.shape[1]
    noise_std = noise_std_last_step / (sequence_length - 1)**0.5  

    # Generate random noise with the calculated standard deviation.
    noise = torch.randn_like(position_sequence) * noise_std

    # Apply the random walk to the noise.
    return torch.cumsum(noise, dim=1)