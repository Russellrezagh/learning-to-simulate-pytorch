import functools
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Dict, List, Tuple

class TrajectoryDataset(Dataset):
    """Dataset for trajectories stored as NumPy arrays."""

    def __init__(self, data_dir: str, max_rollout_length: int):
        """
        Initializes the dataset.

        Args:
            data_dir: Directory containing the NumPy array files.
            max_rollout_length: Maximum length of trajectories.
        """
        self.data_dir = data_dir
        self.max_rollout_length = max_rollout_length
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.files.sort()  # Ensure consistent order

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a trajectory from the dataset.

        Args:
            idx: Index of the trajectory to return.

        Returns:
            A dictionary containing the trajectory data.
        """
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = np.load(file_path)

        trajectory = {
            "position": torch.tensor(data["position"], dtype=torch.float32),
            "velocity": torch.tensor(data["velocity"], dtype=torch.float32),
            "acceleration": torch.tensor(data["acceleration"], dtype=torch.float32),
            "particle_type": torch.tensor(data["particle_type"], dtype=torch.int64),
        }
        
        # Pad trajectories to max_rollout_length
        for key, value in trajectory.items():
          if key == "particle_type":
            padding = (0, 0, 0, self.max_rollout_length - value.shape[0])
          else:
            padding = (0, 0, 0, 0, 0, self.max_rollout_length - value.shape[0])
          trajectory[key] = torch.nn.functional.pad(value, padding)

        return trajectory

def get_output_shapes(dataset: TrajectoryDataset) -> Dict[str, Tuple]:
  """Returns a dictionary of the shapes of the given dataset.

  Args:
    dataset: Dataset to extract shapes.

  Returns:
    A dictionary with the shapes of the elements of `dataset`.
  """
  shapes = {}
  first_trajectory = dataset[0]
  for key, value in first_trajectory.items():
      shapes[key] = tuple(value.shape)
  return shapes

def get_input_trajectory(trajectory: Dict[str, torch.Tensor], rollout_length: int, num_time_steps: int) -> Dict[str, torch.Tensor]:
  """Returns the first `num_time_steps` of a given trajectory.

  If `rollout_length` is shorter than `num_time_steps`, it will be padded with
  the first frame as many times as necessary.

  Args:
    trajectory: Trajectory of features to extract inputs.
    rollout_length: Length of the given trajectory.
    num_time_steps: Number of time steps to extract.

  Returns:
    The first `num_time_steps` of the given trajectory, padded if necessary.
  """
  s = slice(0, num_time_steps)
  inputs = {key: value[s] for key, value in trajectory.items()}

  pad_with_first = num_time_steps - rollout_length
  if pad_with_first > 0:
      for key, value in inputs.items():
          if key == "particle_type":
            padding = (0, 0, 0, pad_with_first)
          else:
            padding = (0, 0, 0, 0, 0, pad_with_first)
          inputs[key] = torch.nn.functional.pad(value, padding, mode="replicate")

  return inputs

def prepare_inputs(trajectory: Dict[str, torch.Tensor], particle_types: torch.Tensor, is_training: bool, noise_std: float,
                   num_input_frames: int, sample_random_times: bool) -> Dict[str, torch.Tensor]:
  """Extracts an input trajectory from a given trajectory.

  The input trajectory is extracted by taking the first `num_input_frames`
  positions, adding noise to them, and concatenating the particle type.

  Args:
    trajectory: Trajectory of positions, velocities and particle types.
    particle_types: Types of the particles.
    is_training: Whether the trajectory is used for training.
    noise_std: Standard deviation of the noise applied to the positions.
    num_input_frames: Number of frames used as input.
    sample_random_times: Whether to sample random time steps as input.

  Returns:
    A dictionary containing the input positions, velocities and types.
  """
  if is_training and sample_random_times:
    pad = 1
    length = trajectory["position"].shape[0]
    start = torch.randint(0, length - num_input_frames - pad, (1,)).item()
    s = slice(start, start + num_input_frames)
    inputs = {key: value[s] for key, value in trajectory.items()}
  else:
    inputs = get_input_trajectory(trajectory,
                                  rollout_length=num_input_frames,
                                  num_time_steps=num_input_frames)
  pos = inputs["position"]
  velocity = inputs["velocity"]
  acceleration = inputs["acceleration"]
  if is_training:
    noise = torch.randn_like(pos) * noise_std
    mask = torch.rand((pos.shape[0], 1, 1), dtype=pos.dtype, device=pos.device)
    noise = torch.where(mask < 1.0, noise, torch.zeros_like(noise))
    pos += noise

  inputs["position"] = pos
  inputs["velocity"] = velocity
  inputs["acceleration"] = acceleration
  inputs["particle_type"] = particle_types.repeat(num_input_frames, 1)
  return inputs

def collate_fn_for_learning(batch: List[Dict[str, torch.Tensor]], output_shapes: Dict[str, Tuple], rollout_length: int, input_length: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
  """Collates and pads a batch of trajectories for use in learning algorithms.

  Args:
    batch: A list of dictionaries containing trajectories of particles.
    output_shapes: Shapes of the elements of the given dataset, without considering the batch and time dimensions.
    rollout_length: Length of trajectories for training.
    input_length: Length of trajectories for input.

  Returns:
    A tuple containing:
      - inputs: A dictionary of batched and padded input trajectories.
      - target_acceleration: A tensor of batched and padded target accelerations.
      - target_velocity: A tensor of batched and padded target velocities.
  """
  # Extract inputs and add noise
  inputs_list = []
  trajectory_list = []
  for trajectory in batch:
      inputs = prepare_inputs(trajectory, trajectory["particle_type"][0], True, 1e-2, input_length, True)
      inputs_list.append(inputs)
      trajectory_list.append(trajectory)

  # Pad trajectories to the desired rollout length
  padded_trajectories = []
  for trajectory in trajectory_list:
      padded_trajectory = {}
      for key, value in trajectory.items():
          if key == "particle_type":
            pad_size = rollout_length - value.shape[0]
            padding = (0, 0, 0, pad_size)
          else:
            pad_size = rollout_length - value.shape[0]
            padding = (0, 0, 0, 0, 0, pad_size)
          padded_trajectory[key] = torch.nn.functional.pad(value, padding)
      padded_trajectories.append(padded_trajectory)

  # Batch trajectories
  batched_trajectories = {}
  for key in padded_trajectories[0].keys():
      if key == "particle_type":
        batched_trajectories[key] = torch.stack([traj[key][0:1] for traj in padded_trajectories])
      else:
        batched_trajectories[key] = torch.stack([traj[key] for traj in padded_trajectories])
  
  # Pad inputs
  padded_inputs = {}
  for key, value in inputs_list[0].items():
      if key == "particle_type":
        pad_size = input_length - value.shape[0]
        padding = (0, 0, 0, pad_size)
      else:
        pad_size = input_length - value.shape[0]
        padding = (0, 0, 0, 0, 0, pad_size)
      padded_inputs[key] = torch.nn.functional.pad(value, padding)
  
  # Batch inputs
  batched_inputs = {}
  for key in padded_inputs.keys():
      batched_inputs[key] = torch.stack([inputs[key] for inputs in inputs_list])

  return batched_inputs, batched_trajectories["acceleration"], batched_trajectories["velocity"]

def prepare_data_for_learning(data_dir: str,
                              batch_size: int,
                              max_rollout_length: int,
                              rollout_length: int,
                              input_length: int,
                              shuffle_buffer_size: int, # keep it for consistency
                              num_parallel_reads: int, # keep it for consistency
                              num_parallel_batches: int # keep it for consistency
                              ) -> DataLoader:
  """Creates a DataLoader for use in learning algorithms."""
  dataset = TrajectoryDataset(data_dir, max_rollout_length)
  output_shapes = get_output_shapes(dataset)
  
  collate_fn = functools.partial(collate_fn_for_learning, output_shapes=output_shapes, rollout_length=rollout_length, input_length=input_length)

  # Use PyTorch's DataLoader for batching and shuffling
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
  
  return dataloader

def collate_fn_for_evaluation(batch: List[Dict[str, torch.Tensor]], output_shapes: Dict[str, Tuple]) -> Tuple[Dict[str, torch.Tensor], List[int]]:
  """Batches and pads a batch of trajectories for use in evaluation.

  Args:
    batch: A list of dictionaries containing trajectories of particles.
    output_shapes: Shapes of the elements of the given dataset, without considering the batch and time dimensions.

  Returns:
    A tuple containing:
      - trajectories: A dictionary of batched and padded trajectories.
      - lengths: A list of the lengths of the original trajectories (before padding).
  """
  # Find the length of each trajectory in the batch
  lengths = [traj["position"].shape[0] for traj in batch]

  # Find the maximum trajectory length in the batch
  max_length = max(lengths)

  # Pad each trajectory to the maximum length
  padded_trajectories = []
  for trajectory in batch:
      padded_trajectory = {}
      for key, value in trajectory.items():
          if key == "particle_type":
            pad_size = max_length - value.shape[0]
            padding = (0, 0, 0, pad_size)
          else:
            pad_size = max_length - value.shape[0]
            padding = (0, 0, 0, 0, 0, pad_size)
          padded_trajectory[key] = torch.nn.functional.pad(value, padding)
      padded_trajectories.append(padded_trajectory)

  # Batch trajectories
  batched_trajectories = {}
  for key in padded_trajectories[0].keys():
      batched_trajectories[key] = torch.stack([traj[key] for traj in padded_trajectories])

  return batched_trajectories, lengths

def prepare_data_for_evaluation(data_dir: str,
                                batch_size: int,
                                max_rollout_length: int,
                                shuffle_buffer_size: int, # keep it for consistency
                                num_parallel_reads: int, # keep it for consistency
                                num_parallel_batches: int # keep it for consistency
                                ) -> DataLoader:
  """Creates a DataLoader for use in evaluation."""
  dataset = TrajectoryDataset(data_dir, max_rollout_length)
  output_shapes = get_output_shapes(dataset)

  collate_fn = functools.partial(collate_fn_for_evaluation, output_shapes=output_shapes)

  # Use PyTorch's DataLoader for batching (no shuffling for evaluation)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

  return dataloader