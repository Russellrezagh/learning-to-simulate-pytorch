import argparse
import functools
import json
import os
import pickle
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import graph_network
import learned_simulator
import reading_utils
import normalization
import model_config

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

def _read_metadata(data_path: str) -> Dict[str, Any]:
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())

def prepare_inputs(trajectory: Dict[str, torch.Tensor], particle_types: torch.Tensor, is_training: bool, noise_std: float,
                   num_input_frames: int, sample_random_times: bool) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Prepares an input trajectory by adding noise to the position.

    Args:
        trajectory: A dict with the trajectory data.
        is_training: Whether the model is being trained.

    Returns:
        A dict with the processed trajectory data.
    """
    inputs = reading_utils.prepare_inputs(trajectory, particle_types, is_training, noise_std, num_input_frames, sample_random_times)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = {
        "acceleration": trajectory["acceleration"].to(device),
        "velocity": trajectory["velocity"].to(device),
    }
    return inputs, labels

def build_model(config: Dict[str, Any], metadata: Dict[str, Any], noise_std: float) -> learned_simulator.LearnedSimulator:
    """Builds the model."""
    # Model parameters.
    model_config = graph_network.get_model_config(config.get('model_config_name', 'gns_model'))
    num_dimensions = metadata['dim']
    if config.get('keypoint_detector_model', False):
        model_config = model_config._replace(output_size=model_config.output_size)
    else:
        model_config = model_config._replace(output_size=num_dimensions)
    graph_network_kwargs = dict(model_config._asdict())

    # Normalization parameters.
    normalization_stats = {
        'acceleration': {
            'mean': torch.tensor(metadata['acc_mean'], dtype=torch.float32),
            'std': torch.sqrt(torch.tensor(metadata['acc_std'], dtype=torch.float32)**2 + noise_std**2),
        },
        'velocity': {
            'mean': torch.tensor(metadata['vel_mean'], dtype=torch.float32),
            'std': torch.sqrt(torch.tensor(metadata['vel_std'], dtype=torch.float32)**2 + noise_std**2),
        },
    }
    if 'context_mean' in metadata:
        normalization_stats['context'] = {
            'mean': torch.tensor(metadata['context_mean'], dtype=torch.float32),
            'std': torch.tensor(metadata['context_std'], dtype=torch.float32),
        }

    return learned_simulator.LearnedSimulator(
        num_dimensions=num_dimensions,
        connectivity_radius=metadata['default_connectivity_radius'],
        graph_network_kwargs=graph_network_kwargs,
        boundaries=metadata['bounds'],
        num_particle_types=NUM_PARTICLE_TYPES,
        normalization_stats=normalization_stats,
        particle_type_embedding_size=config.get('particle_type_embedding_size', 16),
        num_input_frames=INPUT_SEQUENCE_LENGTH)

def get_learning_rate(global_step: int, config: Dict[str, Any]) -> float:
    """Get learning rate for the current step."""
    # Learning rate decays in two phases.
    # Phase 1: decays from 1e-4 to 1e-5 over 10^6 steps.
    # Phase 2: decays from 1e-5 to 1e-6 over 10^7 steps.
    number_of_phases = 2
    boundaries = [0, int(1e6), int(1e7)]
    values = [0., config['learning_rate'], config['learning_rate'] * 1e-1, config['learning_rate'] * 1e-2]

    index = np.searchsorted(boundaries, global_step, side='right')
    return values[index]

def create_optimizer(learning_rate: float) -> torch.optim.Optimizer:
    """Creates an optimizer."""
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_model(model: learned_simulator.LearnedSimulator, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                num_steps: int, noise_std: float, log_every_n_steps: int, summary_writer: SummaryWriter, save_every_n_steps: int,
                checkpoint_dir: str):
    """Trains the model."""
    global_step = 0
    start_time = time.time()
    
    # Restore checkpoint if available.
    global_step = restore_latest_checkpoint(model, optimizer, checkpoint_dir)

    model.train()
    for step, trajectory in enumerate(dataloader):
        if step >= num_steps:
            break

        # Set learning rate.
        learning_rate = get_learning_rate(global_step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        # Prepare inputs.
        inputs, targets = prepare_inputs(trajectory, trajectory["particle_type"][0], True, noise_std, 5, True)

        # Forward pass.
        predicted_acc = model._learned_simulator_step(inputs, is_training=True)[0]

        # Loss.
        loss = ((predicted_acc - targets['acceleration']) ** 2).mean()

        # Backward pass and optimization.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1

        # Log training information.
        if step % log_every_n_steps == 0:
            elapsed_time = time.time() - start_time
            print(f'Step {global_step}/{num_steps} - Loss: {loss.item():.4f} - LR: {learning_rate:.6f} '
                  f'- Elapsed: {elapsed_time:.2f}s')
            summary_writer.add_scalar('train_loss', loss.item(), global_step)
            summary_writer.add_scalar('learning_rate', learning_rate, global_step)

        # Save checkpoint.
        if step % save_every_n_steps == 0:
            save_checkpoint(model, optimizer, global_step, checkpoint_dir)

    # Save the final checkpoint.
    save_checkpoint(model, optimizer, global_step, checkpoint_dir)
    print('Training finished.')

def evaluate_model(model: learned_simulator.LearnedSimulator, dataloader: DataLoader, num_eval_steps: int,
                   summary_writer: SummaryWriter, noise_std: float):
    """Evaluates the model."""
    model.eval()
    losses = []

    with torch.no_grad():
        for step, trajectory in enumerate(dataloader):
            if step == num_eval_steps:
                break

            # Prepare inputs.
            inputs, targets = prepare_inputs(trajectory, trajectory["particle_type"][0], False, noise_std, 5, False)

            # Forward pass.
            predicted_acc = model._learned_simulator_step(inputs, is_training=False)[0]

            # Loss.
            loss = ((predicted_acc - targets['acceleration']) ** 2).mean()
            losses.append(loss.item())

            # Log evaluation information.
            print(f'Eval Step {step}/{num_eval_steps} - Loss: {loss.item():.4f}')
            summary_writer.add_scalar('eval_loss', loss.item(), step)

    # Calculate and log average evaluation loss.
    avg_loss = np.mean(losses)
    print(f'Average evaluation loss: {avg_loss:.4f}')
    summary_writer.add_scalar('avg_eval_loss', avg_loss, 0)

def save_checkpoint(model: learned_simulator.LearnedSimulator, optimizer: torch.optim.Optimizer, step: int, checkpoint_dir: str):
    """Saves a checkpoint of the model and optimizer."""
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{step}.pth')
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')

def restore_latest_checkpoint(model: learned_simulator.LearnedSimulator, optimizer: torch.optim.Optimizer, checkpoint_dir: str) -> int:
    """Restores the latest checkpoint from the given directory."""
    # Find all checkpoint files.
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pth')]
    if not checkpoint_files:
        print('No checkpoint found.')
        return 0

    # Find the checkpoint with the highest step number.
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    # Load the checkpoint.
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    print(f'Checkpoint restored from {checkpoint_path} at step {step}')

    return step

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval', 'eval_rollout'],
                        help='Train model, one step evaluation or rollout evaluation.')
    parser.add_argument('--data_path', type=str, required=True, help='The dataset directory.')
    parser.add_argument('--model_path', type=str, required=True,
                        help=('The path for saving checkpoints of the model. '
                              'Defaults to a temporary directory.'))
    parser.add_argument('--output_path', type=str, default=None,
                        help='The path for saving outputs (e.g. rollouts).')
    parser.add_argument('--batch_size', type=int, default=2, help='The batch size.')
    parser.add_argument('--num_steps', type=int, default=int(2e7), help='Number of steps of training.')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='Number of steps of evaluation.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='The learning rate.')
    parser.add_argument('--noise_std', type=float, default=3e-4, help='The std deviation of the noise.')
    parser.add_argument('--config', type=str, help='The path to the model configuration file.')
    parser.add_argument('--eval_split', type=str, default='test', help='Split to use when running evaluation.')
    parser.add_argument('--log_every_n_steps', type=int, default=100, help='Log training information every n steps.')
    parser.add_argument('--save_every_n_steps', type=int, default=1000, help='Save checkpoint every n steps.')
    args = parser.parse_args()
    
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config file.
    if args.config is None:
      args.config = os.path.join(args.model_path, 'config.json')
    with open(args.config, 'rt') as f:
        config = json.loads(f.read())

    # Load metadata.
    metadata = _read_metadata(args.data_path)

    # Build the model.
    model = build_model(config, metadata, args.noise_std)
    model.to(device)

    # Create a summary writer for TensorBoard.
    summary_writer = SummaryWriter(args.model_path)

    # Get the dataset.
    if args.mode == 'train':
        get_data_fn = reading_utils.prepare_data_for_learning
        data_path = os.path.join(args.data_path, 'train')
        dataset = get_data_fn(
            data_path,
            batch_size=args.batch_size,
            max_rollout_length=config.get('max_rollout_length', 50), # Set here to use for padding
            rollout_length=config.get('rollout_length', 50),
            input_length=INPUT_SEQUENCE_LENGTH,
            shuffle_buffer_size=config.get('shuffle_buffer_size', 1000),
            num_parallel_reads=config.get('num_parallel_reads', 4),
            num_parallel_batches=config.get('num_parallel_batches', 4),
        )
    elif args.mode == 'eval':
        get_data_fn = reading_utils.prepare_data_for_evaluation
        data_path = os.path.join(args.data_path, args.eval_split)
        dataset = get_data_fn(
            data_path,
            batch_size=args.batch_size,
            max_rollout_length=config.get('max_rollout_length', 50),
            shuffle_buffer_size=config.get('shuffle_buffer_size', 1000),
            num_parallel_reads=config.get('num_parallel_reads', 1),
            num_parallel_batches=config.get('num_parallel_batches', 1),
        )

    if args.mode == 'train':
        # Build the optimizer.
        optimizer = create_optimizer(args.learning_rate)
        train_dataloader = DataLoader(dataset, batch_size=None, shuffle=True)
        train_model(model, train_dataloader, optimizer, args.num_steps, args.noise_std, args.log_every_n_steps,
                    summary_writer, args.save_every_n_steps, args.model_path)
    elif args.mode == 'eval':
        eval_dataloader = DataLoader(dataset, batch_size=None, shuffle=False)
        restore_latest_checkpoint(model, optimizer, args.model_path)
        evaluate_model(model, eval_dataloader, args.eval_steps, summary_writer, args.noise_std)
    elif args.mode == 'eval_rollout':
        pass # TODO: implement rollout evaluation
    else:
        raise ValueError('Unknown mode {}'.format(args.mode))

if __name__ == '__main__':
    main()