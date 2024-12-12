import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

def render_rollout(rollout_data, output_dir, sim_name, show_trajectories=False):
    """Renders a rollout trajectory as a series of images and optionally creates an animation (gif).

    Args:
      rollout_data: A dictionary containing the rollout data, including:
        - initial_positions: An array of shape [num_particles, dim] representing the initial positions.
        - predicted_rollout: An array of shape [num_steps, num_particles, dim] representing the predicted positions.
        - target_rollout: An array of shape [num_steps, num_particles, dim] representing the target positions.
      output_dir: The directory to save the rendered images and animation.
      sim_name: The name of the simulation.
      show_trajectories: Whether to show the full trajectories as lines.
    """

    initial_positions = rollout_data["initial_positions"]
    predicted_rollout = rollout_data["predicted_rollout"]
    target_rollout = rollout_data["target_rollout"]
    num_steps = predicted_rollout.shape[0]
    num_particles = predicted_rollout.shape[1]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get bounds for consistent visualization
    all_positions = np.concatenate([initial_positions[np.newaxis, ...], predicted_rollout, target_rollout], axis=0)
    min_bound = np.min(all_positions, axis=(0, 1))
    max_bound = np.max(all_positions, axis=(0, 1))

    # Get particle types (if available)
    metadata = rollout_data.get("metadata")
    if metadata is not None:
        try:
            metadata = pickle.loads(metadata)
            particle_types = metadata.get("particle_type")
            num_particle_types = metadata.get("num_particle_types")
            if particle_types is not None:
                # Normalize particle types to be in the range [0, 1]
                particle_types = particle_types.astype(np.float32) / (num_particle_types - 1)
        except:
            particle_types = None
    else:
        particle_types = None

    # Create colormap for particle types
    if particle_types is not None:
        cmap = plt.cm.get_cmap("jet", num_particle_types)  # You can use other colormaps
    else:
        cmap = None

    # Function to update the plot for each frame
    def update_plot(frame):
        ax.cla()  # Clear the previous frame

        # Plot predicted positions
        if particle_types is not None:
          colors = cmap(particle_types)
          ax.scatter(predicted_rollout[frame, :, 0], predicted_rollout[frame, :, 1], predicted_rollout[frame, :, 2], c=colors, s=10, alpha=0.7, label='Predicted')
        else:
          ax.scatter(predicted_rollout[frame, :, 0], predicted_rollout[frame, :, 1], predicted_rollout[frame, :, 2], c='r', s=10, alpha=0.7, label='Predicted')

        # Plot target positions (if available)
        if target_rollout is not None:
          if particle_types is not None:
            ax.scatter(target_rollout[frame, :, 0], target_rollout[frame, :, 1], target_rollout[frame, :, 2], c=colors, s=10, marker='x', alpha=0.7, label='Target')
          else:
            ax.scatter(target_rollout[frame, :, 0], target_rollout[frame, :, 1], target_rollout[frame, :, 2], c='b', s=10, marker='x', alpha=0.7, label='Target')
            
        # Plot trajectories if requested
        if show_trajectories:
          for p in range(num_particles):
            if particle_types is not None:
                ax.plot(predicted_rollout[:frame+1, p, 0], predicted_rollout[:frame+1, p, 1], predicted_rollout[:frame+1, p, 2], c=cmap(particle_types[p]), alpha=0.5)
            else:
                ax.plot(predicted_rollout[:frame+1, p, 0], predicted_rollout[:frame+1, p, 1], predicted_rollout[:frame+1, p, 2], c='r', alpha=0.5)
            if target_rollout is not None:
                if particle_types is not None:
                  ax.plot(target_rollout[:frame+1, p, 0], target_rollout[:frame+1, p, 1], target_rollout[:frame+1, p, 2], c=cmap(particle_types[p]), alpha=0.5)
                else:
                  ax.plot(target_rollout[:frame+1, p, 0], target_rollout[:frame+1, p, 1], target_rollout[:frame+1, p, 2], c='b', alpha=0.5)

        # Set plot limits
        ax.set_xlim(min_bound[0], max_bound[0])
        ax.set_ylim(min_bound[1], max_bound[1])
        ax.set_zlim(min_bound[2], max_bound[2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{sim_name} - Frame {frame + 1}")
        ax.legend()

    # Create the figure and axes for the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Create the animation
    ani = animation.FuncAnimation(fig, update_plot, frames=num_steps, interval=100)

    # Save the animation as a GIF
    output_path = os.path.join(output_dir, f"{sim_name}_rollout.gif")
    ani.save(output_path, writer="pillow", fps=10)

    # Save individual frames as images
    for frame in range(num_steps):
        update_plot(frame)
        output_path = os.path.join(output_dir, f"{sim_name}_frame_{frame:04d}.png")
        plt.savefig(output_path)

    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for images and animation.")
    parser.add_argument("--input_pickle", type=str, required=True, help="Input path for trajectory data.")
    parser.add_argument("--sim_name", type=str, required=True, help="Name of the simulation to render.")
    parser.add_argument("--show_trajectories", action="store_true", help="Show full trajectories as lines.")
    args = parser.parse_args()

    with open(args.input_pickle, "rb") as f:
        rollout_data = pickle.load(f)

    render_rollout(rollout_data, args.output_dir, args.sim_name, args.show_trajectories)

if __name__ == "__main__":
    main()