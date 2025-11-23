import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def simulate(num_particles=300, num_steps=100, dt=0.05):
    """
    Runs the toilet flush simulation.

    Args:
        num_particles: Number of water particles.
        num_steps: Number of simulation steps.
        dt: Time step.

    Returns:
        A dictionary containing:
        - "initial_positions": (num_particles, 3)
        - "predicted_rollout": (num_steps, num_particles, 3)
        - "target_rollout": (num_steps, num_particles, 3) (Dummy data)
    """
    # Simulation parameters
    gravity = np.array([0, 0, -9.8])

    # Toilet geometry parameters
    bowl_radius = 1.0
    drain_radius = 0.15
    bowl_height = 1.0
    drain_z = -0.5

    # Initialize particles
    initial_positions = []
    for _ in range(num_particles):
        # Random positions in a cylinder above the bowl
        r = np.sqrt(np.random.rand()) * bowl_radius * 0.8
        theta = np.random.rand() * 2 * np.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.rand() * 0.5 + 0.5 # Start slightly above z=0
        initial_positions.append([x, y, z])

    positions = np.array(initial_positions)
    velocities = np.zeros_like(positions)

    rollout = []
    # We don't append initial position to predicted_rollout to match render_rollout expectations usually
    # But let's keep tracking all steps.

    k = 1.5
    z_base = -0.5

    for step in range(num_steps):
        # Update velocities with gravity
        velocities += gravity * dt

        # Update positions
        positions += velocities * dt

        # Collision handling
        # Surface equation: z_surf = k * (x^2 + y^2) + z_base

        for i in range(num_particles):
            p = positions[i]
            x, y, z = p

            # Check for drain (hole in the middle)
            r2 = x*x + y*y
            if r2 < drain_radius**2:
                # Inside the drain radius.
                if z < z_base:
                     if r2 > (drain_radius * 0.9)**2: # Close to pipe wall
                        # Push back radially
                        normal = np.array([-x, -y, 0])
                        norm_mag = np.linalg.norm(normal)
                        if norm_mag > 1e-8:
                            normal /= norm_mag

                        # Reflect velocity
                        v_dot_n = np.dot(velocities[i], normal)
                        if v_dot_n < 0:
                            velocities[i] -= 1.5 * v_dot_n * normal
                            positions[i] += normal * 0.01
                continue # Don't collide with bowl

            z_surf = k * r2 + z_base

            if z < z_surf:
                # Collision detected
                # Normal = (-2kx, -2ky, 1) normalized
                normal = np.array([-2*k*x, -2*k*y, 1.0])
                norm_mag = np.linalg.norm(normal)
                if norm_mag > 1e-8:
                    normal /= norm_mag

                # Push out
                penetration = z_surf - z
                positions[i] += normal * (penetration * normal[2] + 0.01)

                # Reflect velocity
                v = velocities[i]
                v_dot_n = np.dot(v, normal)

                if v_dot_n < 0:
                    # Inelastic collision + friction
                    restitution = 0.3
                    friction = 0.9

                    v_normal = v_dot_n * normal
                    v_tangent = v - v_normal

                    velocities[i] = v_tangent * friction - v_normal * restitution

                    # Add swirl
                    radius_vec = np.array([x, y, 0])
                    r_mag = np.linalg.norm(radius_vec)
                    if r_mag > 1e-8:
                        radius_vec /= r_mag

                    tangent_vec = np.array([-y, x, 0])
                    t_mag = np.linalg.norm(tangent_vec)
                    if t_mag > 1e-8:
                        tangent_vec /= t_mag

                    velocities[i] += tangent_vec * 0.5 * dt

        rollout.append(positions.copy())

    rollout_np = np.array(rollout)

    return {
        "initial_positions": np.array(initial_positions),
        "predicted_rollout": rollout_np,
        "target_rollout": rollout_np # Dummy target
    }

def animate(rollout_data, sim_name="Simulation", figsize=(10, 8), dpi=80):
    """
    Creates an animation from the rollout data.

    Args:
        rollout_data: Dictionary with 'initial_positions', 'predicted_rollout'.
        sim_name: Title of the simulation.
        figsize: Tuple (width, height).
        dpi: Dots per inch.

    Returns:
        matplotlib.animation.FuncAnimation object.
    """
    initial_positions = rollout_data["initial_positions"]
    predicted_rollout = rollout_data["predicted_rollout"]

    num_steps = predicted_rollout.shape[0]
    num_particles = predicted_rollout.shape[1]

    # Calculate bounds
    all_positions = np.concatenate([initial_positions[np.newaxis, ...], predicted_rollout], axis=0)
    min_bound = np.min(all_positions, axis=(0, 1))
    max_bound = np.max(all_positions, axis=(0, 1))

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    def update_plot(frame):
        ax.cla()
        ax.set_xlim(min_bound[0], max_bound[0])
        ax.set_ylim(min_bound[1], max_bound[1])
        ax.set_zlim(min_bound[2], max_bound[2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{sim_name} - Frame {frame}")

        # Plot particles
        # Use a simple color gradient based on index or just blue
        ax.scatter(predicted_rollout[frame, :, 0],
                   predicted_rollout[frame, :, 1],
                   predicted_rollout[frame, :, 2],
                   c='b', s=10, alpha=0.6)

    ani = animation.FuncAnimation(fig, update_plot, frames=num_steps, interval=50)
    plt.close(fig) # Prevent double display in notebooks
    return ani

def show_in_colab(anim):
    """
    Displays the animation in a Google Colab notebook.

    Usage:
        data = simulate()
        anim = animate(data)
        show_in_colab(anim)
    """
    from IPython.display import HTML
    return HTML(anim.to_jshtml())
