import numpy as np
import pickle
import os

def generate_toilet_flush_simulation():
    # Simulation parameters
    num_particles = 300
    num_steps = 100
    dt = 0.05
    gravity = np.array([0, 0, -9.8])

    # Toilet geometry parameters
    bowl_radius = 1.0
    drain_radius = 0.15
    bowl_height = 1.0
    drain_z = -0.5

    # Initialize particles
    # Place them in a cylinder shape above the bowl initially (the tank/water source)
    # Or just fill the bowl partially.
    # Let's start them in a cloud above the bowl.
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
    rollout.append(positions.copy())

    print("Starting simulation...")

    for step in range(num_steps):
        # Update velocities with gravity
        velocities += gravity * dt

        # Update positions
        positions += velocities * dt

        # Collision handling with Bowl
        # Bowl shape: z = 2 * (x^2 + y^2) - 1.0 (Approximate parabola)
        # Let's define the surface as F(x, y, z) = z - k*(x^2 + y^2) - z_base = 0
        # If F < 0, we are "under" the surface (outside the bowl volume if we consider the bowl holds water "above" the surface).
        # Wait, water sits ON TOP of the bowl surface. So if z < surface_z, we collide.

        # Surface equation: z_surf = 0.8 * (x^2 + y^2) - 0.5
        k = 1.5
        z_base = -0.5

        for i in range(num_particles):
            p = positions[i]
            x, y, z = p

            # Check for drain (hole in the middle)
            r2 = x*x + y*y
            if r2 < drain_radius**2:
                # Inside the drain radius.
                # Let it fall, but maybe constrain it to a pipe?
                # Pipe: x^2 + y^2 < drain_radius^2.
                # If it tries to go outside the pipe while in the drain (z < z_base), bounce off pipe walls.
                if z < z_base:
                     if r2 > (drain_radius * 0.9)**2: # Close to pipe wall
                        # Push back radially
                        normal = np.array([-x, -y, 0])
                        normal /= np.linalg.norm(normal) + 1e-8

                        # Reflect velocity
                        v_dot_n = np.dot(velocities[i], normal)
                        if v_dot_n < 0:
                            velocities[i] -= 1.5 * v_dot_n * normal
                            positions[i] += normal * 0.01
                continue # Don't collide with bowl

            z_surf = k * r2 + z_base

            if z < z_surf:
                # Collision detected
                # Calculate Normal: gradient of F(x,y,z) = z - k(x^2+y^2) - z_base
                # Grad F = (-2kx, -2ky, 1)
                # We need normal pointing INTO the fluid (upwards/inwards).
                # The gradient (-2kx, -2ky, 1) points UP (positive z component). This is correct.

                normal = np.array([-2*k*x, -2*k*y, 1.0])
                normal /= np.linalg.norm(normal)

                # Project position back to surface
                # Simple approximation: move in direction of normal until satisfied?
                # Or just reset z? Resetting z is easiest but inaccurate for steep slopes.
                # Let's push along normal.
                penetration = z_surf - z
                # This is vertical distance. Distance along normal is approx penetration * (n.z).
                # Actually, just push out.
                positions[i] += normal * (penetration * normal[2] + 0.01)

                # Reflect velocity
                v = velocities[i]
                v_dot_n = np.dot(v, normal)

                if v_dot_n < 0:
                    # Inelastic collision + friction
                    restitution = 0.3 # Dampen bouncing
                    friction = 0.9 # Friction along surface

                    v_normal = v_dot_n * normal
                    v_tangent = v - v_normal

                    velocities[i] = v_tangent * friction - v_normal * restitution

                    # Add some swirl? To simulate flushing?
                    # Tangential force perpendicular to radius
                    radius_vec = np.array([x, y, 0])
                    radius_vec /= np.linalg.norm(radius_vec) + 1e-8
                    tangent_vec = np.array([-y, x, 0]) # Counter-clockwise
                    tangent_vec /= np.linalg.norm(tangent_vec) + 1e-8

                    # Add swirl velocity
                    velocities[i] += tangent_vec * 0.5 * dt

        rollout.append(positions.copy())

    print("Simulation complete.")

    # Convert to numpy arrays
    rollout_np = np.array(rollout) # Shape: (steps+1, particles, 3)

    # Organize for render_rollout.py
    # initial_positions: (N, 3)
    # predicted_rollout: (T, N, 3)
    # target_rollout: (T, N, 3)

    initial_pos = rollout_np[0]
    pred_rollout = rollout_np[1:]
    target_rollout = rollout_np[1:] # Dummy target

    data = {
        "initial_positions": initial_pos,
        "predicted_rollout": pred_rollout,
        "target_rollout": target_rollout
    }

    with open("toilet_flush_rollout.pkl", "wb") as f:
        pickle.dump(data, f)

    print("Saved rollout to toilet_flush_rollout.pkl")

if __name__ == "__main__":
    generate_toilet_flush_simulation()
