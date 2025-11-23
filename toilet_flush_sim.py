import argparse
import pickle
import toilet_simulation

def main():
    parser = argparse.ArgumentParser(description="Run and optionally animate the toilet flush simulation.")
    parser.add_argument("--output", type=str, default="toilet_flush_rollout.pkl", help="Output pickle file.")
    parser.add_argument("--gif", type=str, default=None, help="Output GIF file (optional).")
    parser.add_argument("--particles", type=int, default=300, help="Number of particles.")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps.")
    args = parser.parse_args()

    print(f"Running simulation with {args.particles} particles for {args.steps} steps...")
    data = toilet_simulation.simulate(num_particles=args.particles, num_steps=args.steps)

    print(f"Saving simulation data to {args.output}...")
    with open(args.output, "wb") as f:
        pickle.dump(data, f)

    if args.gif:
        print(f"Rendering animation to {args.gif}...")
        anim = toilet_simulation.animate(data, sim_name="Toilet Flush")
        anim.save(args.gif, writer="pillow", fps=10)
        print("Done.")

if __name__ == "__main__":
    main()
