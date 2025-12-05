"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a packing of 26 circles inside the unit square.
    Starts from a 5×5 regular grid (25 circles) with radius 0.1,
    then adds a 26th circle in the largest empty inter‑grid gap.
    """
    import numpy as np

    # 5×5 grid points at 0.1, 0.3, …, 0.9
    coords = np.arange(0.1, 1.0, 0.2)
    grid = np.array([[x, y] for y in coords for x in coords], dtype=float)  # (25, 2)

    # radii for the grid circles – maximal equal radius for this arrangement
    radii_grid = np.full(25, 0.1, dtype=float)

    # ------------------------------------------------------------
    # Find the best position for the 26th circle:
    #   candidate centres are the centres of the 4×4 empty squares
    #   formed by the grid points.
    # ------------------------------------------------------------
    best_center = None
    best_radius = -1.0

    for i in range(len(coords) - 1):
        for j in range(len(coords) - 1):
            cx = (coords[i] + coords[i + 1]) / 2.0
            cy = (coords[j] + coords[j + 1]) / 2.0

            # distance to the four walls
            wall_dist = min(cx, cy, 1.0 - cx, 1.0 - cy)

            # distance to each existing grid circle, reduced by its radius
            dists = np.linalg.norm(grid - np.array([cx, cy]), axis=1) - radii_grid
            max_r = min(wall_dist, dists.min())

            if max_r > best_radius:
                best_radius = max_r
                best_center = np.array([cx, cy])

    # If no positive radius was found (should not happen), fall back to a dummy circle
    if best_center is None or best_radius <= 0.0:
        extra_center = grid[0].copy()
        extra_radius = 0.0
    else:
        extra_center = best_center
        extra_radius = best_radius

    # Assemble final arrays
    centers = np.vstack([grid, extra_center])
    radii = np.concatenate([radii_grid, [extra_radius]])

    sum_radii = float(radii.sum())
    return centers, radii, sum_radii
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)
