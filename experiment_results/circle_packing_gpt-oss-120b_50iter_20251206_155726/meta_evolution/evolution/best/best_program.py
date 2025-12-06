"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    """
    Construct a packing of 26 circles in the unit square.
    Uses a 5×5 regular grid (spacing 0.2, offset 0.1) giving 25 circles of
    radius 0.1, and inserts one additional small circle in the centre of a
    grid cell to increase the total sum of radii while keeping the packing
    valid.
    """
    # Grid parameters
    rows, cols = 5, 5
    spacing = 0.2
    offset = 0.1  # distance from each border = radius of grid circles

    # Generate 5×5 grid points
    xs = offset + np.arange(cols) * spacing
    ys = offset + np.arange(rows) * spacing
    grid_centers = np.array([[x, y] for y in ys for x in xs])

    # Radii for the grid circles (maximal given spacing and border)
    grid_radii = np.full(grid_centers.shape[0], spacing / 2.0)  # 0.1

    # ------------------------------------------------------------------
    # Add a 26th circle in the centre of a grid cell.
    # Candidate positions are the centres of the (rows‑1)×(cols‑1) cells.
    # We pick the first candidate that yields a positive radius.
    # ------------------------------------------------------------------
    cell_offset = offset + spacing / 2.0
    extra_center = None
    extra_radius = 0.0

    for i in range(rows - 1):
        for j in range(cols - 1):
            cand = np.array([cell_offset + j * spacing,
                             cell_offset + i * spacing])
            # distance to the nearest existing grid centre
            dists = np.linalg.norm(grid_centers - cand, axis=1)
            d_min = dists.min()
            # maximal radius that fits together with a grid circle of radius 0.1
            r_candidate = d_min - spacing / 2.0
            if r_candidate > 1e-6:  # accept the first feasible spot
                extra_center = cand
                extra_radius = r_candidate
                break
        if extra_center is not None:
            break

    # If for some reason no spot was found (should not happen), fall back to a tiny circle
    if extra_center is None:
        extra_center = np.array([0.5, 0.5])
        extra_radius = 0.01

    # Combine grid circles with the extra one
    centers = np.vstack([grid_centers, extra_center])
    radii = np.concatenate([grid_radii, np.array([extra_radius])])

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
