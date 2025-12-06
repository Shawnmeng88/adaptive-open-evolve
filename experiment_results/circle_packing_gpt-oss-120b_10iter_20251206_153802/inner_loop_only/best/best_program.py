"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a layout of 26 circles inside the unit square and compute
    the optimal radii that maximize the total sum while respecting
    non‑overlap and border constraints.
    """
    # Create a 5×5 grid (25 circles) uniformly spaced inside the square
    grid = np.linspace(0.1, 0.9, 5)
    xv, yv = np.meshgrid(grid, grid)
    centers = np.column_stack((xv.ravel(), yv.ravel()))

    # Add a central circle as the 26th element
    centers = np.vstack((centers, [0.5, 0.5]))

    # Compute the radii that give the maximal total sum
    radii = compute_optimal_radii(centers)

    return centers, radii, radii.sum()


def compute_optimal_radii(centers):
    """
    Solve a linear program to find the largest possible radii for the given
    circle centers.

    Constraints:
        * 0 ≤ r_i ≤ distance from centre i to the nearest square edge
        * r_i + r_j ≤ distance between centres i and j  (no overlap)

    Returns:
        np.ndarray of optimal radii (length = number of centers)
    """
    n = len(centers)

    # Maximum radius allowed by the square borders for each centre
    border_limits = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )

    # Pairwise centre distances
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)

    # Build inequality matrix A_ub and vector b_ub such that A_ub @ r <= b_ub
    A_rows = []
    b_vals = []

    # Border constraints: r_i <= border_limits[i]
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1
        A_rows.append(row)
        b_vals.append(border_limits[i])

    # Non‑overlap constraints: r_i + r_j <= dists[i, j]
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_rows.append(row)
            b_vals.append(dists[i, j])

    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)

    # Objective: maximize sum(r)  →  minimize -sum(r)
    c = -np.ones(n)

    # Solve the LP
    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(0, None)] * n,
        method="highs",
    )

    # If the solver fails, fall back to the border‑limited radii
    if not result.success:
        return border_limits

    return result.x
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
