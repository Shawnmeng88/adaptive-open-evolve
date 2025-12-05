"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a dense arrangement of 26 circles using a regular grid
    and compute the optimal radii via linear programming.
    """
    # Regular 5×5 grid with spacing 0.2, starting at 0.1
    xs = np.arange(0.1, 1.0, 0.2)  # [0.1, 0.3, 0.5, 0.7, 0.9]
    ys = xs.copy()
    grid_points = np.array([[x, y] for y in ys for x in xs])  # 25 points

    # Add one extra point to reach 26 circles
    extra_point = np.array([[0.2, 0.8]])
    centers = np.vstack([grid_points, extra_point])

    # Compute maximal radii for this configuration
    radii = compute_max_radii(centers)

    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """
    Solve a linear program to maximize the sum of radii subject to
    border and non‑overlap constraints.
    """
    n = centers.shape[0]

    # Distance from each center to the square borders
    border_dist = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
    )

    # Pairwise Euclidean distances
    diff = centers[:, None, :] - centers[None, :, :]
    pair_dist = np.sqrt(np.sum(diff ** 2, axis=2))

    # Build inequality matrix A_ub * r <= b_ub
    # Border constraints: r_i <= border_dist_i
    A_rows = []
    b_vals = []

    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_rows.append(row)
        b_vals.append(border_dist[i])

    # Non‑overlap constraints: r_i + r_j <= d_ij
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_vals.append(pair_dist[i, j])

    A_ub = np.vstack(A_rows)
    b_ub = np.array(b_vals)

    # Objective: maximize sum(r)  → minimize -sum(r)
    c = -np.ones(n)

    # Bounds: radii are non‑negative
    bounds = [(0.0, None)] * n

    # Solve the LP using the high‑performance HiGHS solver
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.success:
        return result.x
    # Fallback to the simple scaling method if LP fails
    return _fallback_compute_max_radii(centers)


def _fallback_compute_max_radii(centers):
    """Simple pairwise scaling fallback (original implementation)."""
    n = centers.shape[0]
    radii = np.ones(n)

    # Border limits
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Pairwise scaling
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii
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
