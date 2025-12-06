"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    using a linear program:
        maximize sum(r_i)
        subject to:
            r_i <= distance to each square side
            r_i + r_j <= distance between centers i and j
            r_i >= 0
    Returns:
        np.ndarray of radii (shape: (n,))
    """
    import numpy as np
    from scipy.optimize import linprog

    n = centers.shape[0]

    # distance from each center to the four borders
    border_limits = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )

    # Build inequality matrix A_ub * r <= b_ub
    # 1) border constraints: r_i <= border_limits_i
    A_ub = np.eye(n)
    b_ub = border_limits.copy()

    # 2) pairwise non‑overlap constraints: r_i + r_j <= dist_ij
    pair_rows = []
    pair_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            pair_rows.append(row)
            pair_vals.append(dist)

    if pair_rows:
        A_ub = np.vstack([A_ub, np.array(pair_rows)])
        b_ub = np.concatenate([b_ub, np.array(pair_vals)])

    # Objective: maximize sum(r_i)  => minimize -sum(r_i)
    c = -np.ones(n)

    # Bounds: radii are non‑negative
    bounds = [(0, None)] * n

    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.success:
        return res.x
    # Fallback (should rarely happen): use border limits
    return border_limits


def construct_packing():
    """
    Construct a packing of 26 circles inside the unit square.
    Uses a 5×5 regular grid (25 circles) plus a dummy circle at a corner.
    Radii are obtained by solving a linear program that maximises their sum
    while respecting border and non‑overlap constraints.
    Returns:
        centers (np.ndarray, shape (26,2)),
        radii   (np.ndarray, shape (26,)),
        sum_radii (float)
    """
    import numpy as np

    # 5×5 grid spaced evenly between 0.1 and 0.9 (inclusive)
    grid_vals = np.linspace(0.1, 0.9, 5)
    xv, yv = np.meshgrid(grid_vals, grid_vals)
    grid_points = np.column_stack([xv.ravel(), yv.ravel()])  # 25 points

    # Add an extra dummy circle at a corner (radius will be forced to 0)
    extra_point = np.array([[0.0, 0.0]])
    centers = np.vstack([grid_points, extra_point])  # shape (26,2)

    # Compute optimal radii for these fixed centers
    radii = compute_max_radii(centers)

    sum_radii = np.sum(radii)
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
