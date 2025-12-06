"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a layout for 26 circles inside the unit square and compute the
    maximum possible radii using a linear‑programming formulation.
    """
    n = 26

    # --- Generate a reasonably spread set of centre positions -----------------
    # Use a hexagonal‑like grid with a little random jitter to avoid exact
    # symmetries that could limit the LP solution.
    rows = [5, 6, 5, 5, 5]          # total 26 points
    y_coords = np.linspace(0.1, 0.9, len(rows))
    centers = []

    for r, y in zip(rows, y_coords):
        x_vals = np.linspace(0.1, 0.9, r)
        for x in x_vals:
            # add a tiny random offset (deterministic seed for reproducibility)
            rng = np.random.default_rng(42)
            jitter = rng.uniform(-0.02, 0.02, size=2)
            centers.append(np.clip([x, y] + jitter, 0.01, 0.99))

    centers = np.array(centers[:n])   # ensure exactly n points

    # --- Compute the optimal radii via linear programming --------------------
    radii = _max_radii_lp(centers)

    sum_radii = float(radii.sum())
    return centers, radii, sum_radii


def _max_radii_lp(centers):
    """
    Solve the LP:
        maximise   Σ r_i
        subject to r_i <= distance to the four borders,
                   r_i + r_j <= distance between centres i and j,
                   r_i >= 0
    Returns the optimal radii as a NumPy array.
    """
    n = centers.shape[0]

    # Border constraints
    border_limits = np.minimum.reduce([
        centers[:, 0],                     # distance to left
        centers[:, 1],                     # distance to bottom
        1.0 - centers[:, 0],               # distance to right
        1.0 - centers[:, 1]                # distance to top
    ])

    # Pairwise distance constraints
    pair_idxs = []
    pair_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            pair_idxs.append((i, j))
            pair_dists.append(d)

    # Build A_ub and b_ub
    # Border constraints: r_i <= border_limits[i]
    A_ub = np.zeros((n + len(pair_idxs), n))
    b_ub = np.empty(n + len(pair_idxs))

    # border rows
    for i in range(n):
        A_ub[i, i] = 1.0
        b_ub[i] = border_limits[i]

    # pairwise rows: r_i + r_j <= d_ij
    offset = n
    for k, (i, j) in enumerate(pair_idxs):
        A_ub[offset + k, i] = 1.0
        A_ub[offset + k, j] = 1.0
        b_ub[offset + k] = pair_dists[k]

    # Objective: maximise sum r_i  → minimise -sum r_i
    c = -np.ones(n)

    # Bounds: r_i >= 0 (no explicit upper bound, handled by constraints)
    bounds = [(0, None) for _ in range(n)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not res.success:
        # fallback: very small radii to keep validity
        return np.full(n, 1e-6)

    return np.maximum(res.x, 0.0)  # guard against tiny negative values due to tolerance
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
