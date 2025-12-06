"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a deterministic arrangement of 26 circles in a unit square
    and compute the optimal radii using a linear program that maximizes
    the total sum of radii while respecting border and non‑overlap constraints.
    """
    # ---- generate 26 deterministic centre positions (grid, first 26 cells) ----
    grid_x = np.linspace(0.1, 0.9, 6)   # 6 columns
    grid_y = np.linspace(0.1, 0.9, 5)   # 5 rows
    centers = np.array([[x, y] for y in grid_y for x in grid_x])[:26]

    # ---- compute optimal radii via linear programming ----
    radii = compute_max_radii(centers)

    # ---- total sum of radii ----
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Solve a linear program to obtain the largest possible radii for the given
    centre positions.  Constraints:
        * 0 ≤ r_i ≤ distance from centre i to the nearest square edge
        * r_i + r_j ≤ Euclidean distance between centres i and j  (no overlap)
    The objective is to maximise Σ r_i.
    """
    n = centers.shape[0]

    # ----- border constraints (r_i ≤ min distance to any side) -----
    border_limits = np.minimum.reduce([
        centers[:, 0],               # distance to left side
        centers[:, 1],               # distance to bottom side
        1.0 - centers[:, 0],         # distance to right side
        1.0 - centers[:, 1]          # distance to top side
    ])

    # ----- pairwise distance constraints (r_i + r_j ≤ d_ij) -----
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))

    # Build A_ub and b_ub for linprog (A_ub @ r ≤ b_ub)
    # Start with border constraints
    A_ub = np.eye(n)
    b_ub = border_limits.copy()

    # Append pairwise constraints
    pair_rows = []
    pair_bounds = []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            pair_rows.append(row)
            pair_bounds.append(dists[i, j])

    if pair_rows:
        A_ub = np.vstack([A_ub, np.array(pair_rows)])
        b_ub = np.concatenate([b_ub, np.array(pair_bounds)])

    # Objective: maximize sum(r)  → minimize -sum(r)
    c = -np.ones(n)

    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')

    if res.success:
        return res.x
    else:
        # Fallback to a simple heuristic if LP fails
        radii = np.minimum(border_limits, 0.05)
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
