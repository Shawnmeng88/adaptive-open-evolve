"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct an arrangement of 26 circles in a unit square
    and compute radii that maximise the total sum while keeping
    circles inside the square and non‑overlapping.
    """
    import numpy as np

    n = 26
    # ------------------------------------------------------------------
    # Generate a hexagonal‑like lattice of points inside the unit square.
    # The pattern is deterministic and yields 26 distinct positions.
    # ------------------------------------------------------------------
    spacing = 0.20
    centers = []
    y = 0.10
    row = 0
    while len(centers) < n:
        cols = 5 if row % 2 == 0 else 4
        for c in range(cols):
            x = 0.10 + (c + (0.5 if row % 2 else 0.0)) * spacing
            if x > 0.90:
                continue
            centers.append([x, y])
            if len(centers) == n:
                break
        y += spacing * np.sqrt(3) / 2  # vertical offset for hex packing
        row += 1

    centers = np.array(centers[:n])

    # ------------------------------------------------------------------
    # Solve a linear program that maximises the sum of radii.
    # Variables: r_i for each circle.
    # Constraints:
    #   * r_i <= distance from centre to the four borders
    #   * r_i + r_j <= distance between centres i and j
    # ------------------------------------------------------------------
    radii = _max_sum_radii_lp(centers)

    sum_radii = radii.sum()
    return centers, radii, sum_radii


def _max_sum_radii_lp(centers):
    """
    Linear‑programming solution for the maximal total radius.
    Returns an array of radii (length = number of centres).
    """
    import numpy as np
    from scipy.optimize import linprog

    n = len(centers)

    # Border limits for each centre
    border_limits = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )

    # Pairwise centre distances
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)

    # Objective: maximise sum(r)  -> minimise -sum(r)
    c = -np.ones(n)

    # Inequality matrix A_ub * r <= b_ub
    # 1) Border constraints  r_i <= border_limits_i
    A_rows = [np.eye(n)]
    b_vals = [border_limits]

    # 2) Non‑overlap constraints  r_i + r_j <= d_ij  (for i < j)
    pair_rows = []
    pair_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            pair_rows.append(row)
            pair_vals.append(dists[i, j])

    if pair_rows:
        A_rows.append(np.vstack(pair_rows))
        b_vals.append(np.array(pair_vals))

    A_ub = np.vstack(A_rows)
    b_ub = np.concatenate(b_vals)

    bounds = [(0, None)] * n

    # Use the high‑performance HiGHS solver
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.success:
        return result.x
    # Fallback: use the most restrictive border limits if LP fails
    return border_limits
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
