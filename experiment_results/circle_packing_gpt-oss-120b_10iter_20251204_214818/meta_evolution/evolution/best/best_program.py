# EVOLVE-BLOCK-START
"""Improved circle packing for n=26 circles.

The layout is a two‑ring pattern (inner and outer) with a central circle.
Radii are now obtained by solving a linear program that maximises the
total sum of radii while respecting border and non‑overlap constraints.
"""
import numpy as np
from scipy.optimize import linprog


def construct_packing():
    """
    Construct an arrangement of 26 circles inside the unit square.

    Returns
    -------
    centers : np.ndarray, shape (26, 2)
        (x, y) coordinates of the circle centres.
    radii : np.ndarray, shape (26,)
        Optimised radii for the given centres.
    sum_radii : float
        Sum of all radii (the optimisation objective).
    """
    n = 26
    centers = np.zeros((n, 2))

    # 1) central circle
    centers[0] = [0.5, 0.5]

    # 2) inner ring – 8 circles
    inner_radius = 0.25          # distance from centre to inner‑ring centres
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [
            0.5 + inner_radius * np.cos(angle),
            0.5 + inner_radius * np.sin(angle),
        ]

    # 3) outer ring – 16 circles
    outer_radius = 0.45          # distance from centre to outer‑ring centres
    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [
            0.5 + outer_radius * np.cos(angle),
            0.5 + outer_radius * np.sin(angle),
        ]

    # Ensure every centre stays inside the square with a tiny safety margin.
    centers = np.clip(centers, 0.01, 0.99)

    # Compute the optimal radii for this fixed set of centres.
    radii = compute_max_radii(centers)

    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Solve a linear programme that maximises the sum of radii subject to
    border and non‑overlap constraints.

    Parameters
    ----------
    centers : np.ndarray, shape (n, 2)
        Circle centre coordinates.

    Returns
    -------
    radii : np.ndarray, shape (n,)
        The optimal radii (or a fallback greedy estimate if the LP fails).
    """
    n = centers.shape[0]

    # ---- 1) Upper bounds imposed by the square borders -----------------
    # distance from each centre to the four sides of the unit square
    border_limits = np.minimum.reduce(
        [
            centers[:, 0],          # left side
            centers[:, 1],          # bottom side
            1.0 - centers[:, 0],    # right side
            1.0 - centers[:, 1],    # top side
        ]
    )

    # ---- 2) Pairwise distance constraints (r_i + r_j <= d_ij) ---------
    # Compute the full distance matrix once.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)

    # Build the A_ub matrix (one row per unordered pair i<j)
    A_ub = []
    b_ub = []
    for i in range(n):
        for j in range(i + 1, n):
            coeff = np.zeros(n)
            coeff[i] = 1.0
            coeff[j] = 1.0
            A_ub.append(coeff)
            b_ub.append(dists[i, j])

    A_ub = np.array(A_ub, dtype=float)
    b_ub = np.array(b_ub, dtype=float)

    # ---- 3) Linear programme -------------------------------------------
    # maximise sum(r)  <=>  minimise -sum(r)
    c = -np.ones(n, dtype=float)

    bounds = [(0.0, border_limits[i]) for i in range(n)]

    # Using the HiGHS solver (fast and deterministic)
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )

    if res.success:
        radii = np.maximum(res.x, 0.0)          # guard against tiny negatives
    else:
        # Fallback: the original greedy scaling approach (still feasible)
        radii = _greedy_radii(centers, border_limits)

    return radii


def _greedy_radii(centers, border_limits):
    """
    Simple greedy fallback used only when the linear programme fails.
    It mimics the original behaviour: start with the border limits and
    repeatedly scale overlapping pairs.
    """
    n = centers.shape[0]
    radii = border_limits.copy()

    # Pairwise scaling – repeat a few passes to propagate adjustments.
    for _ in range(5):
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

    plt.title(f"Circle Packing (n={len(centers)}, sum={np.sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")

    # Uncomment to visualise the result:
    # visualize(centers, radii)