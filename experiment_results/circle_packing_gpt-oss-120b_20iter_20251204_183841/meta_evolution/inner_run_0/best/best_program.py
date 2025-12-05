# EVOLVE-BLOCK-START
"""Improved constructor for n=26 circles using a hexagonal lattice and LP radii optimisation"""

import numpy as np
from scipy.optimize import linprog


def construct_packing():
    """
    Construct a dense arrangement of 26 circles inside the unit square.
    Positions are generated on a hexagonal (triangular) lattice and the
    radii are maximised by solving a linear programme:
        maximise   sum_i r_i
        subject to r_i + r_j <= dist(i,j)   (non‑overlap)
                   r_i <= border_i          (stay inside the square)
                   r_i >= 0
    Returns:
        centers (np.ndarray, shape (26,2))
        radii   (np.ndarray, shape (26,))
        sum_radii (float)
    """
    # ---------- 1. Generate a hexagonal lattice of 26 points ----------
    spacing = 0.20                     # centre‑to‑centre distance in x‑direction
    dy = np.sqrt(3) / 2 * spacing      # vertical offset for a triangular lattice
    margin = spacing / 2               # keep points at least this far from the border

    centers = []
    row = 0
    while len(centers) < 26:
        y = margin + dy * row
        if y > 1 - margin:            # stop if we would leave the square
            break
        # offset every second row by half the horizontal spacing
        x_offset = 0.0 if row % 2 == 0 else spacing / 2
        xs = np.arange(margin + x_offset, 1 - margin + 1e-9, spacing)
        for x in xs:
            centers.append([x, y])
            if len(centers) == 26:
                break
        row += 1

    # If the lattice produced fewer than 26 points (unlikely), pad with a centre point
    while len(centers) < 26:
        centers.append([0.5, 0.5])

    centers = np.array(centers, dtype=float)

    # ---------- 2. Optimise radii with a linear programme ----------
    radii = compute_max_radii(centers)

    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Linear‑programming based computation of the largest possible radii
    for a fixed set of centres, respecting the unit‑square borders and
    pairwise non‑overlap constraints.
    """
    n = centers.shape[0]

    # ---- 2.1. Border limits (upper bounds for each radius) ----
    border_limits = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )

    # ---- 2.2. Pairwise distance constraints (r_i + r_j <= d_ij) ----
    # Build A_ub * r <= b_ub
    # There are n*(n-1)/2 constraints, each with two non‑zero entries.
    pair_cnt = n * (n - 1) // 2
    A_ub = np.zeros((pair_cnt, n), dtype=float)
    b_ub = np.zeros(pair_cnt, dtype=float)

    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist_ij = np.linalg.norm(centers[i] - centers[j])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist_ij
            idx += 1

    # ---- 2.3. Linear programme: maximise sum(r)  <=> minimise -sum(r) ----
    c = -np.ones(n)                     # minimise -∑r_i
    bounds = [(0.0, border_limits[i]) for i in range(n)]

    # Use the HiGHS solver (fast & reliable for moderate sizes)
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )

    if res.success:
        return res.x
    # ---- 2.4. Fallback (very unlikely) : simple min‑distance heuristic ----
    radii = border_limits.copy()
    # enforce pairwise constraints greedily
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            max_pair = min(radii[i] + radii[j], d)
            if radii[i] + radii[j] > d:
                # shrink both proportionally
                scale = d / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale
    return radii


# EVOLVE-BLOCK-END


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
        circle = Circle(center, radius, alpha=0.5, edgecolor="k")
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center", fontsize=8)

    plt.title(f"Circle Packing (n={len(centers)}, sum={np.sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # Uncomment to visualise the result:
    # visualize(centers, radii)