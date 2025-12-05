# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def _hex_grid_positions(r):
    """Generate hexagonal lattice points inside the unit square with margin r."""
    dx = 2 * r
    dy = np.sqrt(3) * r
    pts = []
    y = r
    row = 0
    while y <= 1 - r + 1e-12:
        offset = 0.0 if row % 2 == 0 else r
        x = r + offset
        while x <= 1 - r + 1e-12:
            pts.append([x, y])
            x += dx
        row += 1
        y += dy
    return np.array(pts)

def construct_packing():
    """
    Construct a packing of 26 circles inside the unit square.
    Uses a hexagonal lattice to obtain 26 well‑spaced centers,
    then solves a linear program to maximise individual radii.
    Returns:
        centers (np.ndarray): (26, 2) array of circle centres
        radii   (np.ndarray): (26,) array of radii
        sum_radii (float)   : total sum of radii
    """
    # --- binary search for maximal lattice spacing that yields ≥26 points ---
    lo, hi = 0.0, 0.5
    best_r = 0.0
    best_pts = None
    for _ in range(30):
        mid = (lo + hi) / 2.0
        pts = _hex_grid_positions(mid)
        if pts.shape[0] >= 26:
            best_r = mid
            best_pts = pts
            lo = mid
        else:
            hi = mid

    if best_pts is None:
        raise RuntimeError("Failed to generate any packing points.")

    # deterministic ordering (by y then x) and pick first 26 points
    idx = np.lexsort((best_pts[:, 0], best_pts[:, 1]))
    centers = best_pts[idx][:26].copy()

    # maximise radii for these fixed centres using LP
    radii = compute_max_radii_lp(centers)
    sum_radii = float(radii.sum())
    return centers, radii, sum_radii

def compute_max_radii_lp(centers):
    """
    Solve a linear program that maximises the sum of radii
    subject to wall and non‑overlap constraints.
    """
    n = centers.shape[0]
    c = -np.ones(n)                     # maximise sum → minimise -sum
    # Wall limits
    wall_limits = np.minimum.reduce([
        centers[:, 0],
        1.0 - centers[:, 0],
        centers[:, 1],
        1.0 - centers[:, 1]
    ])
    A_wall = np.eye(n)
    # Pairwise constraints
    pair_rows = []
    pair_rhs = []
    for i in range(n):
        for j in range(i + 1, n):
            dij = np.linalg.norm(centers[i] - centers[j])
            coeff = np.zeros(n)
            coeff[i] = 1.0
            coeff[j] = 1.0
            pair_rows.append(coeff)
            pair_rhs.append(dij)
    if pair_rows:
        A_pair = np.vstack(pair_rows)
        A_ub = np.vstack([A_wall, A_pair])
        b_ub = np.concatenate([wall_limits, np.array(pair_rhs)])
    else:
        A_ub = A_wall
        b_ub = wall_limits
    bounds = [(0, None) for _ in range(n)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not res.success:
        # fallback to wall‑limited radii
        return np.minimum.reduce([
            centers[:, 0],
            1.0 - centers[:, 0],
            centers[:, 1],
            1.0 - centers[:, 1]
        ])
    return res.x
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
