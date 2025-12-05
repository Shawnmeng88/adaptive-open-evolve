"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a packing of 26 equal circles inside the unit square.
    Uses a hexagonal lattice, selects the most interior 26 points,
    and solves a linear program that maximises the sum of (potentially
    non‑uniform) radii while respecting border and non‑overlap constraints.
    Returns:
        centers (np.ndarray shape (26,2))
        radii   (np.ndarray shape (26,))
        sum_radii (float)
    """
    import numpy as np
    from scipy.optimize import linprog

    # ------------------------------------------------------------------
    # 1. Generate a hexagonal lattice with a spacing that yields at least 26 points
    # ------------------------------------------------------------------
    def _hex_grid(d):
        """Return points of a hexagonal lattice with spacing d inside [0,1]^2."""
        margin = d / 2.0
        y_step = d * np.sqrt(3) / 2.0
        pts = []
        row = 0
        y = margin
        while y <= 1.0 - margin + 1e-12:
            offset = (d / 2.0) if (row % 2) else 0.0
            x = margin + offset
            while x <= 1.0 - margin + 1e-12:
                pts.append([x, y])
                x += d
            y += y_step
            row += 1
        return np.array(pts)

    # binary search for the largest spacing that still provides ≥26 points
    lo, hi = 0.0, 0.3
    for _ in range(30):
        mid = (lo + hi) / 2.0
        if _hex_grid(mid).shape[0] >= 26:
            lo = mid
        else:
            hi = mid
    d_opt = lo
    all_pts = _hex_grid(d_opt)

    # ------------------------------------------------------------------
    # 2. Pick the 26 points with the largest distance to the square border
    # ------------------------------------------------------------------
    border_limits = np.minimum.reduce(
        [all_pts[:, 0], all_pts[:, 1], 1.0 - all_pts[:, 0], 1.0 - all_pts[:, 1]]
    )
    # indices of points sorted by border margin (largest first)
    idx_sorted = np.argsort(-border_limits)
    selected_idx = idx_sorted[:26]
    centers = all_pts[selected_idx]
    n = centers.shape[0]

    # ------------------------------------------------------------------
    # 3. Build LP to maximise the sum of radii
    # ------------------------------------------------------------------
    # Border constraints: ri ≤ distance to nearest side
    b_border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
    )
    A_border = np.eye(n)

    # Pairwise distance constraints: ri + rj ≤ dist_ij
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))

    pair_rows = []
    pair_b = []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            pair_rows.append(row)
            pair_b.append(dists[i, j])

    if pair_rows:
        A_pair = np.vstack(pair_rows)
        b_pair = np.array(pair_b)
        A_ub = np.vstack((A_border, A_pair))
        b_ub = np.concatenate((b_border, b_pair))
    else:
        A_ub = A_border
        b_ub = b_border

    # Objective: maximise sum(radii)  → minimise -sum(radii)
    c = -np.ones(n)
    bounds = [(0, None) for _ in range(n)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.success:
        radii = res.x
    else:
        # fallback: uniform radius limited by spacing and border
        uniform_r = min(d_opt / 2.0, np.min(b_border))
        radii = np.full(n, uniform_r)

    sum_radii = float(np.sum(radii))
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
