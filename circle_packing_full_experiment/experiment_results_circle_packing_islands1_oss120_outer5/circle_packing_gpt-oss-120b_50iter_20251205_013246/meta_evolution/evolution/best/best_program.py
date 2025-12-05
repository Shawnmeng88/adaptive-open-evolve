# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog


def _hex_grid_centers(spacing: float, n: int) -> np.ndarray:
    """Generate at least *n* points on a hexagonal lattice inside the unit square."""
    margin = spacing / 2.0
    v_spacing = spacing * np.sqrt(3) / 2.0
    pts = []
    row = 0
    y = margin
    while y <= 1 - margin and len(pts) < n:
        offset = margin + (row % 2) * (spacing / 2.0)
        x = offset
        while x <= 1 - margin and len(pts) < n:
            pts.append([x, y])
            x += spacing
        row += 1
        y += v_spacing
    return np.array(pts[:n])


def _optimal_radii_lp(centers: np.ndarray) -> np.ndarray:
    """Solve a linear program that maximises the sum of radii for fixed centres."""
    n = centers.shape[0]

    # distance to the four walls
    wall = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )

    # pairwise centre distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))

    rows = []
    rhs = []

    # wall constraints  r_i <= wall_i
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        rows.append(row)
        rhs.append(wall[i])

    # non‑overlap constraints  r_i + r_j <= d_ij
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs.append(dists[i, j])

    A = np.vstack(rows)
    b = np.array(rhs)

    # maximise sum(r)  → minimise -sum(r)
    c = -np.ones(n)
    var_bounds = [(0, None)] * n

    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=var_bounds, method="highs")
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass

    # fallback heuristic
    return compute_max_radii(centers)


def compute_max_radii(centers: np.ndarray) -> np.ndarray:
    """Simple deterministic fallback: wall distance vs. half nearest neighbour distance."""
    n = centers.shape[0]
    wall = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    nearest = np.min(dists, axis=1)
    return np.minimum(wall, nearest / 2.0)


def construct_packing():
    """
    Construct a packing of 26 circles inside the unit square.
    Several deterministic centre layouts are generated (grid‑based, hexagonal,
    and a dense set of extra‑point candidates). For each layout a linear
    programme yields the optimal radii. The layout with the largest total
    radius sum is returned.
    """
    n = 26
    candidates = []

    # ---------- 5×5 regular grid (margin 0.1) ----------
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 5)
    xv, yv = np.meshgrid(xs, ys)
    grid25 = np.column_stack([xv.ravel(), yv.ravel()])   # shape (25, 2)

    # add a modest set of extra‑point candidates (dense deterministic grid)
    extra_x = np.arange(0.15, 0.86, 0.05)
    extra_y = np.arange(0.15, 0.86, 0.05)
    extra_grid = np.column_stack(np.meshgrid(extra_x, extra_y)).reshape(-1, 2)

    # keep only points that are not already in the 5×5 grid
    def is_new(pt):
        return np.min(np.linalg.norm(grid25 - pt, axis=1)) > 1e-6

    extra_candidates = np.array([pt for pt in extra_grid if is_new(pt)])

    for extra in extra_candidates:
        candidates.append(np.vstack([grid25, extra]))

    # ---------- 6×5 grid without the four corners ----------
    cols, rows = 6, 5
    xs2 = (np.arange(cols) + 0.5) / cols
    ys2 = (np.arange(rows) + 0.5) / rows
    xv2, yv2 = np.meshgrid(xs2, ys2)
    grid30 = np.column_stack([xv2.ravel(), yv2.ravel()])  # 30 points
    corner_idxs = [0, cols - 1, (rows - 1) * cols, rows * cols - 1]
    mask = np.ones(grid30.shape[0], dtype=bool)
    mask[corner_idxs] = False
    candidates.append(grid30[mask])                       # exactly 26 points

    # ---------- Hexagonal lattices with varying spacing ----------
    for spacing in np.arange(0.12, 0.21, 0.005):
        hex_pts = _hex_grid_centers(spacing, n)
        if hex_pts.shape[0] == n:
            candidates.append(hex_pts)

    # ---------- Evaluate all candidates ----------
    best_sum = -1.0
    best_centers = None
    best_radii = None

    for centers in candidates:
        # keep centres safely inside the square
        centers = np.clip(centers, 0.01, 0.99)

        radii = _optimal_radii_lp(centers)
        total = radii.sum()
        if total > best_sum:
            best_sum = total
            best_centers = centers
            best_radii = radii

    return best_centers, best_radii, best_sum
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
