"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a packing of 26 circles inside the unit square.
    Uses multiple random restarts with a hexagonal lattice, hill‑climbing
    refinement and a final tiny‑perturbation stage.
    """
    n_circles = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # number of random restarts
    for _ in range(15):
        # random lattice spacing and orientation
        dx = np.random.uniform(0.08, 0.16)
        pts = _hex_lattice(dx)
        pts = _rotate(pts, np.random.uniform(0.0, np.pi))

        # keep the most interior points
        centers = _select_best(pts, n_circles)

        # hill‑climbing refinement
        centers, radii = _refine(centers, iters=400, step=0.05)

        # tiny random tweaks to escape local plateaus
        centers, radii = _tiny_perturb(centers, radii, tries=30, step=0.01)

        cur_sum = radii.sum()
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_centers = centers
            best_radii = radii

    return best_centers, best_radii, best_sum


def _hex_lattice(dx: float) -> np.ndarray:
    """Generate a hexagonal lattice of points inside the unit square."""
    dy = np.sqrt(3) / 2 * dx
    pts = []
    y = 0.0
    row = 0
    while y <= 1.0:
        offset = (dx / 2) if (row % 2) else 0.0
        x = offset
        while x <= 1.0:
            pts.append([x, y])
            x += dx
        row += 1
        y += dy
    return np.array(pts)


def _rotate(pts: np.ndarray, ang: float) -> np.ndarray:
    """Rotate points around the centre (0.5, 0.5) by angle `ang`."""
    if ang == 0.0:
        return pts
    R = np.array([[np.cos(ang), -np.sin(ang)],
                  [np.sin(ang),  np.cos(ang)]])
    return (pts - 0.5) @ R.T + 0.5


def _select_best(points: np.ndarray, n: int) -> np.ndarray:
    """Select `n` points farthest from the square borders."""
    border_dist = np.minimum.reduce([points[:, 0], points[:, 1],
                                    1 - points[:, 0], 1 - points[:, 1]])
    idx = np.argpartition(-border_dist, n)[:n]
    return points[idx].copy()


def _optimal_radii(centers: np.ndarray) -> np.ndarray:
    """Solve the LP that maximises the sum of radii for given centres."""
    n = len(centers)
    # distance to each side of the square
    border = np.minimum.reduce([centers[:, 0], centers[:, 1],
                               1 - centers[:, 0], 1 - centers[:, 1]])

    # pairwise centre distances
    dists = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))

    # build inequality matrix for r_i + r_j <= d_ij
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(pairs)
    A = np.zeros((m, n))
    b = np.empty(m)
    for k, (i, j) in enumerate(pairs):
        A[k, i] = 1.0
        A[k, j] = 1.0
        b[k] = dists[i, j]

    # linear program: maximise sum(r)  <=> minimise -sum(r)
    c = -np.ones(n)
    bounds = [(0.0, border[i]) for i in range(n)]
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    if res.success:
        return res.x
    # fallback: half of the border distances
    return border * 0.5


def _refine(centers: np.ndarray, iters: int = 200, step: float = 0.04):
    """
    Stochastic hill‑climbing on the centre positions.
    Each move is kept only if it improves the total sum of radii.
    """
    best_c = centers.copy()
    best_r = _optimal_radii(best_c)
    best_sum = best_r.sum()
    n = len(best_c)

    for _ in range(iters):
        i = np.random.randint(n)
        delta = (np.random.rand(2) - 0.5) * 2.0 * step
        new_pos = np.clip(best_c[i] + delta, 0.0, 1.0)

        cand_c = best_c.copy()
        cand_c[i] = new_pos
        cand_r = _optimal_radii(cand_c)
        cand_sum = cand_r.sum()

        if cand_sum > best_sum:
            best_c, best_r, best_sum = cand_c, cand_r, cand_sum

    return best_c, best_r


def _tiny_perturb(centers: np.ndarray, radii: np.ndarray,
                  tries: int = 20, step: float = 0.005):
    """
    Apply very small random adjustments to single centres.
    Keep a change only if it yields a higher total radius sum.
    """
    best_c = centers.copy()
    best_r = radii.copy()
    best_sum = best_r.sum()
    n = len(best_c)

    for _ in range(tries):
        i = np.random.randint(n)
        delta = (np.random.rand(2) - 0.5) * 2.0 * step
        new_pos = np.clip(best_c[i] + delta, 0.0, 1.0)

        cand_c = best_c.copy()
        cand_c[i] = new_pos
        cand_r = _optimal_radii(cand_c)
        cand_sum = cand_r.sum()

        if cand_sum > best_sum:
            best_c, best_r, best_sum = cand_c, cand_r, cand_sum

    return best_c, best_r
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
