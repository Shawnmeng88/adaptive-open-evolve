"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def _pairwise_distances(centers: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix for a set of centers."""
    diff = centers[:, None, :] - centers[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def compute_radii(centers: np.ndarray) -> np.ndarray:
    """
    Simple geometric radii: limited by the square borders and by half the distance
    to the nearest neighbour. Guarantees a feasible packing for the given centers.
    """
    border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
    )
    dists = _pairwise_distances(centers)
    np.fill_diagonal(dists, np.inf)
    nearest = np.min(dists, axis=1)
    return np.minimum(border, nearest / 2.0)


def _initial_centers(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Produce a jittered 5×5 grid (25 points) and one extra random point.
    The grid is placed away from the borders to leave room for radii.
    """
    grid_vals = np.linspace(0.12, 0.88, 5)
    xs, ys = np.meshgrid(grid_vals, grid_vals)
    pts = np.column_stack([xs.ravel(), ys.ravel()])[: n - 1]
    pts += rng.normal(scale=0.02, size=pts.shape)          # slight jitter
    extra = rng.uniform(0.2, 0.8, size=(1, 2))
    centers = np.vstack([pts, extra])
    np.clip(centers, 0.0, 1.0, out=centers)
    return centers


def _lp_optimize(centers: np.ndarray) -> np.ndarray:
    """
    Linear‑programming refinement: maximise sum of radii under
    border and non‑overlap constraints. Falls back to geometric radii
    if the LP fails.
    """
    n = len(centers)
    border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
    )
    dists = _pairwise_distances(centers)
    np.fill_diagonal(dists, np.inf)

    rows = []
    rhs = []

    # border constraints: r_i <= border_i
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        rows.append(row)
        rhs.append(border[i])

    # pairwise non‑overlap: r_i + r_j <= d_ij
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs.append(dists[i, j])

    A = np.vstack(rows)
    b = np.array(rhs)

    res = linprog(
        c=-np.ones(n),          # maximise sum -> minimise negative sum
        A_ub=A,
        b_ub=b,
        bounds=[(0, None)] * n,
        method="highs",
        options={"presolve": True},
    )
    if res.success:
        return res.x
    # fallback
    return compute_radii(centers)


def _valid_radii(centers: np.ndarray, radii: np.ndarray, eps: float = 1e-9) -> bool:
    """Check that radii respect borders and pairwise separation."""
    border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
    )
    if np.any(radii - border > eps):
        return False
    dists = _pairwise_distances(centers)
    i, j = np.triu_indices(len(centers), k=1)
    if np.any(radii[i] + radii[j] - dists[i, j] > eps):
        return False
    return True


def construct_packing():
    """
    Multi‑restart stochastic hill‑climbing.
    Each iteration proposes a small move for a random circle,
    refines radii with LP, and accepts the move only if it improves
    the total sum of radii.
    Returns the best (centers, radii, sum_radii) found.
    """
    n = 26
    rng = np.random.default_rng()
    best_sum = -1.0
    best_centers = None
    best_radii = None

    restarts = 6          # more independent tries
    max_iter = 7000       # iterations per restart
    step_scale = 0.045    # finer perturbations

    for _ in range(restarts):
        centers = _initial_centers(rng, n)
        radii = compute_radii(centers)
        cur_sum = radii.sum()

        for _ in range(max_iter):
            idx = rng.integers(n)
            proposal = centers[idx] + rng.normal(scale=step_scale, size=2)
            np.clip(proposal, 0.0, 1.0, out=proposal)

            new_centers = centers.copy()
            new_centers[idx] = proposal

            # LP gives tighter feasible radii
            new_radii = _lp_optimize(new_centers)

            # safety fallback if LP yields infeasible set
            if not _valid_radii(new_centers, new_radii):
                new_radii = compute_radii(new_centers)

            new_sum = new_radii.sum()
            if new_sum > cur_sum:
                centers, radii, cur_sum = new_centers, new_radii, new_sum

        if cur_sum > best_sum:
            best_sum = cur_sum
            best_centers = centers.copy()
            best_radii = radii.copy()

    # final safety check
    if not _valid_radii(best_centers, best_radii):
        best_radii = compute_radii(best_centers)

    return best_centers, best_radii, float(best_sum)
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
