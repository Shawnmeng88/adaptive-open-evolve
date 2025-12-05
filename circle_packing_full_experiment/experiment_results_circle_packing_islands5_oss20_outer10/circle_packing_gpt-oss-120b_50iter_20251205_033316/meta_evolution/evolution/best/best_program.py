# EVOLVE-BLOCK-START
import numpy as np

def _assert_valid(centers: np.ndarray, radii: np.ndarray, eps: float = 1e-12) -> None:
    """Validate packing constraints; raise AssertionError on violation."""
    if centers.shape[0] != radii.shape[0] or centers.shape[1] != 2:
        raise AssertionError("Shape mismatch between centres and radii")
    if not np.all((centers >= -eps) & (centers <= 1.0 + eps)):
        raise AssertionError("A centre lies outside the unit square")
    if np.any(radii < -eps):
        raise AssertionError("Negative radius encountered")
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.any(radii[:, None] + radii[None, :] - dists > eps):
        raise AssertionError("Overlap detected between circles")

def compute_max_radii(centers: np.ndarray, max_iter: int = 500, eps: float = 1e-12) -> np.ndarray:
    """
    Determine the largest feasible radii for a fixed set of centres.
    Consists of a reduction phase (shrinking offending circles) followed
    by a short expansion phase.
    """
    n = centers.shape[0]

    # 1. wall limits
    edge = np.minimum.reduce([
        centers[:, 0],               # left
        centers[:, 1],               # bottom
        1.0 - centers[:, 0],         # right
        1.0 - centers[:, 1]          # top
    ])
    radii = edge.copy()

    # pairwise distance matrix (inf on diagonal)
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)

    # --------- Reduction phase ----------
    for _ in range(max_iter):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                max_sum = dists[i, j] - eps
                if radii[i] + radii[j] > max_sum:
                    if radii[i] >= radii[j]:
                        new_i = max_sum - radii[j]
                        if new_i < radii[i]:
                            radii[i] = max(new_i, eps)
                            changed = True
                    else:
                        new_j = max_sum - radii[i]
                        if new_j < radii[j]:
                            radii[j] = max(new_j, eps)
                            changed = True
        if not changed:
            break

    # --------- Expansion phase ----------
    for _ in range(5):
        increased = False
        for i in range(n):
            wall_limit = edge[i]
            neighbor_limits = dists[i] - radii
            limit = min(wall_limit, np.min(neighbor_limits))
            if limit > radii[i] + eps:
                radii[i] = limit
                increased = True
        if not increased:
            break

    return np.clip(radii, 0.0, None)

def _hill_climb(centers: np.ndarray, radii: np.ndarray, steps: int = 2000, delta: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple stochastic hill‑climbing: repeatedly propose a small random move
    for a randomly chosen centre and keep it if the total sum of radii
    improves.
    """
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = float(np.sum(best_radii))

    n = centers.shape[0]
    rng = np.random.default_rng()

    for _ in range(steps):
        i = rng.integers(n)
        proposal = best_centers[i] + delta * (rng.random(2) - 0.5)
        proposal = np.clip(proposal, 0.0, 1.0)

        new_centers = best_centers.copy()
        new_centers[i] = proposal
        new_radii = compute_max_radii(new_centers)
        new_sum = float(np.sum(new_radii))

        if new_sum > best_sum + 1e-12:
            best_centers = new_centers
            best_radii = new_radii
            best_sum = new_sum

    return best_centers, best_radii

def construct_packing():
    """
    Construct a packing of 26 circles in the unit square.
    Starts from a 5×5 grid (centre removed) plus two extra circles,
    then refines positions using a hill‑climbing optimisation.
    Returns (centers, radii, sum_radii).
    """
    # 5×5 grid coordinates: 0.1, 0.3, 0.5, 0.7, 0.9
    grid = np.linspace(0.1, 0.9, 5)
    centers = np.array([[x, y] for y in grid for x in grid], dtype=float)

    # discard exact centre (0.5, 0.5)
    mask = ~((np.abs(centers[:, 0] - 0.5) < 1e-12) & (np.abs(centers[:, 1] - 0.5) < 1e-12))
    centers = centers[mask]

    # add two extra circles horizontally displaced from centre
    delta_extra = 0.07
    extra = np.array([[0.5 - delta_extra, 0.5],
                     [0.5 + delta_extra, 0.5]], dtype=float)
    centers = np.vstack([centers, extra])

    # initial radii
    radii = compute_max_radii(centers)

    # stochastic optimisation to increase total radius sum
    centers, radii = _hill_climb(centers, radii, steps=3000, delta=0.025)

    # final safety validation
    _assert_valid(centers, radii)

    return centers, radii, float(np.sum(radii))
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
