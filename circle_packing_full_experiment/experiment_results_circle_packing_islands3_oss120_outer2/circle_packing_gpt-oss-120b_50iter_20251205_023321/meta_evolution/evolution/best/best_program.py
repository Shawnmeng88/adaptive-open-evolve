# EVOLVE-BLOCK-START
import math, random, numpy as np

def _hex_positions(radius):
    """
    Generate circle centers in a hexagonal (triangular) lattice
    with the given equal radius, confined to the unit square.
    """
    eps = 1e-9
    step_x = 2 * radius
    step_y = math.sqrt(3) * radius
    centers = []

    row = 0
    y = radius
    while y <= 1 - radius + eps:
        offset = radius if row % 2 == 0 else radius + radius
        x = offset
        while x <= 1 - radius + eps:
            centers.append((x, y))
            x += step_x
        row += 1
        y += step_y
    return np.array(centers)


def _compute_radii(centers):
    """
    For a fixed set of centers compute the maximal non‑overlapping radii
    limited by the unit‑square borders and by half the pairwise distances.
    """
    # distance to the four sides
    border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )

    # pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)          # ignore self‑distance
    neighbor = np.min(dists, axis=1) / 2.0   # half the nearest neighbour distance

    radii = np.minimum(border, neighbor)
    radii = np.maximum(radii, 0.0)           # guard against tiny negatives
    return radii


def construct_packing():
    """
    Build a packing of 26 circles.
    Starts from a dense hexagonal lattice (uniform radius) and then
    performs a lightweight stochastic local optimisation that moves
    individual centers to increase the total sum of radii.
    Returns (centers, radii, sum_of_radii).
    """
    random.seed(0)
    np.random.seed(0)

    target_n = 26

    # ---- 1) binary search for the largest uniform radius that yields ≥26 points
    lo, hi = 0.0, 0.5
    for _ in range(30):
        mid = (lo + hi) / 2.0
        pts = _hex_positions(mid)
        if pts.shape[0] >= target_n:
            lo = mid
        else:
            hi = mid
    best_r = lo

    # initial centers: first 26 points of the lattice
    centers = _hex_positions(best_r)[:target_n].copy()

    # ---- 2) stochastic local optimisation
    max_iter = 5000
    step = 0.05           # initial perturbation magnitude
    best_centers = centers.copy()
    best_radii = _compute_radii(best_centers)
    best_sum = best_radii.sum()

    for it in range(max_iter):
        # gradually shrink step size
        if it % 1000 == 0 and it > 0:
            step *= 0.5

        i = random.randrange(target_n)
        proposal = best_centers[i] + np.random.uniform(-step, step, size=2)
        proposal = np.clip(proposal, 0.0, 1.0)

        new_centers = best_centers.copy()
        new_centers[i] = proposal

        new_radii = _compute_radii(new_centers)
        new_sum = new_radii.sum()

        if new_sum > best_sum + 1e-8:   # accept only if improvement is noticeable
            best_centers = new_centers
            best_radii = new_radii
            best_sum = new_sum

    sum_radii = float(best_sum)
    return best_centers, best_radii, sum_radii
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
