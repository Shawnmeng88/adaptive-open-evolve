"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def improve_positions(centers, steps=200, lr=0.01):
    """
    Simple force‑directed refinement.
    Moves points slightly away from the nearest neighbour
    or toward the centre when the border is the limiting factor.
    """
    n = centers.shape[0]
    for _ in range(steps):
        # distance to each side of the unit square
        border = np.minimum.reduce([centers[:, 0], centers[:, 1],
                                    1 - centers[:, 0], 1 - centers[:, 1]])

        # pairwise distances
        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)

        nearest_idx = np.argmin(dists, axis=1)
        nearest_dist = dists[np.arange(n), nearest_idx]

        # decide which constraint is tighter
        border_limited = border < nearest_dist / 2

        # 1) border‑limited → nudge toward centre
        direction_to_centre = np.array([0.5, 0.5]) - centers
        centers[border_limited] += lr * direction_to_centre[border_limited]

        # 2) neighbour‑limited → push away from nearest neighbour
        vec = centers - centers[nearest_idx]
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        unit = vec / norm
        centers[~border_limited] += lr * unit[~border_limited]

        # keep points inside the square (with a small safety margin)
        centers = np.clip(centers, 0.01, 0.99)

    return centers


def compute_max_radii(centers):
    """
    For a given set of centre positions compute the largest radii
    that keep all circles inside the unit square and non‑overlapping.
    """
    # distance to the four borders
    border = np.minimum.reduce([centers[:, 0], centers[:, 1],
                                1 - centers[:, 0], 1 - centers[:, 1]])

    # pairwise centre distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)

    # nearest neighbour distance for each circle
    nearest = np.min(dists, axis=1)

    # radius limited by border and by half the nearest neighbour distance
    radii = np.minimum(border, nearest / 2.0)
    return radii


def construct_packing():
    """
    Build a deterministic set of 26 centre positions and refine them
    to obtain a larger total radius sum.
    """
    n = 26
    rng = np.random.default_rng(42)

    # start from a simple 5×5 grid (25 points) plus a centre point
    grid_vals = np.linspace(0.15, 0.85, 5)
    xv, yv = np.meshgrid(grid_vals, grid_vals)
    grid_points = np.column_stack([xv.ravel(), yv.ravel()])  # 25 points
    centers = np.vstack([grid_points, np.array([[0.5, 0.5]])])  # 26 points

    # add a tiny random jitter to avoid perfect symmetry
    centers += rng.uniform(-0.02, 0.02, size=centers.shape)
    centers = np.clip(centers, 0.01, 0.99)

    # refine positions with the force‑directed step
    centers = improve_positions(centers, steps=300, lr=0.015)

    # compute radii for the final layout
    radii = compute_max_radii(centers)
    sum_radii = np.sum(radii)

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
