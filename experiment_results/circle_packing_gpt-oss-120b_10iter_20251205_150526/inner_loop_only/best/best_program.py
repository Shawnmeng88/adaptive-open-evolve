"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    n = 26
    rng = np.random.default_rng(0)
    # deterministic seed (grid + centre)
    g = np.linspace(0.1, 0.9, 5)
    xv, yv = np.meshgrid(g, g)
    base = np.column_stack([xv.ravel(), yv.ravel()])
    base = np.vstack([base, [0.5, 0.5]])

    best_sum = -1.0
    best_centers = None
    best_radii = None

    # perturb the deterministic layout several times
    for _ in range(15):
        centers = base + rng.uniform(-0.02, 0.02, base.shape)
        centers = np.clip(centers, 0.01, 0.99)
        radii = compute_max_radii(centers)
        s = radii.sum()
        if s > best_sum:
            best_sum, best_centers, best_radii = s, centers, radii

    # try a few pure random layouts
    for _ in range(5):
        centers = rng.uniform(0.01, 0.99, (n, 2))
        radii = compute_max_radii(centers)
        s = radii.sum()
        if s > best_sum:
            best_sum, best_centers, best_radii = s, centers, radii

    return best_centers, best_radii, float(best_sum)


def compute_max_radii(centers):
    n = centers.shape[0]
    border = np.minimum.reduce([centers[:, 0], centers[:, 1],
                               1 - centers[:, 0], 1 - centers[:, 1]])

    diff = centers[:, None, :] - centers[None, :, :]
    pair = np.sqrt(np.sum(diff ** 2, axis=2))

    A = np.vstack([np.eye(n),
                   np.eye(n)[np.triu_indices(n, k=1)[0]] + np.eye(n)[np.triu_indices(n, k=1)[1]]])
    b = np.concatenate([border, pair[np.triu_indices(n, k=1)]])

    res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=[(0, None)] * n, method="highs")
    return res.x if res.success else np.minimum(border, np.min(pair + np.eye(n) * np.inf, axis=1) / 2)
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
