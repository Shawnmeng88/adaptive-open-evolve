"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """Generate several candidate layouts and keep the one with the largest total radius."""
    n = 26
    rng = np.random.default_rng()
    # start with the handcrafted layout
    base = np.zeros((n, 2))
    base[0] = [0.5, 0.5]
    for i in range(8):
        a = 2 * np.pi * i / 8
        base[i + 1] = [0.5 + 0.35 * np.cos(a), 0.5 + 0.35 * np.sin(a)]
    for i in range(16):
        a = 2 * np.pi * i / 16
        base[i + 9] = [0.5 + 0.68 * np.cos(a), 0.5 + 0.68 * np.sin(a)]
    candidates = [np.clip(base, 0.001, 0.999)]

    # add a few random layouts
    for _ in range(4):
        pts = rng.uniform(0.05, 0.95, size=(n, 2))
        candidates.append(pts)

    best_sum = -1.0
    best_centers, best_radii = None, None
    for centers in candidates:
        radii = compute_max_radii(centers)
        s = radii.sum()
        if s > best_sum:
            best_sum, best_centers, best_radii = s, centers, radii
    return best_centers, best_radii, best_sum


def compute_max_radii(centers):
    """Linear‑programming maximisation of radii for fixed centres."""
    n = centers.shape[0]
    # border limits
    border = np.minimum.reduce([centers[:, 0], centers[:, 1],
                               1 - centers[:, 0], 1 - centers[:, 1]])
    # pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    A = np.zeros((len(pairs), n))
    b = np.empty(len(pairs))
    for k, (i, j) in enumerate(pairs):
        A[k, i] = A[k, j] = 1
        b[k] = dists[i, j]

    bounds = [(0, border[i]) for i in range(n)]
    c = -np.ones(n)

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    return res.x if res.success else border
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
