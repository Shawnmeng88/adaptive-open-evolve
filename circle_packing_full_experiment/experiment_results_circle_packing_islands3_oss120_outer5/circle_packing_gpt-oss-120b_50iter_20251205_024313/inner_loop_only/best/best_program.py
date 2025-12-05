# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    """Create 26 circles and maximise total radii."""
    n, margin = 5, 0.1
    step = (1 - 2 * margin) / (n - 1)
    pts = [(margin + i * step, margin + j * step) for i in range(n) for j in range(n)]
    pts.append((margin / 2, margin / 2))          # 26th tiny circle
    centers = np.array(pts)

    radii = compute_max_radii(centers)
    radii = enlarge_radii(centers, radii)
    radii = refine_radii(centers, radii, iterations=5)

    # three rounds of position optimisation, each followed by uniform growth & fine‑tuning
    for _ in range(3):
        centers, radii = _local_position_optimize(centers, radii,
                                                 iterations=800, step_scale=0.04)
        radii = enlarge_radii(centers, radii)
        radii = refine_radii(centers, radii, iterations=3)

    return centers, radii, radii.sum()


def compute_max_radii(centers):
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    border = np.minimum.reduce([centers[:, 0], centers[:, 1],
                               1 - centers[:, 0], 1 - centers[:, 1]])
    nearest = dists.min(axis=1) / 2.0
    return np.minimum(border, nearest)


def enlarge_radii(centers, radii):
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    slack_center = (dists - radii[:, None] - radii[None, :]).min()
    border = np.minimum.reduce([centers[:, 0], centers[:, 1],
                               1 - centers[:, 0], 1 - centers[:, 1]])
    slack_border = (border - radii).min()
    delta = max(0.0, 0.5 * min(slack_center, slack_border))
    return radii + delta if delta > 0 else radii


def refine_radii(centers, radii, iterations=5):
    for _ in range(iterations):
        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        border = np.minimum.reduce([centers[:, 0], centers[:, 1],
                                   1 - centers[:, 0], 1 - centers[:, 1]])
        max_allowed = np.minimum(border,
                                 (dists - radii[None, :]).min(axis=1))
        radii = np.maximum(radii, max_allowed)
    return radii


def _local_position_optimize(centers, radii, iterations=200, step_scale=0.02):
    """Hill‑climbing with a cooling step size."""
    rng = np.random.default_rng()
    best_c, best_r = centers.copy(), radii.copy()
    best_sum = best_r.sum()
    n = len(best_c)

    for it in range(iterations):
        i = rng.integers(n)
        scale = step_scale * (0.99 ** (it // (iterations // 5)))
        delta = (rng.random(2) - 0.5) * 2 * scale
        new_center = np.clip(best_c[i] + delta, 0.0, 1.0)

        border = min(new_center[0], new_center[1],
                     1 - new_center[0], 1 - new_center[1])
        others = np.delete(best_c, i, axis=0)
        other_r = np.delete(best_r, i)
        dists = np.linalg.norm(others - new_center, axis=1)
        neigh = (dists - other_r).min() if dists.size else np.inf
        new_radius = min(border, neigh)
        if new_radius <= 0:
            continue

        new_sum = best_sum - best_r[i] + new_radius
        if new_sum > best_sum + 1e-9:
            best_sum = new_sum
            best_c[i] = new_center
            best_r[i] = new_radius

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
