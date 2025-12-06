"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def compute_max_radii(centers):
    n = centers.shape[0]
    wall = np.minimum.reduce([centers[:, 0], centers[:, 1],
                             1 - centers[:, 0], 1 - centers[:, 1]])
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    nearest = np.min(dists, axis=1) / 2.0
    return np.clip(np.minimum(wall, nearest), 0.0, None)


def _hex_initial():
    dx = 0.18
    dy = dx * np.sqrt(3) / 2
    pts = []
    y = dx / 2
    row = 0
    while y < 1 - dx / 2:
        offset = 0.0 if row % 2 == 0 else dx / 2
        x = dx / 2 + offset
        while x < 1 - dx / 2:
            pts.append([x, y])
            x += dx
        y += dy
        row += 1
    pts = np.array(pts)
    if pts.shape[0] >= 26:
        wall = np.minimum.reduce([pts[:, 0], pts[:, 1],
                                 1 - pts[:, 0], 1 - pts[:, 1]])
        idx = np.argsort(-wall)[:26]
        return pts[idx]
    extra = 26 - pts.shape[0]
    rand = np.random.rand(extra, 2) * (1 - dx) + dx / 2
    return np.vstack([pts, rand])


def construct_packing():
    np.random.seed(0)
    centers = _hex_initial()
    # small random jitter to break symmetry
    centers += (np.random.rand(*centers.shape) - 0.5) * 0.02
    centers = np.clip(centers, 0.0, 1.0)

    radii = compute_max_radii(centers)
    best_sum = radii.sum()
    step = 0.04

    for it in range(2000):
        i = np.random.randint(26)
        delta = (np.random.rand(2) - 0.5) * step
        new_c = np.clip(centers[i] + delta, 0.0, 1.0)

        trial = centers.copy()
        trial[i] = new_c
        trial_r = compute_max_radii(trial)
        s = trial_r.sum()

        if s > best_sum:
            centers, radii, best_sum = trial, trial_r, s

        # gradually reduce step size
        if (it + 1) % 200 == 0:
            step *= 0.9

    return centers, radii, best_sum
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
