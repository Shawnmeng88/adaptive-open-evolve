# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    rng = np.random.default_rng(0)
    n = 26
    pts = rng.uniform(0.05, 0.95, (n, 2))

    def radii(p):
        wall = np.minimum(p, 1 - p).min(axis=1)
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        nearest = d.min(axis=1)
        return np.minimum(wall, nearest / 2)

    for _ in range(200):
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        nb = d.argmin(axis=1)
        dir_vec = pts - pts[nb]
        norm = np.linalg.norm(dir_vec, axis=1, keepdims=True) + 1e-12
        pts += 0.02 * dir_vec / norm
        pts = np.clip(pts, 0.01, 0.99)

    best_r = radii(pts)
    best_sum = best_r.sum()

    iters = 2500
    step = 0.08
    step_decay = (0.005 / step) ** (1 / iters)
    temp = 0.001
    temp_decay = (1e-6 / temp) ** (1 / iters)

    for _ in range(iters):
        i = rng.integers(n)
        prop = pts[i] + rng.uniform(-step, step, 2)
        prop = np.clip(prop, 0.01, 0.99)

        old = pts[i].copy()
        pts[i] = prop
        new_r = radii(pts)
        new_sum = new_r.sum()

        if new_sum > best_sum or rng.random() < np.exp((new_sum - best_sum) / temp):
            best_r, best_sum = new_r, new_sum
        else:
            pts[i] = old

        step *= step_decay
        temp *= temp_decay

    return pts, best_r, best_sum
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
