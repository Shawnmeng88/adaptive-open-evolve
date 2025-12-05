# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    n = 26
    rng = np.random.default_rng()
    best_sum = -1.0
    best_centers = best_radii = None
    restarts = 5
    for _ in range(restarts):
        centers = rng.uniform(0.05, 0.95, (n, 2))
        step = 0.08
        cur_centers = centers.copy()
        cur_radii = compute_max_radii(cur_centers)
        cur_sum = cur_radii.sum()
        for it in range(2000):
            i = rng.integers(n)
            prop = np.clip(cur_centers[i] + rng.uniform(-step, step, 2), 0.001, 0.999)
            cand = cur_centers.copy()
            cand[i] = prop
            cand_r = compute_max_radii(cand)
            cand_sum = cand_r.sum()
            if cand_sum > cur_sum:
                cur_centers, cur_radii, cur_sum = cand, cand_r, cand_sum
            if it and it % 500 == 0:
                step *= 0.7
        if cur_sum > best_sum:
            best_sum, best_centers, best_radii = cur_sum, cur_centers, cur_radii
    return best_centers, best_radii, best_sum

def compute_max_radii(centers):
    border = np.minimum.reduce([centers[:, 0], centers[:, 1],
                               1 - centers[:, 0], 1 - centers[:, 1]])
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return np.minimum(border, d.min(axis=1) / 2)
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
