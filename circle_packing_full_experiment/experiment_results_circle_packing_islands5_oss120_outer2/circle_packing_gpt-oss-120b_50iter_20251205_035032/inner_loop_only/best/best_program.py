# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    """
    Start from a 5×5 grid (margin 0.1) plus a centre point,
    then perform a simple stochastic local search to increase the total sum of radii.
    """
    # initial layout
    g = np.linspace(0.1, 0.9, 5)
    xv, yv = np.meshgrid(g, g)
    centers = np.vstack([np.column_stack([xv.ravel(), yv.ravel()]), [0.5, 0.5]])

    # evaluate the starting configuration
    best_c = centers.copy()
    best_r = compute_max_radii(best_c)
    best_sum = best_r.sum()

    rng = np.random.default_rng()
    # number of random perturbations – more iterations give a better chance of improvement
    for _ in range(1200):
        # pick a random circle to move
        i = rng.integers(len(best_c))
        # propose a small move, keep inside the unit square
        delta = 0.05
        cand = best_c.copy()
        cand[i] += (rng.random(2) - 0.5) * delta
        cand[i] = np.clip(cand[i], 0.0, 1.0)

        r = compute_max_radii(cand)
        s = r.sum()
        if s > best_sum:
            best_c, best_r, best_sum = cand, r, s

    return best_c, best_r, best_sum


def compute_max_radii(centers):
    """
    Compute maximal radii limited by square borders and half the distance to the nearest neighbour.
    """
    # distance to the four borders
    border = np.minimum.reduce([
        centers[:, 0],
        1 - centers[:, 0],
        centers[:, 1],
        1 - centers[:, 1]
    ])

    # pairwise centre distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)  # ignore self‑distance
    nearest = dists.min(axis=1)

    radii = np.minimum(border, nearest / 2.0)
    return np.maximum(radii, 0.0)
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
