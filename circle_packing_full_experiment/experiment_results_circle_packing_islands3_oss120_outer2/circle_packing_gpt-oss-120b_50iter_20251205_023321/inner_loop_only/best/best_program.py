# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    g = (np.arange(5) + 0.5) / 5.0
    base = np.array([[x, y] for x in g for y in g])
    np.random.seed(0)
    cand = np.vstack([np.random.rand(2000, 2), [0.5, 0.55]])
    best_sum, best_pt = -1.0, None
    for p in cand:
        s = compute_max_radii(np.vstack([base, p])).sum()
        if s > best_sum:
            best_sum, best_pt = s, p
    step = 0.05
    while step > 1e-4:
        improved = False
        for dx in (-step, 0, step):
            for dy in (-step, 0, step):
                if dx == dy == 0:
                    continue
                pt = best_pt + np.array([dx, dy])
                if (pt < 0).any() or (pt > 1).any():
                    continue
                s = compute_max_radii(np.vstack([base, pt])).sum()
                if s > best_sum:
                    best_sum, best_pt, improved = s, pt, True
        if not improved:
            step *= 0.5
    centers = np.vstack([base, best_pt])
    radii = compute_max_radii(centers)
    return centers, radii, best_sum

def compute_max_radii(centers):
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    b = np.minimum.reduce([centers[:, 0], centers[:, 1],
                          1 - centers[:, 0], 1 - centers[:, 1]])
    return np.minimum(b, d.min(axis=1) / 2.0)
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
