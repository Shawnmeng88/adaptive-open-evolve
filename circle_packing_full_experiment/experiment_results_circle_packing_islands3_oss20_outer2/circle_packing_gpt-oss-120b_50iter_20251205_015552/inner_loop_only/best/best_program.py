# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    rng = np.random.default_rng(42)
    best_c, best_r, best_sum = _grid_packing()
    for _ in range(4000):
        i = rng.integers(len(best_c))
        proposal = np.clip(best_c[i] + rng.normal(scale=0.015, size=2), 0.0, 1.0)
        new_c = best_c.copy()
        new_c[i] = proposal
        new_r = compute_max_radii(new_c)
        new_sum = new_r.sum()
        if new_sum > best_sum:
            best_c, best_r, best_sum = new_c, new_r, new_sum
    return best_c, best_r, best_sum

def _grid_packing():
    dx, dy = 0.2, np.sqrt(3) / 2 * 0.2
    pts = []
    y, row = dy / 2, 0
    while len(pts) < 26 and y < 1:
        off = 0 if row % 2 == 0 else dx / 2
        x = off + dx / 2
        while len(pts) < 26 and x < 1:
            pts.append([x, y])
            x += dx
        y += dy
        row += 1
    c = np.array(pts[:26])
    r = compute_max_radii(c)
    return c, r, r.sum()

def compute_max_radii(c):
    b = np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    n = d.min(axis=1)
    return np.minimum(b, n / 2)
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
