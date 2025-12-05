# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    n = 26
    s = 0.20
    dy = s * np.sqrt(3) / 2
    pts = []
    y, row = s / 2, 0
    while y < 1 and len(pts) < n:
        off = s / 2 if row % 2 else 0
        x = s / 2 + off
        while x < 1 and len(pts) < n:
            pts.append([x, y])
            x += s
        y += dy
        row += 1

    best = np.array(pts[:n])
    best_r = _radii(best)
    best_s = best_r.sum()

    rng = np.random.default_rng()
    for step in range(3000):
        delta = 0.04 * (1 - step / 3000)
        if rng.random() < 0.02:
            delta = 0.1
        i = rng.integers(n)
        cand = best.copy()
        cand[i] = np.clip(cand[i] + rng.uniform(-delta, delta, 2), 0, 1)
        cand_r = _radii(cand)
        cand_s = cand_r.sum()
        if cand_s > best_s:
            best, best_r, best_s = cand, cand_r, cand_s
    return best, best_r, best_s

def _radii(c):
    b = np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    p = d.min(axis=1) / 2
    return np.minimum(b, p)
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
