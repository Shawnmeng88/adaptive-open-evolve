# EVOLVE-BLOCK-START
import numpy as np

def _hex_grid(s):
    v = s * np.sqrt(3) / 2
    pts = []
    y, row = s / 2, 0
    while y <= 1 - s / 2:
        off = 0 if row % 2 == 0 else s / 2
        xs = np.arange(s / 2 + off, 1 - s / 2 + 1e-9, s)
        for x in xs:
            pts.append([x, y])
        y += v
        row += 1
    return np.array(pts)

def _max_radii(c):
    border = np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return np.minimum(border, d.min(axis=1) / 2)

def construct_packing():
    best_sum, best_c, best_r = -1.0, None, None
    for s in np.linspace(0.12, 0.22, 31):
        pts = _hex_grid(s)
        if pts.shape[0] < 26:
            continue
        c = pts[:26]
        r = _max_radii(c)
        total = r.sum()
        if total > best_sum:
            best_sum, best_c, best_r = total, c, r

    # Simple stochastic hill‑climbing refinement
    rng = np.random.default_rng()
    c = best_c.copy()
    cur = best_sum
    stagn = 0
    while stagn < 2000:
        i = rng.integers(26)
        new_center = rng.random(2)
        old = c[i].copy()
        c[i] = new_center
        r = _max_radii(c)
        total = r.sum()
        if total > cur:
            cur, best_c, best_r = total, c.copy(), r.copy()
            stagn = 0
        else:
            c[i] = old
            stagn += 1
    return best_c, best_r, cur
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
