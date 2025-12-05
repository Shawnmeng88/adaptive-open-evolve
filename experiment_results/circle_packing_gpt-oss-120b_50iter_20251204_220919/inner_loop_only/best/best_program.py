# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    n, m, sp = 26, 0.02, 0.25
    # seed: hexagonal lattice
    while True:
        dy = sp * np.sqrt(3) / 2
        rows = int((1 - 2 * m) // dy) + 1
        pts = []
        for i in range(rows):
            y = m + i * dy
            off = (i % 2) * sp / 2
            cols = int((1 - 2 * m - off) // sp) + 1
            for j in range(cols):
                x = m + off + j * sp
                if x <= 1 - m and y <= 1 - m:
                    pts.append((x, y))
        if len(pts) >= n:
            best = np.array(pts[:n])
            break
        sp *= 0.95

    def max_r(c):
        b = np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        return np.minimum(b, d.min(1) / 2)

    rng = np.random.default_rng()
    best_r = max_r(best)
    best_sum = best_r.sum()
    sigma = 0.03

    for it in range(12000):
        i = rng.integers(n)
        cand = best[i] + rng.normal(scale=sigma, size=2)
        cand = np.clip(cand, m, 1 - m)
        new = best.copy()
        new[i] = cand
        r = max_r(new)
        s = r.sum()
        if s > best_sum:
            best, best_r, best_sum = new, r, s
            sigma = 0.03
        else:
            sigma *= 0.9995
        if it % 4000 == 0 and it:
            rnd = rng.random((n, 2)) * (1 - 2 * m) + m
            r2 = max_r(rnd)
            s2 = r2.sum()
            if s2 > best_sum:
                best, best_r, best_sum = rnd, r2, s2
                sigma = 0.03

    return best, best_r, best_sum
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
