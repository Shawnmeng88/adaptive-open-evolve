# EVOLVE-BLOCK-START
import math, numpy as np

def construct_packing():
    # Generate points on a hexagonal grid for a given spacing
    def pts(a):
        p = []
        y = a / 2.0
        row = 0
        while y <= 1 - a / 2.0 and len(p) < 26:
            off = 0 if row % 2 == 0 else a / 2.0
            x = a / 2.0 + off
            while x <= 1 - a / 2.0 and len(p) < 26:
                p.append([x, y])
                x += a
            y += a * math.sqrt(3) / 2.0
            row += 1
        return np.array(p)

    # binary search for the largest spacing that yields at least 26 points
    lo, hi = 0.05, 0.40
    for _ in range(20):
        mid = (lo + hi) / 2.0
        if pts(mid).shape[0] >= 26:
            lo = mid
        else:
            hi = mid

    # ------------------------------------------------------------------
    # local optimisation utilities
    # ------------------------------------------------------------------
    def improve(initial, iters=3000, sigma_start=0.03):
        """hill‑climb improvement of a centre set."""
        best = initial.copy()
        best_sum = compute_max_radii(best).sum()
        sigma = sigma_start
        n = best.shape[0]

        for k in range(iters):
            i = np.random.randint(n)
            cand = best[i] + np.random.normal(scale=sigma, size=2)
            cand = np.clip(cand, 0.0, 1.0)

            new = best.copy()
            new[i] = cand
            s = compute_max_radii(new).sum()

            if s > best_sum:
                best, best_sum = new, s

            # gradually shrink the perturbation radius
            sigma = sigma_start * (1.0 - k / iters)

        return best, best_sum

    # ------------------------------------------------------------------
    # multiple restarts – keep the best result
    # ------------------------------------------------------------------
    best_overall = None
    best_sum_overall = -np.inf

    # first restart uses the hexagonal grid (a good baseline)
    init0 = pts(lo)[:26]
    centers, s = improve(init0)
    if s > best_sum_overall:
        best_overall, best_sum_overall = centers, s

    # additional random restarts
    for _ in range(4):
        rand_init = np.random.rand(26, 2)  # uniform random points in the unit square
        centers, s = improve(rand_init)
        if s > best_sum_overall:
            best_overall, best_sum_overall = centers, s

    radii = compute_max_radii(best_overall)
    return best_overall, radii, best_sum_overall

def compute_max_radii(c):
    border = np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    neigh = 0.5 * np.min(d, axis=1)
    return np.minimum(border, neigh)
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
