# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    """
    Improved stochastic hill‑climb for the 26‑circle packing problem.
    Uses a linear step‑size schedule and a longer search horizon.
    """
    np.random.seed(0)

    # ---- initial hexagonal layout (26 points) ----
    spacing = 0.18
    offset = spacing / 2
    pts = []
    y, row = 0.1, 0
    while y < 0.9:
        x_start = 0.1 + (offset if row % 2 else 0.0)
        x = x_start
        while x < 0.9:
            pts.append([x, y])
            x += spacing
        y += spacing * np.sqrt(3) / 2
        row += 1
    pts = np.array(pts[:26])
    pts = np.clip(pts, 0.01, 0.99)

    # ---- helper to compute feasible radii for a set of centres ----
    def radii_for(centers):
        border = np.minimum.reduce(
            [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
        )
        dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        return np.minimum(border, np.min(dists, axis=1) / 2.0)

    radii = radii_for(pts)
    best_sum = radii.sum()
    best_pts = pts.copy()
    best_radii = radii.copy()

    # ---- stochastic improvement loop ----
    max_iters = 5000
    max_step = 0.07
    min_step = 0.001

    for it in range(max_iters):
        # linear step size decay
        step = max_step * (1 - it / max_iters) + min_step

        i = np.random.randint(len(pts))
        cand = pts.copy()
        delta = (np.random.rand(2) - 0.5) * 2 * step
        cand[i] += delta
        cand[i] = np.clip(cand[i], 0.01, 0.99)

        cand_r = radii_for(cand)
        s = cand_r.sum()

        # accept if improvement over current best
        if s > best_sum:
            best_sum = s
            best_pts = cand
            best_radii = cand_r

        # also keep the working set updated so future moves build on the latest improvement
        pts = cand if s > radii.sum() else pts
        radii = cand_r if s > radii.sum() else radii

    # final refinement with very small steps
    for _ in range(1000):
        i = np.random.randint(len(best_pts))
        cand = best_pts.copy()
        delta = (np.random.rand(2) - 0.5) * 2 * min_step
        cand[i] += delta
        cand[i] = np.clip(cand[i], 0.01, 0.99)

        cand_r = radii_for(cand)
        s = cand_r.sum()
        if s > best_sum:
            best_sum = s
            best_pts = cand
            best_radii = cand_r

    return best_pts, best_radii, best_sum
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
