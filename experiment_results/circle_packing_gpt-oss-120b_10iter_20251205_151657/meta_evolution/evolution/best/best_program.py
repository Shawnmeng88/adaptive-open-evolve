"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    import numpy as np, math, random
    from scipy.optimize import linprog

    n = 26
    eps = 1e-8

    def solve_lp(centers):
        d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
        A, b = [], []
        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i] = row[j] = 1.0
                A.append(row)
                b.append(d[i, j] - eps)
        bounds = [(0.0, min(c[0], c[1], 1 - c[0], 1 - c[1])) for c in centers]
        res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if res.success:
            return res.x.sum(), res.x
        return -1.0, None

    def random_centers():
        pts = []
        while len(pts) < n:
            p = np.random.rand(2) * 0.9 + 0.05
            if all(np.linalg.norm(p - np.array(pts), axis=1) >= 0.07) if pts else True:
                pts.append(p)
        return np.array(pts)

    def hex_grid():
        d = 0.18
        dy = d * math.sqrt(3) / 2.0
        pts, y, row = [], dy, 0
        while y < 1 - dy:
            off = 0.0 if row % 2 == 0 else d / 2.0
            x = off + d / 2.0
            while x < 1 - d / 2.0:
                pts.append([x, y])
                x += d
            y += dy
            row += 1
        pts = np.array(pts)
        if pts.shape[0] >= n:
            return pts[:n]
        extra = random_centers()
        return np.vstack((pts, extra))[:n]

    def local_refine(centers, cur_sum, cur_rad):
        best_sum, best_centers, best_radii = cur_sum, centers.copy(), cur_rad.copy()
        for _ in range(120):
            i = np.random.randint(n)
            step = 0.02
            new_c = best_centers[i] + (np.random.rand(2) - 0.5) * step
            new_c = np.clip(new_c, 0.0, 1.0)
            cand = best_centers.copy()
            cand[i] = new_c
            tot, rad = solve_lp(cand)
            if tot > best_sum:
                best_sum, best_centers, best_radii = tot, cand, rad
        return best_sum, best_centers, best_radii

    best_sum = -1.0
    best_centers = best_radii = None

    seeds = [hex_grid()] + [random_centers() for _ in range(25)]
    for cent in seeds:
        total, rad = solve_lp(cent)
        if total > best_sum:
            best_sum, best_centers, best_radii = total, cent, rad
        # try local improvement on promising seeds
        if total > 0:
            total2, cent2, rad2 = local_refine(cent, total, rad)
            if total2 > best_sum:
                best_sum, best_centers, best_radii = total2, cent2, rad2

    if best_centers is None:
        best_centers = np.zeros((n, 2))
        best_radii = np.zeros(n)
        best_sum = 0.0

    return best_centers, best_radii, best_sum
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
