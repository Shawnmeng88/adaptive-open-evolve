"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def _solve_radii_lp(centers):
    n = centers.shape[0]
    c = -np.ones(n)
    A = []
    b = []
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, y, 1 - x, 1 - y)
        row = np.zeros(n)
        row[i] = 1.0
        A.append(row)
        b.append(max_r)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = row[j] = 1.0
            A.append(row)
            b.append(d)
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b),
                 bounds=[(0, None)] * n, method="highs")
    return res.x if res.success else np.zeros(n)


def construct_packing():
    step = 1.0 / 6.0
    pts = [[i * step, j * step] for i in range(1, 6) for j in range(1, 6)]
    pts.append([0.5, 0.5])
    base = np.array(pts)

    best_r = _solve_radii_lp(base)
    best_sum = best_r.sum()
    best_c = base

    rng = np.random.default_rng()
    for _ in range(200):
        cand = base + rng.normal(scale=0.02, size=base.shape)
        cand = np.clip(cand, 0.0, 1.0)
        r = _solve_radii_lp(cand)
        s = r.sum()
        if s > best_sum:
            best_sum, best_c, best_r = s, cand, r

    return best_c, best_r, best_sum


def compute_max_radii(centers):
    return _solve_radii_lp(centers)
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
