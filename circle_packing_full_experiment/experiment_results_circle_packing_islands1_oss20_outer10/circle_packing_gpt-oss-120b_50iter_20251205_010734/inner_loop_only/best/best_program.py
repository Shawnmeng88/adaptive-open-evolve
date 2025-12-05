# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def construct_packing():
    n = 26
    g = np.linspace(.1, .9, 5)
    pts = np.array([[x, y] for x in g for y in g])
    centers = np.vstack([pts[:25], [.5, .5]])
    rng = np.random.default_rng(42)
    best_c = centers.copy()
    best_r = compute_max_radii(best_c)
    best_sum = best_r.sum()
    step = 0.07
    for it in range(250):
        i = rng.integers(n)
        delta = rng.uniform(-step, step, 2)
        cand = best_c.copy()
        cand[i] = np.clip(cand[i] + delta, 0.01, 0.99)
        r = compute_max_radii(cand)
        s = r.sum()
        if s > best_sum:
            best_c, best_r, best_sum = cand, r, s
        if (it + 1) % 50 == 0 and step > 0.01:
            step *= 0.7
    return best_c, best_r, best_sum

def compute_max_radii(C):
    n = len(C)
    b = np.minimum.reduce([C[:, 0], C[:, 1], 1 - C[:, 0], 1 - C[:, 1]])
    d = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))
    A, B = [], []
    for i in range(n):
        row = np.zeros(n); row[i] = 1
        A.append(row); B.append(b[i])
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n); row[i] = row[j] = 1
            A.append(row); B.append(d[i, j])
    res = linprog(-np.ones(n), A_ub=A, b_ub=B,
                  bounds=[(0, None)] * n, method='highs')
    return res.x if res.success else np.zeros(n)
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
