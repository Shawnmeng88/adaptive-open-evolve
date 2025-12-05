# EVOLVE-BLOCK-START
"""Improved circle packing for n=26 using LP radius optimisation."""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

np.random.seed(0)                     # deterministic randomness (once)

def _lp_radii(centers):
    """Solve a linear programme: maximise Σ r_i  subject to
       r_i ≤ border_i and r_i+r_j ≤ d_ij."""
    n = centers.shape[0]
    # border limits (tiny epsilon avoids fp edge cases)
    eps = 1e-9
    border = np.minimum.reduce([centers[:, 0], centers[:, 1],
                                1 - centers[:, 0], 1 - centers[:, 1]]) - eps
    # pairwise distances
    D = cdist(centers, centers)
    # build A_ub x ≤ b_ub
    rows = []
    b = []
    # border constraints
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1
        rows.append(row)
        b.append(border[i])
    # non‑overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = row[j] = 1
            rows.append(row)
            b.append(D[i, j])
    A = np.vstack(rows)
    b = np.array(b)
    # maximise Σ r_i  → minimise −Σ r_i
    res = linprog(-np.ones(n), A_ub=A, b_ub=b,
                  bounds=[(0, None)] * n, method="highs")
    return res.x if res.success else np.zeros(n)

def _initial_centers(n):
    """Create a well‑spread deterministic seed of n points."""
    # start from a coarse 5×6 grid (30 pts) and take the first n
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 6)
    grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)[:n]
    # add a tiny deterministic jitter
    jitter = 0.02 * np.random.randn(n, 2)
    return np.clip(grid + jitter, 0.01, 0.99)

def construct_packing():
    """Return (centers, radii, sum_of_radii) for 26 circles."""
    n = 26
    best_sum = -1.0
    best_centers = best_radii = None
    # multi‑start (3 deterministic seeds)
    for _ in range(3):
        centers = _initial_centers(n)
        radii = _lp_radii(centers)
        s = radii.sum()
        if s > best_sum:
            best_sum, best_centers, best_radii = s, centers, radii
    return best_centers, best_radii, best_sum
# EVOLVE-BLOCK-END

# ----------------------------------------------------------------------
def run_packing():
    """Run the circle packing constructor for n=26."""
    return construct_packing()

def visualize(centers, radii):
    """Simple Matplotlib visualiser (unchanged)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.grid(True)
    for i, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(c, r, alpha=0.5))
        ax.text(*c, str(i), ha="center", va="center")
    plt.title(f"Circle Packing (n={len(centers)}, sum={radii.sum():.6f})")
    plt.show()

if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s:.6f}")
    # visualize(c, r)   # uncomment to see the layout