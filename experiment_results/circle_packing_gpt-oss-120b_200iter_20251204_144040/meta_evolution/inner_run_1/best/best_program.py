# EVOLVE-BLOCK-START
"""Hex‑lattice + LP refinement circle packing (n=26)"""

import numpy as np
from scipy.optimize import linprog

np.random.seed(0)                     # deterministic randomness
N = 26                                # number of circles
B = 0.02                               # border safety margin


def init_hex_lattice():
    """Generate ≈N points on a hexagonal lattice inside the unit square."""
    # basic spacing; will be scaled later
    dx, dy = 0.18, np.sqrt(3) * 0.18 / 2
    pts = []
    y = dy
    row = 0
    while len(pts) < N:
        offset = 0.0 if row % 2 == 0 else dx / 2
        x = dx + offset
        while x < 1 - dx and len(pts) < N:
            pts.append([x, y])
            x += dx
        y += dy
        row += 1
    pts = np.array(pts)[:N]
    # squeeze into safe interior
    pts = B + (1 - 2 * B) * pts
    return pts


def lp_optimize(centers):
    """Linear‑programming step: maximise Σr subject to border & non‑overlap."""
    n = len(centers)
    # border limits
    border = np.minimum.reduce([centers[:, 0], centers[:, 1],
                               1 - centers[:, 0], 1 - centers[:, 1]])
    # pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))

    # build A_ub·r ≤ b_ub
    rows = []
    b = []

    # border constraints: r_i ≤ border_i
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1
        rows.append(row)
        b.append(border[i])

    # non‑overlap: r_i + r_j ≤ d_ij   (i<j)
    iu = np.triu_indices(n, 1)
    for i, j in zip(*iu):
        row = np.zeros(n)
        row[i] = row[j] = 1
        rows.append(row)
        b.append(dists[i, j])

    A_ub = np.vstack(rows)
    b_ub = np.array(b)

    res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, None)] * n, method='highs')
    return res.x if res.success else np.zeros(n)


def jitter(centers, delta):
    """Randomly move a small subset of centers by ≤δ."""
    new = centers.copy()
    idx = np.random.choice(len(centers), size=5, replace=False)
    new[idx] += np.random.uniform(-delta, delta, size=(5, 2))
    new = np.clip(new, B, 1 - B)
    return new


def check_valid(centers, radii):
    """Return True iff all constraints are satisfied (within tolerance)."""
    eps = 1e-9
    border = np.minimum.reduce([centers[:, 0], centers[:, 1],
                               1 - centers[:, 0], 1 - centers[:, 1]])
    if np.any(radii - border > eps):
        return False
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(len(centers), 1)
    if np.any(radii[iu[0]] + radii[iu[1]] - dists[iu] > eps):
        return False
    return True


def construct_packing():
    """Main constructor – returns (centers, radii, sum_of_radii)."""
    centers = init_hex_lattice()
    radii = lp_optimize(centers)
    best_sum, best_c, best_r = radii.sum(), centers, radii

    delta = 0.05
    while delta > 1e-4:
        improved = False
        for _ in range(30):                     # attempts per δ
            cand_c = jitter(best_c, delta)
            cand_r = lp_optimize(cand_c)
            if cand_r.sum() > best_sum and check_valid(cand_c, cand_r):
                best_sum, best_c, best_r = cand_r.sum(), cand_c, cand_r
                improved = True
        if not improved:
            delta *= 0.95                       # anneal
        else:
            delta *= 0.97                       # slower decay after success
    return best_c, best_r, best_sum


# EVOLVE-BLOCK-END


def run_packing():
    """Fixed wrapper – unchanged."""
    return construct_packing()


def visualize(centers, radii):
    """Optional visualisation (unchanged API)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(c, r, alpha=0.5))
        ax.text(c[0], c[1], str(i), ha='center', va='center')

    plt.title(f'Circle packing (n={len(centers)}, sum={radii.sum():.4f})')
    plt.show()


if __name__ == '__main__':
    c, r, s = run_packing()
    print(f'Sum of radii: {s:.6f}')
    # visualize(c, r)   # uncomment to see the layout