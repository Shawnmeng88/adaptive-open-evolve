# EVOLVE-BLOCK-START
"""Hex‑grid + LP optimiser for 26 circles in the unit square."""
import numpy as np
from scipy.optimize import linprog

# ------------------------------------------------------------
# deterministic seed – never re‑seeded later
np.random.seed(0)

# ------------------------------------------------------------
def _hex_grid(n, angle=0.0):
    """Return at least *n* points of a hexagonal lattice rotated by *angle*."""
    # lattice spacing – chosen so a 6×6 patch comfortably fits in the unit square
    s = 0.18
    # raw (unrotated) coordinates
    pts = []
    dy = np.sqrt(3) / 2 * s
    for i in range(10):                     # enough rows
        y = i * dy
        offset = (i % 2) * s / 2
        for j in range(10):                 # enough columns
            x = j * s + offset
            pts.append([x, y])
    pts = np.array(pts)
    # centre & rotate about (0.5,0.5)
    pts -= 0.5
    c, s_ = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s_], [s_, c]])
    pts = pts @ R.T + 0.5
    # keep only points strictly inside the square
    mask = (pts[:, 0] > 0) & (pts[:, 0] < 1) & (pts[:, 1] > 0) & (pts[:, 1] < 1)
    pts = pts[mask]
    return pts[:n]                     # first *n* points


def _lp_optimize(centers):
    """Linear programme: maximise Σ r_i  subject to border & pairwise constraints."""
    n = centers.shape[0]
    # border limits
    border = np.minimum.reduce([centers[:, 0],
                                centers[:, 1],
                                1 - centers[:, 0],
                                1 - centers[:, 1]])
    # pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    # build A_ub, b_ub
    rows = []
    # r_i ≤ border_i
    rows.append(np.eye(n))
    # r_i + r_j ≤ d_ij  for i<j
    iu = np.triu_indices(n, k=1)
    pair_mat = np.zeros((len(iu[0]), n))
    pair_mat[np.arange(len(iu[0])), iu[0]] = 1
    pair_mat[np.arange(len(iu[0])), iu[1]] = 1
    rows.append(pair_mat)
    A_ub = np.vstack(rows)
    b_ub = np.hstack([border, dists[iu]])
    # solve
    res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, None)] * n, method='highs')
    return res.x if res.success else np.zeros(n)


def _validate(centers, radii, eps=1e-9):
    """Return True iff all geometric constraints are satisfied."""
    # border
    if np.any(radii - np.minimum.reduce([centers[:, 0],
                                         centers[:, 1],
                                         1 - centers[:, 0],
                                         1 - centers[:, 1]]) > eps):
        return False
    # pairwise
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    mask = np.triu(np.ones_like(dists, dtype=bool), k=1)
    if np.any(radii[:, None] + radii[None, :] - dists > eps, where=mask):
        return False
    return True


def construct_packing():
    """Build the best feasible layout found among a few rotated hex‑grids."""
    best_score = -1.0
    best_centers, best_radii = None, None

    # try a handful of rotations (deterministic, cheap)
    for k in range(4):
        angle = (np.pi / 3) * k / 4          # 0 … 60° in 4 steps
        centers = _hex_grid(26, angle=angle)
        radii = _lp_optimize(centers)
        if not _validate(centers, radii):
            continue
        score = radii.sum()
        if score > best_score:
            best_score, best_centers, best_radii = score, centers, radii

        # tiny jitter + re‑optimise (still deterministic)
        jitter = np.random.normal(scale=0.005, size=centers.shape)
        j_cent = np.clip(best_centers + jitter, 0.0, 1.0)
        j_radii = _lp_optimize(j_cent)
        if _validate(j_cent, j_radii) and j_radii.sum() > best_score:
            best_score, best_centers, best_radii = j_radii.sum(), j_cent, j_radii

    return best_centers, best_radii, best_score


# EVOLVE-BLOCK-END


def run_packing():
    """Run the circle‑packing constructor for n=26."""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """Optional visualisation (unchanged from the original template)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(c, r, alpha=0.5))
        ax.text(c[0], c[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={radii.sum():.6f})")
    plt.show()


if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s:.6f}")
    # visualize(c, r)   # uncomment to see the result