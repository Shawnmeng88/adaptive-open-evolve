# EVOLVE-BLOCK-START
"""Hex‑lattice + LP optimiser for 26 circles in a unit square."""
import numpy as np
from scipy.optimize import linprog

np.random.seed(0)                     # deterministic randomness


def init_hex_lattice(n, angle=0.0):
    """Generate a hexagonal lattice, rotate by *angle*, and keep *n* points."""
    # basic hex pattern (rows offset by 0.5, vertical spacing √3/2)
    cols = int(np.ceil(np.sqrt(n))) + 2
    rows = int(np.ceil(np.sqrt(n))) + 2
    xv, yv = np.meshgrid(np.arange(cols), np.arange(rows))
    xv = xv.astype(float)
    yv = yv * np.sqrt(3) / 2
    xv[::2] += 0.5                     # offset every other row
    pts = np.stack([xv.ravel(), yv.ravel()], axis=1)
    # scale to fit inside (0,1) with a margin
    pts -= pts.min(0)
    pts /= pts.max(0)
    pts = 0.02 + 0.96 * pts            # keep 0.02‑0.98 margin
    # rotate about centre (0.5,0.5)
    if angle != 0.0:
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        pts = (pts - 0.5) @ R + 0.5
    return pts[:n]


def lp_optimize(centers):
    """Linear programme: maximise sum(r) with r_i ≤ border & r_i+r_j ≤ d_ij."""
    n = len(centers)
    # border limits
    border = np.minimum.reduce([centers[:, 0], centers[:, 1],
                               1 - centers[:, 0], 1 - centers[:, 1]])
    # pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # build A_ub, b_ub
    rows = []
    # r_i ≤ border_i
    rows.append(np.eye(n))
    # r_i + r_j ≤ d_ij  (i<j)
    iu = np.triu_indices(n, k=1)
    pair_mat = np.zeros((len(iu[0]), n))
    pair_mat[np.arange(len(iu[0])), iu[0]] = 1
    pair_mat[np.arange(len(iu[0])), iu[1]] = 1
    rows.append(pair_mat)
    A_ub = np.vstack(rows)
    b_ub = np.hstack([border, dists[iu]])
    # linprog (maximise sum → minimise -sum)
    c = -np.ones(n)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None),
                  method='highs', options={'presolve': True})
    return res.x if res.success else np.zeros(n)


def validate(centers, radii, eps=1e-9):
    """Return True iff all geometric constraints hold."""
    # border
    border_ok = np.all(radii <= np.minimum.reduce([centers[:, 0],
                                                  centers[:, 1],
                                                  1 - centers[:, 0],
                                                  1 - centers[:, 1]]) + eps)
    if not border_ok:
        return False
    # pairwise
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    sum_r = radii[:, None] + radii[None, :]
    np.fill_diagonal(dists, np.inf)      # ignore self‑distance
    return np.all(sum_r <= dists + eps)


def construct_packing():
    """Return (centers, radii, sum_of_radii) for the best layout found."""
    n = 26
    best_score = -1.0
    best_centers, best_radii = None, None

    # ---- multi‑start (rotated lattices) ----
    for k in range(6):
        angle = 2 * np.pi * k / 6
        centers = init_hex_lattice(n, angle)
        radii = lp_optimize(centers)

        # ---- jitter‑refinement loop ----
        for _ in range(20):
            # small Gaussian jitter
            jitter = np.random.normal(scale=0.01, size=centers.shape)
            newc = np.clip(centers + jitter, 0.02, 0.98)
            newr = lp_optimize(newc)
            if not validate(newc, newr):
                continue
            if newr.sum() > radii.sum():
                centers, radii = newc, newr

        if validate(centers, radii) and radii.sum() > best_score:
            best_score, best_centers, best_radii = radii.sum(), centers, radii

    # fallback (should never happen)
    if best_centers is None:
        best_centers = init_hex_lattice(n)
        best_radii = lp_optimize(best_centers)

    return best_centers, best_radii, best_radii.sum()


# EVOLVE-BLOCK-END


def run_packing():
    """Run the circle packing constructor for n=26."""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """Optional visualiser – unchanged from original."""
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
    # visualize(c, r)   # uncomment to see the layout