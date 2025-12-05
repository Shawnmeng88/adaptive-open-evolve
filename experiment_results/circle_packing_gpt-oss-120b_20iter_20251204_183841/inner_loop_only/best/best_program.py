# EVOLVE-BLOCK-START
"""
Hex‑grid starter + cheap stochastic refinement.
The LP (or greedy fallback) gives the optimal radii for a *fixed* set of centres.
We therefore improve the packing by tweaking the centre positions with a tiny
hill‑climbing loop – each trial re‑solves the radius problem and keeps the
best layout found.
The approach stays deterministic (fixed seed) and stays well under the
character limit while usually raising the total sum of radii a few‑hundredths.
"""

import numpy as np
from typing import Tuple

# ----------------------------------------------------------------------
# 1️⃣  Linear‑program / greedy radii computation (unchanged)
# ----------------------------------------------------------------------
try:                                   # SciPy gives a true LP solution
    from scipy.optimize import linprog
except Exception:                      # fallback to cheap greedy repair
    linprog = None


def _max_radii(centers: np.ndarray) -> np.ndarray:
    """
    Maximise Σ r_i subject to border and non‑overlap constraints.
    Returns a feasible (and optimal when SciPy is present) radius vector.
    """
    n = len(centers)
    # distance to the four square sides
    border = np.minimum.reduce([centers[:, 0], centers[:, 1],
                               1 - centers[:, 0], 1 - centers[:, 1]])
    # pairwise centre distances
    dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)

    # ----- linear programme (if available) --------------------------------
    if linprog is not None:
        c = -np.ones(n)                     # maximise sum → minimise -sum
        A, b = [], []

        # r_i ≤ border_i
        for i, lim in enumerate(border):
            row = np.zeros(n)
            row[i] = 1
            A.append(row); b.append(lim)

        # r_i + r_j ≤ d_ij  (i<j)
        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i] = row[j] = 1
                A.append(row); b.append(dmat[i, j])

        res = linprog(c,
                      A_ub=np.array(A),
                      b_ub=np.array(b),
                      bounds=[(0, None)] * n,
                      method="highs")
        return res.x if res.success else np.zeros(n)

    # ----- greedy repair (fallback) ---------------------------------------
    rad = border.copy()
    for i in range(n):
        for j in range(i + 1, n):
            excess = rad[i] + rad[j] - dmat[i, j]
            if excess > 0:
                # shave the larger radius proportionally
                if rad[i] >= rad[j]:
                    rad[i] -= excess * rad[i] / (rad[i] + rad[j])
                else:
                    rad[j] -= excess * rad[j] / (rad[i] + rad[j])
    return rad


# ----------------------------------------------------------------------
# 2️⃣  Deterministic hexagonal seed
# ----------------------------------------------------------------------
def _hex_centers(n: int) -> np.ndarray:
    """Generate a deterministic hex‑like lattice inside the unit square."""
    rows, cols = 5, 6                     # 5×6 = 30 ≥ 26
    pts = []
    for r in range(rows):
        for c in range(cols):
            if len(pts) == n:
                break
            x = (c + 0.5 * (r % 2)) / cols
            y = (r + 0.5) / rows
            pts.append([x, y])
    return np.clip(np.array(pts), 0.01, 0.99)


# ----------------------------------------------------------------------
# 3️⃣  Tiny stochastic optimiser for the centre positions
# ----------------------------------------------------------------------
def _refine_centers(start: np.ndarray,
                    iters: int = 1500,
                    step: float = 0.03,
                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Hill‑climb the centre layout.
    * start – initial centre array (n,2)
    * iters – number of random proposals
    * step  – Gaussian std‑dev for a move (in unit‑square coordinates)
    * seed  – makes the whole run deterministic
    Returns the best (centres, radii, sum) triple discovered.
    """
    rng = np.random.default_rng(seed)
    best_c = start.copy()
    best_r = _max_radii(best_c)
    best_sum = best_r.sum()

    n = len(start)

    for _ in range(iters):
        # pick a circle and propose a new location
        i = rng.integers(n)
        proposal = best_c.copy()
        proposal[i] += rng.normal(scale=step, size=2)

        # keep inside the safe interior [0.01,0.99]
        proposal[i] = np.clip(proposal[i], 0.01, 0.99)

        # recompute radii for the proposal
        rad = _max_radii(proposal)
        s = rad.sum()
        if s > best_sum:                     # accept only improvements
            best_c, best_r, best_sum = proposal, rad, s

    return best_c, best_r, best_sum


# ----------------------------------------------------------------------
# 4️⃣  Public constructor – unchanged signature
# ----------------------------------------------------------------------
def construct_packing() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Return (centres, radii, sum_of_radii) for 26 circles.
    The function first creates a hex‑grid seed and then performs a short
    stochastic refinement to boost the total radius sum.
    """
    seed = _hex_centers(26)
    ctr, rad, total = _refine_centers(seed, iters=2000, step=0.025, seed=123)
    return ctr, rad, total


# EVOLVE-BLOCK-END


def run_packing():
    """Execute the constructor and return its results."""
    return construct_packing()


def visualize(centers: np.ndarray, radii: np.ndarray):
    """Optional quick Matplotlib visualisation."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for c, r in zip(centers, radii):
        ax.add_patch(Circle(c, r, alpha=0.4))
    plt.title(f'Sum of radii = {radii.sum():.4f}')
    plt.show()


if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s:.6f}")
    # visualize(c, r)   # uncomment to see the packing