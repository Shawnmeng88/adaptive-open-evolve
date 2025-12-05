# EVOLVE-BLOCK-START
"""Optimised constructor for packing 26 circles in a unit square.

Approach
--------
* Initialise 26 centres on a 5×5 grid plus one extra point (as before).
* Run a simple stochastic local‑search (fixed random seed for reproducibility):
  - Randomly perturb a single centre by a small amount.
  - Re‑compute the maximal admissible radii for the new layout.
  - Keep the change if the total sum of radii improves.
* After a fixed number of iterations return the best layout found.

The optimisation works only on the centre positions; the radii are always
computed by the deterministic LP‑style greedy refinement in `compute_max_radii`,
so validity (no overlaps, all circles inside the square) is guaranteed.
"""

import numpy as np


def construct_packing():
    """Return centres, radii and total sum of radii for 26 circles."""
    # ---- 1. initialise a regular 5×5 grid + one extra centre ----------
    grid_vals = np.linspace(0.1, 0.9, 5)                     # 0.1,0.3,…,0.9
    grid = np.array([[x, y] for y in grid_vals for x in grid_vals])
    extra = np.array([[0.5, 0.55]])                         # distinct from (0.5,0.5)
    centres = np.vstack([grid, extra])                     # (26,2)

    # ---- 2. stochastic improvement of centre positions -----------------
    np.random.seed(0)                                      # deterministic run
    best_c = centres.copy()
    best_r = compute_max_radii(best_c)
    best_sum = best_r.sum()

    max_iters = 3000                                       # cheap enough
    step = 0.04                                            # max move per coord

    for _ in range(max_iters):
        cand_c = best_c.copy()
        i = np.random.randint(26)                          # pick a circle
        move = (np.random.rand(2) - 0.5) * step
        cand_c[i] = np.clip(cand_c[i] + move, 0.0, 1.0)     # stay inside square
        cand_r = compute_max_radii(cand_c)
        cand_sum = cand_r.sum()
        if cand_sum > best_sum:                            # accept improvement
            best_c, best_r, best_sum = cand_c, cand_r, cand_sum

    # ---- 3. final radii for the best layout ---------------------------
    radii = compute_max_radii(best_c)
    return best_c, radii, float(radii.sum())


def compute_max_radii(centers, max_iter: int = 500, eps: float = 1e-9):
    """Greedy LP‑style refinement to obtain maximal admissible radii."""
    n = len(centers)
    # start with border‑limited radii
    radii = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )
    for _ in range(max_iter):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d + eps:
                    excess = radii[i] + radii[j] - d
                    if radii[i] >= radii[j]:
                        dec = min(excess, radii[i])
                        radii[i] -= dec
                        excess -= dec
                        if excess > 0:
                            radii[j] = max(radii[j] - excess, 0.0)
                    else:
                        dec = min(excess, radii[j])
                        radii[j] -= dec
                        excess -= dec
                        if excess > 0:
                            radii[i] = max(radii[i] - excess, 0.0)
                    changed = True
        if not changed:
            break
    return np.clip(radii, 0.0, None)


# EVOLVE-BLOCK-END


def run_packing():
    """Run the circle packing constructor for n=26."""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """Optional visualisation of the packing."""
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

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # visualize(centers, radii)   # uncomment to see the packing