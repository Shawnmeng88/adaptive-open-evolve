# EVOLVE-BLOCK-START
"""Improved 26‑circle packing.

The centres start on a 5×6 interior grid (first 26 points).  A tiny
random‑local‑search moves individual centres while always recomputing the
maximal feasible radii (border limit and half the nearest‑neighbour
distance).  The best layout found is returned together with its radii and
the total radius sum.
"""

import numpy as np


def construct_packing():
    """
    Returns
    -------
    centers : np.ndarray, shape (26, 2)
        Optimised centre coordinates.
    radii   : np.ndarray, shape (26,)
        Maximal radii for the returned centres.
    sum_radii : float
        Sum of the radii (the optimisation target).
    """
    # ----- 1️⃣  initialise a regular 5×6 interior grid -----------------
    rows, cols = 5, 6                         # 30 points → keep 26
    xs = np.arange(1, cols + 1) / (cols + 1)   # stay away from the borders
    ys = np.arange(1, rows + 1) / (rows + 1)

    grid = np.array([[x, y] for y in ys for x in xs])
    centres = grid[:26].astype(float)          # (26, 2)

    # ----- 2️⃣  evaluate the starting layout ---------------------------
    radii = compute_max_radii(centres)
    best_sum = radii.sum()
    best_centres = centres.copy()

    # ----- 3️⃣  tiny stochastic hill‑climb ------------------------------
    rng = np.random.default_rng(42)            # deterministic seed
    step = 0.02                                # maximal move per coordinate
    for _ in range(3000):                     # cheap but enough iterations
        i = rng.integers(0, 26)                # pick a circle to jiggle
        cand = best_centres.copy()
        # propose a new position and keep it inside the unit square
        cand[i] += rng.uniform(-step, step, size=2)
        np.clip(cand[i], 0.0, 1.0, out=cand[i])

        # recompute radii for the perturbed layout
        cand_radii = compute_max_radii(cand)
        cand_sum = cand_radii.sum()

        # accept only if the total radius sum improves
        if cand_sum > best_sum:
            best_sum = cand_sum
            best_centres = cand
            radii = cand_radii                     # keep the radii of the best layout

    return best_centres, radii, float(best_sum)


def compute_max_radii(centres: np.ndarray) -> np.ndarray:
    """
    Maximise each radius under the constraints

        r_i ≤ distance to the nearest square side
        r_i ≤ ½·min_{j≠i}‖c_i‑c_j‖

    This yields a feasible (non‑overlapping) packing for the supplied centres.
    """
    # distance to the four sides of the unit square
    border = np.minimum.reduce(
        [centres[:, 0], centres[:, 1], 1 - centres[:, 0], 1 - centres[:, 1]]
    )

    # pairwise centre distances
    diff = centres[:, None, :] - centres[None, :, :]   # (n,n,2)
    dists = np.linalg.norm(diff, axis=2)              # (n,n)

    # ignore self‑distance
    np.fill_diagonal(dists, np.inf)
    nearest = np.min(dists, axis=1)

    # admissible radius is the tighter of the two limits
    return np.minimum(border, nearest / 2.0)


# EVOLVE-BLOCK-END


def run_packing():
    """Run the circle packing constructor for n=26."""
    centres, radii, sum_radii = construct_packing()
    return centres, radii, sum_radii


def visualize(centres, radii):
    """Optional visualisation helper."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centres, radii)):
        circ = Circle(c, r, edgecolor="black", facecolor="C0", alpha=0.5)
        ax.add_patch(circ)
        ax.text(c[0], c[1], str(i), ha="center", va="center", fontsize=8)

    plt.title(f"Circle Packing (n={len(centres)}, sum={np.sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centres, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # visualize(centres, radii)   # uncomment to see a plot