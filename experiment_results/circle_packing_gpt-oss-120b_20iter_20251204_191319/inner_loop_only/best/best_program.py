# EVOLVE-BLOCK-START
"""Deterministic hex‑seed + cheap hill‑climb to boost total radius (n=26)"""

import numpy as np


def _max_radii(pts: np.ndarray) -> np.ndarray:
    """
    Compute the largest admissible radius for each centre.

    The radius of a circle centred at ``p`` is limited by:
      * the distance to the four sides of the unit square, and
      * half the distance to the closest other centre (to avoid overlap).

    Parameters
    ----------
    pts : np.ndarray, shape (n, 2)
        Candidate centre coordinates.

    Returns
    -------
    np.ndarray, shape (n,)
        Maximal radii for the supplied centres.
    """
    # pairwise Euclidean distances
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)                     # ignore self‑distance
    nearest = d.min(axis=1)                         # closest neighbour for each point

    # distance to the four borders of the unit square
    border = np.minimum.reduce([pts[:, 0], pts[:, 1],
                                1 - pts[:, 0], 1 - pts[:, 1]])

    return np.minimum(border, nearest / 2.0)


def _hex_seed() -> np.ndarray:
    """
    Produce the classic 6‑5‑6‑5‑4 hexagonal lattice that exactly yields 26 points
    inside the unit square.
    """
    sp = 0.18                     # horizontal centre distance
    vt = sp * np.sqrt(3) / 2      # vertical offset for a hex lattice
    mr = sp / 2                   # small margin from the square edges
    rows = [6, 5, 6, 5, 4]        # pattern → 26 points

    pts = []
    for r, cols in enumerate(rows):
        y = mr + r * vt
        x_offset = 0 if r % 2 == 0 else sp / 2
        for c in range(cols):
            x = mr + x_offset + c * sp
            pts.append([x, y])
    return np.array(pts[:26])


def construct_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Build a 26‑circle layout.
    Starts from the deterministic hexagonal seed and performs a very cheap
    hill‑climbing search (random local moves) to increase the sum of radii.
    The interface is unchanged: returns (centres, radii, total_radius_sum).
    """
    # ---- 1️⃣  initialise from the exact hex seed -------------------------
    best_pts = _hex_seed()
    best_rad = _max_radii(best_pts)
    best_sum = best_rad.sum()

    # ---- 2️⃣  cheap stochastic hill‑climb ---------------------------------
    #   * 3000 iterations keep runtime well under a few hundred ms
    #   * each step perturbs a single random centre by a tiny amount
    #   * if the new configuration improves the total radius we keep it
    rng = np.random.default_rng()
    for _ in range(3000):
        # pick a centre to move
        i = rng.integers(0, best_pts.shape[0])

        # propose a new location – Gaussian jitter, clipped to the unit square
        cand = best_pts.copy()
        jitter = rng.normal(scale=0.02, size=2)          # small step
        cand[i] = np.clip(cand[i] + jitter, 0.0, 1.0)

        # recompute radii for the candidate configuration
        rad = _max_radii(cand)
        s = rad.sum()

        if s > best_sum:                     # accept only improvements
            best_pts, best_rad, best_sum = cand, rad, s

    return best_pts, best_rad, float(best_sum)


# EVOLVE-BLOCK-END


# -------------------------------------------------------------------------
# Fixed helper / entry‑point (unchanged interface)
def run_packing():
    """Run the circle‑packing constructor for n=26."""
    return construct_packing()


def visualize(centers, radii):
    """Optional visualisation using matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(c, r, alpha=0.5, edgecolor="k"))
        ax.text(c[0], c[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing – sum of radii = {radii.sum():.6f}")
    plt.show()


if __name__ == "__main__":
    centers, radii, total = run_packing()
    print(f"Sum of radii: {total:.6f}")
    # visualize(centers, radii)   # uncomment to see a plot