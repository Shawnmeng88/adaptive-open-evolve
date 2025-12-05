# EVOLVE-BLOCK-START
"""
Improved circle packing for n=26.

Approach
--------
1. Initialise centres on a dense hexagonal lattice (deterministic).
2. Compute maximal feasible radii for those centres.
3. Perform a lightweight deterministic hill‑climbing:
   - Randomly pick a circle and propose a small move.
   - Re‑compute all radii.
   - Accept the move only if the total sum of radii increases.
   - The move size slowly shrinks, allowing finer adjustments later.
4. Return the final centres, radii and the sum of radii.

The algorithm stays well within the 2 s budget (≈ 3 k iterations,
each O(n²) with n=26) and guarantees validity because
`compute_max_radii` always respects the square borders and the
non‑overlap condition.
"""

import numpy as np


def construct_packing():
    """Construct a high‑quality feasible packing of 26 circles.

    Returns
    -------
    centers : np.ndarray, shape (26, 2)
        (x, y) coordinates of the circle centres.
    radii : np.ndarray, shape (26,)
        Radii of the circles, guaranteed to be non‑overlapping and inside
        the unit square.
    sum_of_radii : float
        Sum of all radii – the optimisation target.
    """
    n = 26

    # ------------------------------------------------------------------
    # 1. Hexagonal lattice initialisation (deterministic)
    # ------------------------------------------------------------------
    spacing = 0.20                     # horizontal spacing
    dy = spacing * np.sqrt(3) / 2.0    # vertical spacing for triangular grid

    pts = []
    y = 0.0
    row = 0
    while y <= 1.0 + 1e-9:
        x_offset = (spacing / 2.0) if (row % 2) else 0.0
        x = x_offset
        while x <= 1.0 + 1e-9:
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                pts.append((x, y))
            x += spacing
        row += 1
        y = row * dy

    pts = np.array(pts, dtype=float)
    if pts.shape[0] < n:
        raise RuntimeError("Not enough lattice points generated.")
    centers = pts[:n].copy()

    # ------------------------------------------------------------------
    # 2. Initial radii
    # ------------------------------------------------------------------
    radii = compute_max_radii(centers)
    sum_of_radii = float(radii.sum())

    # ------------------------------------------------------------------
    # 3. Simple deterministic hill‑climbing
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)   # fixed seed → reproducible
    step = 0.06                       # initial move size
    min_step = 0.005
    decay = 0.9995                    # geometric decay per iteration
    max_iter = 3000

    for _ in range(max_iter):
        # pick a circle to perturb
        i = rng.integers(n)
        # propose a random displacement
        theta = rng.random() * 2.0 * np.pi
        delta = np.array([np.cos(theta), np.sin(theta)]) * step
        new_center = centers[i] + delta
        # keep inside the unit square (the radii computation will handle the exact
        # border distance, but we keep the centre inside to avoid pathological cases)
        new_center = np.clip(new_center, 0.0, 1.0)

        # evaluate the move
        old_center = centers[i].copy()
        centers[i] = new_center
        new_radii = compute_max_radii(centers)
        new_sum = float(new_radii.sum())

        if new_sum > sum_of_radii:          # accept only improvements
            radii = new_radii
            sum_of_radii = new_sum
        else:                               # revert
            centers[i] = old_center

        # gradually shrink the step size
        step = max(step * decay, min_step)

    # final safety recomputation (ensures radii correspond to final centres)
    radii = compute_max_radii(centers)
    sum_of_radii = float(radii.sum())

    return centers, radii, sum_of_radii


def compute_max_radii(centers):
    """Compute the largest feasible radii for a given set of centres.

    For each centre the radius is limited by:
      * distance to the four sides of the unit square,
      * half of the distance to the nearest other centre (to avoid overlap).

    The returned radii are guaranteed to be non‑negative, respect the
    square borders and guarantee pairwise non‑overlap.
    """
    # distance to each side of the unit square
    border_dist = np.minimum.reduce([
        centers[:, 0],               # left
        centers[:, 1],               # bottom
        1.0 - centers[:, 0],         # right
        1.0 - centers[:, 1]          # top
    ])

    # start with border distances
    radii = border_dist.copy()

    # pairwise centre distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]   # (n, n, 2)
    dists = np.linalg.norm(diff, axis=2)                         # (n, n)

    # ignore self‑distance
    np.fill_diagonal(dists, np.inf)

    # nearest neighbour distance for each centre
    nearest = np.min(dists, axis=1)

    # enforce non‑overlap: radius ≤ half the nearest neighbour distance
    radii = np.minimum(radii, nearest / 2.0)

    # numerical safety
    radii = np.clip(radii, 0.0, None)
    return radii


# EVOLVE-BLOCK-END


# ----------------------------------------------------------------------
# Public API (unchanged)
# ----------------------------------------------------------------------
def run_packing():
    """Run the circle packing constructor for n=26."""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """Visualize the circle packing."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centers, radii)):
        circ = Circle(c, r, alpha=0.5, edgecolor='k')
        ax.add_patch(circ)
        ax.text(c[0], c[1], str(i), ha="center", va="center", fontsize=8)

    plt.title(f"Circle Packing (n={len(centers)}, sum={np.sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # Uncomment to see the picture:
    # visualize(centers, radii)