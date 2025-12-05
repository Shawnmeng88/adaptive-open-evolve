# EVOLVE-BLOCK-START
"""Hexagonal‑grid circle packing for n=26 circles inside a unit square.

The algorithm searches for the largest possible equal radius r such that at
least 26 circles can be placed on a hexagonal (triangular) lattice that fits
inside the unit square.  The first 26 lattice points are then returned together
with the common radius.  This gives a much larger total sum of radii than the
previous handcrafted layout while guaranteeing:
    * every circle lies completely inside the square,
    * circles never overlap,
    * the function signature stays identical to the original program.
"""

import numpy as np
import math


def _hex_lattice_points(r: float):
    """
    Generate all lattice points of a hexagonal (triangular) grid with
    spacing 2 r that fit inside the unit square [0, 1]² while staying at
    distance r from the borders.

    Parameters
    ----------
    r: float
        Candidate radius.

    Returns
    -------
    List[Tuple[float, float]]
        All admissible centre coordinates.
    """
    pts = []
    # vertical distance between rows in a hexagonal packing
    dy = r * math.sqrt(3)

    # start from the bottom border + r
    y = r
    row = 0
    while y <= 1 - r + 1e-12:          # tiny epsilon for floating‑point safety
        # horizontal offset: even rows start at r, odd rows are shifted by r
        x_start = r if row % 2 == 0 else 2 * r
        x = x_start
        while x <= 1 - r + 1e-12:
            pts.append((x, y))
            x += 2 * r
        y += dy
        row += 1
    return pts


def _max_equal_radius(n: int, iters: int = 30) -> float:
    """
    Binary‑search the largest radius r for which at least *n* circles can be
    placed on a hexagonal grid inside the unit square.

    Parameters
    ----------
    n: int
        Desired number of circles.
    iters: int, optional
        Number of binary‑search iterations (default 30 → ~1e‑9 precision).

    Returns
    -------
    float
        The maximal feasible radius.
    """
    lo, hi = 0.0, 0.5          # radius can never exceed 0.5 in a unit square
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        cnt = len(_hex_lattice_points(mid))
        if cnt >= n:
            lo = mid           # mid is feasible → try larger
        else:
            hi = mid           # too large → shrink
    return lo


def construct_packing():
    """
    Build a packing of 26 circles using the optimal hexagonal grid found by
    ``_max_equal_radius``.  The function keeps the original return signature:

    Returns
    -------
    centers : np.ndarray, shape (26, 2)
        (x, y) coordinates of the circle centres.
    radii   : np.ndarray, shape (26,)
        Radii of the circles (all equal).
    sum_of_radii : float
        Sum of all radii (the primary fitness component).
    """
    n = 26

    # 1️⃣ Find the largest equal radius that accommodates at least 26 circles.
    best_r = _max_equal_radius(n)

    # 2️⃣ Generate the full lattice for that radius and keep the first 26 points.
    all_pts = _hex_lattice_points(best_r)
    selected = np.array(all_pts[:n])

    # 3️⃣ Radii are all identical.
    radii = np.full(n, best_r)

    # 4️⃣ Compute the fitness‑relevant metric.
    sum_radii = float(np.sum(radii))

    return selected, radii, sum_radii


# ----------------------------------------------------------------------
# The original helper (kept for compatibility – not used by the new strategy)
def compute_max_radii(centers):
    """
    Legacy routine retained for API compatibility.  It computes a feasible
    radius for each centre independently, but the new packing already guarantees
    validity, so the function simply returns the uniform radius used.
    """
    n = centers.shape[0]
    # all circles share the same radius – infer it from the first centre's
    # distance to the nearest border.
    x, y = centers[0]
    r = min(x, y, 1 - x, 1 - y)
    return np.full(n, r)


# EVOLVE-BLOCK-END


# ----------------------------------------------------------------------
# Fixed (non‑evolved) driver code – unchanged from the original program.
def run_packing():
    """Run the circle packing constructor for n=26."""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualise the circle packing.

    Parameters
    ----------
    centers : np.ndarray, shape (n, 2)
        Circle centre coordinates.
    radii   : np.ndarray, shape (n,)
        Circle radii.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(c, r, alpha=0.5, edgecolor='black'))
        ax.text(c[0], c[1], str(i), ha="center", va="center", fontsize=8)

    plt.title(f"Hexagonal Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # Uncomment to visualise the result:
    # visualize(centers, radii)