# EVOLVE-BLOCK-START
"""
Improved constructor for packing 26 circles in a unit square.

Strategy
--------
* Use a hexagonal (close‑packed) lattice – the densest packing of equal circles.
* Perform a binary search for the largest possible equal radius *r* such that at
  least 26 lattice points fit inside the unit square while staying at distance
  *r* from the borders.
* The first 26 lattice points are taken as the centres.  Because the lattice
  respects the minimal centre‑to‑centre distance of *2·r*, the circles are
  guaranteed not to overlap.
* The fitness proxy is the sum of the radii (26 · r).  This deterministic scheme
  yields a higher sum than the previous rectangular‑grid approach while keeping
  validity = 1.0.
"""

import numpy as np
import math


def _hex_lattice_points(r: float):
    """
    Generate centres of a hexagonal lattice with spacing that respects a
    given radius *r*.  Points are guaranteed to be at least *r* away from the
    unit‑square borders.

    Parameters
    ----------
    r : float
        Desired circle radius.

    Returns
    -------
    pts : np.ndarray, shape (m, 2)
        All lattice points that fit inside the square (m ≥ 0).
    """
    # vertical step of a hexagonal row
    v_step = math.sqrt(3) * r
    # horizontal step within a row
    h_step = 2 * r

    pts = []
    y = r
    row = 0
    while y <= 1 - r + 1e-12:          # include the last feasible row
        # odd rows are offset by one horizontal step
        x_start = r if row % 2 == 0 else r + h_step / 2
        x = x_start
        while x <= 1 - r + 1e-12:
            pts.append((x, y))
            x += h_step
        y += v_step
        row += 1

    return np.array(pts)


def _max_equal_radius(num_circles: int = 26, eps: float = 1e-7) -> float:
    """
    Binary‑search the largest radius that allows at least *num_circles*
    points of a hexagonal lattice to fit into the unit square.

    Parameters
    ----------
    num_circles : int
        Desired number of circles (default 26).
    eps : float
        Desired precision of the radius.

    Returns
    -------
    r_opt : float
        Maximal feasible radius.
    """
    lo, hi = 0.0, 0.5          # radius cannot exceed 0.5 in a unit square
    for _ in range(50):       # enough iterations for double‑precision accuracy
        mid = (lo + hi) / 2.0
        pts = _hex_lattice_points(mid)
        if pts.shape[0] >= num_circles:
            lo = mid           # feasible → try larger
        else:
            hi = mid           # not enough points → shrink
        if hi - lo < eps:
            break
    return lo


def construct_packing():
    """
    Build a deterministic arrangement of 26 equal circles inside the unit square
    using a hexagonal lattice with the maximal possible radius.

    Returns
    -------
    centers : np.ndarray, shape (26, 2)
        (x, y) coordinates of the circle centres.
    radii   : np.ndarray, shape (26,)
        Identical admissible radius for each centre.
    sum_radii : float
        Sum of all radii (the value used for fitness evaluation).
    """
    # 1️⃣ Find the largest feasible equal radius
    r_opt = _max_equal_radius(26)

    # 2️⃣ Generate the lattice and keep the first 26 points
    all_centres = _hex_lattice_points(r_opt)
    centres = all_centres[:26]

    # 3️⃣ Radii are all identical
    radii = np.full(26, r_opt, dtype=float)

    # 4️⃣ Sum of radii – the fitness proxy
    sum_radii = float(radii.sum())

    return centres, radii, sum_radii
# EVOLVE-BLOCK-END


def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5, edgecolor='k')
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center", fontsize=8)

    plt.title(f"Circle Packing (n={len(centers)}, sum={np.sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")

    # Uncomment to visualise the result:
    # visualize(centers, radii)