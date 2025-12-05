"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Build a hexagonal lattice that fits exactly 26 circles inside the unit square,
    then locally improve the placement by nudging circles to increase their
    individual radii.  The improvement loop is lightweight and deterministic
    enough to stay within the evaluation budget while typically raising the
    total sum of radii.
    Returns:
        centers (np.ndarray): (26, 2) array of circle centres.
        radii   (np.ndarray): (26,) array of per‑circle radii.
        sum_radii (float)  : total of all radii.
    """
    n = 26
    lo, hi = 0.0, 1.0
    best_centers = None

    # --- binary search for the tightest hex lattice spacing that still yields n circles ---
    for _ in range(40):
        s = (lo + hi) / 2.0
        centers = _place_hex_lattice(s, n)
        if centers.shape[0] == n:
            lo = s
            best_centers = centers
        else:
            hi = s

    if best_centers is None:                     # fallback – should never happen
        best_centers = _place_hex_lattice(lo, n)

    centers = best_centers.copy()

    # --- initial radii (boundary + neighbour constraints) ---
    radii = _max_radii(centers)

    # --- local improvement: try to move each circle slightly to enlarge its radius ---
    _local_improve(centers, radii, iterations=1500, step=0.015)

    sum_radii = radii.sum()
    return centers, radii, sum_radii


def _max_radii(centers):
    """Compute the maximal feasible radius for each centre given current positions."""
    # distance to the four sides of the unit square
    dist_boundary = np.minimum.reduce([
        centers[:, 0],               # left
        1.0 - centers[:, 0],         # right
        centers[:, 1],               # bottom
        1.0 - centers[:, 1]          # top
    ])

    # pairwise centre distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)          # ignore self‑distance
    min_center_dist = dists.min(axis=1)       # nearest neighbour distance

    # radius limited by boundary and by half the nearest neighbour distance
    return np.minimum(dist_boundary, min_center_dist / 2.0)


def _local_improve(centers, radii, iterations=1000, step=0.01):
    """
    Simple stochastic hill‑climbing: repeatedly pick a circle, propose a small
    random displacement, and keep the move if it does not violate constraints
    and yields a non‑decreasing total sum of radii.
    """
    n = centers.shape[0]
    rng = np.random.default_rng(42)

    for _ in range(iterations):
        i = rng.integers(0, n)                     # circle to move
        # propose a displacement bounded by `step`
        delta = rng.uniform(-step, step, size=2)
        new_center = centers[i] + delta

        # enforce staying inside the unit square (using current radius as margin)
        r_i = radii[i]
        if not (r_i <= new_center[0] <= 1.0 - r_i and r_i <= new_center[1] <= 1.0 - r_i):
            continue

        # compute new radius for this circle given the proposed centre
        # distance to boundaries
        dist_boundary = min(new_center[0],
                            1.0 - new_center[0],
                            new_center[1],
                            1.0 - new_center[1])

        # distance to other centres
        other_idxs = np.arange(n) != i
        dists_to_others = np.linalg.norm(centers[other_idxs] - new_center, axis=1)
        min_neighbour = dists_to_others.min() if dists_to_others.size else np.inf
        new_radius_i = min(dist_boundary, min_neighbour / 2.0)

        if new_radius_i < r_i:          # we only accept moves that do not shrink this circle
            continue

        # compute radii of the affected neighbours (they may shrink because we moved closer)
        # we conservatively recompute all radii; the cost is acceptable for n=26
        old_sum = radii.sum()
        old_center = centers[i].copy()
        centers[i] = new_center
        new_radii = _max_radii(centers)
        new_sum = new_radii.sum()

        if new_sum >= old_sum:          # accept if total sum does not decrease
            radii[:] = new_radii
        else:
            centers[i] = old_center      # revert


def _place_hex_lattice(s, max_circles):
    """
    Generate up to `max_circles` centres on a hexagonal lattice with spacing `s`.
    The lattice is trimmed to stay completely inside the unit square.
    """
    row_height = s * np.sqrt(3) / 2.0
    margin = s / 2.0

    centers = []
    y = margin
    row = 0
    while y <= 1.0 - margin and len(centers) < max_circles:
        x_start = margin if row % 2 == 0 else margin + s / 2.0
        x = x_start
        while x <= 1.0 - margin and len(centers) < max_circles:
            centers.append([x, y])
            x += s
        y += row_height
        row += 1

    return np.array(centers) if centers else np.empty((0, 2))
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
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
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)
