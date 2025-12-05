"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a packing of 26 equal circles inside the unit square using a
    hexagonal (triangular) lattice.  The binary‑search finds the largest
    common radius that allows at least 26 circles to fit.
    """
    n = 26
    lo, hi = 0.0, 0.5  # radius cannot exceed 0.5 in a unit square

    for _ in range(50):               # enough iterations for high precision
        mid = (lo + hi) / 2.0
        if _can_place(n, mid):
            lo = mid
        else:
            hi = mid

    r = lo
    centers = _generate_positions(n, r)
    radii = np.full(n, r)
    sum_radii = float(r * n)
    return centers, radii, sum_radii


def _can_place(n, r):
    """Return True if a hexagonal lattice with radius *r* can accommodate *n* circles."""
    if r <= 0:
        return False

    vert = r * np.sqrt(3)          # vertical distance between rows
    y = r
    row = 0
    count = 0
    while y + r <= 1.0 + 1e-12:
        offset = 0.0 if row % 2 == 0 else r
        x = r + offset
        while x + r <= 1.0 + 1e-12:
            count += 1
            if count >= n:
                return True
            x += 2.0 * r
        y += vert
        row += 1
    return False


def _generate_positions(n, r):
    """Create up to *n* centre positions for circles of radius *r* on the lattice."""
    positions = []
    vert = r * np.sqrt(3)
    y = r
    row = 0
    while y + r <= 1.0 + 1e-12 and len(positions) < n:
        offset = 0.0 if row % 2 == 0 else r
        x = r + offset
        while x + r <= 1.0 + 1e-12 and len(positions) < n:
            positions.append([x, y])
            x += 2.0 * r
        y += vert
        row += 1
    return np.array(positions)


def compute_max_radii(centers):
    """
    Compatibility placeholder – all circles share the same radius already
    computed in ``construct_packing``.  Return that uniform radius.
    """
    if centers.size == 0:
        return np.array([])
    # Minimum distance to the four borders determines the feasible uniform radius
    mins = np.minimum.reduce([centers[:, 0], centers[:, 1],
                             1.0 - centers[:, 0], 1.0 - centers[:, 1]])
    r = mins.min()
    return np.full(len(centers), r)
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
