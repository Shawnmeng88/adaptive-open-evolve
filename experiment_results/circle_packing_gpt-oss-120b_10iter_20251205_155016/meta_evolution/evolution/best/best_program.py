"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def _place_centers(r, n=26):
    """
    Try to place up to `n` points on a hexagonal lattice with
    centre‑to‑centre distance 2*r inside the unit square.
    Returns an (m,2) array with m≤n points if successful,
    otherwise None.
    """
    centers = []
    row = 0
    y = r
    vert_step = np.sqrt(3) * r
    while y <= 1 - r + 1e-12 and len(centers) < n:
        # even rows start at x=r, odd rows are offset by r (i.e. 2r)
        offset = r if row % 2 == 0 else r + r
        x = offset
        while x <= 1 - r + 1e-12 and len(centers) < n:
            centers.append((x, y))
            x += 2 * r
        row += 1
        y += vert_step
    if len(centers) >= n:
        return np.array(centers[:n])
    return None


def construct_packing():
    """
    Construct a packing of 26 equal circles in the unit square.
    The algorithm searches for the largest feasible radius `r`
    using a binary search on a hexagonal lattice layout.
    Returns:
        centers (np.ndarray shape (26,2))
        radii   (np.ndarray shape (26,))
        sum_radii (float)
    """
    n = 26
    # binary search for maximal radius
    lo, hi = 0.0, 0.25  # radius cannot exceed 0.5, 0.25 is safe upper bound
    best_r = 0.0
    best_centers = None
    for _ in range(30):  # ~1e-9 precision
        mid = (lo + hi) / 2.0
        cand = _place_centers(mid, n)
        if cand is not None:
            best_r = mid
            best_centers = cand
            lo = mid
        else:
            hi = mid
    # safety fallback – if binary search failed (should not happen)
    if best_centers is None:
        # fall back to a simple grid
        xs = np.linspace(0.05, 0.95, 5)
        ys = np.linspace(0.05, 0.95, 6)
        xv, yv = np.meshgrid(xs, ys)
        pts = np.column_stack([xv.ravel(), yv.ravel()])[:n]
        best_centers = pts
        best_r = np.min([
            np.min(best_centers[:, 0]),               # distance to left wall
            np.min(best_centers[:, 1]),               # distance to bottom wall
            1 - np.max(best_centers[:, 0]),           # right wall
            1 - np.max(best_centers[:, 1])            # top wall
        ])
    radii = np.full(n, best_r)
    sum_radii = n * best_r
    return best_centers, radii, sum_radii
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
