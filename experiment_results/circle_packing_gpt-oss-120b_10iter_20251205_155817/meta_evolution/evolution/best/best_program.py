"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a packing of 26 equal circles inside the unit square
    using a hexagonal (triangular) lattice. The radius is maximized
    via binary search while ensuring at least 26 circles fit.
    Returns:
        centers (np.ndarray): shape (26, 2)
        radii   (np.ndarray): shape (26,)
        sum_radii (float): sum of all radii
    """
    import numpy as np

    def generate_positions(r):
        """Generate hexagonal lattice points for a given radius r."""
        pts = []
        dy = np.sqrt(3) * r
        y = r
        row = 0
        while y <= 1 - r + 1e-12:
            if row % 2 == 0:
                x_start = r
            else:
                x_start = r + r  # offset by half the horizontal spacing
            x = x_start
            while x <= 1 - r + 1e-12:
                pts.append((x, y))
                x += 2 * r
            y += dy
            row += 1
        return pts

    # Binary search for maximal radius allowing at least 26 circles
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2.0
        cnt = len(generate_positions(mid))
        if cnt >= 26:
            lo = mid
        else:
            hi = mid
    r_opt = lo

    # Generate positions and keep first 26
    all_pts = generate_positions(r_opt)
    centers = np.array(all_pts[:26])
    radii = np.full(26, r_opt)
    sum_radii = float(radii.sum())
    return centers, radii, sum_radii

def compute_max_radii(centers):
    """
    Fallback routine kept for compatibility; not used in the
    optimized construction. Returns radii limited by borders
    and pairwise distances.
    """
    import numpy as np
    n = centers.shape[0]
    radii = np.ones(n)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale
    return radii
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
