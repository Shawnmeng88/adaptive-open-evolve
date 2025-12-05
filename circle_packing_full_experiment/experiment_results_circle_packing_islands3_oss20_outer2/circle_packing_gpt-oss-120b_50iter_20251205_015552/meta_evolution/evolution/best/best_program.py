# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    """
    Build a packing of 26 circles inside the unit square using a hexagonal lattice.
    The function binary‑searches the largest uniform radius that fits at least 26 circles.
    Returns:
        centers (np.ndarray): shape (26, 2)
        radii   (np.ndarray): shape (26,)
        sum_of_radii (float)
    """
    n = 26

    def generate_hex_positions(r):
        """Generate hexagonal lattice points for a given circle radius r."""
        spacing = 2.0 * r
        vert = spacing * np.sqrt(3.0) / 2.0
        pts = []
        y = r
        row = 0
        while y <= 1.0 - r + 1e-12:
            offset = (row % 2) * r
            x = r + offset
            while x <= 1.0 - r + 1e-12:
                pts.append([x, y])
                x += spacing
            y += vert
            row += 1
        return np.array(pts)

    # Binary search for the maximum feasible uniform radius
    lo, hi = 0.0, 0.5  # radius cannot exceed 0.5 in a unit square
    best_r = 0.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        pts = generate_hex_positions(mid)
        if pts.shape[0] >= n:
            best_r = mid
            lo = mid
        else:
            hi = mid

    # Final placement using the best radius found
    all_pts = generate_hex_positions(best_r)
    centers = all_pts[:n]
    radii = np.full(n, best_r)
    sum_radii = radii.sum()
    return centers, radii, sum_radii
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
