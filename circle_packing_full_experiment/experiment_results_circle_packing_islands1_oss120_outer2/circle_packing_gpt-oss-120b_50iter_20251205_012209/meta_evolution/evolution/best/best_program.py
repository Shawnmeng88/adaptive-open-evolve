# EVOLVE-BLOCK-START
import math
import numpy as np

def _generate_hex_grid(radius: float) -> np.ndarray:
    """Generate a hexagonal lattice of circle centers for a given radius."""
    if radius <= 0.0:
        return np.empty((0, 2), dtype=float)

    dx = 2.0 * radius
    dy = math.sqrt(3.0) * radius
    eps = 1e-12

    points = []
    y = radius
    row = 0
    while y <= 1.0 - radius + eps:
        x_offset = 0.0 if (row % 2 == 0) else radius
        x = radius + x_offset
        while x <= 1.0 - radius + eps:
            points.append((x, y))
            x += dx
        y += dy
        row += 1

    return np.asarray(points, dtype=float)


def _can_place(radius: float, required: int) -> bool:
    """Return True if a hexagonal grid with the given radius can contain at least `required` circles."""
    return _generate_hex_grid(radius).shape[0] >= required


def construct_packing():
    """
    Construct a dense equal‑radius hexagonal packing of 26 circles.
    The maximal feasible radius is found via binary search.
    Returns:
        centers (np.ndarray): shape (26, 2)
        radii   (np.ndarray): shape (26,)
        sum_radii (float)
    """
    n = 26
    lo, hi = 0.0, 0.5  # radius cannot exceed 0.5 in a unit square

    # binary search with high precision
    for _ in range(60):
        mid = (lo + hi) * 0.5
        if _can_place(mid, n):
            lo = mid
        else:
            hi = mid

    radius = lo
    centers = _generate_hex_grid(radius)[:n]
    radii = np.full(centers.shape[0], radius, dtype=float)
    return centers, radii, float(radii.sum())
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
