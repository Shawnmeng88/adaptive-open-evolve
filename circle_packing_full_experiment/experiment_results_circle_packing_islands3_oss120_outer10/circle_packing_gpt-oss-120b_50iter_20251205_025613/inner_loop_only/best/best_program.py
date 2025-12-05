# EVOLVE-BLOCK-START
import numpy as np

def hex_points(r):
    """Generate hexagonal lattice points with spacing 2r inside the unit square."""
    if r == 0:
        return []
    pts = []
    dy = np.sqrt(3) * r
    y, row = r, 0
    while y <= 1 - r + 1e-12:
        offset = r if row % 2 == 0 else 2 * r
        x = offset
        while x <= 1 - r + 1e-12:
            pts.append((x, y))
            x += 2 * r
        y += dy
        row += 1
    return pts

def construct_packing():
    """Construct a packing of 26 circles using a hexagonal lattice."""
    n = 26
    lo, hi = 0.0, 0.5
    for _ in range(40):  # refine radius estimate
        mid = (lo + hi) / 2
        if len(hex_points(mid)) >= n:
            lo = mid
        else:
            hi = mid
    r = lo
    centers = np.array(hex_points(r)[:n])
    radii = np.full(len(centers), r)
    return centers, radii, np.sum(radii)
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
