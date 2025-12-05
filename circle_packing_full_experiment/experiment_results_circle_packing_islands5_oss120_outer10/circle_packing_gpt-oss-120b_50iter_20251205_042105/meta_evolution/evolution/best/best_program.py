# EVOLVE-BLOCK-START
import numpy as np


def construct_packing():
    """Construct a packing of 26 circles in the unit square.

    Uses a 5×5 regular square lattice with spacing 1/5 (so each circle
    has radius 0.1) and adds a 26th circle in a corner where it
    minimally reduces the radii of the existing circles.
    """
    k = 5
    spacing = 1.0 / k          # distance between neighbouring lattice points
    margin = spacing / 2.0      # distance from the outermost points to the square border

    xs = margin + np.arange(k) * spacing
    ys = margin + np.arange(k) * spacing
    grid_centers = np.array([(x, y) for y in ys for x in xs])

    # candidate extra positions – corners just inside the border
    extra_candidates = np.array([
        [margin / 2.0, margin / 2.0],
        [1.0 - margin / 2.0, margin / 2.0],
        [margin / 2.0, 1.0 - margin / 2.0],
        [1.0 - margin / 2.0, 1.0 - margin / 2.0],
    ])

    best_sum = -1.0
    best_centers = None
    best_radii = None

    for extra in extra_candidates:
        centers = np.vstack([grid_centers, extra])
        radii = compute_max_radii(centers)
        s = radii.sum()
        if s > best_sum:
            best_sum = s
            best_centers = centers
            best_radii = radii

    # Fallback (should never happen)
    if best_centers is None:
        extra = np.array([margin / 2.0, margin / 2.0])
        best_centers = np.vstack([grid_centers, extra])
        best_radii = compute_max_radii(best_centers)
        best_sum = best_radii.sum()

    return best_centers, best_radii, float(best_sum)


def compute_max_radii(centers):
    """Compute a safe radius for each centre."""
    border_dist = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    nearest = np.min(dists, axis=1)
    radii = np.minimum(border_dist, 0.5 * nearest)
    return np.clip(radii, 0.0, None)
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
