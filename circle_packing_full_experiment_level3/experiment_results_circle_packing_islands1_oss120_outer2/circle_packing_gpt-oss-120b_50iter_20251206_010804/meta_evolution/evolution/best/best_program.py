"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    """
    Construct a packing of 26 circles inside the unit square.
    Uses a regular 5×5 square grid (spacing 0.2, radius 0.1) for 25 circles
    and adds one extra small circle in a grid cell to increase the total sum.
    Returns:
        centers (np.ndarray): shape (26, 2) array of circle centers.
        radii   (np.ndarray): shape (26,) array of circle radii.
        sum_radii (float): sum of radii.
    """
    # --- base square grid -------------------------------------------------
    # spacing chosen so that circles just touch each other and the borders
    spacing = 0.2
    base_radius = spacing / 2.0  # 0.1

    # generate grid coordinates starting at base_radius and stepping by spacing
    coords = np.arange(base_radius, 1.0 - base_radius + 1e-12, spacing)
    xv, yv = np.meshgrid(coords, coords)
    grid_centers = np.column_stack((xv.ravel(), yv.ravel()))  # 25 points

    # --- extra circle -----------------------------------------------------
    # pick the centre of the first grid cell (0.2, 0.2) which is not occupied
    extra_center = np.array([base_radius + spacing / 2.0,
                             base_radius + spacing / 2.0])

    # distance to the square borders
    border_dist = min(extra_center[0], extra_center[1],
                      1.0 - extra_center[0], 1.0 - extra_center[1])

    # distance to existing grid circles, subtracting their radius
    dists = np.linalg.norm(grid_centers - extra_center, axis=1) - base_radius
    neighbor_limit = dists.min()

    extra_radius = min(border_dist, neighbor_limit)
    if extra_radius <= 0:
        # fallback: if for any reason the extra circle is infeasible,
        # just omit it (should not happen with the chosen parameters)
        extra_center = None
        extra_radius = 0.0

    # --- assemble final arrays --------------------------------------------
    if extra_center is not None:
        centers = np.vstack((grid_centers, extra_center))
        radii = np.full(26, base_radius)
        radii[-1] = extra_radius
    else:
        centers = grid_centers
        radii = np.full(25, base_radius)

    sum_radii = float(radii.sum())
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
