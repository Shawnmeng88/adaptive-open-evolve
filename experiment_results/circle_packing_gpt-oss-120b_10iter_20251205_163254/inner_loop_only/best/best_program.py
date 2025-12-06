"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a packing of 26 circles inside the unit square.
    Uses a regular 5×5 grid (25 points) plus one extra point.
    Returns the centers array, radii array and their sum.
    """
    # ----- generate centers -----
    # 5×5 grid with spacing 0.2 and offset 0.1 → points at 0.1,0.3,…,0.9
    grid_vals = np.linspace(0.1, 0.9, 5)
    centers = np.array(
        [(x, y) for y in grid_vals for x in grid_vals], dtype=float
    )  # shape (25, 2)

    # add one extra point (center of the top edge) that is not already in the grid
    extra = np.array([[0.5, 0.25]])
    centers = np.vstack([centers, extra])  # now (26, 2)

    # ----- compute radii -----
    radii = compute_max_radii(centers)

    return centers, radii, np.sum(radii)


def compute_max_radii(centers, eps=1e-8, max_iter=1000):
    """
    Compute a feasible set of radii that maximises their sum.
    Starts from the distance to the square borders and then
    iteratively reduces radii to satisfy pairwise non‑overlap constraints.
    """
    n = centers.shape[0]

    # initial radii limited only by the square borders
    radii = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )

    # iterative refinement
    for _ in range(max_iter):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                # distance between centres
                d = np.linalg.norm(centers[i] - centers[j])
                # allowed sum of radii for non‑overlap
                max_sum = d
                cur_sum = radii[i] + radii[j]
                if cur_sum > max_sum + eps:
                    # reduce the larger radius first; keep the smaller unchanged if possible
                    excess = cur_sum - max_sum
                    if radii[i] >= radii[j]:
                        reduction_i = min(excess, radii[i] - eps)
                        radii[i] -= reduction_i
                        excess -= reduction_i
                        if excess > eps:
                            radii[j] = max(radii[j] - excess, eps)
                    else:
                        reduction_j = min(excess, radii[j] - eps)
                        radii[j] -= reduction_j
                        excess -= reduction_j
                        if excess > eps:
                            radii[i] = max(radii[i] - excess, eps)
                    changed = True
        if not changed:
            break

    # final safety clamp (no negative radii)
    radii = np.clip(radii, 0.0, None)
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
