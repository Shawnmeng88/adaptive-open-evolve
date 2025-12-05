"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a layout for 26 circles inside the unit square and compute
    the optimal radii using linear programming to maximize the total sum.
    """
    n = 26

    # Hexagonal (triangular) lattice placement – generally yields larger gaps
    # between points than a simple rectangular grid.
    margin = 0.02          # small margin from the square edges
    dx = 0.2               # horizontal spacing
    dy = dx * (3 ** 0.5) / 2  # vertical spacing for hexagonal packing

    points = []
    row = 0
    y = margin
    while y <= 1 - margin:
        offset = (row % 2) * (dx / 2)
        x = margin + offset
        while x <= 1 - margin:
            points.append([x, y])
            x += dx
        row += 1
        y = margin + row * dy

    points = np.array(points)

    # If we have more than needed, randomly pick n of them for diversity.
    if len(points) > n:
        np.random.shuffle(points)
        centers = points[:n]
    else:
        centers = points[:n]

    radii = compute_max_radii(centers)
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Solve a linear program that maximises the sum of radii subject to:
      * each radius ≤ distance to the nearest square side,
      * for every pair i, j: r_i + r_j ≤ distance between centres i and j,
      * radii ≥ 0.
    """
    n = centers.shape[0]

    # Distance from each centre to the four sides of the unit square
    border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )

    # Pairwise centre distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))

    # Build inequality matrix A_ub * r ≤ b_ub
    pair_i, pair_j = np.triu_indices(n, k=1)
    m = n + len(pair_i)                     # total number of constraints
    A = np.zeros((m, n))
    b = np.zeros(m)

    # Border constraints: r_i ≤ border_i
    for i in range(n):
        A[i, i] = 1.0
        b[i] = border[i]

    # Pairwise non‑overlap constraints: r_i + r_j ≤ d_ij
    row = n
    for i, j in zip(pair_i, pair_j):
        A[row, i] = 1.0
        A[row, j] = 1.0
        b[row] = dists[i, j]
        row += 1

    # Objective: maximise sum(r) → minimise -sum(r)
    c = -np.ones(n)

    # Solve the LP
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * n, method="highs")
    if not res.success:
        # In the unlikely event of failure, fall back to zero radii
        return np.zeros(n)

    return res.x
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
