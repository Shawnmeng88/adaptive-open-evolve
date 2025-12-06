"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def generate_hex_grid(n):
    """
    Generate up to n points arranged in a hexagonal (triangular) lattice
    inside the unit square. Points are kept away from the borders
    by a small margin to allow non‑zero radii.
    """
    pts = []
    # spacing chosen to roughly accommodate 26 points
    spacing_x = 0.18
    spacing_y = spacing_x * np.sqrt(3) / 2
    margin = 0.01

    y = margin + spacing_y / 2
    row = 0
    while y < 1 - margin and len(pts) < n:
        offset = 0 if row % 2 == 0 else spacing_x / 2
        x = margin + offset + spacing_x / 2
        while x < 1 - margin and len(pts) < n:
            pts.append([x, y])
            x += spacing_x
        y += spacing_y
        row += 1
    return np.array(pts[:n])


def solve_optimal_radii_lp(centers):
    """
    Solve a linear program to obtain the maximal radii for a fixed set
    of circle centres inside the unit square.

    Maximises sum(r_i) subject to:
        r_i >= 0
        r_i <= distance to each wall
        r_i + r_j <= distance between centres i and j
    """
    n = centers.shape[0]

    # Wall distance upper bounds
    wall_bounds = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )

    # Pairwise centre distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))

    # Build inequality matrix A_ub * r <= b_ub
    # 1) wall constraints
    A_wall = np.eye(n)
    b_wall = wall_bounds

    # 2) pairwise non‑overlap constraints
    pair_indices = np.triu_indices(n, k=1)
    i_idx, j_idx = pair_indices
    A_pair = np.zeros((len(i_idx), n))
    A_pair[np.arange(len(i_idx)), i_idx] = 1
    A_pair[np.arange(len(i_idx)), j_idx] = 1
    b_pair = dists[i_idx, j_idx]

    # Concatenate constraints
    A_ub = np.vstack([A_wall, A_pair])
    b_ub = np.concatenate([b_wall, b_pair])

    # Objective: maximize sum(r)  => minimize -sum(r)
    c = -np.ones(n)

    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n, method="highs")
    if res.success:
        return res.x
    # Fallback to simple geometric radii if LP fails
    return compute_max_radii(centers)


def compute_max_radii(centers):
    """
    Simple fallback: start from wall limits and iteratively scale
    overlapping circles proportionally.
    """
    n = centers.shape[0]
    radii = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )
    # Iterative pairwise scaling (few passes are enough)
    for _ in range(3):
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d:
                    scale = d / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
    return radii


def construct_packing():
    """
    Construct a packing of 26 circles in the unit square.
    Positions follow a hexagonal lattice; radii are obtained
    by solving a linear program that maximises their total sum.
    """
    n = 26
    centers = generate_hex_grid(n)

    # Ensure all centres are strictly inside the square
    centers = np.clip(centers, 0.001, 0.999)

    radii = solve_optimal_radii_lp(centers)

    sum_radii = np.sum(radii)
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
