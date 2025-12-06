"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Build a 26‑circle packing inside the unit square.
    Uses a hexagonal lattice for the centers and solves a linear program
    to obtain the maximal feasible radii for those centers.
    """
    np.random.seed(0)

    # ----- generate a hexagonal lattice of points -----
    spacing = 0.18  # distance between neighbouring lattice points
    points = []
    row = 0
    y = spacing / 2
    while y < 1 - spacing / 2:
        offset = 0 if row % 2 == 0 else spacing / 2
        x = spacing / 2 + offset
        while x < 1 - spacing / 2:
            points.append([x, y])
            x += spacing
        y += spacing * np.sqrt(3) / 2
        row += 1

    centers = np.array(points)
    # Trim or pad to exactly 26 circles
    if centers.shape[0] > 26:
        centers = centers[:26]
    elif centers.shape[0] < 26:
        extra = 26 - centers.shape[0]
        rand = np.random.rand(extra, 2) * (1 - spacing) + spacing / 2
        centers = np.vstack([centers, rand])

    # ----- compute optimal radii via linear programming -----
    radii = compute_max_radii(centers)

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Solve a linear program that maximises the sum of radii
    subject to:
        * each circle stays inside the unit square,
        * circles do not overlap (ri + rj <= distance_ij).
    If the LP fails for any reason, fall back to the simple
    scaling heuristic used previously.
    """
    n = centers.shape[0]

    # Upper bounds from the square borders
    border_ub = np.minimum.reduce([
        centers[:, 0],               # distance to left side
        1 - centers[:, 0],           # distance to right side
        centers[:, 1],               # distance to bottom
        1 - centers[:, 1]            # distance to top
    ])

    # Pairwise distance constraints: ri + rj <= d_ij
    pair_rows = []
    pair_rhs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            pair_rows.append(row)
            pair_rhs.append(d)

    if pair_rows:
        A_ub = np.vstack(pair_rows)
        b_ub = np.array(pair_rhs)
    else:
        A_ub = None
        b_ub = None

    # Linear programme: maximise sum(r)  <=> minimise -sum(r)
    c = -np.ones(n)

    bounds = [(0, ub) for ub in border_ub]

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass  # fall back if linprog raises

    # ----- fallback: simple iterative scaling (original approach) -----
    radii = border_ub.copy()
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > d:
                scale = d / (radii[i] + radii[j])
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
