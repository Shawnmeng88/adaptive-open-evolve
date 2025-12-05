"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def _hex_grid(step, offset_start=0.0):
    """Generate points of a hexagonal lattice inside the unit square.
    Returns up to 26 points; the pattern can be shifted by offset_start."""
    dy = np.sqrt(3) / 2 * step
    margin = step / 2
    points = []
    row = 0
    y = margin
    while y <= 1 - margin and len(points) < 26:
        # alternate rows are shifted by half a step; add global offset_start too
        offset = ((row % 2) * step / 2) + offset_start
        x = margin + offset
        while x <= 1 - margin and len(points) < 26:
            points.append([x, y])
            x += step
        row += 1
        y += dy
    return np.array(points)

def _solve_lp(centers, eps=1e-9):
    """Linear program: maximize sum of radii under non‑overlap & border constraints.
    eps is subtracted from each upper bound to give a tiny safety margin."""
    n = len(centers)
    c = -np.ones(n)  # maximize sum r_i

    A_ub = []
    b_ub = []

    # border constraints
    for i, (x, y) in enumerate(centers):
        for bound in (x, y, 1 - x, 1 - y):
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(bound - eps)

    # pairwise non‑overlap constraints
    dists = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j] - eps)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    bounds = [(0, None) for _ in range(n)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        # fallback: zero radii (should not happen)
        return np.zeros(n)
    radii = res.x

    # tiny uniform scaling to consume any remaining slack (optional safety boost)
    # compute max factor s such that s * radii still satisfies all constraints
    Ax = A_ub @ radii
    feasible_factors = b_ub / np.where(Ax > 0, Ax, np.inf)
    scale = feasible_factors.min()
    if scale > 1.0:
        radii *= scale
    return radii

def construct_packing():
    """
    Construct a packing of 26 circles in the unit square.
    Tries a fine grid of hexagonal lattice spacings and two offset phases,
    keeping the configuration that yields the largest total radius sum.
    """
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # explore step sizes from 0.15 to 0.23 (inclusive) with small increments
    for step in np.linspace(0.15, 0.23, 17):
        for offset in (0.0, step / 4):
            centers = _hex_grid(step, offset_start=offset)
            if centers.shape[0] < 26:
                continue
            radii = _solve_lp(centers)
            total = radii.sum()
            if total > best_sum:
                best_sum = total
                best_centers = centers
                best_radii = radii

    # safety fallback: simple uniform grid if nothing succeeded
    if best_centers is None:
        xs = np.linspace(0.07, 0.93, 6)
        ys = np.linspace(0.07, 0.93, 5)
        grid = np.array([[x, y] for y in ys for x in xs])[:26]
        best_centers = grid
        best_radii = _solve_lp(best_centers)
        best_sum = best_radii.sum()

    return best_centers, best_radii, best_sum
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
