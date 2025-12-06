"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def construct_packing():
    """
    Construct a packing of 26 circles inside the unit square.
    Starts from a regular 5×5 grid (spacing 0.2) plus a dummy circle.
    Then repeatedly perturbs the grid and solves a linear program that
    maximises the total sum of radii while respecting border and
    non‑overlap constraints. The best configuration found is returned.
    """
    # ----- base grid -----
    n_grid = 5
    offset = 0.1  # ensures points are 0.1 away from each border
    xs = np.linspace(offset, 1 - offset, n_grid)
    ys = np.linspace(offset, 1 - offset, n_grid)
    grid_centers = np.array([[x, y] for y in ys for x in xs])  # 25 points

    # dummy circle (will receive radius 0)
    dummy = np.array([[0.0, 0.0]])

    base_centers = np.vstack([grid_centers, dummy])  # shape (26, 2)

    # ----- optimisation helpers -----
    def solve_radii(centers):
        """
        Given fixed centres, solve a linear program:
            maximise sum r_i
            s.t. 0 <= r_i <= border_i
                 r_i + r_j <= dist(i,j)   for all i<j
        Returns the optimal radii as a 1‑D ndarray.
        """
        n = centers.shape[0]

        # border limits
        border_limits = np.minimum.reduce(
            [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
        )

        # pairwise distances
        diff = centers[:, None, :] - centers[None, :, :]  # (n, n, 2)
        dists = np.sqrt(np.sum(diff ** 2, axis=2))       # (n, n)

        # Build inequality matrix A_ub * r <= b_ub
        # 1) border constraints: r_i <= border_i
        A_border = np.eye(n)
        b_border = border_limits

        # 2) pairwise constraints: r_i + r_j <= d_ij
        pair_rows = []
        pair_bounds = []
        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                pair_rows.append(row)
                pair_bounds.append(dists[i, j])

        if pair_rows:
            A_pair = np.vstack(pair_rows)
            b_pair = np.array(pair_bounds)
            A_ub = np.vstack([A_border, A_pair])
            b_ub = np.concatenate([b_border, b_pair])
        else:
            A_ub = A_border
            b_ub = b_border

        # Objective: maximise sum r_i  => minimise -sum r_i
        c = -np.ones(n)

        # Bounds for each variable: (0, None) – upper bound already in A_ub
        bounds = [(0, None) for _ in range(n)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:
            # fallback to zero radii (should not happen)
            return np.zeros(n)
        return res.x

    # ----- initial solution -----
    best_centers = base_centers.copy()
    best_radii = solve_radii(best_centers)
    best_sum = best_radii.sum()

    # ----- local perturbation loop -----
    rng = np.random.default_rng()
    for _ in range(30):  # number of perturbation attempts
        # jitter non‑dummy points
        perturbed = best_centers.copy()
        jitter = rng.normal(scale=0.02, size=perturbed.shape)
        jitter[-1] = 0.0                     # keep dummy unchanged
        perturbed += jitter
        # keep points inside the unit square
        perturbed[:, 0] = np.clip(perturbed[:, 0], 0.0, 1.0)
        perturbed[:, 1] = np.clip(perturbed[:, 1], 0.0, 1.0)

        radii = solve_radii(perturbed)
        total = radii.sum()
        if total > best_sum + 1e-6:          # accept only genuine improvements
            best_sum = total
            best_centers = perturbed
            best_radii = radii

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
