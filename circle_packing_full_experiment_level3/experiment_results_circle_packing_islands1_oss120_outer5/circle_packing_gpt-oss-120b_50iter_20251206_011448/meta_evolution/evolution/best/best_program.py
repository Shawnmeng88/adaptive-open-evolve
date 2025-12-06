"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Build a 5×5 grid (25 points) and optimise the position of the 26th
    circle.  For each candidate start position we run a tiny deterministic
    hill‑climbing search that repeatedly perturbs the extra point and keeps
    the change if it improves the total sum of radii obtained from the LP.
    The best layout found is returned.
    """
    # ----- base 5×5 grid -------------------------------------------------
    grid_vals = np.linspace(0.1, 0.9, 5)          # 0.1,0.3,…,0.9
    xv, yv = np.meshgrid(grid_vals, grid_vals)
    grid_centers = np.column_stack((xv.ravel(), yv.ravel()))   # (25, 2)

    eps = 0.02  # offset from the borders for candidate starts
    # corner candidates + edge‑mid candidates
    candidate_extras = np.array([
        [eps, eps],
        [eps, 1 - eps],
        [1 - eps, eps],
        [1 - eps, 1 - eps],
        [0.5, eps],
        [0.5, 1 - eps],
        [eps, 0.5],
        [1 - eps, 0.5]
    ])

    best_sum = -1.0
    best_centers = None
    best_radii = None

    for start_extra in candidate_extras:
        centers, radii, total = _optimise_extra_point(grid_centers, start_extra)
        if total > best_sum:
            best_sum = total
            best_centers = centers
            best_radii = radii

    # final safety margin
    best_radii = np.maximum(best_radii - 1e-7, 0.0)
    return best_centers, best_radii, float(best_sum)


def _optimise_extra_point(grid_centers, init_extra, max_iter=200, step=0.02):
    """
    Hill‑climbing optimisation of the extra point while keeping the grid
    points fixed.  Returns the final centres, radii and total sum.
    """
    rng = np.random.default_rng(0)  # deterministic seed
    extra = init_extra.copy()
    centers = np.vstack((grid_centers, extra[None, :]))
    radii = _max_radii_lp(centers)
    best_sum = radii.sum()
    best_extra = extra.copy()
    best_radii = radii

    for _ in range(max_iter):
        proposal = best_extra + rng.uniform(-step, step, size=2)
        proposal = np.clip(proposal, 0.0, 1.0)   # keep inside the unit square
        trial = np.vstack((grid_centers, proposal[None, :]))
        trial_radii = _max_radii_lp(trial)
        total = trial_radii.sum()
        if total > best_sum + 1e-9:
            best_sum = total
            best_extra = proposal
            best_radii = trial_radii

    final_centers = np.vstack((grid_centers, best_extra[None, :]))
    return final_centers, best_radii, best_sum


def _max_radii_lp(centers):
    """
    Linear‑programming optimiser for a fixed set of centres.
    Maximises Σ r_i subject to:
        0 ≤ r_i ≤ distance to the nearest square side,
        r_i + r_j ≤ distance(centers_i, centers_j).
    Returns an array of radii (length = n).
    """
    n = centers.shape[0]

    # Upper bounds from the four borders
    border_limits = np.minimum.reduce([
        centers[:, 0],                # distance to left side
        centers[:, 1],                # distance to bottom side
        1.0 - centers[:, 0],          # distance to right side
        1.0 - centers[:, 1]           # distance to top side
    ])

    # Pairwise non‑overlap constraints: r_i + r_j ≤ dist(i, j)
    pair_cnt = n * (n - 1) // 2
    A = np.zeros((pair_cnt, n))
    b = np.zeros(pair_cnt)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[k, i] = 1.0
            A[k, j] = 1.0
            b[k] = np.linalg.norm(centers[i] - centers[j])
            k += 1

    # Objective: maximise sum(r) → minimise -sum(r)
    c = -np.ones(n)

    bounds = [(0.0, float(limit)) for limit in border_limits]

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    if res.success:
        return np.maximum(res.x, 0.0)

    # Fallback – use the conservative border limits (always feasible)
    return border_limits.copy()
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
