"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    """
    Generate several diverse initial centre sets and improve each with
    stochastic hill‑climbing that occasionally solves a linear program.
    Returns the best centre/radius configuration found.
    """
    np.random.seed(0)          # reproducibility
    n = 26

    # diverse seeds
    seeds = [
        _hex_lattice_initial(n, spacing=0.18),
        _grid_initial(n, spacing=0.2),
        _random_initial(n, margin=0.02)
    ]

    best_sum = -1.0
    best_centers = None
    best_radii = None

    for centers in seeds:
        c, r, s = _refine_centers(
            centers,
            iterations=3500,
            init_step=0.04,
            lp_interval=600
        )
        if s > best_sum:
            best_sum, best_centers, best_radii = s, c, r

    return best_centers, best_radii, best_sum


def _hex_lattice_initial(n, spacing=0.18):
    """Hexagonal lattice trimmed to the unit square."""
    pts = []
    row = 0
    y = spacing / 2.0
    vstep = spacing * np.sqrt(3) / 2.0
    while y < 1 - spacing / 2.0 and len(pts) < n:
        offset = 0.0 if row % 2 == 0 else spacing / 2.0
        x = spacing / 2.0 + offset
        while x < 1 - spacing / 2.0 and len(pts) < n:
            pts.append([x, y])
            x += spacing
        row += 1
        y += vstep
    return np.clip(np.array(pts[:n]), 0.005, 0.995)


def _grid_initial(n, spacing=0.2):
    """Regular square grid, padded with random points if needed."""
    xs = np.arange(spacing / 2, 1, spacing)
    ys = np.arange(spacing / 2, 1, spacing)
    pts = np.array([[x, y] for y in ys for x in xs])
    if pts.shape[0] < n:
        extra = _random_initial(n - pts.shape[0], margin=0.01)
        pts = np.vstack([pts, extra])
    return np.clip(pts[:n], 0.005, 0.995)


def _random_initial(n, margin=0.01):
    """Uniform random points respecting a margin from the borders."""
    return np.random.uniform(margin, 1 - margin, size=(n, 2))


def _max_radii(centers, eps=1e-9):
    """
    Exact maximal radii for a fixed centre set:
    r_i = min( distance to walls, 0.5 * distance to any other centre )
    """
    n = centers.shape[0]
    # distance to walls
    wall = np.minimum(centers, 1 - centers)
    radii = np.min(wall, axis=1)

    # pairwise half‑distances
    for i in range(n):
        diffs = centers[i] - centers[i + 1 :]
        dists = np.linalg.norm(diffs, axis=1)
        if dists.size:
            half = dists / 2.0
            radii[i] = min(radii[i], np.min(half))
            radii[i + 1 :] = np.minimum(radii[i + 1 :], half)

    return np.maximum(radii - eps, 0.0)


def _compute_max_radii_lp(centers):
    """
    Linear‑program that maximises Σ r_i under containment and non‑overlap.
    Falls back to the exact geometric radii if LP fails.
    """
    from scipy.optimize import linprog

    n = centers.shape[0]
    c = -np.ones(n)                     # maximise sum -> minimise negative sum

    A = []
    b = []

    # wall constraints
    for i, (x, y) in enumerate(centers):
        for bound in (x, y, 1 - x, 1 - y):
            row = np.zeros(n)
            row[i] = 1.0
            A.append(row)
            b.append(bound)

    # pairwise non‑overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A.append(row)
            b.append(dist)

    A = np.array(A)
    b = np.array(b)

    bounds = [(0.0, None)] * n
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if res.success:
        return np.maximum(res.x, 0.0)
    # fallback
    return _max_radii(centers)


def _refine_centers(centers, iterations=3000, init_step=0.03, lp_interval=500):
    """
    Hill‑climbing with adaptive step size and periodic LP refinement.
    """
    best_c = centers.copy()
    best_r = _max_radii(best_c)
    best_sum = best_r.sum()
    n = best_c.shape[0]

    step = init_step
    decay = 0.9995  # gradual annealing

    for it in range(iterations):
        # propose a move for a random centre
        i = np.random.randint(n)
        trial = best_c.copy()
        delta = np.random.normal(scale=step, size=2)
        trial[i] = np.clip(trial[i] + delta, 0.001, 0.999)

        # evaluate using exact radii (fast)
        trial_r = _max_radii(trial)
        s = trial_r.sum()
        if s > best_sum + 1e-8:
            best_c, best_r, best_sum = trial, trial_r, s

        # periodic LP escape / polishing
        if (it + 1) % lp_interval == 0:
            lp_r = _compute_max_radii_lp(best_c)
            lp_sum = lp_r.sum()
            if lp_sum > best_sum + 1e-8:
                best_r, best_sum = lp_r, lp_sum

        # anneal step size
        step *= decay
        if step < 1e-4:
            step = 1e-4

    return best_c, best_r, best_sum
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
