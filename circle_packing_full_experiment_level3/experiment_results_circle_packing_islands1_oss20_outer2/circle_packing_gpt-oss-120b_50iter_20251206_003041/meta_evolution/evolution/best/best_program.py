"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def _lp_optimal_radii(centers):
    """Solve LP to maximize sum of radii for given centers."""
    n = centers.shape[0]
    wall = np.minimum.reduce([centers[:, 0], centers[:, 1],
                             1.0 - centers[:, 0], 1.0 - centers[:, 1]])
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    rows, cols = np.triu_indices(n, k=1)
    m = len(rows)
    if m:
        A = np.zeros((m, n))
        A[np.arange(m), rows] = 1.0
        A[np.arange(m), cols] = 1.0
        b = dists[rows, cols]
    else:
        A = None
        b = None
    c = -np.ones(n)
    bounds = [(0.0, w) for w in wall]
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    return res.x if res.success else np.zeros(n)


def _grid_centers():
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 5)
    xv, yv = np.meshgrid(xs, ys)
    grid = np.column_stack([xv.ravel(), yv.ravel()])
    extra = np.array([[0.5, 0.2]])
    return np.vstack([grid, extra])[:26]


def _hex_centers():
    margin = 0.05
    rows, cols = 5, 6
    y_vals = np.linspace(margin, 1.0 - margin, rows)
    pts = []
    for i, y in enumerate(y_vals):
        offset = (i % 2) * (0.5 / cols)
        x_vals = np.linspace(margin + offset, 1.0 - margin + offset, cols)
        for x in x_vals:
            if margin <= x <= 1.0 - margin:
                pts.append([x, y])
    pts = np.array(pts)
    if pts.shape[0] > 26:
        pts = pts[:26]
    elif pts.shape[0] < 26:
        rng = np.random.default_rng(0)
        extra = rng.uniform(margin, 1.0 - margin, size=(26 - pts.shape[0], 2))
        pts = np.vstack([pts, extra])
    return pts


def _perturb(centers, idx, sigma, rng):
    """Return a copy of centers with point idx perturbed by Gaussian noise."""
    trial = centers.copy()
    delta = rng.normal(scale=sigma, size=2)
    trial[idx] = np.clip(trial[idx] + delta, 0.0, 1.0)
    return trial


def construct_packing():
    """Construct a high‑score packing of 26 circles."""
    rng = np.random.default_rng(42)

    # ---- initial candidate layouts ----
    candidates = [_grid_centers(), _hex_centers()]

    # many random seeds for a broader start
    for _ in range(200):
        candidates.append(rng.uniform(0.05, 0.95, size=(26, 2)))

    best_sum = -1.0
    best_centers = None
    best_radii = None

    # evaluate all seeds
    for centres in candidates:
        radii = _lp_optimal_radii(centres)
        s = radii.sum()
        if s > best_sum:
            best_sum, best_centers, best_radii = s, centres.copy(), radii.copy()

    # ---- adaptive hill‑climb on the current best layout ----
    sigma = 0.018
    for step in range(3000):
        # gradually shrink step size
        if step and step % 600 == 0:
            sigma *= 0.7

        i = rng.integers(0, 26)

        # occasional big jump to escape plateaus
        if rng.random() < 0.03:
            trial = best_centers.copy()
            trial[i] = rng.uniform(0.05, 0.95, size=2)
        else:
            trial = _perturb(best_centers, i, sigma, rng)

        radii = _lp_optimal_radii(trial)
        s = radii.sum()
        if s > best_sum + 1e-9:
            best_sum, best_centers, best_radii = s, trial, radii

    # ---- short random restarts with brief polishing ----
    for _ in range(60):
        trial_centers = rng.uniform(0.05, 0.95, size=(26, 2))
        radii = _lp_optimal_radii(trial_centers)
        s = radii.sum()
        if s > best_sum:
            best_sum, best_centers, best_radii = s, trial_centers, radii

        # brief local search
        for __ in range(120):
            i = rng.integers(0, 26)
            trial = _perturb(best_centers, i, 0.02, rng)
            radii = _lp_optimal_radii(trial)
            s = radii.sum()
            if s > best_sum + 1e-9:
                best_sum, best_centers, best_radii = s, trial, radii

    # ---- final deterministic polishing (8‑direction search) ----
    step_size = 0.005
    dirs = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]
    improved = True
    while improved:
        improved = False
        for i in range(26):
            best_local = best_sum
            best_trial = None
            for dx, dy in dirs:
                trial = best_centers.copy()
                delta = np.array([dx, dy]) * step_size
                trial[i] = np.clip(trial[i] + delta, 0.0, 1.0)
                radii = _lp_optimal_radii(trial)
                s = radii.sum()
                if s > best_local + 1e-9:
                    best_local = s
                    best_trial = (trial, radii)
            if best_trial is not None:
                best_sum, best_centers, best_radii = best_local, best_trial[0], best_trial[1]
                improved = True

    return best_centers, best_radii, float(best_sum)
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
