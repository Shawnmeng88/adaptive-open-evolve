"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a packing of 26 equal circles inside the unit square.
    The algorithm explores many candidate centre configurations,
    solves a linear programme for each to obtain optimal radii,
    and then refines the best configuration with an extended
    hill‑climbing phase.
    """
    import numpy as np
    from scipy.optimize import linprog

    # ------------------------------------------------------------------
    # Layout generators
    # ------------------------------------------------------------------
    def hex_grid(jitter=0.0):
        """Staggered hex‑like grid (up to 26 points) with optional jitter."""
        rows, cols = 5, 6
        y_vals = np.linspace(0.12, 0.88, rows)
        x_base = np.linspace(0.12, 0.88, cols)
        pts = []
        for i, y in enumerate(y_vals):
            offset = 0.0 if i % 2 == 0 else 0.04
            xs = x_base + offset
            xs = xs[(xs > 0.001) & (xs < 0.999)]
            for x in xs:
                pts.append([x, y])
        pts = np.array(pts)
        if pts.shape[0] < 26:
            extra = np.random.rand(26 - pts.shape[0], 2) * 0.76 + 0.12
            pts = np.vstack([pts, extra])
        else:
            pts = pts[:26]
        if jitter:
            pts += np.random.uniform(-jitter, jitter, pts.shape)
            pts = np.clip(pts, 0.001, 0.999)
        return pts

    def square_grid(jitter=0.0):
        """Regular square grid (6×5) with optional jitter."""
        xs = np.linspace(0.12, 0.88, 6)
        ys = np.linspace(0.12, 0.88, 5)
        pts = np.array([(x, y) for y in ys for x in xs])
        if pts.shape[0] < 26:
            extra = np.random.rand(26 - pts.shape[0], 2) * 0.76 + 0.12
            pts = np.vstack([pts, extra])
        else:
            pts = pts[:26]
        if jitter:
            pts += np.random.uniform(-jitter, jitter, pts.shape)
            pts = np.clip(pts, 0.001, 0.999)
        return pts

    def random_layout():
        """Uniform random points (26) inside the square."""
        return np.random.rand(26, 2) * 0.96 + 0.02

    # ------------------------------------------------------------------
    # Linear programme: maximise sum of radii for fixed centres
    # ------------------------------------------------------------------
    def solve_radii(centers):
        n = centers.shape[0]

        # wall distances (upper bound for each radius)
        wall_dist = np.min(
            np.stack([centers[:, 0], centers[:, 1],
                      1 - centers[:, 0], 1 - centers[:, 1]]),
            axis=0
        )

        # pairwise non‑overlap constraints: r_i + r_j <= d_ij
        rows = []
        rhs = []
        for i in range(n):
            diffs = centers[i + 1:] - centers[i]
            dists = np.linalg.norm(diffs, axis=1)
            for k, d in enumerate(dists):
                j = i + 1 + k
                row = np.zeros(n)
                row[i] = row[j] = 1.0
                rows.append(row)
                rhs.append(d)

        A_ub = np.vstack([np.eye(n)] + rows) if rows else np.eye(n)
        b_ub = np.concatenate([wall_dist, rhs]) if rows else wall_dist

        c = -np.ones(n)                     # maximise sum → minimise -sum
        bounds = [(0, None)] * n

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method='highs', options={'presolve': True})

        if not res.success:
            return None
        # tiny safety margin
        return np.maximum(res.x * 0.9999, 0.0)

    # ------------------------------------------------------------------
    # Feasibility test (numeric tolerance)
    # ------------------------------------------------------------------
    def feasible(centers, radii, eps=1e-9):
        if radii is None:
            return False
        # wall constraints
        if np.any(radii - np.min(
                np.stack([centers[:, 0], centers[:, 1],
                          1 - centers[:, 0], 1 - centers[:, 1]]), axis=0) > eps):
            return False
        # pairwise constraints
        n = centers.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] - d > eps:
                    return False
        return True

    # ------------------------------------------------------------------
    # Exploration of many candidate layouts
    # ------------------------------------------------------------------
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # a richer set of jitter levels
    jitter_levels = np.concatenate([
        np.linspace(0.0, 0.08, 13),            # systematic sweep
        np.random.rand(10) * 0.10             # random samples
    ])

    # 1) Hex grid variants
    for jitter in jitter_levels:
        centers = hex_grid(jitter=float(jitter))
        radii = solve_radii(centers)
        if feasible(centers, radii):
            s = radii.sum()
            if s > best_sum:
                best_sum, best_centers, best_radii = s, centers.copy(), radii.copy()

    # 2) Square grid variants
    for jitter in jitter_levels:
        centers = square_grid(jitter=float(jitter))
        radii = solve_radii(centers)
        if feasible(centers, radii):
            s = radii.sum()
            if s > best_sum:
                best_sum, best_centers, best_radii = s, centers.copy(), radii.copy()

    # 3) Pure random layouts
    for _ in range(40):
        centers = random_layout()
        radii = solve_radii(centers)
        if feasible(centers, radii):
            s = radii.sum()
            if s > best_sum:
                best_sum, best_centers, best_radii = s, centers.copy(), radii.copy()

    # ------------------------------------------------------------------
    # Extended hill‑climbing refinement
    # ------------------------------------------------------------------
    if best_centers is not None:
        total_iters = 500
        for it in range(total_iters):
            # perturb a random subset of points (1‑4 points)
            trial = best_centers.copy()
            n_perturb = np.random.choice([1, 2, 3, 4])
            idxs = np.random.choice(trial.shape[0], n_perturb, replace=False)
            # annealing sigma: start ~0.05, decay to ~0.001
            sigma = 0.05 * (1.0 - it / total_iters) + 0.001
            for idx in idxs:
                trial[idx] += np.random.normal(scale=sigma, size=2)
                trial[idx] = np.clip(trial[idx], 0.001, 0.999)

            radii = solve_radii(trial)
            if feasible(trial, radii):
                s = radii.sum()
                if s > best_sum:
                    best_sum, best_centers, best_radii = s, trial, radii

    # fallback (should never happen)
    if best_centers is None:
        best_centers = hex_grid()
        best_radii = np.zeros(26)
        best_sum = 0.0

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
