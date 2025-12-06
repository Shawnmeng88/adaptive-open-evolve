"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Generate several well‑distributed centre sets, keep the one with the largest
    LP‑optimised total radius and then refine it with stochastic hill‑climbing.
    """
    candidates = _candidate_layouts()
    best_sum = -1.0
    best_centers = None
    best_radii = None

    for pts in candidates:
        radii = _max_radii_via_lp(pts)
        total = radii.sum()
        if total > best_sum:
            best_sum = total
            best_centers = pts.copy()
            best_radii = radii.copy()

    # fine‑tune the best layout
    best_centers, best_radii = _hill_climb(best_centers, best_radii,
                                          iterations=250, step=0.025)

    return best_centers, best_radii, float(best_radii.sum())


def _candidate_layouts():
    """Return a list of diverse centre configurations."""
    rng = np.random.default_rng()
    n = 26
    layouts = []

    # 1. deterministic hex grid
    layouts.append(_hex_grid_points(margin=0.08))

    # 2. jittered hex grid
    hex_pts = _hex_grid_points(margin=0.08)
    jitter = rng.uniform(-0.015, 0.015, size=hex_pts.shape)
    layouts.append(np.clip(hex_pts + jitter, 0.0, 1.0))

    # 3. low‑discrepancy Halton points
    layouts.append(_halton_points(n, margin=0.02))

    # 4. uniform random points with margin
    layouts.append(rng.uniform(0.02, 0.98, size=(n, 2)))

    return layouts


def _halton_points(num, margin=0.02):
    """Generate `num` 2‑D Halton points scaled into [margin, 1‑margin]."""
    bases = (2, 3)
    seq = np.empty((num, 2))
    for dim, base in enumerate(bases):
        for i in range(1, num + 1):
            f, denom, n = 0.0, 1, i
            while n:
                n, r = divmod(n, base)
                denom *= base
                f += r / denom
            seq[i - 1, dim] = f
    return np.clip(seq, margin, 1.0 - margin)


def _hill_climb(centers, radii, iterations=200, step=0.02):
    """
    Stochastic hill‑climbing: move a random centre, re‑optimise radii via LP,
    keep the move if the total radius sum improves.
    """
    best_c = centers.copy()
    best_r = radii.copy()
    best_sum = best_r.sum()
    n = best_c.shape[0]
    wall_margin = 0.005
    rng = np.random.default_rng()

    for _ in range(iterations):
        i = rng.integers(n)
        proposal = best_c[i] + rng.uniform(-step, step, size=2)
        proposal = np.clip(proposal, wall_margin, 1.0 - wall_margin)

        new_c = best_c.copy()
        new_c[i] = proposal
        new_r = _max_radii_via_lp(new_c)
        new_sum = new_r.sum()

        if new_sum > best_sum + 1e-7:
            best_c, best_r, best_sum = new_c, new_r, new_sum

    return best_c, best_r


def _max_radii_via_lp(centers):
    """
    Linear programme:
        maximise Σ r_i
        s.t.   r_i ≤ distance to walls
               r_i + r_j ≤ distance between centres
               r_i ≥ 0
    """
    n = centers.shape[0]
    eps = 1e-9
    c = -np.ones(n)

    A = []
    b = []

    # wall constraints
    for i in range(n):
        x, y = centers[i]
        A.append(np.eye(1, n, i)[0])
        b.append(min(x, y, 1 - x, 1 - y) - eps)

    # pairwise non‑overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j]) - eps
            row = np.zeros(n)
            row[i] = row[j] = 1.0
            A.append(row)
            b.append(d)

    res = linprog(c, A_ub=np.vstack(A), b_ub=np.array(b),
                  bounds=[(0.0, None)] * n, method="highs")
    return np.maximum(res.x, 0.0) if res.success else _greedy_max_radii(centers)


def _greedy_max_radii(centers):
    """Fallback greedy scaling when the LP fails."""
    n = centers.shape[0]
    r = np.minimum.reduce([centers[:, 0], centers[:, 1],
                           1 - centers[:, 0], 1 - centers[:, 1]])
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if r[i] + r[j] > d:
                s = d / (r[i] + r[j])
                r[i] *= s
                r[j] *= s
    return r


def _hex_grid_points(margin=0.08):
    """
    Deterministic staggered (hexagonal) arrangement of 26 points inside the unit
    square.  Five rows with column counts [5,5,5,5,6] are used; alternate rows are
    shifted by half the horizontal spacing.
    """
    rows = 5
    cols_per_row = [5, 5, 5, 5, 6]
    y_vals = np.linspace(margin, 1.0 - margin, rows)
    points = []

    max_cols = max(cols_per_row)
    for r, cols in enumerate(cols_per_row):
        x_vals = np.linspace(margin, 1.0 - margin, cols)
        if r % 2 == 1:
            spacing = (1.0 - 2 * margin) / (max_cols - 1)
            x_vals = np.clip(x_vals + spacing / 2.0, margin, 1.0 - margin)
        for x in x_vals:
            points.append([x, y_vals[r]])

    return np.array(points)
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
