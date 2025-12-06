"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def generate_centers(margin=0.02, invert=False):
    """
    Produce 26 deterministic points in a hexagonal‑like lattice
    inside the unit square with a given margin.
    If invert is True, the horizontal offset for odd rows is reversed.
    """
    row_counts = [5, 5, 5, 5, 6]          # 5‑5‑5‑5‑6 = 26
    n_rows = len(row_counts)

    dy = (1.0 - 2 * margin) / (n_rows - 1)
    max_cols = max(row_counts)
    dx = (1.0 - 2 * margin) / (max_cols - 1)

    pts = []
    for r, cols in enumerate(row_counts):
        y = margin + r * dy
        # normal offset is dx/2 for odd rows; invert flips the sign
        offset = (dx / 2.0) if (r % 2 == 1) else 0.0
        if invert and (r % 2 == 1):
            offset = -offset
        for c in range(cols):
            x = margin + offset + c * dx
            # safety clamp
            x = min(max(x, margin), 1.0 - margin)
            pts.append([x, y])
    return np.array(pts, dtype=float)


def compute_optimal_radii(centers):
    """
    Solve a linear program to maximise the sum of radii for the given centers.
    Returns radii scaled by a tiny safety factor.
    """
    n = centers.shape[0]

    # distance to each square side
    border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
    )

    # pairwise Euclidean distances
    diff = centers[:, None, :] - centers[None, :, :]          # (n, n, 2)
    dists = np.sqrt(np.sum(diff ** 2, axis=2))               # (n, n)

    rows = []
    rhs = []

    # border constraints: r_i <= border_i
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        rows.append(row)
        rhs.append(border[i])

    # non‑overlap constraints: r_i + r_j <= dist_ij
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs.append(dists[i, j])

    A_ub = np.vstack(rows)
    b_ub = np.array(rhs)

    # maximise sum(r) → minimise -sum(r)
    c = -np.ones(n)

    bounds = [(0.0, None)] * n

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        # conservative fallback
        return np.full(n, 1e-6)

    # tiny safety factor to stay strictly feasible
    return res.x * 0.999


def _local_search(start_centers, start_sum, margin, rng, max_iter=250):
    """
    Simple stochastic hill‑climbing: repeatedly perturb a random circle,
    recompute radii via LP and keep the move if it improves the total sum.
    """
    best_centers = start_centers.copy()
    best_sum = start_sum

    n = best_centers.shape[0]
    step_scale = 0.003  # typical perturbation magnitude

    for _ in range(max_iter):
        i = rng.integers(n)                     # pick a circle to move
        perturb = rng.normal(scale=step_scale, size=2)
        cand = best_centers.copy()
        cand[i] += perturb
        # enforce margin bounds
        cand[i, 0] = np.clip(cand[i, 0], margin, 1.0 - margin)
        cand[i, 1] = np.clip(cand[i, 1], margin, 1.0 - margin)

        radii = compute_optimal_radii(cand)
        total = float(radii.sum())
        if total > best_sum:
            best_sum = total
            best_centers = cand
            # gradually shrink step size to fine‑tune
            step_scale *= 0.97

    return best_centers, best_sum


def construct_packing():
    """
    Explore a grid of lattice parameters, then perform a cheap stochastic
    refinement to improve the total radius sum.
    Returns the best centres, radii and their summed radius.
    """
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # Expanded margin grid and inversion flag
    margin_vals = [0.017, 0.018, 0.019, 0.020, 0.021, 0.022, 0.023]
    invert_options = [False, True]

    # Deterministic jitter patterns (including none)
    jitter_patterns = [
        None,
        np.full((26, 2), 0.001),          # outward shift
        np.full((26, 2), -0.001),         # inward shift
    ]

    for margin in margin_vals:
        for invert in invert_options:
            base = generate_centers(margin=margin, invert=invert)
            for jitter in jitter_patterns:
                if jitter is not None:
                    cand = np.clip(base + jitter, margin, 1.0 - margin)
                else:
                    cand = base
                radii = compute_optimal_radii(cand)
                total = float(radii.sum())
                if total > best_sum:
                    best_sum = total
                    best_centers = cand
                    best_radii = radii

    # ----- stochastic local improvement -----
    rng = np.random.default_rng(12345)
    refined_centers, refined_sum = _local_search(
        best_centers, best_sum, margin=margin_vals[0], rng=rng, max_iter=300
    )
    refined_radii = compute_optimal_radii(refined_centers)

    # Keep the better of refined vs original
    if refined_sum > best_sum:
        return refined_centers, refined_radii, refined_sum
    else:
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
