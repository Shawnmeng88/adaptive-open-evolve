# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog


def _initial_centers():
    """
    Produce a deterministic staggered (hex‑like) layout of 26 points.
    Rows are arranged as [5,5,5,5,6] with alternating horizontal offset.
    Points are kept away from the square border by a small margin.
    """
    row_counts = [5, 5, 5, 5, 6]  # sum = 26
    n_rows = len(row_counts)
    margin = 0.04
    centers = []

    for r, cnt in enumerate(row_counts):
        # y coordinate evenly spaced inside the square (respecting margin)
        y = margin + (1 - 2 * margin) * (r + 0.5) / n_rows

        # x positions evenly spaced between margins
        if cnt == 1:
            xs = np.array([0.5])
        else:
            xs = np.linspace(margin, 1 - margin, cnt)

        # offset every other row by half the intra‑row spacing
        if r % 2 == 1:
            spacing = (1 - 2 * margin) / (cnt - 1) if cnt > 1 else 0.0
            xs = xs + spacing / 2.0
            xs = np.clip(xs, margin, 1 - margin)

        for x in xs:
            centers.append([x, y])

    return np.array(centers, dtype=float)


def _solve_radii(centers):
    """
    Solve the linear program that maximises the sum of radii for a fixed set
    of centres. Returns an array of radii (may be slightly infeasible due to
    numeric tolerances – a tiny safety scaling is applied by the caller).
    """
    n = centers.shape[0]

    # border limits
    border_ub = np.min(
        np.stack([centers[:, 0], centers[:, 1],
                  1 - centers[:, 0], 1 - centers[:, 1]]),
        axis=0
    )

    # pairwise distance constraints
    pair_i, pair_j = np.triu_indices(n, k=1)
    dists = np.linalg.norm(centers[pair_i] - centers[pair_j], axis=1)

    m = len(dists)
    A = np.zeros((m, n))
    A[np.arange(m), pair_i] = 1.0
    A[np.arange(m), pair_j] = 1.0
    b = dists

    # maximise sum r_i  -> minimise -sum r_i
    c = -np.ones(n)

    bounds = [(0.0, ub) for ub in border_ub]

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if res.success:
        return np.maximum(res.x, 0.0)
    # fallback – simple greedy heuristic (always feasible)
    radii = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )
    for i, j, d in zip(pair_i, pair_j, dists):
        if radii[i] + radii[j] > d:
            scale = d / (radii[i] + radii[j])
            radii[i] *= scale
            radii[j] *= scale
    return radii


def _local_search(start_centers, iterations=250, step=0.03, seed=0):
    """
    Simple stochastic hill‑climb: move one circle at a time, re‑solve the LP,
    and keep the improvement if the total radius sum grows.
    """
    np.random.seed(seed)
    best_centers = start_centers.copy()
    best_radii = _solve_radii(best_centers)
    best_sum = best_radii.sum()
    n = best_centers.shape[0]

    for _ in range(iterations):
        i = np.random.randint(n)
        proposal = best_centers[i] + np.random.uniform(-step, step, size=2)
        proposal = np.clip(proposal, 0.01, 0.99)

        cand_centers = best_centers.copy()
        cand_centers[i] = proposal

        cand_radii = _solve_radii(cand_centers)
        cand_sum = cand_radii.sum()

        if cand_sum > best_sum:
            best_sum = cand_sum
            best_centers = cand_centers
            best_radii = cand_radii

    return best_centers, best_radii, best_sum


def construct_packing():
    """
    Build a packing of 26 circles.
    Multiple jittered starts are tried; the best configuration (by sum of radii)
    is returned. A final safety scaling (0.9999) is applied to guard against
    floating‑point rounding errors.
    """
    base_centers = _initial_centers()
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # try a few different jitter seeds
    for seed in range(5):
        jitter = np.random.RandomState(seed).uniform(-0.015, 0.015, size=base_centers.shape)
        start = np.clip(base_centers + jitter, 0.01, 0.99)

        centers, radii, total = _local_search(start, iterations=300, step=0.035, seed=seed + 100)
        if total > best_sum:
            best_sum = total
            best_centers = centers
            best_radii = radii

    # safety margin to avoid tiny violations due to numerical tolerance
    best_radii *= 0.9999
    best_sum = float(best_radii.sum())

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
