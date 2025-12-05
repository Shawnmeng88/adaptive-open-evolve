# EVOLVE-BLOCK-START
"""
Compact 26‑circle packing optimiser.

Main ideas
----------
* Start from a simple regular grid (5×5) plus one extra point.
* Perform a long simulated‑annealing (SA) search on the centre positions.
  – Random Gaussian moves with a temperature schedule.
  – Accept improvements always; accept worsening moves with probability
    exp(Δ/temperature).
* Finish with a short deterministic local‑search for each centre.
* If SciPy is available, polish the radii with a tiny linear‑program.
"""

import numpy as np

# ----------------------------------------------------------------------
# Greedy radii – always yields a feasible packing (used during search)
# ----------------------------------------------------------------------
def _greedy_radii(centers: np.ndarray) -> np.ndarray:
    n = centers.shape[0]
    # distance to the four walls
    radii = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )
    # shrink overlapping pairs, always reducing the larger circle
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.hypot(dx, dy)
            if radii[i] + radii[j] > d:
                excess = radii[i] + radii[j] - d
                if radii[i] >= radii[j]:
                    radii[i] -= excess
                else:
                    radii[j] -= excess
                radii[i] = max(radii[i], 0.0)
                radii[j] = max(radii[j], 0.0)
    return radii


def compute_max_radii(centers: np.ndarray) -> np.ndarray:
    """Wrapper used by the optimiser."""
    return _greedy_radii(centers)


# ----------------------------------------------------------------------
# Optional LP refinement – gives the true optimum for a fixed centre set
# ----------------------------------------------------------------------
def _optimal_radii_lp(centers: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import linprog
    except Exception:                     # SciPy missing – fall back to greedy
        return _greedy_radii(centers)

    n = centers.shape[0]

    # maximise sum r_i  →  minimise -sum r_i
    c = -np.ones(n)

    # wall limits
    walls = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )
    bounds = [(0.0, w) for w in walls]

    # pairwise constraints r_i + r_j ≤ distance
    m = n * (n - 1) // 2
    A = np.zeros((m, n))
    b = np.zeros(m)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            b[idx] = np.hypot(*(centers[i] - centers[j]))
            idx += 1

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if res.success:
        return np.clip(res.x, 0.0, None)
    return _greedy_radii(centers)


# ----------------------------------------------------------------------
# Packing constructor
# ----------------------------------------------------------------------
def construct_packing():
    """
    Returns
    -------
    centers : np.ndarray, shape (26, 2)
    radii   : np.ndarray, shape (26,)
    sum_radii : float
    """
    rng = np.random.default_rng(42)

    # ---- 1. Simple deterministic start (5×5 grid + extra point) ----------
    step, margin = 0.2, 0.1
    xs = np.arange(margin, 1.0 - margin + 1e-9, step)
    ys = np.arange(margin, 1.0 - margin + 1e-9, step)
    base = [(x, y) for y in ys for x in xs]          # 25 points
    base.append((0.5, 0.2))                         # 26th point
    centers = np.array(base, dtype=float)

    # ---- 2. Initial greedy radii -----------------------------------------
    best_radii = compute_max_radii(centers)
    best_sum = best_radii.sum()
    best_centers = centers.copy()

    # ---- 3. Simulated‑annealing search ------------------------------------
    N_ITER = 60000
    temp_start, temp_end = 0.015, 0.0003
    for it in range(N_ITER):
        # temperature schedule (linear in log‑space)
        t = np.exp(
            np.log(temp_start) + (np.log(temp_end) - np.log(temp_start)) * (it / N_ITER)
        )
        i = rng.integers(0, 26)
        # proposal: Gaussian step whose scale shrinks with temperature
        scale = 0.07 * (t / temp_start) + 0.003
        proposal = np.clip(best_centers[i] + rng.normal(scale=scale, size=2), 0.0, 1.0)

        cand = best_centers.copy()
        cand[i] = proposal
        cand_radii = compute_max_radii(cand)
        cand_sum = cand_radii.sum()

        delta = cand_sum - best_sum
        if delta > 0 or rng.random() < np.exp(delta / t):
            best_sum, best_centers, best_radii = cand_sum, cand, cand_radii

    # ---- 4. Deterministic local refinement --------------------------------
    LOCAL_TRIALS, LOCAL_STEP = 30, 0.012
    for i in range(26):
        improved = True
        while improved:
            improved = False
            cur_center = best_centers[i]
            cur_sum = best_sum
            for _ in range(LOCAL_TRIALS):
                prop = np.clip(
                    cur_center + rng.normal(scale=LOCAL_STEP, size=2), 0.0, 1.0
                )
                cand = best_centers.copy()
                cand[i] = prop
                cand_radii = compute_max_radii(cand)
                cand_sum = cand_radii.sum()
                if cand_sum > cur_sum + 1e-10:
                    best_sum, best_centers, best_radii = cand_sum, cand, cand_radii
                    cur_center, cur_sum = prop, cand_sum
                    improved = True

    # ---- 5. Final LP polishing (optional) ---------------------------------
    final_radii = _optimal_radii_lp(best_centers)
    final_sum = final_radii.sum()

    return best_centers, final_radii, final_sum


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
