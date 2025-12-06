"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a packing of 26 circles inside the unit square.
    Uses several diversified starts (jittered lattice, far‑point sampling,
    random interior) and a hill‑climb that moves a single centre with an
    exponentially decaying step size.  Occasionally a global shake (re‑placing
    one centre) helps escape plateaus.  The best layout among all restarts is
    returned.
    """
    import numpy as np
    from scipy.optimize import linprog

    rng = np.random.default_rng()
    n = 26

    # ------------------------------------------------------------------
    # LP: maximise total radius for fixed centres
    # ------------------------------------------------------------------
    def optimal_radii(centers: np.ndarray) -> np.ndarray:
        m = centers.shape[0]

        # pairwise non‑overlap constraints r_i + r_j <= d_ij
        pair_cnt = m * (m - 1) // 2
        A = np.zeros((pair_cnt, m))
        b = np.empty(pair_cnt)
        k = 0
        for i in range(m):
            for j in range(i + 1, m):
                A[k, i] = 1.0
                A[k, j] = 1.0
                b[k] = np.linalg.norm(centers[i] - centers[j])
                k += 1

        # wall distance bounds
        bounds = [
            (0.0, min(c[0], c[1], 1.0 - c[0], 1.0 - c[1]))
            for c in centers
        ]

        # maximise sum(r)  → minimise -sum(r)
        res = linprog(-np.ones(m), A_ub=A, b_ub=b,
                      bounds=bounds, method="highs",
                      options={"presolve": True})
        if res.success:
            return np.maximum(res.x, 0.0)
        # fallback – wall distances (always feasible)
        return np.array([ub for _, ub in bounds])

    # ------------------------------------------------------------------
    # Diverse initial layouts
    # ------------------------------------------------------------------
    def jittered_lattice() -> np.ndarray:
        vals = np.linspace(0.15, 0.85, 5)
        gx, gy = np.meshgrid(vals, vals)
        pts = np.column_stack([gx.ravel(), gy.ravel()])   # 25 points
        pts = np.vstack([pts, [0.5, 0.5]])                # 26 points
        pts += 0.01 * rng.standard_normal(pts.shape)    # tiny jitter
        return np.clip(pts, 0.0, 1.0)

    def farthest_point_sampling(k: int) -> np.ndarray:
        # dense grid of candidates
        grid_vals = np.linspace(0.1, 0.9, 9)
        gx, gy = np.meshgrid(grid_vals, grid_vals)
        candidates = np.column_stack([gx.ravel(), gy.ravel()])
        selected = [np.array([0.5, 0.5])]
        mask = np.ones(len(candidates), dtype=bool)

        while len(selected) < k:
            remaining = candidates[mask]
            dists = np.min(
                np.linalg.norm(
                    remaining[:, None, :] - np.array(selected)[None, :, :],
                    axis=2),
                axis=1)
            idx = np.argmax(dists)
            selected.append(remaining[idx])
            global_idx = np.where(mask)[0][idx]
            mask[global_idx] = False
        return np.vstack(selected)

    def random_inside() -> np.ndarray:
        pts = rng.uniform(0.08, 0.92, size=(n, 2))
        return pts

    # ------------------------------------------------------------------
    # Hill‑climb for a single start
    # ------------------------------------------------------------------
    def hill_climb(start: np.ndarray,
                   total_iters: int = 2500,
                   step_start: float = 0.05,
                   step_end: float = 0.003) -> tuple[np.ndarray, np.ndarray, float]:
        centres = start.copy()
        radii = optimal_radii(centres)
        best_sum = radii.sum()
        best_c, best_r = centres, radii

        for it in range(total_iters):
            # exponential decay of step size
            sigma = step_start * (step_end / step_start) ** (it / total_iters)

            # ---- single‑point move -------------------------------------------------
            idx = rng.integers(0, n)
            cand = best_c.copy()
            delta = rng.normal(scale=sigma, size=2)
            cand[idx] = np.clip(cand[idx] + delta, 0.0, 1.0)

            # reject moves that hit the walls
            if np.min([cand[idx, 0], 1.0 - cand[idx, 0],
                       cand[idx, 1], 1.0 - cand[idx, 1]]) < 1e-8:
                continue

            cand_r = optimal_radii(cand)
            cand_sum = cand_r.sum()
            if cand_sum > best_sum:
                best_c, best_r, best_sum = cand, cand_r, cand_sum
                continue

            # ---- occasional global shake -----------------------------------------
            if rng.random() < 0.002:   # ~5 shakes per run
                shake_idx = rng.integers(0, n)
                shake_pos = rng.uniform(0.05, 0.95, size=2)
                cand2 = best_c.copy()
                cand2[shake_idx] = shake_pos
                cand2_r = optimal_radii(cand2)
                cand2_sum = cand2_r.sum()
                if cand2_sum > best_sum:
                    best_c, best_r, best_sum = cand2, cand2_r, cand2_sum

        return best_c, best_r, float(best_sum)

    # ------------------------------------------------------------------
    # Run several independent restarts and keep the overall best
    # ------------------------------------------------------------------
    best_overall_sum = -1.0
    best_overall_c = None
    best_overall_r = None

    starters = [
        jittered_lattice(),
        farthest_point_sampling(n),
        random_inside()
    ]

    for start in starters:
        c, r, s = hill_climb(start,
                             total_iters=2600,
                             step_start=0.05,
                             step_end=0.003)
        if s > best_overall_sum:
            best_overall_sum, best_overall_c, best_overall_r = s, c, r

    # final fine‑tuning around the best layout
    if best_overall_c is not None:
        c, r, s = hill_climb(best_overall_c,
                             total_iters=1500,
                             step_start=0.008,
                             step_end=0.0008)
        if s > best_overall_sum:
            best_overall_sum, best_overall_c, best_overall_r = s, c, r

    return best_overall_c, best_overall_r, float(best_overall_sum)
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
