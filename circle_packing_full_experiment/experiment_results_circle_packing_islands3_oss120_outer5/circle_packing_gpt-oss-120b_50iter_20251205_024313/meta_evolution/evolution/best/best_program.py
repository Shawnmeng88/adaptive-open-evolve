# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

_EPS = 1e-9
_TOL = 1e-10
_MAX_ITER_GLOBAL = 500
_GLOBAL_STEP = 0.025
_MIN_GLOBAL_STEP = 0.001
_MAX_GLOBAL_STEP = 0.05
_REFINE_ITER = 50
_REFINE_STEP = 0.02
_EXTRA_GRID_N = 70
_EARLY_STOP_PATIENCE = 60
_GREEDY_TRIALS = 2000


def _build_lp_matrices(centers: np.ndarray):
    """Construct LP matrices for the given centre positions."""
    n = centers.shape[0]

    # distance from each centre to the square sides
    border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
    )

    # pairwise centre distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)

    # maximise sum(r) → minimise -sum(r)
    c = -np.ones(n)

    # border constraints: r_i <= border_i
    A = [np.eye(n)]
    b = [border]

    # non‑overlap constraints: r_i + r_j <= d_ij
    rows = []
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            vals.append(dists[i, j] - _EPS)  # safety margin
    if rows:
        A.append(np.vstack(rows))
        b.append(np.array(vals))

    A_ub = np.vstack(A)
    b_ub = np.concatenate(b)
    bounds = [(0.0, None)] * n
    return c, A_ub, b_ub, bounds


def _check_constraints(centers: np.ndarray, radii: np.ndarray) -> float:
    """Return the maximum constraint violation (0 if feasible)."""
    max_allowed = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
    )
    border_violation = np.maximum(radii - max_allowed, 0.0).max()

    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    sum_r = radii[:, None] + radii[None, :]
    mask = np.triu(np.ones_like(dists, dtype=bool), k=1)
    overlap_violation = np.maximum(sum_r - dists, 0.0)[mask].max(initial=0.0)

    return max(border_violation, overlap_violation)


def _solve_radius_lp(centers: np.ndarray) -> np.ndarray:
    """Solve the LP for a given centre set, returning feasible radii."""
    c, A_ub, b_ub, bounds = _build_lp_matrices(centers)
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
        options={"tol": _TOL},
    )
    if not res.success:
        return np.zeros(centers.shape[0])

    radii = np.maximum(res.x, 0.0)

    # post‑solve sanity check
    if _check_constraints(centers, radii) > _EPS:
        return np.zeros(centers.shape[0])

    return radii


def _find_extra_center(base_centers: np.ndarray,
                       base_radii: np.ndarray,
                       grid_n: int = _EXTRA_GRID_N) -> np.ndarray:
    """Search a uniform grid for the point with maximal clearance."""
    xs = np.linspace(0.02, 0.98, grid_n)
    ys = np.linspace(0.02, 0.98, grid_n)

    best_pt = None
    best_val = -1.0
    bc, br = base_centers, base_radii

    for x in xs:
        for y in ys:
            pt = np.array([x, y])
            border = min(x, y, 1.0 - x, 1.0 - y) - _EPS
            dists = np.linalg.norm(bc - pt, axis=1) - br
            clearance = min(border, dists.min())
            if clearance > best_val:
                best_val = clearance
                best_pt = pt
    return best_pt


def _greedy_extra_center(base_centers: np.ndarray,
                         base_radii: np.ndarray,
                         trials: int = _GREEDY_TRIALS) -> np.ndarray:
    """Fallback: random sampling for a feasible extra centre."""
    rng = np.random.default_rng()
    best_pt = None
    best_val = -1.0
    bc, br = base_centers, base_radii

    for _ in range(trials):
        pt = rng.uniform(0.02, 0.98, size=2)
        border = min(pt[0], pt[1], 1.0 - pt[0], 1.0 - pt[1]) - _EPS
        dists = np.linalg.norm(bc - pt, axis=1) - br
        clearance = min(border, dists.min())
        if clearance > best_val:
            best_val = clearance
            best_pt = pt
    return best_pt


def _validate_solution(centers: np.ndarray, radii: np.ndarray):
    """Raise if any containment or overlap violation occurs."""
    eps = 1e-9
    max_allowed = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
    )
    if not np.all(radii <= max_allowed + eps):
        raise AssertionError("Containment violation")
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    sum_r = radii[:, None] + radii[None, :]
    mask = np.triu(np.ones_like(dists, dtype=bool), k=1)
    if not np.all(dists[mask] + eps >= sum_r[mask]):
        raise AssertionError("Overlap violation")


def construct_packing():
    """Construct a 26‑circle packing with adaptive stochastic refinement."""
    rng = np.random.default_rng(12345)

    # ---- deterministic 5×5 grid (baseline) ----
    grid = np.linspace(0.1, 0.9, 5)
    xv, yv = np.meshgrid(grid, grid)
    base_centers = np.column_stack((xv.ravel(), yv.ravel()))  # 25 points

    # Radii for the base grid (used for extra‑point search)
    base_radii = _solve_radius_lp(base_centers)

    # ---- locate the 26th centre ----
    extra_center = _find_extra_center(base_centers, base_radii)
    if extra_center is None:
        extra_center = _greedy_extra_center(base_centers, base_radii)

    # ---- local refinement of the extra centre ----
    best_center = extra_center
    best_sum = -1.0
    no_improve = 0
    for _ in range(_REFINE_ITER):
        cand = best_center + rng.uniform(-_REFINE_STEP, _REFINE_STEP, size=2)
        cand = np.clip(cand, 0.02, 0.98)
        all_c = np.vstack([base_centers, cand])
        rad = _solve_radius_lp(all_c)
        s = rad.sum()
        if s > best_sum + 1e-12:
            best_sum = s
            best_center = cand
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= _EARLY_STOP_PATIENCE:
            break

    # start from the best configuration found so far
    centers = np.vstack([base_centers, best_center])
    radii = _solve_radius_lp(centers)
    best_sum = radii.sum()

    # ---- global stochastic improvement with adaptive step ----
    global_step = _GLOBAL_STEP
    fail_streak = 0
    for it in range(_MAX_ITER_GLOBAL):
        idx = rng.integers(0, centers.shape[0])
        pert = centers[idx] + rng.uniform(-global_step, global_step, size=2)
        pert = np.clip(pert, 0.02, 0.98)

        new_centers = centers.copy()
        new_centers[idx] = pert
        new_radii = _solve_radius_lp(new_centers)
        new_sum = new_radii.sum()

        if new_sum > best_sum + 1e-12:
            centers, radii, best_sum = new_centers, new_radii, new_sum
            fail_streak = 0
            # enlarge step if we made a significant jump
            if new_sum - best_sum > 0.02:
                global_step = min(global_step * 1.1, _MAX_GLOBAL_STEP)
        else:
            fail_streak += 1
            if fail_streak >= 3:
                global_step = max(global_step / 2.0, _MIN_GLOBAL_STEP)
                fail_streak = 0

        # periodic safety validation
        if it % 25 == 0:
            try:
                _validate_solution(centers, radii)
            except AssertionError:
                # rollback to last known good state
                centers = np.vstack([base_centers, best_center])
                radii = _solve_radius_lp(centers)
                best_sum = radii.sum()
                global_step = _GLOBAL_STEP
                fail_streak = 0

    # final validation
    _validate_solution(centers, radii)

    return centers, radii, float(best_sum)
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
