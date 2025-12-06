"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    High‑quality packing for 26 circles.
    1. Generate a dense triangular‑lattice candidate set.
    2. Greedy removal using a linear programme that maximises the sum of radii.
    3. Optional SLSQP refinement that moves centres slightly while keeping
       all constraints, yielding a higher total radius.
    Returns:
        centres (np.ndarray shape (26,2)),
        radii   (np.ndarray shape (26,)),
        total   (float) sum of radii after refinement.
    """
    import numpy as np
    from scipy.optimize import linprog, minimize

    TARGET_N = 26

    # ------------------------------------------------------------------
    # 1. Generate a dense triangular (hexagonal) lattice of candidate points
    # ------------------------------------------------------------------
    def hex_lattice(step):
        """
        Produce points of a triangular lattice with horizontal spacing `step`.
        The vertical spacing is sqrt(3)/2 * step. Points are kept inside [0,1].
        """
        dy = np.sqrt(3) * step / 2.0
        points = []
        y = step / 2.0
        row = 0
        while y <= 1 - step / 2.0:
            offset = 0.0 if row % 2 == 0 else step / 2.0
            x = step / 2.0 + offset
            while x <= 1 - step / 2.0:
                points.append([x, y])
                x += step
            y += dy
            row += 1
        return np.array(points)

    # a fairly fine lattice – about 250‑300 candidates
    cand_pts = hex_lattice(step=0.06)

    # ------------------------------------------------------------------
    # 2. LP solver for a fixed set of centres
    # ------------------------------------------------------------------
    def solve_lp(centers):
        m = centers.shape[0]

        # border limits (upper bound for each radius)
        borders = np.minimum.reduce(
            [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
        )
        bounds = [(0.0, float(b)) for b in borders]

        # pairwise distance constraints: r_i + r_j <= d_ij
        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.linalg.norm(diff, axis=2)

        A = []
        b = []
        for i in range(m):
            for j in range(i + 1, m):
                row = np.zeros(m)
                row[i] = row[j] = 1.0
                A.append(row)
                b.append(dists[i, j])

        c = -np.ones(m)                     # maximise sum(r) → minimise -sum(r)
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds,
                      method="highs", options={"presolve": True})
        return res.x if res.success else np.zeros(m)

    # ------------------------------------------------------------------
    # 3. Greedy removal until exactly TARGET_N centres remain
    # ------------------------------------------------------------------
    active_idx = list(range(len(cand_pts)))
    while len(active_idx) > TARGET_N:
        cur_centers = cand_pts[active_idx]
        cur_radii = solve_lp(cur_centers)
        # discard the most constrained circle (smallest radius)
        remove_pos = int(np.argmin(cur_radii))
        del active_idx[remove_pos]

    centres = cand_pts[active_idx]
    radii = solve_lp(centres)

    # ------------------------------------------------------------------
    # 4. Optional SLSQP refinement (move centres slightly)
    # ------------------------------------------------------------------
    def refine_slsqp(centers, radii):
        n = centers.shape[0]
        # initial variable vector: [x1,y1,...,xn,yn, r1,...,rn]
        x0 = np.hstack([centers.ravel(), radii])

        # indices for convenience
        idx_xy = np.arange(2 * n)
        idx_r = np.arange(2 * n, 3 * n)

        # pair indices for constraints
        pair_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]

        # objective: maximise sum(r) -> minimise negative sum
        def objective(v):
            return -np.sum(v[idx_r])

        # inequality constraints
        cons = []

        # border constraints: x_i - r_i >= 0, (1-x_i) - r_i >= 0, similarly for y
        for i in range(n):
            def left_fun(v, i=i):
                return v[2 * i] - v[2 * n + i]
            cons.append({"type": "ineq", "fun": left_fun})

            def right_fun(v, i=i):
                return (1.0 - v[2 * i]) - v[2 * n + i]
            cons.append({"type": "ineq", "fun": right_fun})

            def bottom_fun(v, i=i):
                return v[2 * i + 1] - v[2 * n + i]
            cons.append({"type": "ineq", "fun": bottom_fun})

            def top_fun(v, i=i):
                return (1.0 - v[2 * i + 1]) - v[2 * n + i]
            cons.append({"type": "ineq", "fun": top_fun})

        # pairwise non‑overlap constraints
        for i, j in pair_idx:
            def pair_fun(v, i=i, j=j):
                xi, yi = v[2 * i], v[2 * i + 1]
                xj, yj = v[2 * j], v[2 * j + 1]
                ri, rj = v[2 * n + i], v[2 * n + j]
                return np.hypot(xi - xj, yi - yj) - (ri + rj)
            cons.append({"type": "ineq", "fun": pair_fun})

        # bounds: 0<=x,y<=1, r>=0 (no explicit upper bound – borders are enforced by constraints)
        bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, None)] * n

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 200, "ftol": 1e-9, "disp": False},
        )
        if not result.success:
            return centers, radii
        v_opt = result.x
        new_centers = v_opt[: 2 * n].reshape(n, 2)
        new_radii = v_opt[2 * n :]
        return new_centers, new_radii

    # run a few random perturbations before the SLSQP to escape local minima
    rng = np.random.default_rng()
    best_sum = radii.sum()
    best_centers, best_radii = centres, radii

    for _ in range(8):
        # jitter each centre a little
        jitter = rng.normal(scale=0.015, size=best_centers.shape)
        perturbed = np.clip(best_centers + jitter, 0.0, 1.0)
        pert_radii = solve_lp(perturbed)
        # if LP already improves, keep it
        if pert_radii.sum() > best_sum:
            best_sum = pert_radii.sum()
            best_centers, best_radii = perturbed, pert_radii

    # final continuous optimisation
    refined_centers, refined_radii = refine_slsqp(best_centers, best_radii)

    total = float(refined_radii.sum())
    return refined_centers, refined_radii, total
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
