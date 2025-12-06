"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a high‑quality packing of 26 circles in the unit square.
    Strategy:
        1. Generate dense candidate sets (hexagonal grids + large random pools).
        2. Prune each set to exactly 26 points using an LP that maximises radii.
        3. Refine the selected 26‑point configuration with a short SLSQP optimisation.
        4. Keep the best result over many stochastic restarts.
    """
    import numpy as np
    from scipy.optimize import linprog, minimize

    # ----- linear programme for a fixed set of centres --------------------
    def lp_opt(centers):
        n = centers.shape[0]
        c = -np.ones(n)                     # maximise sum(r) → minimise -sum(r)
        bounds = [(0, None)] * n
        A, b = [], []
        # border constraints
        for i, (x, y) in enumerate(centers):
            for d in (x, y, 1.0 - x, 1.0 - y):
                row = np.zeros(n)
                row[i] = 1.0
                A.append(row)
                b.append(d)
        # non‑overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                row = np.zeros(n)
                row[i] = row[j] = 1.0
                A.append(row)
                b.append(d)
        res = linprog(c, A_ub=np.asarray(A), b_ub=np.asarray(b),
                      bounds=bounds, method='highs')
        return res.x if res.success else np.zeros(n)

    # ----- prune a candidate pool to exactly `target` points -------------
    def prune_to_n(pool, target, rng):
        pts = pool.copy()
        while pts.shape[0] > target:
            radii = lp_opt(pts)
            if np.all(radii == 0):
                drop = rng.integers(pts.shape[0])
            else:
                drop = np.argmin(radii)
            pts = np.delete(pts, drop, axis=0)
        return pts, lp_opt(pts)

    # ----- SLSQP refinement ---------------------------------------------
    def refine(centers, radii0):
        n = centers.shape[0]
        x0 = np.column_stack([centers, radii0]).ravel()
        bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, None)] * n

        def border(v):
            xs, ys, rs = v[0::3], v[1::3], v[2::3]
            return np.concatenate([xs - rs, 1.0 - xs - rs,
                                   ys - rs, 1.0 - ys - rs])

        def overlap(v):
            xs, ys, rs = v[0::3], v[1::3], v[2::3]
            cons = []
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    cons.append(dx * dx + dy * dy - (rs[i] + rs[j]) ** 2)
            return np.array(cons)

        constraints = [
            {'type': 'ineq', 'fun': border},
            {'type': 'ineq', 'fun': overlap}
        ]

        def obj(v):
            return -np.sum(v[2::3])

        res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': 500, 'ftol': 1e-9, 'disp': False})
        if res.success:
            v = res.x
            return np.column_stack([v[0::3], v[1::3]]), v[2::3], float(v[2::3].sum())
        return centers, radii0, float(radii0.sum())

    # ----- hexagonal lattice generator ------------------------------------
    def hex_grid(spacing):
        dy = spacing * np.sqrt(3) / 2.0
        pts = []
        row = 0
        while True:
            y = row * dy + spacing / 2.0
            if y > 1 - spacing / 2.0:
                break
            offset = spacing / 2.0 if (row % 2) else 0.0
            col = 0
            while True:
                x = offset + col * spacing + spacing / 2.0
                if x > 1 - spacing / 2.0:
                    break
                pts.append([x, y])
                col += 1
            row += 1
        return np.array(pts)

    rng = np.random.default_rng()
    best_sum = -1.0
    best_centers = best_radii = None

    # 1. Hex‑grid candidates (dense grids, then prune)
    for spacing in np.linspace(0.12, 0.22, 9):
        pool = hex_grid(spacing)
        if pool.shape[0] <= 26:
            continue
        centers, radii0 = prune_to_n(pool, 26, rng)
        c, r, s = refine(centers, radii0)
        if s > best_sum:
            best_sum, best_centers, best_radii = s, c, r
        # jittered variants of the pruned set
        for _ in range(4):
            jitter = rng.uniform(-0.015, 0.015, centers.shape)
            cand = np.clip(centers + jitter, 0.0, 1.0)
            radii0_j = lp_opt(cand)
            c, r, s = refine(cand, radii0_j)
            if s > best_sum:
                best_sum, best_centers, best_radii = s, c, r

    # 2. Large random pools (150 points) pruned to 26
    for _ in range(80):
        pool = rng.uniform(0.0, 1.0, (150, 2))
        centers, radii0 = prune_to_n(pool, 26, rng)
        c, r, s = refine(centers, radii0)
        if s > best_sum:
            best_sum, best_centers, best_radii = s, c, r

    # 3. Direct 26‑point random attempts (fallback)
    for _ in range(100):
        cand = rng.uniform(0.0, 1.0, (26, 2))
        radii0 = lp_opt(cand)
        c, r, s = refine(cand, radii0)
        if s > best_sum:
            best_sum, best_centers, best_radii = s, c, r

    # guaranteed valid result
    if best_centers is None:
        cand = rng.uniform(0.0, 1.0, (26, 2))
        radii0 = lp_opt(cand)
        best_centers, best_radii, best_sum = refine(cand, radii0)

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
