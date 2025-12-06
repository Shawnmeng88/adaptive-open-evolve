"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a high‑quality packing of 26 circles in the unit square.
    Strategy:
      1. Generate many hexagonal candidate grids with modest random spacing and jitter.
      2. Solve a linear programme for all candidates to obtain feasible radii.
      3. Keep the 26 circles with the largest radii as a starting layout.
      4. Refine positions and radii jointly with SLSQP.
      5. Perform a few local‑perturbation rounds on the best layout.
    Returns
    -------
    centers : np.ndarray shape (26, 2)
    radii   : np.ndarray shape (26,)
    total   : float   (sum of radii)
    """
    import numpy as np
    from scipy.optimize import linprog, minimize

    # ----------------------------------------------------------------------
    # Helper: hexagonal lattice generator
    # ----------------------------------------------------------------------
    def hex_grid(spacing: float) -> np.ndarray:
        """Points of a hexagonal lattice confined to the unit square."""
        dy = spacing * np.sqrt(3.0) / 2.0
        pts = []
        row = 0
        while True:
            y = row * dy + spacing / 2.0
            if y > 1.0 - spacing / 2.0:
                break
            x_off = spacing / 2.0 if (row % 2) else 0.0
            col = 0
            while True:
                x = x_off + col * spacing + spacing / 2.0
                if x > 1.0 - spacing / 2.0:
                    break
                pts.append([x, y])
                col += 1
            row += 1
        return np.asarray(pts)

    # ----------------------------------------------------------------------
    # Helper: linear programme for fixed centres
    # ----------------------------------------------------------------------
    def lp_max_radii(centers: np.ndarray) -> np.ndarray:
        """Return radii that maximise total sum for given centres."""
        n = centers.shape[0]
        c = -np.ones(n)                     # maximise sum(r) → minimise -sum(r)
        bounds = [(0.0, None)] * n

        A = []
        b = []

        # border constraints
        for i, (x, y) in enumerate(centers):
            for d in (x, y, 1.0 - x, 1.0 - y):
                row = np.zeros(n)
                row[i] = 1.0
                A.append(row)
                b.append(d)

        # pairwise non‑overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                row = np.zeros(n)
                row[i] = row[j] = 1.0
                A.append(row)
                b.append(d)

        res = linprog(c, A_ub=np.array(A), b_ub=np.array(b),
                      bounds=bounds, method='highs', options={'presolve': True})
        return res.x if res.success else np.zeros(n)

    # ----------------------------------------------------------------------
    # Helper: SLSQP refinement of centres & radii
    # ----------------------------------------------------------------------
    def refine_layout(centers: np.ndarray, radii0: np.ndarray):
        """Jointly optimise centres and radii using SLSQP."""
        n = centers.shape[0]
        x0 = np.column_stack([centers, radii0]).ravel()
        bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, None)] * n

        def border(v):
            xs, ys, rs = v[0::3], v[1::3], v[2::3]
            return np.concatenate([xs - rs,
                                   1.0 - xs - rs,
                                   ys - rs,
                                   1.0 - ys - rs])

        def overlap(v):
            xs, ys, rs = v[0::3], v[1::3], v[2::3]
            dx = xs[:, None] - xs
            dy = ys[:, None] - ys
            dist2 = dx ** 2 + dy ** 2
            rad2 = (rs[:, None] + rs) ** 2
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            return (dist2 - rad2)[mask]

        cons = [{'type': 'ineq', 'fun': border},
                {'type': 'ineq', 'fun': overlap}]

        def objective(v):
            return -np.sum(v[2::3])          # maximise total radius

        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=cons,
                       options={'maxiter': 300, 'ftol': 1e-8, 'disp': False})
        if res.success:
            v = res.x
            centers_opt = np.column_stack([v[0::3], v[1::3]])
            radii_opt = v[2::3]
            return centers_opt, radii_opt, float(radii_opt.sum())
        # fallback to the LP solution
        return centers, radii0, float(radii0.sum())

    rng = np.random.default_rng()
    best_sum = -1.0
    best_centers = best_radii = None

    # ----------------------------------------------------------------------
    # Main search – many randomised grid attempts
    # ----------------------------------------------------------------------
    for attempt in range(20):
        # modest spacing range that reliably yields enough points
        spacing = rng.uniform(0.11, 0.16)
        candidates = hex_grid(spacing)

        # ensure we have enough points; if not, fall back to a finer grid
        if candidates.shape[0] < 30:
            candidates = hex_grid(0.10)

        # gentle jitter to break symmetry without destroying feasibility
        jitter = rng.uniform(-0.008, 0.008, candidates.shape)
        candidates = np.clip(candidates + jitter, 0.0, 1.0)

        # LP radii for all candidates
        radii_all = lp_max_radii(candidates)

        if candidates.shape[0] < 26:
            continue

        # select 26 circles with largest LP radii
        idx = np.argpartition(radii_all, -26)[-26:]
        idx = idx[np.argsort(-radii_all[idx])]   # sort descending for determinism
        centers_sel = candidates[idx]
        radii_sel = radii_all[idx]

        # refinement via SLSQP
        cen_opt, rad_opt, total = refine_layout(centers_sel, radii_sel)
        if total > best_sum:
            best_sum, best_centers, best_radii = total, cen_opt, rad_opt

    # ----------------------------------------------------------------------
    # Local perturbation rounds on the current best layout
    # ----------------------------------------------------------------------
    if best_centers is not None:
        for _ in range(6):
            pert = rng.uniform(-0.003, 0.003, best_centers.shape)
            cand_centers = np.clip(best_centers + pert, 0.0, 1.0)

            cand_radii0 = lp_max_radii(cand_centers)

            cen_opt, rad_opt, total = refine_layout(cand_centers, cand_radii0)
            if total > best_sum:
                best_sum, best_centers, best_radii = total, cen_opt, rad_opt

    # ----------------------------------------------------------------------
    # Fallback – pure random points if nothing succeeded
    # ----------------------------------------------------------------------
    if best_centers is None:
        rand = rng.uniform(0.0, 1.0, (26, 2))
        rad0 = lp_max_radii(rand)
        best_centers, best_radii, best_sum = refine_layout(rand, rad0)

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
