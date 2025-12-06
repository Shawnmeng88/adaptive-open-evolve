"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a packing of 26 circles inside the unit square.
    Starts from a hexagonal lattice, solves a linear program for the radii,
    then performs a cheap stochastic local search to improve the total radius.
    Returns (centers, radii, sum_of_radii).
    """
    import numpy as np
    from scipy.optimize import linprog

    # ------------------------------------------------------------------
    # Helper: generate a hexagonal lattice inside the unit square
    # ------------------------------------------------------------------
    def generate_hex_lattice(spacing):
        dy = spacing * np.sqrt(3) / 2.0
        start_y = spacing / 2.0
        y_vals = np.arange(start_y, 1.0 + 1e-9, dy)
        pts = []
        for row_idx, y in enumerate(y_vals):
            offset = 0.0 if row_idx % 2 == 0 else spacing / 2.0
            start_x = spacing / 2.0 + offset
            x_vals = np.arange(start_x, 1.0 + 1e-9, spacing)
            for x in x_vals:
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    pts.append([x, y])
        return np.array(pts)

    # ------------------------------------------------------------------
    # Helper: solve LP that maximises sum of radii for a fixed set of centres
    # ------------------------------------------------------------------
    def optimal_radii_lp(centers):
        n = centers.shape[0]
        # maximal radius limited by the four borders
        border = np.minimum.reduce([centers[:, 0], centers[:, 1],
                                    1.0 - centers[:, 0], 1.0 - centers[:, 1]])

        # pairwise distances
        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))

        rows = []
        b = []

        # border constraints: r_i <= border_i
        for i in range(n):
            row = np.zeros(n)
            row[i] = 1.0
            rows.append(row)
            b.append(border[i])

        # non‑overlap constraints: r_i + r_j <= d_ij
        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                rows.append(row)
                b.append(dists[i, j])

        A_ub = np.array(rows)
        b_ub = np.array(b)

        # maximise sum(r_i)  <=> minimise -sum(r_i)
        c = -np.ones(n)

        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(0.0, None)] * n, method='highs')
        if res.success:
            return res.x
        # fallback – simple geometric bound
        return np.minimum(border, np.min(dists + np.eye(n) * 1e9, axis=1) / 2.0)

    # ------------------------------------------------------------------
    # Phase 1 – coarse search over hex lattice spacings
    # ------------------------------------------------------------------
    best_sum = -1.0
    best_centers = None
    best_radii = None

    for spacing in np.linspace(0.12, 0.30, 12):
        cand_pts = generate_hex_lattice(spacing)
        if cand_pts.shape[0] < 26:
            continue

        # Prefer points farther from the border (they can host larger circles)
        border_dist = np.minimum.reduce([cand_pts[:, 0], cand_pts[:, 1],
                                         1.0 - cand_pts[:, 0], 1.0 - cand_pts[:, 1]])
        idx = np.argsort(-border_dist)          # descending order
        centers = cand_pts[idx[:26]]

        radii = optimal_radii_lp(centers)
        total = radii.sum()
        if total > best_sum:
            best_sum = total
            best_centers = centers.copy()
            best_radii = radii.copy()

    # ------------------------------------------------------------------
    # Phase 2 – random initialisations (quick diversification)
    # ------------------------------------------------------------------
    rng = np.random.default_rng()
    for _ in range(20):
        centers = rng.random((26, 2))
        radii = optimal_radii_lp(centers)
        total = radii.sum()
        if total > best_sum:
            best_sum = total
            best_centers = centers.copy()
            best_radii = radii.copy()

    # ------------------------------------------------------------------
    # Phase 3 – stochastic local refinement of the current best layout
    # ------------------------------------------------------------------
    if best_centers is not None:
        current_centers = best_centers.copy()
        current_radii = best_radii.copy()
        current_sum = best_sum

        for _ in range(300):
            i = rng.integers(26)
            # propose a small Gaussian move
            proposal = current_centers[i] + rng.normal(scale=0.02, size=2)
            proposal = np.clip(proposal, 0.0, 1.0)

            new_centers = current_centers.copy()
            new_centers[i] = proposal

            new_radii = optimal_radii_lp(new_centers)
            new_sum = new_radii.sum()

            if new_sum > current_sum + 1e-6:
                current_sum = new_sum
                current_centers = new_centers
                current_radii = new_radii

        # keep the refined solution if it improved
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = current_centers
            best_radii = current_radii

    # ------------------------------------------------------------------
    # Fallback – very simple pattern (should never be needed)
    # ------------------------------------------------------------------
    if best_centers is None:
        n = 26
        centers = np.zeros((n, 2))
        centers[0] = [0.5, 0.5]
        for i in range(8):
            ang = 2 * np.pi * i / 8
            centers[i + 1] = [0.5 + 0.3 * np.cos(ang), 0.5 + 0.3 * np.sin(ang)]
        for i in range(16):
            ang = 2 * np.pi * i / 16
            centers[i + 9] = [0.5 + 0.7 * np.cos(ang), 0.5 + 0.7 * np.sin(ang)]
        centers = np.clip(centers, 0.01, 0.99)
        best_centers = centers
        best_radii = compute_max_radii(centers)
        best_sum = best_radii.sum()

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
