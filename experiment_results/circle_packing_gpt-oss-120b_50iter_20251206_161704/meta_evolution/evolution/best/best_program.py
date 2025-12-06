"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def construct_packing():
    n = 26
    rng = np.random.default_rng()

    # --------------------------------------------------------------
    # LP: maximise total radius for a fixed set of centres
    # --------------------------------------------------------------
    def optimal_radii(cent):
        m = cent.shape[0]
        c = -np.ones(m)                     # maximise sum r_i → minimise -sum
        bounds = [(0.0, None)] * m

        rows, rhs = [], []

        # wall constraints: r_i <= distance to each side
        for i, (x, y) in enumerate(cent):
            for d in (x, y, 1.0 - x, 1.0 - y):
                row = np.zeros(m)
                row[i] = 1.0
                rows.append(row)
                rhs.append(d)

        # pairwise non‑overlap constraints
        for i in range(m):
            for j in range(i + 1, m):
                d = np.linalg.norm(cent[i] - cent[j])
                row = np.zeros(m)
                row[i] = row[j] = 1.0
                rows.append(row)
                rhs.append(d)

        A = np.vstack(rows)
        b = np.array(rhs)

        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        return res.x if res.success else np.zeros(m)

    # --------------------------------------------------------------
    # Layout generators
    # --------------------------------------------------------------
    def grid_layout():
        vals = np.linspace(0.12, 0.88, 5)
        gx, gy = np.meshgrid(vals, vals)
        pts = np.column_stack([gx.ravel(), gy.ravel()])[:25]
        return np.vstack([pts, [0.5, 0.5]])

    def jittered_grid(jitter=0.02):
        base = grid_layout()
        noise = rng.uniform(-jitter, jitter, size=base.shape)
        return np.clip(base + noise, 0.01, 0.99)

    def hex_layout(spacing=0.2):
        dx, dy = spacing, np.sqrt(3) * spacing / 2
        pts = []
        y, row = 0.15, 0
        while y < 0.85 and len(pts) < n:
            offset = 0.0 if row % 2 == 0 else dx / 2
            x = 0.15 + offset
            while x < 0.85 and len(pts) < n:
                pts.append([x, y])
                x += dx
            y += dy
            row += 1
        while len(pts) < n:
            pts.append([0.5, 0.5])
        return np.array(pts[:n])

    def random_layout():
        return rng.uniform(0.05, 0.95, size=(n, 2))

    # --------------------------------------------------------------
    # Adaptive local search – small random moves
    # --------------------------------------------------------------
    def local_search(cent, cur_sum, steps=250, init_scale=0.025):
        best_c = cent.copy()
        best_sum = cur_sum
        scale = init_scale
        for _ in range(steps):
            i = rng.integers(n)
            proposal = best_c.copy()
            delta = rng.normal(scale=scale, size=2)
            proposal[i] = np.clip(proposal[i] + delta, 0.01, 0.99)
            rad = optimal_radii(proposal)
            s = rad.sum()
            if s > best_sum:
                best_sum, best_c = s, proposal
                scale = max(scale * 0.95, 0.004)   # tighten after improvement
        return best_c, best_sum

    # --------------------------------------------------------------
    # Deterministic fine‑grained refinement
    # --------------------------------------------------------------
    def fine_refine(cent, cur_sum, passes=4, delta=0.004):
        best_c = cent.copy()
        best_sum = cur_sum
        for _ in range(passes):
            improved = False
            for i in range(n):
                for d in [(delta, 0), (-delta, 0), (0, delta), (0, -delta)]:
                    cand = best_c.copy()
                    cand[i] = np.clip(cand[i] + d, 0.01, 0.99)
                    s = optimal_radii(cand).sum()
                    if s > best_sum + 1e-9:
                        best_c, best_sum = cand, s
                        improved = True
            if not improved:
                break
        return best_c, best_sum

    # --------------------------------------------------------------
    # Main multi‑restart optimisation loop
    # --------------------------------------------------------------
    best_centers = grid_layout()
    best_radii = optimal_radii(best_centers)
    best_sum = best_radii.sum()

    total_iters = 1500
    for it in range(total_iters):
        # switch base layout periodically
        if it % 180 == 0:
            mode = (it // 180) % 4
            if mode == 0:
                cand = grid_layout()
            elif mode == 1:
                cand = hex_layout(spacing=0.2)
            elif mode == 2:
                cand = hex_layout(spacing=0.16)
            else:
                cand = random_layout()
        else:
            decay = 0.07 * (0.94 ** (it // 30))
            cand = best_centers + rng.uniform(-decay, decay, size=best_centers.shape)

        cand = np.clip(cand, 0.01, 0.99)

        cand_radii = optimal_radii(cand)
        cand_sum = cand_radii.sum()

        # focus local search on promising candidates
        if cand_sum > best_sum * 0.99:
            cand, cand_sum = local_search(cand, cand_sum, steps=250, init_scale=0.025)

        if cand_sum > best_sum:
            best_sum = cand_sum
            best_centers = cand
            best_radii = optimal_radii(best_centers)

    # final deterministic polishing
    best_centers, best_sum = fine_refine(best_centers, best_sum, passes=4, delta=0.004)
    best_radii = optimal_radii(best_centers)

    return best_centers, best_radii, float(best_sum)
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
