"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    n = 26
    rng = np.random.default_rng()

    def solve(cent):
        m = cent.shape[0]
        c = -np.ones(m)
        rows, b = [], []
        for i, (x, y) in enumerate(cent):
            for lim in (x, 1 - x, y, 1 - y):
                row = np.zeros(m); row[i] = 1
                rows.append(row); b.append(lim)
        for i in range(m):
            for j in range(i + 1, m):
                d = np.linalg.norm(cent[i] - cent[j])
                row = np.zeros(m); row[i] = row[j] = 1
                rows.append(row); b.append(d)
        A = np.vstack(rows); B = np.array(b)
        res = linprog(c, A_ub=A, b_ub=B, bounds=[(0, None)] * m, method='highs')
        return res.x if res.success else np.zeros(m)

    def lattice():
        xs = np.linspace(0.12, 0.88, 5)
        ys = np.linspace(0.12, 0.88, 5)
        g = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)[:25]
        return np.vstack([g, [[0.5, 0.5]]])

    def random_layout(min_dist=0.02):
        pts = []
        while len(pts) < n:
            p = rng.random(2)
            if not pts or np.all(np.linalg.norm(np.array(pts) - p, axis=1) > min_dist):
                pts.append(p)
        return np.array(pts)

    # ---- initial exploration ----
    best_sum = -1.0
    best_c = best_r = None
    for start in [lattice()] + [random_layout() for _ in range(200)]:
        r = solve(start)
        s = r.sum()
        if s > best_sum:
            best_sum, best_c, best_r = s, start.copy(), r.copy()

    # ---- iterative refinement ----
    init_step, decay = 0.07, 0.93
    for outer in range(30):
        step = init_step * (decay ** outer)
        improved = False
        for i in range(n):
            # try several perturbations for circle i
            K = 6
            deltas = (rng.random((K, 2)) - 0.5) * 2 * step
            cand_pos = np.clip(best_c[i] + deltas, 0.001, 0.999)
            for pos in cand_pos:
                trial = best_c.copy()
                trial[i] = pos
                r = solve(trial)
                s = r.sum()
                if s > best_sum:
                    best_sum, best_c, best_r = s, trial, r
                    improved = True
        if not improved:
            # occasional global shake to escape plateaus
            shake = best_c + (rng.random(best_c.shape) - 0.5) * 0.02
            shake = np.clip(shake, 0.001, 0.999)
            r = solve(shake)
            s = r.sum()
            if s > best_sum:
                best_sum, best_c, best_r = s, shake, r

    best_c = np.clip(best_c, 0.001, 0.999)
    return best_c, best_r, float(best_sum)
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
