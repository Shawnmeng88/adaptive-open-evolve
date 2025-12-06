"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    n = 26

    # ----- initial deterministic layout (center + two rings) -----
    centers = np.zeros((n, 2))
    centers[0] = [0.5, 0.5]
    r_inner, r_outer = 0.25, 0.45
    for i in range(8):
        a = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + r_inner * np.cos(a),
                         0.5 + r_inner * np.sin(a)]
    for i in range(16):
        a = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + r_outer * np.cos(a),
                         0.5 + r_outer * np.sin(a)]
    centers = np.clip(centers, 0.001, 0.999)

    # ----- LP to maximise sum of radii for a fixed centre set -----
    def optimal_radii(cs: np.ndarray) -> np.ndarray:
        border = np.minimum.reduce([cs[:, 0], cs[:, 1], 1 - cs[:, 0], 1 - cs[:, 1]])

        rows, rhs = [], []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(cs[i] - cs[j])
                row = np.zeros(n)
                row[i] = row[j] = 1
                rows.append(row)
                rhs.append(d)

        A_ub = np.array(rows) if rows else None
        b_ub = np.array(rhs) if rhs else None
        c = -np.ones(n)
        bounds = [(0, border[i]) for i in range(n)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method='highs', options={'presolve': True})
        return res.x if res.success else border

    # ----- hill‑climbing search on centre positions -----
    rng = np.random.default_rng(0)
    best_c = centers.copy()
    best_r = optimal_radii(best_c)
    best_sum = best_r.sum()

    total_iters = 3000
    max_step = 0.025
    min_step = 0.001

    for it in range(total_iters):
        # linearly decreasing step size
        sigma = min_step + (max_step - min_step) * (1 - it / total_iters)

        # occasional multi‑circle moves to escape local optima
        k = 1 if (it % 200) else rng.integers(2, 5)
        idxs = rng.choice(n, size=k, replace=False)

        cand = best_c.copy()
        for idx in idxs:
            cand[idx] = np.clip(
                cand[idx] + rng.uniform(-sigma, sigma, 2),
                0.001, 0.999)

        r = optimal_radii(cand)
        s = r.sum()

        if s > best_sum:
            best_sum, best_c, best_r = s, cand, r

    return best_c, best_r, best_sum
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
