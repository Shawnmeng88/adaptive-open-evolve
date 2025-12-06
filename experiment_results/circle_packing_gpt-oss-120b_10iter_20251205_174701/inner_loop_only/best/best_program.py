"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    import numpy as np, math
    from scipy.optimize import linprog

    def optimal_radii(centers):
        n = len(centers)
        border = np.minimum(centers, 1 - centers).min(axis=1)
        rows, b = [], []
        for i in range(n):
            row = np.zeros(n); row[i] = 1
            rows.append(row); b.append(border[i])
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                row = np.zeros(n); row[i] = row[j] = 1
                rows.append(row); b.append(d)
        A = np.vstack(rows); b = np.array(b)
        res = linprog(-np.ones(n), A_ub=A, b_ub=b,
                     bounds=[(0, None)] * n, method="highs")
        return res.x if res.success else np.zeros(n)

    n = 26
    rng = np.random.default_rng(0)
    best_sum, best_c, best_r = -1, None, None

    # deterministic rings with a few radius choices
    for inner in (0.15, 0.2, 0.25):
        for outer in (0.4, 0.45, 0.5):
            centers = np.zeros((n, 2))
            centers[0] = [0.5, 0.5]
            for i in range(8):
                a = 2 * math.pi * i / 8
                centers[i + 1] = [0.5 + inner * math.cos(a),
                                 0.5 + inner * math.sin(a)]
            for i in range(16):
                a = 2 * math.pi * i / 16
                centers[i + 9] = [0.5 + outer * math.cos(a),
                                 0.5 + outer * math.sin(a)]
            # jitter a few times around this pattern
            for _ in range(3):
                cand = np.clip(centers + rng.normal(scale=0.02,
                                 size=centers.shape), 0.01, 0.99)
                r = optimal_radii(cand)
                s = r.sum()
                if s > best_sum:
                    best_sum, best_c, best_r = s, cand, r

    # some pure random layouts
    for _ in range(5):
        cand = rng.uniform(0.02, 0.98, size=(n, 2))
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
