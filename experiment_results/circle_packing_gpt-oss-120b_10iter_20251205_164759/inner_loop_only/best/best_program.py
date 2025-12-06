"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """Try several layouts and keep the one with the largest total radius."""
    import numpy as np

    best_sum = -1.0
    best_centers = best_radii = None

    def update(c):
        nonlocal best_sum, best_centers, best_radii
        r = compute_max_radii(c)
        s = r.sum()
        if s > best_sum:
            best_sum, best_centers, best_radii = s, c.copy(), r.copy()

    # 1) 5×5 grid + extra point
    gv = np.arange(0.1, 1.0, 0.2)
    g = np.array([[x, y] for x in gv for y in gv], float)
    g = np.vstack([g, [[0.2, 0.2]]])
    update(g)

    # 2) two‑ring pattern (center + inner + outer)
    n = 26
    cen = np.array([0.5, 0.5])
    inner, outer = 0.3, 0.55
    r2 = np.zeros((n, 2))
    r2[0] = cen
    for i in range(8):
        a = 2 * np.pi * i / 8
        r2[i + 1] = cen + inner * np.array([np.cos(a), np.sin(a)])
    for i in range(16):
        a = 2 * np.pi * i / 16
        r2[i + 9] = cen + outer * np.array([np.cos(a), np.sin(a)])
    r2 = np.clip(r2, 0.001, 0.999)
    update(r2)

    # 3) a few random layouts
    rng = np.random.default_rng(42)
    for _ in range(5):
        rand = rng.random((n, 2))
        update(rand)

    return best_centers, best_radii, best_sum


def compute_max_radii(centers):
    """Linear‑programming max‑sum‑radii for fixed centre positions."""
    import numpy as np
    from scipy.optimize import linprog

    n = centers.shape[0]
    # border limits
    bl = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )
    c = -np.ones(n)

    # pairwise non‑overlap constraints
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(pairs)
    A = np.zeros((m + n, n))
    b = np.empty(m + n)

    row = 0
    for i, j in pairs:
        A[row, i] = A[row, j] = 1.0
        b[row] = np.linalg.norm(centers[i] - centers[j])
        row += 1

    # border constraints
    A[row:] = np.eye(n)
    b[row:] = bl

    bounds = [(0, None)] * n
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    return res.x if res.success else bl
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
