"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    import numpy as np
    from itertools import product
    from scipy.optimize import linprog

    rng = np.random.default_rng(12345)
    n = 26

    def _solve_lp(centers):
        b = np.minimum.reduce([centers[:,0], centers[:,1],
                               1-centers[:,0], 1-centers[:,1]])
        d = np.linalg.norm(centers[:,None,:]-centers[None,:, :], axis=2)
        m = n + n*(n-1)//2
        A = np.zeros((m, n))
        B = np.empty(m)
        rows = np.arange(n)
        A[rows, rows] = 1
        B[:n] = b
        k = n
        for i in range(n):
            for j in range(i+1, n):
                A[k,i] = A[k,j] = 1
                B[k] = d[i,j]
                k += 1
        res = linprog(-np.ones(n), A_ub=A, b_ub=B,
                      bounds=[(0, None)]*n, method='highs')
        return np.clip(res.x,0,None) if res.success else np.zeros(n)

    def _max_scale(centers, radii):
        eps = 1e-12
        border = np.minimum.reduce([centers[:,0], centers[:,1],
                                   1-centers[:,0], 1-centers[:,1]])
        with np.errstate(divide='ignore', invalid='ignore'):
            sb = np.where(radii>eps, border/radii, np.inf)
        d = np.linalg.norm(centers[:,None,:]-centers[None,:, :], axis=2)
        sr = radii[:,None]+radii[None,:]
        np.fill_diagonal(d, np.inf); np.fill_diagonal(sr, np.inf)
        with np.errstate(divide='ignore', invalid='ignore'):
            sp = np.where(sr>eps, d/sr, np.inf)
        return max(min(np.min(sb), np.min(sp))*0.999999, 0.0)

    xs = np.linspace(0.1,0.9,5)
    grid = np.array(list(product(xs,xs)))
    extra = np.array([[0.5,0.2]])
    base = np.vstack([grid, extra])

    best_sum = -1.0
    best_c = best_r = None

    for t in range(30):
        if t==0:
            c = base.copy()
        elif t<6:
            c = np.clip(base + rng.uniform(-0.02,0.02,base.shape),0.01,0.99)
        elif t<12:
            extra_pt = rng.uniform([0.3,0.05],[0.7,0.35],(1,2))
            c = np.vstack([grid, extra_pt])
        else:
            c = rng.uniform(0.01,0.99,(n,2))
        r0 = _solve_lp(c)
        r = r0 * _max_scale(c, r0)
        s = r.sum()
        if s>best_sum:
            best_sum, best_c, best_r = s, c.copy(), r.copy()

    for _ in range(20):
        i = rng.integers(n)
        cand = best_c.copy()
        cand[i] = np.clip(cand[i] + rng.normal(scale=0.008, size=2),0.01,0.99)
        r0 = _solve_lp(cand)
        r = r0 * _max_scale(cand, r0)
        s = r.sum()
        if s>best_sum:
            best_sum, best_c, best_r = s, cand.copy(), r.copy()

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
