"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def optimal_radii(centers):
    """Compute radii that maximise total sum for given centre positions."""
    m = centers.shape[0]
    # distance to square borders
    border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
    )
    # pairwise centre distances
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)

    rows = []
    rhs = []

    # border constraints: r_i <= border_i
    for i in range(m):
        row = np.zeros(m)
        row[i] = 1.0
        rows.append(row)
        rhs.append(border[i])

    # non‑overlap constraints: r_i + r_j <= d_ij
    for i in range(m):
        for j in range(i + 1, m):
            row = np.zeros(m)
            row[i] = row[j] = 1.0
            rows.append(row)
            rhs.append(dists[i, j])

    A = np.array(rows)
    b = np.array(rhs)
    c = -np.ones(m)                     # maximise sum(r) → minimise -sum(r)

    res = linprog(c, A_ub=A, b_ub=b,
                 bounds=[(0, None)] * m,
                 method="highs", options={"presolve": True})
    return res.x if res.success else border * 0.5


def construct_packing():
    n = 26

    # generate hexagonal lattice points for a given centre spacing d
    def gen(d):
        r = d / 2.0
        pts = []
        row = 0
        y = r
        vert = d * np.sqrt(3) / 2.0
        while y <= 1 - r + 1e-12 and len(pts) < n:
            offset = 0.0 if row % 2 == 0 else d / 2.0
            x = r + offset
            while x <= 1 - r + 1e-12 and len(pts) < n:
                pts.append([x, y])
                x += d
            y += vert
            row += 1
        return np.array(pts)

    # binary‑search the largest spacing that still fits n points
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2.0
        if len(gen(mid)) >= n:
            lo = mid
        else:
            hi = mid
    spacing = lo
    base = gen(spacing)[:n]

    rng = np.random.default_rng()

    # optimisation parameters – a bit more thorough than previous version
    num_restarts = 6
    local_iters = 800
    init_jitter = spacing * 0.07

    best_sum = -1.0
    best_centers = None
    best_radii = None

    for _ in range(num_restarts):
        centers = base + rng.uniform(-init_jitter, init_jitter, size=base.shape)
        centers = np.clip(centers, 0.0, 1.0)

        radii = optimal_radii(centers)
        cur_sum = radii.sum()
        cur_centers, cur_radii = centers, radii

        for it in range(local_iters):
            scale = spacing * 0.12 * (1.0 - it / local_iters)
            idx = rng.integers(n)
            cand = cur_centers.copy()
            cand[idx] += rng.uniform(-scale, scale, size=2)
            cand = np.clip(cand, 0.0, 1.0)

            cand_radii = optimal_radii(cand)
            s = cand_radii.sum()
            if s > cur_sum:
                cur_sum, cur_centers, cur_radii = s, cand, cand_radii

        if cur_sum > best_sum:
            best_sum, best_centers, best_radii = cur_sum, cur_centers, cur_radii

    # final fine‑tuning around the best layout
    fine_iters = 300
    for it in range(fine_iters):
        jitter = spacing * 0.04 * (1.0 - it / fine_iters)
        idx = rng.integers(n)
        cand = best_centers.copy()
        cand[idx] += rng.uniform(-jitter, jitter, size=2)
        cand = np.clip(cand, 0.0, 1.0)

        cand_radii = optimal_radii(cand)
        s = cand_radii.sum()
        if s > best_sum:
            best_sum, best_centers, best_radii = s, cand, cand_radii

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
