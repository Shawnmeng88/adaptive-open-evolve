"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Stochastic + multi‑scale local search for 26‑circle packing.
    """
    # ---- initial grid + centre ----
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 5)
    xv, yv = np.meshgrid(xs, ys)
    centers = np.column_stack([xv.ravel(), yv.ravel()])
    centers = np.vstack([centers, [0.55, 0.55]])

    best_radii = compute_max_radii(centers)
    best_sum = best_radii.sum()
    best_centers = centers.copy()
    rng = np.random.default_rng(0)

    # ---- phase‑1: cooling stochastic search ----
    for i in range(120):
        sigma = 0.05 * (1 - i / 120) + 0.005
        cand = np.clip(best_centers + rng.normal(scale=sigma,
                                                size=best_centers.shape), 0, 1)
        r = compute_max_radii(cand)
        s = r.sum()
        if s > best_sum:
            best_sum, best_radii, best_centers = s, r, cand

    # ---- helper for deterministic refinement ----
    def refine(centers, cur_sum, step):
        dirs = np.array([[0, 0],
                         [ step, 0], [-step, 0],
                         [0,  step], [0, -step],
                         [ step,  step], [ step, -step],
                         [-step,  step], [-step, -step]])
        improved = False
        for idx in range(centers.shape[0]):
            cur = centers[idx].copy()
            local_best_sum, local_best_pos = cur_sum, cur
            for d in dirs:
                p = np.clip(cur + d, 0, 1)
                tmp = centers.copy()
                tmp[idx] = p
                r = compute_max_radii(tmp)
                s = r.sum()
                if s > local_best_sum + 1e-9:
                    local_best_sum, local_best_pos = s, p
            if local_best_sum > cur_sum + 1e-9:
                centers[idx] = local_best_pos
                cur_sum = local_best_sum
                improved = True
        return centers, cur_sum, improved

    # ---- phase‑2: coarse refinement (step 0.01) ----
    while True:
        best_centers, best_sum, changed = refine(best_centers, best_sum, 0.01)
        if not changed:
            break

    # ---- phase‑3: fine refinement (step 0.004) ----
    while True:
        best_centers, best_sum, changed = refine(best_centers, best_sum, 0.004)
        if not changed:
            break

    best_radii = compute_max_radii(best_centers)
    return best_centers, best_radii, best_sum


def compute_max_radii(centers):
    """
    Linear‑program to maximise sum of radii under border and non‑overlap constraints.
    """
    n = centers.shape[0]
    border = np.minimum.reduce([centers[:, 0], centers[:, 1],
                               1 - centers[:, 0], 1 - centers[:, 1]])
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))

    A = [np.eye(n)]
    b = [border]
    rows = []
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = row[j] = 1.0
            rows.append(row)
            vals.append(dists[i, j])
    if rows:
        A.append(np.array(rows))
        b.append(np.array(vals))

    A_ub = np.vstack(A)
    b_ub = np.concatenate(b)
    c = -np.ones(n)
    bounds = [(0, None)] * n
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    return res.x if res.success else np.zeros(n)
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
