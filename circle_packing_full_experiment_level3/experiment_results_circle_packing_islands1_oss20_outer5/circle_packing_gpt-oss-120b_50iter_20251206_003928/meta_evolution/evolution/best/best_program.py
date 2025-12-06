"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def hex_grid(d, ox=0.0, oy=0.0):
    dx, dy = d, d * np.sqrt(3.0) / 2.0
    pts, row, y = [], 0, oy
    while y <= 1.0:
        off = (dx / 2.0) if (row % 2) else 0.0
        x = ox + off
        while x <= 1.0:
            pts.append([x, y])
            x += dx
        row += 1
        y = oy + row * dy
    return np.array(pts)


def farthest_points(pts, k):
    pts = np.asarray(pts)
    n = pts.shape[0]
    if k >= n:
        return np.arange(n)
    wall = np.minimum.reduce([pts[:, 0], pts[:, 1], 1 - pts[:, 0], 1 - pts[:, 1]])
    first = np.argmax(wall)
    sel = [first]
    dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    min_d = dmat[first].copy()
    for _ in range(1, k):
        min_d[sel] = -1.0
        nxt = np.argmax(min_d)
        sel.append(nxt)
        min_d = np.minimum(min_d, dmat[nxt])
    return np.array(sel, dtype=int)


def optimal_radii(centers):
    n = centers.shape[0]
    c = -np.ones(n)
    A, b = [], []
    for i, (x, y) in enumerate(centers):
        row = np.zeros(n); row[i] = 1.0
        A.append(row); b.append(min(x, y, 1 - x, 1 - y))
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n); row[i] = row[j] = 1.0
            A.append(row); b.append(np.linalg.norm(centers[i] - centers[j]))
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b),
                  bounds=[(0.0, None)] * n, method="highs")
    return res.x if res.success else np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )


def _eval(centers):
    r = optimal_radii(centers)
    return float(r.sum()), r


def construct_packing():
    n, rng = 26, np.random.default_rng(42)
    best_sum, best_c, best_r = -1.0, None, None

    # hex grids with offsets
    for d in np.linspace(0.07, 0.30, 25):
        for ox_f in (0.0, 0.5):
            for oy_f in (0.0, 0.5):
                pts = hex_grid(d, ox_f * d, oy_f * d * np.sqrt(3) / 2)
                if pts.shape[0] < n:
                    continue
                idx = farthest_points(pts, n)
                centers = pts[idx]
                s, r = _eval(centers)
                if s > best_sum:
                    best_sum, best_c, best_r = s, centers.copy(), r.copy()

    # square 5×5 + extra point
    gv = (np.arange(5) + 0.5) / 5.0
    xv, yv = np.meshgrid(gv, gv)
    grid = np.column_stack((xv.ravel(), yv.ravel()))
    extra = np.array([[0.5, 0.2]])
    centers = np.vstack([grid, extra])
    s, r = _eval(centers)
    if s > best_sum:
        best_sum, best_c, best_r = s, centers.copy(), r.copy()

    # random seeds
    for _ in range(12):
        centers = rng.uniform(0.0, 1.0, size=(n, 2))
        s, r = _eval(centers)
        if s > best_sum:
            best_sum, best_c, best_r = s, centers.copy(), r.copy()

    # local refinement
    cur_c, cur_s = best_c.copy(), best_sum
    sigma = 0.03
    no_imp = 0
    for _ in range(3000):
        i = rng.integers(n)
        if rng.random() < 0.02:
            prop = rng.uniform(0.0, 1.0, size=2)
        else:
            prop = cur_c[i] + rng.normal(scale=sigma, size=2)
        prop = np.clip(prop, 0.0, 1.0)
        new_c = cur_c.copy()
        new_c[i] = prop
        s, r = _eval(new_c)
        if s > cur_s:
            cur_c, cur_s = new_c, s
            if s > best_sum:
                best_sum, best_c, best_r = s, new_c.copy(), r.copy()
            sigma = min(sigma * 1.07, 0.07)
            no_imp = 0
        else:
            no_imp += 1
            sigma *= 0.995
        if no_imp > 300:
            sigma = min(sigma * 1.5, 0.08)
            no_imp = 0
        sigma = max(sigma, 4e-4)

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
