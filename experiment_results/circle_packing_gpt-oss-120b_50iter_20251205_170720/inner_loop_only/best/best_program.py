"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
import numpy as np, math
from scipy.optimize import linprog

def _solve_lp(centers):
    n = len(centers)
    c = -np.ones(n)
    A, b = [], []
    for i, (x, y) in enumerate(centers):
        row = np.zeros(n)
        row[i] = 1
        A.append(row)
        b.append(min(x, y, 1 - x, 1 - y))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = row[j] = 1
            A.append(row)
            b.append(d)
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b),
                  bounds=[(0, None)] * n, method="highs")
    if res.success:
        return res.x, res.x.sum(), True
    return None, 0.0, False

def _fallback_packing():
    # Very simple fallback: 26 points on a coarse grid
    n = 26
    xs = np.linspace(0.1, 0.9, int(np.sqrt(n)))
    ys = np.linspace(0.1, 0.9, int(np.sqrt(n)))
    xv, yv = np.meshgrid(xs, ys)
    centers = np.column_stack((xv.ravel(), yv.ravel()))[:n]
    radii = np.full(n, 0.05)
    return centers, radii, radii.sum()

def construct_packing():
    target = 26
    best = None
    for r in np.arange(0.20, 0.005, -0.001):
        centers = []
        y = r
        row = 0
        while y + r <= 1.0:
            offset = r if row % 2 else 0.0
            x = r + offset
            while x + r <= 1.0:
                centers.append((x, y))
                if len(centers) >= target:
                    break
                x += 2 * r
            if len(centers) >= target:
                break
            row += 1
            y += math.sqrt(3) * r
        if len(centers) >= target:
            best = (r, np.array(centers[:target]))
            break

    if best is None:
        return _fallback_packing()

    _, centers = best
    n = target
    radii, total, ok = _solve_lp(centers)
    if not ok:
        r = best[0]
        radii = np.full(n, r)
        return centers, radii, radii.sum()

    rng = np.random.default_rng()
    step = 0.03
    for _ in range(50):
        i = rng.integers(n)
        new_center = np.clip(centers[i] + rng.uniform(-step, step, 2), 0.0, 1.0)
        if not (radii[i] <= new_center[0] <= 1 - radii[i] and
                radii[i] <= new_center[1] <= 1 - radii[i]):
            continue
        trial_centers = centers.copy()
        trial_centers[i] = new_center
        tr, ts, ok = _solve_lp(trial_centers)
        if ok and ts > total + 1e-6:
            centers, radii, total = trial_centers, tr, ts

    return centers, radii, total
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
