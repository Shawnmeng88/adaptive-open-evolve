"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def compute_max_radii_lp(centers):
    """Maximize sum of radii for fixed centers via linear programming."""
    import numpy as np
    from scipy.optimize import linprog

    n = centers.shape[0]
    c = -np.ones(n)                     # maximize sum(r) → minimize -sum(r)

    rows, b = [], []
    for i, (x, y) in enumerate(centers):
        for d in (x, y, 1 - x, 1 - y):
            row = np.zeros(n); row[i] = 1
            rows.append(row); b.append(d)

    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n); row[i] = row[j] = 1
            rows.append(row); b.append(d)

    A, B = np.vstack(rows), np.array(b)
    res = linprog(c, A_ub=A, b_ub=B, bounds=[(0, None)] * n, method="highs")
    if res.success:
        return res.x

    # fallback scaling
    rad = np.minimum.reduce([centers[:,0], 1-centers[:,0],
                             centers[:,1], 1-centers[:,1]])
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i]-centers[j])
            if rad[i]+rad[j] > d:
                s = d/(rad[i]+rad[j]); rad[i]*=s; rad[j]*=s
    return rad


def _initial_grid(n):
    """Uniform grid inside [0.1,0.9]²."""
    import numpy as np
    m = int(np.ceil(np.sqrt(n)))
    xs = np.linspace(0.1, 0.9, m)
    ys = np.linspace(0.1, 0.9, m)
    pts = [(x, y) for y in ys for x in xs]
    return np.array(pts[:n], float)


def construct_packing():
    """Stochastic search over centers with LP radius optimisation."""
    import numpy as np
    n, rng = 26, np.random.default_rng(42)

    cur = _initial_grid(n)
    cur_r = compute_max_radii_lp(cur)
    cur_sum = cur_r.sum()
    best_c, best_r, best_sum = cur.copy(), cur_r.copy(), cur_sum

    step, patience = 0.05, 0
    for it in range(2500):
        i = rng.integers(n)
        prop = cur.copy()
        prop[i] = np.clip(prop[i] + rng.normal(scale=step, size=2), 0, 1)

        rad = compute_max_radii_lp(prop)
        s = rad.sum()

        # accept better or occasional worse move
        if s > cur_sum or rng.random() < 0.005:
            cur, cur_sum = prop, s
            patience = 0
        else:
            patience += 1

        if s > best_sum:
            best_c, best_r, best_sum = prop.copy(), rad.copy(), s
            patience = 0

        # adapt step size
        if it % 200 == 0 and step > 0.005:
            step *= 0.9
        # random restart if stuck
        if patience > 300:
            cur = _initial_grid(n)
            cur_sum = compute_max_radii_lp(cur).sum()
            patience = 0

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
