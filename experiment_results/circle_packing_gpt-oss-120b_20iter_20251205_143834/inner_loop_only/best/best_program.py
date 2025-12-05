"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    n = 26
    base = np.zeros((n, 2))
    base[0] = [0.5, 0.5]
    for i in range(8):
        a = 2 * np.pi * i / 8
        base[i + 1] = [0.5 + 0.3 * np.cos(a), 0.5 + 0.3 * np.sin(a)]
    for i in range(16):
        a = 2 * np.pi * i / 16
        base[i + 9] = [0.5 + 0.7 * np.cos(a), 0.5 + 0.7 * np.sin(a)]
    base = np.clip(base, 0.01, 0.99)

    rng = np.random.default_rng()
    best_sum = -1.0
    best_c = best_r = None

    # broad random search
    for _ in range(200):
        pert = base + rng.normal(scale=0.05, size=base.shape)
        pert = np.clip(pert, 0.01, 0.99)
        rad = _optimal_radii_lp(pert)
        s = rad.sum()
        if s > best_sum:
            best_sum, best_c, best_r = s, pert, rad

    # simple hill‑climb refinement
    delta = 0.01
    for _ in range(5):
        improved = False
        for i in range(n):
            for dx, dy in ((delta, 0), (-delta, 0), (0, delta), (0, -delta)):
                cand = best_c.copy()
                cand[i] += [dx, dy]
                cand[i] = np.clip(cand[i], 0.01, 0.99)
                rad = _optimal_radii_lp(cand)
                s = rad.sum()
                if s > best_sum:
                    best_sum, best_c, best_r = s, cand, rad
                    improved = True
        if not improved:
            break

    return best_c, best_r, best_sum


def _optimal_radii_lp(centers):
    n = len(centers)
    border = np.minimum.reduce([centers[:, 0], centers[:, 1],
                               1 - centers[:, 0], 1 - centers[:, 1]])

    rows, caps = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = row[j] = 1
            rows.append(row)
            caps.append(d)
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1
        rows.append(row)
        caps.append(border[i])

    A = np.vstack(rows)
    b = np.array(caps)

    res = linprog(c=-np.ones(n), A_ub=A, b_ub=b,
                  bounds=(0, None), method='highs')
    if res.success:
        return res.x

    # fallback: scale down overlapping radii
    radii = border.copy()
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > d:
                scale = d / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale
    return radii
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
