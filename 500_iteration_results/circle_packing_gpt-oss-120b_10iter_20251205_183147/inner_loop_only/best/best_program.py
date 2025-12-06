"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Hill‑climb optimisation of circle centres.
    Starts from a symmetric layout and repeatedly perturbs a single
    centre, keeping changes that increase the LP‑maximised sum of radii.
    """
    n = 26
    # symmetric seed layout
    centers = np.zeros((n, 2))
    centers[0] = [0.5, 0.5]
    for i in range(8):
        a = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(a), 0.5 + 0.3 * np.sin(a)]
    for i in range(16):
        a = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.7 * np.cos(a), 0.5 + 0.7 * np.sin(a)]

    def optimal_radii(c):
        """Linear‑programming optimum radii for fixed centres."""
        bd = np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])
        rows, rhs = [], []
        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i] = row[j] = 1
                rows.append(row)
                rhs.append(np.linalg.norm(c[i] - c[j]))
        A = np.array(rows)
        b = np.array(rhs)
        res = linprog(-np.ones(n), A_ub=A, b_ub=b,
                      bounds=[(0, bd[i]) for i in range(n)],
                      method='highs')
        return res.x if res.success else np.zeros(n)

    # initialise best solution
    best_c = centers.copy()
    best_r = optimal_radii(best_c)
    best_sum = best_r.sum()

    # hill‑climbing loop
    for it in range(2500):
        # pick a random circle and jitter its centre
        i = np.random.randint(n)
        step = (np.random.rand(2) - 0.5) * 0.06  # up to ±0.03
        cand = best_c.copy()
        cand[i] = np.clip(cand[i] + step, 0.01, 0.99)

        rad = optimal_radii(cand)
        s = rad.sum()
        if s > best_sum:
            best_sum, best_c, best_r = s, cand, rad

        # occasional global jitter to escape plateaus
        if it % 500 == 0 and it > 0:
            jitter = (np.random.rand(*best_c.shape) - 0.5) * 0.04
            cand = np.clip(best_c + jitter, 0.01, 0.99)
            rad = optimal_radii(cand)
            s = rad.sum()
            if s > best_sum:
                best_sum, best_c, best_r = s, cand, rad

    return best_c, best_r, best_sum


def compute_max_radii(centers):
    """Legacy fallback – returns the LP‑optimised radii."""
    return compute_optimal_radii(centers) if 'compute_optimal_radii' in globals() else np.zeros_like(centers[:, 0])
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
