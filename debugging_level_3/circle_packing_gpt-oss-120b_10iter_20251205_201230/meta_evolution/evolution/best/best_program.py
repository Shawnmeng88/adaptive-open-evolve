"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Build a hexagonal lattice for 26 circles, then locally optimise the
    centre positions using a simple hill‑climber that solves a linear program
    for the radii at each step.  The LP maximises the sum of radii while
    respecting wall and non‑overlap constraints.
    Returns
    -------
    centers : np.ndarray, shape (k, 2)
        Final circle centres (k may be ≤ 26 if some points are dropped).
    radii   : np.ndarray, shape (k,)
        Optimised radii for the centres.
    sum_radii : float
        Sum of the radii (fitness component).
    """
    import random
    import numpy as np
    from scipy.optimize import linprog

    # ---- 1. Initialise a regular hexagonal lattice -----------------
    n_target = 26
    cols, rows = 5, 6
    sqrt3_over_2 = 0.8660254037844386

    # spacing limited by width and height
    s_max_width = 1.0 / cols
    s_max_height = 1.0 / ((rows - 1) * sqrt3_over_2 + 1.0)
    s = min(s_max_width, s_max_height)
    r0 = s / 2.0                     # initial uniform radius (used only for margin)

    centres = []
    for row in range(rows):
        y = r0 + row * sqrt3_over_2 * s
        x_offset = (s / 2.0) if (row % 2 == 1) else 0.0
        for col in range(cols):
            x = r0 + x_offset + col * s
            if x - r0 < 0 or x + r0 > 1 or y - r0 < 0 or y + r0 > 1:
                continue
            centres.append([x, y])
            if len(centres) == n_target:
                break
        if len(centres) == n_target:
            break

    centres = np.array(centres, dtype=float)

    # ---- 2. Linear‑program to obtain optimal radii for a fixed centre set ----
    def optimal_radii(points):
        """Return LP‑optimal radii for the given points."""
        n = len(points)
        # objective: maximise sum(r)  → minimise -sum(r)
        c = -np.ones(n)

        # wall constraints: r_i ≤ distance to closest wall
        wall_dist = np.minimum.reduce([
            points[:, 0],          # left
            1.0 - points[:, 0],    # right
            points[:, 1],          # bottom
            1.0 - points[:, 1]     # top
        ])

        A = []
        b = []

        for i in range(n):
            a = np.zeros(n)
            a[i] = 1.0
            A.append(a)
            b.append(wall_dist[i])

        # pairwise non‑overlap: r_i + r_j ≤ d_ij
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(points[i] - points[j])
                a = np.zeros(n)
                a[i] = 1.0
                a[j] = 1.0
                A.append(a)
                b.append(d)

        bounds = [(0, None)] * n
        res = linprog(c, A_ub=np.array(A), b_ub=np.array(b),
                      bounds=bounds, method='highs', options={'presolve': True})
        if res.success:
            return res.x
        # fallback – uniform radius limited by walls
        return np.minimum(wall_dist, np.full(n, np.min(wall_dist)))

    # ---- 3. Simple hill‑climber to improve centre placement -------------
    best_centres = centres.copy()
    best_radii = optimal_radii(best_centres)
    best_sum = best_radii.sum()

    max_iters = 250
    step_size = 0.05

    for _ in range(max_iters):
        i = random.randrange(len(best_centres))
        # propose a random move for centre i
        proposal = best_centres[i] + np.random.uniform(-step_size, step_size, size=2)
        proposal = np.clip(proposal, 0.0, 1.0)

        # quick wall‑distance check – centre must stay at least a tiny epsilon
        # away from the boundary to keep the LP feasible.
        eps = 1e-6
        if (proposal[0] < eps or proposal[0] > 1 - eps or
                proposal[1] < eps or proposal[1] > 1 - eps):
            continue

        new_centres = best_centres.copy()
        new_centres[i] = proposal
        new_radii = optimal_radii(new_centres)
        new_sum = new_radii.sum()

        if new_sum > best_sum + 1e-6:          # accept only if improvement
            best_centres = new_centres
            best_radii = new_radii
            best_sum = new_sum

    return best_centres, best_radii, float(best_sum)
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
