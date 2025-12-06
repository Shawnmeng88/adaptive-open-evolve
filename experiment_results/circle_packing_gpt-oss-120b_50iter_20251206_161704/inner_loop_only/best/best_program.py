"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def compute_max_radii(centers):
    """
    Solve a linear programme that maximises the sum of radii subject to:
      * each radius ≤ distance to the square edges,
      * ri + rj ≤ distance between centres i and j.
    Returns the optimal radii as a NumPy array, or a greedy fallback on failure.
    """
    n = centers.shape[0]

    # distance to nearest border for each centre
    border = np.minimum.reduce([centers[:, 0],
                               centers[:, 1],
                               1 - centers[:, 0],
                               1 - centers[:, 1]])

    # pairwise distance constraints
    idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(idx)
    A = np.zeros((m, n))
    b = np.empty(m)
    for k, (i, j) in enumerate(idx):
        A[k, i] = 1.0
        A[k, j] = 1.0
        b[k] = np.linalg.norm(centers[i] - centers[j])

    # maximise sum(r) → minimise -sum(r)
    c = -np.ones(n)
    bounds = [(0.0, border[i]) for i in range(n)]

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds,
                  method='highs', options={'presolve': True})

    if res.success:
        return np.maximum(res.x, 0.0)

    # fallback: greedy scaling
    radii = np.copy(border)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > d:
                scale = d / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale
    return radii


def _local_refine(start_centers, start_sum, rng, steps, radius):
    """Hill‑climbing refinement."""
    n = start_centers.shape[0]
    best_c, best_s = start_centers.copy(), start_sum

    for _ in range(steps):
        i = rng.integers(n)
        cand = best_c.copy()
        cand[i] = np.clip(cand[i] + rng.uniform(-radius, radius, 2), 0.01, 0.99)
        rad = compute_max_radii(cand)
        s = rad.sum()
        if s > best_s:
            best_c, best_s = cand, s
    return best_c, best_s


def _hex_pattern(n, scale=0.7):
    """
    Generate a hexagonal lattice inside the unit square.
    The pattern is centred and then clipped to stay within [0.01, 0.99].
    """
    # approximate rows/cols to hold n points
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    dx = scale / (cols - 1)
    dy = scale * np.sqrt(3) / 2 / (rows - 1)

    points = []
    for r in range(rows):
        y = 0.5 - scale / 2 + r * dy
        offset = 0.5 * dx if r % 2 else 0.0
        for c in range(cols):
            x = 0.5 - scale / 2 + c * dx + offset
            points.append([x, y])
            if len(points) == n:
                break
        if len(points) == n:
            break
    pts = np.array(points)
    return np.clip(pts, 0.01, 0.99)


def construct_packing():
    """
    Explore several deterministic and stochastic centre configurations,
    then apply a multi‑stage refinement to maximise the total sum of radii.
    """
    n = 26
    rng = np.random.default_rng()
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # ------------------------------------------------------------------
    # Candidate generators
    # ------------------------------------------------------------------
    def base_pattern():
        c = np.zeros((n, 2))
        c[0] = [0.5, 0.5]
        for i in range(8):
            a = 2 * np.pi * i / 8
            c[i + 1] = [0.5 + 0.30 * np.cos(a), 0.5 + 0.30 * np.sin(a)]
        for i in range(16):
            a = 2 * np.pi * i / 16
            c[i + 9] = [0.5 + 0.70 * np.cos(a), 0.5 + 0.70 * np.sin(a)]
        return np.clip(c, 0.01, 0.99)

    def random_pattern():
        return rng.uniform(0.05, 0.95, size=(n, 2))

    # ------------------------------------------------------------------
    # 1) Jittered deterministic patterns
    # ------------------------------------------------------------------
    for _ in range(120):          # more than previous 100
        centers = base_pattern()
        jitter = rng.uniform(-0.035, 0.035, size=centers.shape)
        centers = np.clip(centers + jitter, 0.01, 0.99)
        rad = compute_max_radii(centers)
        s = rad.sum()
        if s > best_sum:
            best_sum, best_centers, best_radii = s, centers, rad

    for _ in range(80):           # hexagonal base + jitter
        centers = _hex_pattern(n, scale=0.78)
        jitter = rng.uniform(-0.03, 0.03, size=centers.shape)
        centers = np.clip(centers + jitter, 0.01, 0.99)
        rad = compute_max_radii(centers)
        s = rad.sum()
        if s > best_sum:
            best_sum, best_centers, best_radii = s, centers, rad

    # ------------------------------------------------------------------
    # 2) Pure random patterns – larger sample
    # ------------------------------------------------------------------
    for _ in range(800):
        centers = random_pattern()
        rad = compute_max_radii(centers)
        s = rad.sum()
        if s > best_sum:
            best_sum, best_centers, best_radii = s, centers, rad

    # ------------------------------------------------------------------
    # 3) Multi‑stage local refinement
    # ------------------------------------------------------------------
    if best_centers is not None:
        # coarse
        best_centers, best_sum = _local_refine(best_centers, best_sum, rng,
                                              steps=1000, radius=0.025)
        # medium
        best_centers, best_sum = _local_refine(best_centers, best_sum, rng,
                                              steps=800, radius=0.015)
        # fine
        best_centers, best_sum = _local_refine(best_centers, best_sum, rng,
                                              steps=600, radius=0.008)
        # ultra‑fine (very cheap)
        best_centers, best_sum = _local_refine(best_centers, best_sum, rng,
                                              steps=400, radius=0.004)

        best_radii = compute_max_radii(best_centers)

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------
    if best_centers is None:
        best_centers = base_pattern()
        best_radii = compute_max_radii(best_centers)
        best_sum = best_radii.sum()

    return best_centers, best_radii, best_sum
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
