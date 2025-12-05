# EVOLVE-BLOCK-START
import numpy as np, math, random
from scipy.optimize import linprog

def _optimal_radii(centers: np.ndarray) -> np.ndarray:
    """
    Solve a linear programme that maximises the total sum of radii
    for a fixed set of centres.
    Constraints:
        - r_i <= distance to each wall
        - r_i + r_j <= distance between centres i and j
    Returns the optimal radii (or a fallback heuristic if LP fails).
    """
    n = len(centers)
    # wall limits
    wall = np.minimum.reduce([
        centers[:, 0],
        centers[:, 1],
        1.0 - centers[:, 0],
        1.0 - centers[:, 1]
    ])

    # pairwise distances
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))

    # Objective: maximise sum(r)  → minimise -sum(r)
    c = -np.ones(n)

    rows = []
    bounds = []

    # wall constraints r_i <= wall_i
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        rows.append(row)
        bounds.append(wall[i])

    # non‑overlap constraints r_i + r_j <= d_ij
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = row[j] = 1.0
            rows.append(row)
            bounds.append(dists[i, j])

    A_ub = np.vstack(rows)
    b_ub = np.array(bounds)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, None)] * n,
                  method='highs',
                  options={'presolve': True})
    if res.success:
        return np.maximum(res.x, 0.0)
    # fallback – use the fast heuristic
    wall = np.minimum.reduce([centers[:, 0], centers[:, 1],
                             1.0 - centers[:, 0], 1.0 - centers[:, 1]])
    np.fill_diagonal(dists, np.inf)
    nearest = np.min(dists, axis=1) / 2.0
    return np.minimum(wall, nearest)


def _hex_grid(n_points: int, rng: np.random.Generator, spacing: float = 0.18) -> np.ndarray:
    """
    Generate a hexagonal lattice inside the unit square and return the first n_points.
    """
    dy = spacing * math.sqrt(3) / 2.0
    pts = []
    y = spacing / 2.0
    row = 0
    while y <= 1.0 - spacing / 2.0:
        offset = (spacing / 2.0) if (row % 2) else 0.0
        x = spacing / 2.0 + offset
        while x <= 1.0 - spacing / 2.0:
            pts.append([x, y])
            x += spacing
        row += 1
        y += dy
    pts = np.array(pts)
    rng.shuffle(pts)
    if len(pts) < n_points:
        extra = rng.uniform(0.0, 1.0, size=(n_points - len(pts), 2))
        pts = np.vstack([pts, extra])
    return pts[:n_points].astype(float)


def _anneal(start_centers: np.ndarray,
            rng: np.random.Generator,
            max_iter: int = 20000,
            step_max: float = 0.03,
            step_min: float = 0.004,
            T0: float = 0.004) -> tuple[np.ndarray, float]:
    """
    Simulated‑annealing hill‑climb that searches for a good centre layout.
    Returns the best centres found and the corresponding sum of radii (computed via LP).
    """
    n = len(start_centers)
    best_c = start_centers.copy()
    best_r = _optimal_radii(best_c)
    best_sum = float(best_r.sum())

    cur_c = best_c.copy()
    cur_sum = best_sum
    eps = 1e-12

    for it in range(max_iter):
        step = step_max - (step_max - step_min) * (it / max_iter)
        T = T0 * (1.0 - it / max_iter)

        i = rng.integers(n)
        proposal = cur_c[i] + rng.uniform(-step, step, size=2)
        proposal = np.clip(proposal, 0.0, 1.0)

        trial_c = cur_c.copy()
        trial_c[i] = proposal
        trial_r = _optimal_radii(trial_c)
        trial_sum = float(trial_r.sum())

        delta = trial_sum - cur_sum
        accept = False
        if delta > eps:
            accept = True
        elif T > 0:
            prob = math.exp(delta / T) if delta < 0 else 1.0
            accept = rng.random() < prob

        if accept:
            cur_c = trial_c
            cur_sum = trial_sum
            if trial_sum > best_sum + eps:
                best_c, best_sum = trial_c, trial_sum

    return best_c, best_sum


def _refine_lp(centers: np.ndarray,
               rng: np.random.Generator,
               max_iter: int = 500,
               step_max: float = 0.01,
               step_min: float = 0.001) -> tuple[np.ndarray, float]:
    """
    Deterministic hill‑climb (no temperature) that performs a short
    local search using the exact LP radii after each move.
    """
    n = len(centers)
    cur_c = centers.copy()
    cur_r = _optimal_radii(cur_c)
    cur_sum = float(cur_r.sum())
    eps = 1e-12

    for it in range(max_iter):
        step = step_max - (step_max - step_min) * (it / max_iter)
        i = rng.integers(n)
        proposal = cur_c[i] + rng.uniform(-step, step, size=2)
        proposal = np.clip(proposal, 0.0, 1.0)

        trial_c = cur_c.copy()
        trial_c[i] = proposal
        trial_r = _optimal_radii(trial_c)
        trial_sum = float(trial_r.sum())

        if trial_sum > cur_sum + eps:
            cur_c, cur_sum = trial_c, trial_sum

    return cur_c, cur_sum


def construct_packing():
    """
    Build a packing of 26 circles in the unit square.
    Uses simulated annealing on centre positions with exact LP radii,
    followed by a brief deterministic refinement.
    Returns:
        centres (np.ndarray shape (26,2)),
        radii   (np.ndarray shape (26,)),
        sum_radii (float)
    """
    # deterministic RNGs for reproducibility
    rng_hex = np.random.default_rng(0)
    rng_rand = np.random.default_rng(1)
    rng_jit = np.random.default_rng(2)

    seeds = [
        _hex_grid(26, rng_hex, spacing=0.18),
        rng_rand.uniform(0.0, 1.0, size=(26, 2)),
        _hex_grid(26, rng_jit, spacing=0.18) + rng_jit.normal(scale=0.02, size=(26, 2)),
    ]

    # keep all points inside the unit square
    seeds = [np.clip(s, 0.0, 1.0) for s in seeds]

    best_sum = -1.0
    best_c = None

    # coarse optimisation for each seed
    for idx, seed in enumerate(seeds):
        rng = np.random.default_rng(10 + idx)
        c, s = _anneal(seed, rng, max_iter=20000)
        if s > best_sum:
            best_sum, best_c = s, c

    # final fine‑tuning on the current best
    rng_final = np.random.default_rng(99)
    best_c, best_sum = _anneal(best_c, rng_final, max_iter=30000,
                              step_max=0.025, step_min=0.003, T0=0.003)

    # short deterministic LP‑based refinement
    best_c, best_sum = _refine_lp(best_c, rng_final, max_iter=500,
                                 step_max=0.008, step_min=0.0005)

    # final radii from LP
    best_r = _optimal_radii(best_c)

    return best_c, best_r, float(best_r.sum())
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
