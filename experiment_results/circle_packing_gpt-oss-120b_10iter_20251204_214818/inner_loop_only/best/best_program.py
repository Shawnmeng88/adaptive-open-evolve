# EVOLVE-BLOCK-START
"""
Enhanced circle‑packing for n = 26.

The original version used a fixed 5×5 grid (plus a centre) and solved a
linear programme (LP) for the radii.  Here we keep the same LP
formulation (so the solution is always valid) but we *optimise the
centres* with a very cheap stochastic hill‑climber.  Each centre is
perturbed a little, the LP is resolved, and the change is kept only if
the total sum of radii increases.  The algorithm is deterministic (fixed
random seed) and runs in a few hundred LP solves – well within the time
budget of the evaluation harness.

The public interface (`run_packing`) and the visualisation helper are
unchanged.
"""
import numpy as np
from scipy.optimize import linprog

# ----------------------------------------------------------------------
# 1️⃣  Fixed layout – a sensible deterministic start point
# ----------------------------------------------------------------------
def _grid_centers():
    """Return 25 points on a 5×5 grid plus a centre point (26 total)."""
    pts = np.linspace(0.12, 0.88, 5)          # keep a margin for the radii
    grid = np.array(np.meshgrid(pts, pts)).T.reshape(-1, 2)
    centre = np.array([[0.5, 0.5]])           # extra circle
    return np.vstack([grid, centre])


# ----------------------------------------------------------------------
# 2️⃣  LP that, for a *given* set of centres, yields the maximal radii
# ----------------------------------------------------------------------
def _max_radii_lp(centers: np.ndarray) -> np.ndarray:
    """
    Solve the linear programme

        maximise   Σ r_i
        subject to r_i ≤ wall_i,
                   r_i + r_j ≤ d_ij,
                   r_i ≥ 0

    where ``wall_i`` is the distance from centre i to the closest side of
    the unit square and ``d_ij`` the Euclidean distance between centres i
    and j.
    """
    n = len(centers)

    # distance to the four walls (vectorised)
    wall = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )

    # pairwise centre distances
    dists = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))

    # LP data
    c = -np.ones(n)                               # maximise sum → minimise -sum
    bounds = [(0.0, w) for w in wall]             # 0 ≤ r_i ≤ wall_i

    # constraints r_i + r_j ≤ d_ij
    A, b = [], []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A.append(row)
            b.append(dists[i, j])

    if A:                                         # normal case (n > 1)
        A = np.array(A)
        b = np.array(b)
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    else:                                         # degenerate case n == 1
        res = linprog(c, bounds=bounds, method="highs")

    # If the solver fails for any reason, fall back to the wall limits.
    if not res.success:
        return wall
    return res.x


# ----------------------------------------------------------------------
# 3️⃣  Very cheap stochastic hill‑climber on the centre positions
# ----------------------------------------------------------------------
def _optimise_centres(
    centres: np.ndarray,
    *,
    iters: int = 1500,
    step: float = 0.025,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Starting from ``centres`` perform a simple hill‑climb:

    * pick a random circle,
    * move its centre by a small Gaussian perturbation,
    * recompute the optimal radii with the LP,
    * keep the move if the total sum of radii increases.

    The routine returns the best (centres, radii, sum_of_radii) found.
    """
    rng = np.random.default_rng(seed)

    # initialise
    best_c = centres.copy()
    best_r = _max_radii_lp(best_c)
    best_sum = float(best_r.sum())
    n = len(best_c)

    for _ in range(iters):
        i = rng.integers(n)                     # which circle to move
        proposal = best_c[i] + rng.normal(scale=step, size=2)
        # keep inside the admissible box (a tiny margin avoids numerical trouble)
        proposal = np.clip(proposal, 0.01, 0.99)

        cand_c = best_c.copy()
        cand_c[i] = proposal

        cand_r = _max_radii_lp(cand_c)
        cand_sum = float(cand_r.sum())

        if cand_sum > best_sum:                 # accept only improvements
            best_c, best_r, best_sum = cand_c, cand_r, cand_sum

    return best_c, best_r, best_sum


# ----------------------------------------------------------------------
# 4️⃣  Public constructor – deterministic entry point
# ----------------------------------------------------------------------
def construct_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Build a deterministic arrangement of 26 circles, optimise the centre
    positions with a tiny stochastic search and return
    (centres, radii, sum_of_radii).
    """
    start = _grid_centers()
    start = np.clip(start, 0.01, 0.99)            # safety margin
    centres, radii, total = _optimise_centres(
        start,
        iters=1500,          # a few hundred LP solves – fast enough
        step=0.025,
        seed=42,             # fixed seed → reproducible result
    )
    return centres, radii, total


# EVOLVE-BLOCK-END


# ----------------------------------------------------------------------
# Fixed helper / visualisation (unchanged by the evolution engine)
# ----------------------------------------------------------------------
def run_packing():
    """Run the circle‑packing constructor for n = 26."""
    return construct_packing()


def visualize(centers: np.ndarray, radii: np.ndarray):
    """Draw the circles inside the unit square."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(c, r, alpha=0.5, edgecolor="k"))
        ax.text(*c, str(i), ha="center", va="center", fontsize=8)

    plt.title(f"Circle packing (n={len(centers)}, sum={radii.sum():.4f})")
    plt.show()


if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s:.6f}")
    # visualize(c, r)   # uncomment to see the layout