# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog


def _compute_max_radii(centres: np.ndarray) -> np.ndarray:
    """
    Solve the LP that maximises the sum of radii under wall and
    non‑overlap constraints, then aggressively inflate the radii
    using up to 90 % of the remaining slack.  This mirrors the
    best‑performing implementation from earlier generations.
    """
    n = centres.shape[0]

    # distance to the four walls
    walls = np.minimum.reduce(
        [centres[:, 0], centres[:, 1], 1.0 - centres[:, 0], 1.0 - centres[:, 1]]
    )

    # pairwise centre distances
    diff = centres[:, None, :] - centres[None, :, :]
    dists = np.linalg.norm(diff, axis=2)

    # ----- linear programme (maximise sum of radii) -----
    c = -np.ones(n)                     # maximise -> minimise -sum
    A_wall = np.eye(n)
    b_wall = walls

    pair_i, pair_j = np.triu_indices(n, k=1)
    m = pair_i.size
    A_pair = np.zeros((m, n))
    A_pair[np.arange(m), pair_i] = 1.0
    A_pair[np.arange(m), pair_j] = 1.0
    b_pair = dists[pair_i, pair_j] - 1e-12  # tiny safety margin

    A_ub = np.vstack([A_wall, A_pair])
    b_ub = np.hstack([b_wall, b_pair])
    bounds = [(0.0, None)] * n

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.success:
        radii = np.maximum(res.x, 0.0)
    else:
        # geometric fallback
        np.fill_diagonal(dists, np.inf)
        radii = np.minimum(walls, np.min(dists, axis=1) * 0.5)

    # ----- aggressive inflation (up to 90 % of remaining slack) -----
    wall_slack = walls - radii

    # neighbour slack: for each i, minimum distance to any other centre
    # after subtracting that other's radius
    neigh_slack = np.full(n, np.inf)
    for i in range(n):
        slack_i = dists[i] - radii  # d_ij - r_j
        slack_i[i] = np.inf
        neigh_slack[i] = np.min(slack_i) - radii[i]

    total_slack = np.minimum(wall_slack, neigh_slack)
    increase = np.clip(0.9 * total_slack, 0.0, None)
    radii += increase
    radii = np.minimum(radii, walls - 1e-12)   # final safety margin
    return radii


def _hex_lattice(spacing: float):
    """
    Generator yielding points of a hexagonal lattice that respect a
    margin of ``spacing/2`` from the square boundary.
    """
    dy = spacing * np.sqrt(3) / 2.0
    y = spacing / 2.0
    row = 0
    while y < 1.0 - spacing / 2.0:
        offset = spacing / 2.0 if (row % 2) else 0.0
        x = spacing / 2.0 + offset
        while x < 1.0 - spacing / 2.0:
            yield np.array([x, y])
            x += spacing
        y += dy
        row += 1


def _farthest_point_sampling(n: int, grid_res: int = 30):
    """
    Deterministic farthest‑point sampling on a uniform grid.
    Provides a well‑spread initial layout.
    """
    xs = np.linspace(0.01, 0.99, grid_res)
    xv, yv = np.meshgrid(xs, xs)
    candidates = np.column_stack([xv.ravel(), yv.ravel()])

    selected = [np.array([0.5, 0.5])]
    for _ in range(n - 1):
        sel_arr = np.stack(selected)
        dists = np.linalg.norm(candidates[:, None, :] - sel_arr[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        idx = np.argmax(min_dists)
        selected.append(candidates[idx])
        candidates = np.delete(candidates, idx, axis=0)
    return np.stack(selected)


def construct_packing():
    """
    Construct a packing of 26 circles inside the unit square.
    Two independent initial layouts (hexagonal lattice and farthest‑point
    sampling) are optimised with a two‑phase stochastic hill‑climbing
    search that repeatedly re‑optimises radii via the LP‑based
    ``_compute_max_radii`` routine.
    Returns (centres, radii, sum_radii).
    """
    np.random.seed(0)
    n = 26

    # ---- generate two candidate initial centre sets ----
    # 1) Hexagonal lattice with just‑sufficient spacing
    spacing = 0.18
    while True:
        pts = np.array(list(_hex_lattice(spacing)))
        if pts.shape[0] >= n:
            break
        spacing += 0.01
    hex_initial = np.clip(pts[:n], 0.01, 0.99)

    # 2) Deterministic farthest‑point sampling
    fpps_initial = _farthest_point_sampling(n, grid_res=30)

    best_overall = None  # (centres, radii, sum)

    for init_centres in (hex_initial, fpps_initial):
        centres = init_centres.copy()
        radii = _compute_max_radii(centres)
        best_sum = radii.sum()
        best_centres, best_radii = centres.copy(), radii.copy()

        # ----- coarse stage: quadratic decay, many iterations -----
        coarse_iters = 5000
        initial_step = 0.07
        for it in range(coarse_iters):
            idx = np.random.randint(n)
            # quadratic decay for larger moves early
            step = initial_step * (1.0 - (it / coarse_iters) ** 2)
            proposal = best_centres[idx] + (np.random.rand(2) - 0.5) * step
            proposal = np.clip(proposal, 0.01, 0.99)

            trial_centres = best_centres.copy()
            trial_centres[idx] = proposal
            trial_radii = _compute_max_radii(trial_centres)
            trial_sum = trial_radii.sum()

            if trial_sum > best_sum + 1e-12:
                best_sum = trial_sum
                best_centres, best_radii = trial_centres, trial_radii

        # ----- fine stage: linear decay -----
        fine_iters = 3000
        fine_step = 0.015
        for it in range(fine_iters):
            idx = np.random.randint(n)
            step = fine_step * (1.0 - it / fine_iters)
            proposal = best_centres[idx] + (np.random.rand(2) - 0.5) * step
            proposal = np.clip(proposal, 0.01, 0.99)

            trial_centres = best_centres.copy()
            trial_centres[idx] = proposal
            trial_radii = _compute_max_radii(trial_centres)
            trial_sum = trial_radii.sum()

            if trial_sum > best_sum + 1e-12:
                best_sum = trial_sum
                best_centres, best_radii = trial_centres, trial_radii

        # final recomputation to capture any remaining slack
        best_radii = _compute_max_radii(best_centres)
        best_sum = best_radii.sum()

        if best_overall is None or best_sum > best_overall[2]:
            best_overall = (best_centres, best_radii, best_sum)

    return best_overall[0], best_overall[1], best_overall[2]
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
