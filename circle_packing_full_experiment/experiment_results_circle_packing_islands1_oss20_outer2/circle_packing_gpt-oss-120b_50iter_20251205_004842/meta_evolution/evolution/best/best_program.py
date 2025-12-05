# EVOLVE-BLOCK-START
import numpy as np

def _shrink_and_tighten(centers: np.ndarray, max_iter: int = 200, tol: float = 1e-7) -> np.ndarray:
    """
    Compute a feasible set of radii for the given centres.
    1. Start from the wall‑limited radii.
    2. Repeatedly shrink the larger circle of every violating pair
       just enough to satisfy the non‑overlap constraint.
    3. After no violations remain, tighten each radius by taking the
       maximum allowed by the walls and the current neighbours.
    The process converges quickly for ≤30 circles.
    """
    n = centers.shape[0]

    # 1️⃣ wall‑limited start
    radii = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )

    # pairwise distances (inf on diagonal)
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)

    # ---------- shrink phase ----------
    for _ in range(max_iter):
        # violation matrix
        viol = radii[:, None] + radii[None, :] - dists
        i_idx, j_idx = np.triu_indices(n, k=1)
        mask = viol[i_idx, j_idx] > 0
        if not np.any(mask):
            break

        # resolve each violating pair
        for i, j in zip(i_idx[mask], j_idx[mask]):
            d = dists[i, j]
            if radii[i] >= radii[j]:
                # shrink i just enough
                radii[i] = max(0.0, d - radii[j])
            else:
                radii[j] = max(0.0, d - radii[i])

        # re‑apply wall limits (might have been violated by reductions)
        wall = np.minimum.reduce(
            [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
        )
        radii = np.minimum(radii, wall)

    # ---------- tighten phase ----------
    # now radii are feasible; try to enlarge each circle as much as possible
    for _ in range(10):
        improved = False
        # compute current neighbour limits
        neigh_limit = dists - radii[None, :]          # (n, n) max radius for i given j
        max_allowed = np.minimum.reduce(
            [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
        )
        max_allowed = np.minimum(max_allowed, neigh_limit.min(axis=1))
        # only increase if we can
        inc = max_allowed - radii
        if np.max(inc) > tol:
            radii += inc * 0.5          # damped increase to stay safe
            improved = True
        if not improved:
            break

    return radii


def _hex_grid_centers(rows: int = 6, cols: int = 5, target_n: int = 26):
    """
    Deterministic hexagonal lattice that fits inside the unit square.
    Returns the first ``target_n`` points.
    """
    r_horiz = 1.0 / (2 * cols)
    r_vert = 1.0 / (2 + (rows - 1) * np.sqrt(3))
    radius = min(r_horiz, r_vert)

    dx = 2 * radius
    dy = np.sqrt(3) * radius

    pts = []
    for i in range(rows):
        y = radius + i * dy
        x_off = radius if i % 2 == 0 else radius + radius
        for j in range(cols):
            x = x_off + j * dx
            if x <= 1 - radius and y <= 1 - radius:
                pts.append([x, y])
            if len(pts) >= target_n:
                break
        if len(pts) >= target_n:
            break
    return np.array(pts[:target_n]), radius


def construct_packing(num_circles: int = 26, seed: int | None = None):
    """
    Stochastic search for a high‑sum feasible packing.
    1️⃣ Start from a deterministic hex lattice (good baseline).
    2️⃣ Perform many random‑restart trials.
    3️⃣ Refine the best candidate with a simple local‑move hill‑climber.
    Returns (centers, radii, sum_of_radii).
    """
    rng = np.random.default_rng(seed if seed is not None else 12345)

    # ---- baseline from hex grid ------------------------------------------------
    best_centers, _ = _hex_grid_centers()
    best_radii = _shrink_and_tighten(best_centers)
    best_sum = best_radii.sum()

    # ---- random‑restart phase -------------------------------------------------
    margin = 0.02  # safety margin from the walls
    for _ in range(2500):
        centres = rng.random((num_circles, 2)) * (1 - 2 * margin) + margin
        radii = _shrink_and_tighten(centres)
        cur_sum = radii.sum()
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_centers = centres
            best_radii = radii

    # ---- local hill‑climbing refinement ----------------------------------------
    for _ in range(800):
        i = rng.integers(num_circles)
        delta = (rng.random(2) - 0.5) * 0.04  # ±0.02
        new_center = best_centers[i] + delta
        if np.any(new_center < margin) or np.any(new_center > 1 - margin):
            continue

        trial_centers = best_centers.copy()
        trial_centers[i] = new_center
        trial_radii = _shrink_and_tighten(trial_centers)
        trial_sum = trial_radii.sum()

        if trial_sum > best_sum + 1e-7:
            best_sum = trial_sum
            best_centers = trial_centers
            best_radii = trial_radii

    return best_centers, best_radii, float(best_sum)
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
