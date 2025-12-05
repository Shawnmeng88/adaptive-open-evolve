# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import math

def _solve_lp(centers: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve the linear program for a given set of centers.

    Returns:
        radii (np.ndarray): optimal radii for the centers
        total (float): sum of radii
    """
    n = centers.shape[0]

    # maximum radius limited by the square borders
    ub = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )

    # pairwise centre distances
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))

    # Build inequality matrix A_ub * r <= b_ub
    # 1) border constraints: r_i <= ub_i
    A = [np.eye(n)]
    b = [ub]

    # 2) non‑overlap constraints: r_i + r_j <= d_ij   (i < j)
    rows = np.triu(np.ones((n, n), dtype=bool), k=1)
    i_idx, j_idx = np.where(rows)
    pair_mat = np.zeros((len(i_idx), n))
    pair_mat[np.arange(len(i_idx)), i_idx] = 1
    pair_mat[np.arange(len(i_idx)), j_idx] = 1
    A.append(pair_mat)
    b.append(d[i_idx, j_idx])

    A_ub = np.vstack(A)
    b_ub = np.concatenate(b)

    # maximise sum(r)  → minimise -sum(r)
    res = linprog(
        -np.ones(n),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=(0, None),
        method="highs",
    )
    radii = res.x if res.success else np.zeros(n)
    return radii, radii.sum()


def construct_packing():
    """
    Build a packing of 26 circles.
    Starts from a 5×5 uniform grid (25 points) and tries a few
    candidate positions for the 26th circle, keeping the layout
    that yields the largest total radius after solving the LP.
    """
    # --- base 5×5 grid -------------------------------------------------
    pts = np.linspace(0.1, 0.9, 5)          # 0.1, 0.3, 0.5, 0.7, 0.9
    xv, yv = np.meshgrid(pts, pts)
    base_centers = np.column_stack([xv.ravel(), yv.ravel()])   # 25 points

    # --- candidate positions for the extra circle -----------------------
    # a modest set of points spread across the square
    extra_candidates = np.array(
        [
            [0.2, 0.2],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.8, 0.8],
            [0.5, 0.2],
            [0.5, 0.8],
            [0.2, 0.5],
            [0.8, 0.5],
            [0.5, 0.5],   # centre (already present in grid, will be ignored)
            [0.52, 0.52], # original extra point (kept as a fallback)
        ]
    )

    best_sum = -np.inf
    best_centers = None
    best_radii = None

    # Evaluate each candidate (skip if it duplicates an existing centre)
    for cand in extra_candidates:
        # avoid exact duplicates
        if np.any(np.all(np.isclose(base_centers, cand), axis=1)):
            continue

        centers = np.vstack([base_centers, cand])
        radii, total = _solve_lp(centers)

        if total > best_sum:
            best_sum = total
            best_centers = centers
            best_radii = radii

    # Fallback – if something went wrong, use the original layout
    if best_centers is None:
        best_centers = np.vstack([base_centers, [0.52, 0.52]])
        best_radii, best_sum = _solve_lp(best_centers)

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
