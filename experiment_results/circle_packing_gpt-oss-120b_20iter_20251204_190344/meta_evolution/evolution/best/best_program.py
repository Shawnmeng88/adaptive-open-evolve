# EVOLVE-BLOCK-START
"""Greedy vectorised construction of 26 non‑overlapping circles
   inside the unit square. The algorithm places circles one‑by‑one
   on a dense candidate grid, each time choosing the position that
   yields the largest feasible radius (respecting the square borders
   and all previously placed circles)."""

import numpy as np


def _candidate_grid(step: float = 0.05) -> np.ndarray:
    """Create a uniform grid of candidate centre points.

    The grid is kept away from the borders by `step` to guarantee a
    non‑zero initial radius for every candidate.

    Returns
    -------
    np.ndarray of shape (M, 2) with (x, y) coordinates.
    """
    coords = np.arange(step, 1.0 - step + 1e-12, step)
    xv, yv = np.meshgrid(coords, coords)
    return np.column_stack([xv.ravel(), yv.ravel()])


def _max_radius_for_candidates(candidates: np.ndarray,
                               placed_centers: np.ndarray,
                               placed_radii: np.ndarray) -> np.ndarray:
    """Vectorised computation of the largest admissible radius for each candidate.

    Parameters
    ----------
    candidates : (M, 2) array of candidate centre positions.
    placed_centers : (k, 2) array of already placed circle centres.
    placed_radii : (k,) array of radii of the already placed circles.

    Returns
    -------
    radii : (M,) array – the maximal radius each candidate could have
            without violating the square borders or overlapping any
            already placed circle.
    """
    # distance to the four square edges
    edge_dist = np.minimum.reduce(
        [candidates[:, 0],                # left edge
         candidates[:, 1],                # bottom edge
         1.0 - candidates[:, 0],          # right edge
         1.0 - candidates[:, 1]]          # top edge
    )

    if placed_centers.size == 0:
        # no circles placed yet – the edge distance is the only limitation
        return edge_dist

    # pairwise distances from each candidate to every placed centre
    # shape (k, M)
    dists = np.linalg.norm(
        placed_centers[:, np.newaxis, :] - candidates[np.newaxis, :, :],
        axis=2
    )
    # minimal clearance: distance to an existing centre minus that circle's radius
    clearance = dists - placed_radii[:, np.newaxis]
    # the most restrictive clearance across all already placed circles
    min_clearance = np.min(clearance, axis=0)

    # feasible radius is the smaller of the edge distance and the clearance
    return np.minimum(edge_dist, min_clearance)


def construct_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Greedy construction of 26 circles.

    Returns
    -------
    centers : (26, 2) array of (x, y) positions.
    radii   : (26,)   array of circle radii.
    sum_radii : float – total of all radii (used for the fitness score).
    """
    n_circles = 26
    # a fairly dense grid – more points give the algorithm more freedom
    candidates = _candidate_grid(step=0.05)

    placed_centers = np.empty((0, 2), float)
    placed_radii = np.empty((0,), float)

    for _ in range(n_circles):
        # compute the admissible radius for every remaining candidate
        radii = _max_radius_for_candidates(candidates, placed_centers, placed_radii)

        # choose the candidate that yields the largest radius
        best_idx = np.argmax(radii)
        best_radius = radii[best_idx]

        # if the best radius is non‑positive we cannot place more circles
        if best_radius <= 0:
            break

        # store the chosen circle
        best_center = candidates[best_idx]
        placed_centers = np.vstack([placed_centers, best_center])
        placed_radii = np.append(placed_radii, best_radius)

        # remove the used candidate from the pool
        candidates = np.delete(candidates, best_idx, axis=0)

    sum_radii = float(placed_radii.sum())
    return placed_centers, placed_radii, sum_radii


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
