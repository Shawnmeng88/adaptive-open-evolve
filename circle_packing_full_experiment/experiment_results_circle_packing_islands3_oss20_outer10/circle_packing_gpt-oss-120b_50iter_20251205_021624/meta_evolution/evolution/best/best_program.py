# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    """
    Deterministic 5×5 base grid (25 circles) with an exhaustive
    search for the 26th centre.  First a coarse 0.01 lattice is scanned,
    then the best candidate is refined on a 0.001 lattice in a
    ±0.01 neighbourhood.  This yields a slightly larger total sum of
    radii while keeping validity = 1.0.
    Returns:
        centers (np.ndarray): shape (26, 2)
        radii   (np.ndarray): shape (26,)
        sum_radii (float)
    """
    # ----- base 5×5 grid -------------------------------------------------
    xs = np.linspace(0.1, 0.9, 5)          # 0.1, 0.3, …, 0.9
    ys = np.linspace(0.1, 0.9, 5)
    xv, yv = np.meshgrid(xs, ys)
    base_centers = np.column_stack((xv.ravel(), yv.ravel()))  # (25, 2)

    # ----- pre‑computations for the fixed base points --------------------
    base_border = np.minimum.reduce(
        [base_centers[:, 0], base_centers[:, 1],
         1.0 - base_centers[:, 0], 1.0 - base_centers[:, 1]]
    )
    base_diff = base_centers[:, None, :] - base_centers[None, :, :]
    base_dists = np.linalg.norm(base_diff, axis=2)
    np.fill_diagonal(base_dists, np.inf)

    # ----- coarse search (step = 0.01) ----------------------------------
    step_coarse = 0.01
    cand_vals = np.arange(step_coarse, 1.0, step_coarse)   # 0.01 … 0.99
    best_sum = -np.inf
    best_extra = None

    for cx in cand_vals:
        for cy in cand_vals:
            extra = np.array([cx, cy])
            # skip if extra coincides with an existing centre
            if np.any(np.linalg.norm(base_centers - extra, axis=1) < 1e-12):
                continue

            # border distances
            border = np.append(base_border,
                               min(cx, cy, 1.0 - cx, 1.0 - cy))

            # distances: reuse base matrix and compute distances to extra point
            d_extra = np.linalg.norm(base_centers - extra, axis=1)
            dists = np.block([
                [base_dists,          d_extra[:, None]],
                [d_extra[None, :],    np.array([[np.inf]])]
            ])

            nearest_half = np.min(dists, axis=1) * 0.5
            radii = np.minimum(border, nearest_half)
            s = radii.sum()

            if s > best_sum:
                best_sum = s
                best_extra = extra

    # ----- fine refinement around the coarse optimum --------------------
    if best_extra is not None:
        step_fine = 0.001
        delta_range = np.arange(-10, 11) * step_fine   # ±0.01
        cx0, cy0 = best_extra
        for dx in delta_range:
            for dy in delta_range:
                cx, cy = cx0 + dx, cy0 + dy
                if not (0.0 < cx < 1.0 and 0.0 < cy < 1.0):
                    continue
                extra = np.array([cx, cy])
                if np.any(np.linalg.norm(base_centers - extra, axis=1) < 1e-12):
                    continue

                border = np.append(base_border,
                                   min(cx, cy, 1.0 - cx, 1.0 - cy))

                d_extra = np.linalg.norm(base_centers - extra, axis=1)
                dists = np.block([
                    [base_dists,          d_extra[:, None]],
                    [d_extra[None, :],    np.array([[np.inf]])]
                ])

                nearest_half = np.min(dists, axis=1) * 0.5
                radii = np.minimum(border, nearest_half)
                s = radii.sum()

                if s > best_sum:
                    best_sum = s
                    best_extra = extra

    # ----- fallback ------------------------------------------------------
    if best_extra is None:
        best_extra = np.array([0.5, 0.5])

    # ----- final layout --------------------------------------------------
    centers = np.vstack((base_centers, best_extra))
    radii = compute_max_radii(centers)
    sum_radii = float(radii.sum())
    return centers, radii, sum_radii

def compute_max_radii(centers, max_iter=200, eps=1e-12):
    """
    Vectorised radius computation.
    For each centre the feasible radius is the minimum of:
        • distance to the closest square side,
        • half the distance to the nearest other centre.
    The additional parameters are kept for compatibility but unused.
    """
    border_dist = np.minimum.reduce(
        [centers[:, 0], centers[:, 1],
         1.0 - centers[:, 0], 1.0 - centers[:, 1]]
    )
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    nearest_half = np.min(dists, axis=1) * 0.5
    radii = np.minimum(border_dist, nearest_half)
    radii = np.clip(radii, 0.0, None)
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
