# EVOLVE-BLOCK-START
import numpy as np

def _hex_lattice_points(spacing, required):
    """
    Generate points on a hexagonal lattice with given spacing.
    Returns a list of points staying at least spacing/2 from the unit‑square borders.
    The list may contain more than `required` points.
    """
    if spacing <= 0:
        return []
    r = spacing / 2.0
    dy = spacing * np.sqrt(3) / 2.0
    points = []
    y = r
    row = 0
    while y <= 1.0 - r + 1e-12:
        offset = (row % 2) * (spacing / 2.0)
        x = r + offset
        while x <= 1.0 - r + 1e-12:
            points.append([x, y])
            if len(points) >= required * 3:  # generous oversampling
                break
            x += spacing
        if len(points) >= required * 3:
            break
        row += 1
        y += dy
    return points

def _refine_radii(centers, init_radii, max_iter=200):
    """
    Given fixed centres, iteratively enlarge each radius to the maximal
    feasible value respecting borders and other circles.
    """
    n = centers.shape[0]
    border = np.minimum.reduce([
        centers[:, 0],
        centers[:, 1],
        1.0 - centers[:, 0],
        1.0 - centers[:, 1]
    ])
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)

    radii = init_radii.copy()
    for _ in range(max_iter):
        allowed = np.minimum(border, np.min(dists - radii[None, :], axis=1))
        allowed = np.maximum(allowed, 0.0)
        if np.allclose(radii, allowed, atol=1e-9):
            break
        radii = allowed
    return radii

def construct_packing():
    """
    Construct a packing of 26 circles inside the unit square.
    Starts from a maximal uniform hexagonal lattice, refines radii,
    then performs local jitter optimisation to increase the total sum of radii.
    Returns:
        centers (np.ndarray): shape (26, 2)
        radii   (np.ndarray): shape (26,)
        sum_of_radii (float)
    """
    n = 26

    # --- Step 1: find maximal uniform spacing that yields at least n points ---
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2.0
        pts = _hex_lattice_points(mid, n)
        if len(pts) >= n:
            lo = mid
        else:
            hi = mid
    spacing = lo

    # --- Step 2: initialise centres on the lattice ---
    centers = np.array(_hex_lattice_points(spacing, n)[:n], dtype=float)

    # --- Step 3: start with uniform radii and refine ---
    radii = np.full(n, spacing / 2.0, dtype=float)
    radii = _refine_radii(centers, radii, max_iter=200)

    # --- Step 4: local jitter optimisation ---
    rng = np.random.default_rng(0)
    eps = 0.02  # maximum jitter amplitude
    for _ in range(500):
        i = rng.integers(n)
        delta = rng.uniform(-eps, eps, size=2)
        new_center = centers[i] + delta
        new_center = np.clip(new_center, 0.0, 1.0)

        # compute maximal feasible radius at the new position
        border = min(new_center[0], new_center[1],
                     1.0 - new_center[0], 1.0 - new_center[1])
        other_mask = np.arange(n) != i
        if other_mask.any():
            dists = np.linalg.norm(centers[other_mask] - new_center, axis=1) - radii[other_mask]
            max_r = min(border, dists.min())
        else:
            max_r = border

        if max_r > radii[i] + 1e-4:
            centers[i] = new_center
            radii[i] = max_r
            # re‑refine all radii after the move
            radii = _refine_radii(centers, radii, max_iter=200)

    sum_radii = float(radii.sum())
    return centers, radii, sum_radii
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
