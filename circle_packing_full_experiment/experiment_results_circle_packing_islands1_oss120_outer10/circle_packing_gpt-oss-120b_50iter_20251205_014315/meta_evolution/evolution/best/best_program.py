# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    """
    Build a packing of 26 circles inside the unit square.
    Tries two strategies (square lattice with an extra point, and hexagonal lattice)
    and returns the one with the larger total radius sum.
    """
    # ----- Square‑grid strategy (25 points + best extra point) -----
    spacing_sq = _max_spacing_for_target(25, low=0.05, high=0.5, eps=1e-5)
    grid_sq = _grid_points(spacing_sq)[:25]

    # generate candidate positions for the 26‑th circle
    half = spacing_sq / 2.0
    grid_set = {tuple(p) for p in grid_sq.tolist()}
    candidates = []

    for x, y in grid_sq:
        # horizontal neighbour centre
        if x + spacing_sq <= 1.0 - half + 1e-12:
            cand = (x + half, y)
            if cand not in grid_set:
                candidates.append(cand)
        # vertical neighbour centre
        if y + spacing_sq <= 1.0 - half + 1e-12:
            cand = (x, y + half)
            if cand not in grid_set:
                candidates.append(cand)
        # diagonal centre of a cell
        if (x + spacing_sq <= 1.0 - half + 1e-12) and (y + spacing_sq <= 1.0 - half + 1e-12):
            cand = (x + half, y + half)
            if cand not in grid_set:
                candidates.append(cand)

    # evaluate each candidate using current radii of the grid points
    core_radii = _max_feasible_radii(grid_sq)
    best_cand = None
    best_r = -1.0
    for cand in candidates:
        r = _max_radius_for_point(np.array(cand), grid_sq, core_radii)
        if r > best_r + 1e-12:
            best_r = r
            best_cand = cand
        elif abs(r - best_r) <= 1e-12 and best_cand is not None:
            # deterministic tie‑break: smallest x, then y
            if cand < best_cand:
                best_cand = cand

    if best_cand is None:
        best_cand = (0.5, 0.5)

    centers_sq = np.vstack([grid_sq, best_cand])
    radii_sq = _max_feasible_radii(centers_sq)
    sum_sq = float(radii_sq.sum())

    # ----- Hexagonal‑grid strategy (26 points) -----
    spacing_hex = _max_spacing_for_n(26, low=0.05, high=0.5, eps=1e-4)
    centres_hex = _generate_hex_grid(spacing_hex, 26)

    # centre the hex pattern inside the unit square
    mins = centres_hex.min(axis=0)
    maxs = centres_hex.max(axis=0)
    shift = (1.0 - (maxs - mins)) / 2.0 - mins
    centres_hex = centres_hex + shift

    radii_hex = compute_max_radii(centres_hex)
    sum_hex = float(radii_hex.sum())

    # ----- Choose the better packing -----
    if sum_hex > sum_sq:
        return centres_hex, radii_hex, sum_hex
    else:
        return centers_sq, radii_sq, sum_sq


def _max_spacing_for_target(target, low=0.05, high=0.5, eps=1e-5):
    """Binary search for the largest square‑grid spacing yielding ≥ target points."""
    while high - low > eps:
        mid = (low + high) / 2.0
        if _count_grid_points(mid) >= target:
            low = mid
        else:
            high = mid
    return low


def _count_grid_points(spacing):
    """Count how many points a square grid with given spacing would contain."""
    start = spacing / 2.0
    if start > 1.0 - start:
        return 0
    cnt = 0
    y = start
    while y <= 1.0 - start + 1e-12:
        x = start
        while x <= 1.0 - start + 1e-12:
            cnt += 1
            x += spacing
        y += spacing
    return cnt


def _grid_points(spacing):
    """Generate all points of a square lattice inside the unit square."""
    start = spacing / 2.0
    pts = []
    y = start
    while y <= 1.0 - start + 1e-12:
        x = start
        while x <= 1.0 - start + 1e-12:
            pts.append([x, y])
            x += spacing
        y += spacing
    return np.array(pts)


def _max_feasible_radii(centers):
    """Maximum feasible radius for each centre respecting walls and other circles."""
    wall = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
    )
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    nearest = np.min(dists, axis=1)
    radii = np.minimum(wall, nearest / 2.0)
    return radii


def _max_radius_for_point(pt, centers, radii):
    """Maximum radius for a single point given existing centres and their radii."""
    wall = min(pt[0], pt[1], 1.0 - pt[0], 1.0 - pt[1])
    if centers.size == 0:
        return wall
    dists = np.linalg.norm(centers - pt, axis=1) - radii
    return min(wall, np.min(dists))


def compute_max_radii(centers):
    """Legacy alias – same as _max_feasible_radii."""
    return _max_feasible_radii(centers)


def _max_spacing_for_n(n, low=0.05, high=0.5, eps=1e-4):
    """Binary search for the largest hex‑grid spacing yielding ≥ n points."""
    while high - low > eps:
        mid = (low + high) / 2.0
        if _count_hex_points(mid, n) >= n:
            low = mid
        else:
            high = mid
    return low


def _count_hex_points(spacing, target):
    """Count points produced by a hexagonal grid up to `target`."""
    dy = spacing * np.sqrt(3) / 2.0
    y = 0.0
    row = 0
    cnt = 0
    while y <= 1.0 and cnt < target:
        offset = (spacing / 2.0) if (row % 2 == 1) else 0.0
        x = offset
        while x <= 1.0 and cnt < target:
            cnt += 1
            x += spacing
        row += 1
        y += dy
    return cnt


def _generate_hex_grid(spacing, n):
    """First `n` points of a hexagonal lattice with the given spacing."""
    dy = spacing * np.sqrt(3) / 2.0
    points = []
    y = 0.0
    row = 0
    while y <= 1.0 and len(points) < n:
        offset = (spacing / 2.0) if (row % 2 == 1) else 0.0
        x = offset
        while x <= 1.0 and len(points) < n:
            points.append([x, y])
            x += spacing
        row += 1
        y += dy
    return np.array(points[:n])
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
