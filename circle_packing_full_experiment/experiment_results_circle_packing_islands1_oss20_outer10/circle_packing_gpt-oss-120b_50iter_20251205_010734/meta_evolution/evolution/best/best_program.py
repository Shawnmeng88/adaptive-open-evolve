# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    """
    Deterministic construction of 26 circles inside the unit square.
    Searches hexagonal and square lattices over a fine spacing grid,
    applies two deterministic selection strategies (radius ranking and
    farthest‑point sampling), optionally refines the layout with a tiny
    deterministic push‑out, and returns the configuration with the largest
    total radius sum.
    """
    target_n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # fine spacing sweep – enough candidates for a rich search
    spacings = np.linspace(0.04, 0.30, 260)

    for spacing in spacings:
        # ---------- hexagonal lattice ----------
        base_hex = _hex_lattice(spacing)
        v_step = spacing * np.sqrt(3.0) / 2.0
        hex_offsets = [
            (0.0, 0.0),
            (spacing / 2.0, 0.0),
            (0.0, v_step),
            (spacing / 2.0, v_step),
        ]
        for ox, oy in hex_offsets:
            pts = base_hex + np.array([ox, oy])
            mask = (pts[:, 0] >= 0.0) & (pts[:, 0] <= 1.0) & (pts[:, 1] >= 0.0) & (pts[:, 1] <= 1.0)
            cand = pts[mask]
            if cand.shape[0] < target_n:
                continue

            # strategy 1: radius ranking
            centers, radii = _select_top_n(cand, target_n)
            centers, radii = _push_out(centers, radii)
            total = radii.sum()
            if total > best_sum:
                best_sum, best_centers, best_radii = total, centers, radii

            # strategy 2: farthest‑point sampling
            centers, radii = _select_fps(cand, target_n)
            centers, radii = _push_out(centers, radii)
            total = radii.sum()
            if total > best_sum:
                best_sum, best_centers, best_radii = total, centers, radii

        # ---------- square lattice ----------
        base_sq = _square_lattice(spacing)
        sq_offsets = [
            (0.0, 0.0),
            (spacing / 2.0, 0.0),
            (0.0, spacing / 2.0),
            (spacing / 2.0, spacing / 2.0),
        ]
        for ox, oy in sq_offsets:
            pts = base_sq + np.array([ox, oy])
            mask = (pts[:, 0] >= 0.0) & (pts[:, 0] <= 1.0) & (pts[:, 1] >= 0.0) & (pts[:, 1] <= 1.0)
            cand = pts[mask]
            if cand.shape[0] < target_n:
                continue

            centers, radii = _select_top_n(cand, target_n)
            centers, radii = _push_out(centers, radii)
            total = radii.sum()
            if total > best_sum:
                best_sum, best_centers, best_radii = total, centers, radii

            centers, radii = _select_fps(cand, target_n)
            centers, radii = _push_out(centers, radii)
            total = radii.sum()
            if total > best_sum:
                best_sum, best_centers, best_radii = total, centers, radii

    # fallback – deterministic hex lattice with farthest‑point sampling.
    if best_centers is None:
        fallback = _hex_lattice(0.12)
        best_centers, best_radii = _select_fps(fallback, target_n)
        best_centers, best_radii = _push_out(best_centers, best_radii)
        best_sum = best_radii.sum()

    return best_centers, best_radii, float(best_sum)


def _hex_lattice(spacing: float) -> np.ndarray:
    """
    Generate points on a hexagonal (triangular) lattice inside the unit square.
    Points are kept at least `spacing/2` away from each wall.
    """
    margin = spacing / 2.0
    vert_step = spacing * np.sqrt(3.0) / 2.0
    pts = []
    y = margin
    row = 0
    while y <= 1.0 - margin + 1e-12:
        offset = 0.0 if row % 2 == 0 else spacing / 2.0
        x = margin + offset
        while x <= 1.0 - margin + 1e-12:
            pts.append([x, y])
            x += spacing
        y += vert_step
        row += 1
    return np.array(pts, dtype=np.float64)


def _square_lattice(spacing: float) -> np.ndarray:
    """
    Generate points on a square lattice inside the unit square.
    Points are kept at least `spacing/2` away from each wall.
    """
    margin = spacing / 2.0
    xs = np.arange(margin, 1.0 - margin + 1e-12, spacing)
    ys = np.arange(margin, 1.0 - margin + 1e-12, spacing)
    xv, yv = np.meshgrid(xs, ys)
    return np.column_stack([xv.ravel(), yv.ravel()])


def _select_top_n(points: np.ndarray, n: int):
    """
    Choose `n` points with the largest feasible radii.
    Radii are first computed for all candidates, the points are sorted by
    descending radius (ties broken by original index), the top `n` are kept,
    and radii are recomputed for the final subset.
    """
    radii_all = _compute_radii(points)
    order = np.lexsort((-radii_all, np.arange(len(radii_all))))
    selected = points[order[:n]]
    final_radii = _compute_radii(selected)
    return selected, final_radii


def _select_fps(points: np.ndarray, n: int):
    """
    Select `n` points using deterministic farthest‑point sampling,
    then compute their maximal admissible radii.
    """
    selected = _farthest_point_sampling(points, n)
    radii = _compute_radii(selected)
    return selected, radii


def _farthest_point_sampling(candidates: np.ndarray, k: int) -> np.ndarray:
    """
    Deterministic greedy farthest‑point sampling.
    Starts from the candidate farthest from the square walls, then repeatedly
    adds the candidate whose minimal distance to the already selected set is
    maximal. Ties are resolved by the smallest index.
    """
    if candidates.shape[0] <= k:
        return candidates.copy()

    wall_dist = np.minimum.reduce([
        candidates[:, 0],
        candidates[:, 1],
        1.0 - candidates[:, 0],
        1.0 - candidates[:, 1]
    ])
    first_idx = int(np.argmax(wall_dist))
    selected_idxs = [first_idx]

    remaining = np.ones(len(candidates), dtype=bool)
    remaining[first_idx] = False

    while len(selected_idxs) < k:
        rem_idxs = np.where(remaining)[0]
        sel = candidates[selected_idxs]
        diffs = candidates[rem_idxs][:, None, :] - sel[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)          # shape (rem, sel)
        min_to_sel = dists.min(axis=1)
        next_rel = int(np.argmax(min_to_sel))
        next_abs = rem_idxs[next_rel]

        selected_idxs.append(next_abs)
        remaining[next_abs] = False

    return candidates[selected_idxs]


def _compute_radii(centers: np.ndarray) -> np.ndarray:
    """
    Compute the maximal admissible radius for each centre:
    the minimum of the distance to the nearest wall and half the distance to
    the nearest neighbour.
    """
    wall_dist = np.minimum.reduce([
        centers[:, 0],
        centers[:, 1],
        1.0 - centers[:, 0],
        1.0 - centers[:, 1]
    ])

    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    nearest = np.min(dists, axis=1)

    return np.minimum(wall_dist, nearest / 2.0)


def _push_out(centers: np.ndarray, radii: np.ndarray,
              iterations: int = 5, step: float = 0.001):
    """
    Deterministic tiny push‑out refinement.
    For a few iterations each circle is nudged a small step away from its nearest
    neighbour (if it stays inside the square and does not create a new overlap).
    Radii are recomputed after each full pass.
    """
    n = centers.shape[0]
    for _ in range(iterations):
        for i in range(n):
            c = centers[i]
            r = radii[i]

            # vector away from nearest neighbour
            diffs = centers - c
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf
            nearest_idx = int(np.argmin(dists))
            direction = c - centers[nearest_idx]
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            direction /= norm
            new_c = c + step * direction

            # check containment
            if not (r <= new_c[0] <= 1.0 - r and r <= new_c[1] <= 1.0 - r):
                continue

            # check no new overlaps
            others = np.delete(centers, i, axis=0)
            if np.all(np.linalg.norm(others - new_c, axis=1) >= 2 * r - 1e-12):
                centers[i] = new_c

        radii = _compute_radii(centers)

    return centers, radii
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
