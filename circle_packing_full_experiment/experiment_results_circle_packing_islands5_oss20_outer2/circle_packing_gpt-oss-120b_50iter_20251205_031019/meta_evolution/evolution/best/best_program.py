# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import distance


def _optimal_radii(centers: np.ndarray) -> np.ndarray:
    """Linear program maximizing sum of radii for fixed centres."""
    n = centers.shape[0]
    border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )
    dists = distance.cdist(centers, centers)

    rows = []
    rhs = []

    # border constraints
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        rows.append(row)
        rhs.append(border[i])

    # pairwise non‑overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = row[j] = 1.0
            rows.append(row)
            rhs.append(dists[i, j])

    A_ub = np.vstack(rows)
    b_ub = np.array(rhs)

    c = -np.ones(n)                # maximise sum(r) → minimise -sum(r)
    bounds = [(0, None)] * n

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.success:
        return res.x
    # fallback – half of the border distance (always feasible)
    return border * 0.5


def _hex_lattice_points(num_points: int, margin: float = 0.02) -> np.ndarray:
    """Generate points on a hexagonal lattice inside the unit square."""
    dx = 0.18
    dy = np.sqrt(3) / 2 * dx
    pts = []
    y = margin
    row = 0
    while y <= 1 - margin and len(pts) < num_points * 2:
        offset = 0.0 if row % 2 == 0 else dx / 2
        x = margin + offset
        while x <= 1 - margin:
            pts.append([x, y])
            if len(pts) >= num_points * 2:
                break
            x += dx
        row += 1
        y += dy
    return np.array(pts)[:num_points]


def _farthest_sampling(candidates: np.ndarray, k: int) -> np.ndarray:
    """Greedy far‑point sampling to obtain well‑spaced centres."""
    start_idx = np.argmin(np.sum((candidates - 0.5) ** 2, axis=1))
    selected = [start_idx]
    while len(selected) < k:
        remaining = np.setdiff1d(np.arange(len(candidates)), selected)
        dists = distance.cdist(candidates[remaining], candidates[selected])
        min_dists = dists.min(axis=1)
        next_idx = remaining[np.argmax(min_dists)]
        selected.append(next_idx)
    return candidates[selected]


def _base_candidate_sets() -> list:
    """Create a diverse pool of deterministic centre configurations."""
    cand = []

    # 5×5 regular grid + centre point
    step = 0.2
    coords = np.arange(0.1, 0.9 + 1e-9, step)
    grid = np.array([[x, y] for y in coords for x in coords])  # 25 points
    extra = np.array([0.5, 0.5])
    cand.append(np.vstack([grid, extra])[:26])

    # Hexagonal lattice
    cand.append(_hex_lattice_points(26, margin=0.02))

    # Farthest sampling from a dense 7×7 grid
    xs = np.linspace(0.125, 0.875, 7)
    ys = np.linspace(0.125, 0.875, 7)
    dense = np.array([[x, y] for y in ys for x in xs])
    cand.append(_farthest_sampling(dense, 26))

    # Additional deterministic set: jittered grid (small random offset)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.02, 0.02, size=grid.shape)
    jittered = np.clip(grid + jitter, 0.02, 0.98)
    cand.append(np.vstack([jittered, extra])[:26])

    return cand


def _hill_climb(initial: np.ndarray,
                iters: int = 400,
                step_start: float = 0.04,
                margin: float = 0.01,
                rng: np.random.Generator = None) -> tuple:
    """Hill‑climbing that perturbs one centre at a time and keeps improvements."""
    if rng is None:
        rng = np.random.default_rng()
    best_centers = initial.copy()
    best_radii = _optimal_radii(best_centers)
    best_sum = best_radii.sum()
    step = step_start

    for _ in range(iters):
        i = rng.integers(0, len(best_centers))
        proposal = best_centers.copy()
        delta = rng.uniform(-step, step, size=2)
        proposal[i] = np.clip(proposal[i] + delta, margin, 1 - margin)

        radii = _optimal_radii(proposal)
        total = radii.sum()
        if total > best_sum:
            best_sum = total
            best_centers = proposal
            best_radii = radii
            step = min(step_start, step * 1.02)   # keep step relatively large after success
        else:
            step = max(0.001, step * 0.98)        # slowly shrink step

    return best_centers, best_radii, float(best_sum)


def _refine_candidate(centers: np.ndarray,
                      rng: np.random.Generator,
                      margin: float = 0.01) -> tuple:
    """Three‑phase refinement: coarse → medium → fine hill‑climb."""
    # Phase 1 – aggressive exploration
    c1, r1, s1 = _hill_climb(
        centers,
        iters=800,
        step_start=0.05,
        margin=margin,
        rng=rng,
    )
    # Phase 2 – medium‑scale polishing
    c2, r2, s2 = _hill_climb(
        c1,
        iters=600,
        step_start=0.025,
        margin=margin,
        rng=rng,
    )
    # Phase 3 – fine‑grained polishing
    c3, r3, s3 = _hill_climb(
        c2,
        iters=400,
        step_start=0.012,
        margin=margin,
        rng=rng,
    )
    # Return the best of the three phases
    best = max([(s1, c1, r1), (s2, c2, r2), (s3, c3, r3)], key=lambda x: x[0])
    return best[1], best[2], best[0]


def _scale_radii(centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Uniformly scale radii up to the first violated constraint."""
    n = len(radii)
    if np.any(radii == 0):
        return radii
    border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )
    # Border scaling factors
    border_factors = border / radii
    # Pairwise scaling factors
    dists = distance.cdist(centers, centers)
    sum_r = radii[:, None] + radii[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        pair_factors = np.where(sum_r > 0, dists / sum_r, np.inf)
    # Ignore diagonal
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    min_pair_factor = np.min(pair_factors[mask])
    factor = min(border_factors.min(), min_pair_factor)
    if factor > 1.0:
        return radii * factor
    return radii


def construct_packing():
    """
    Generate deterministic and random centre configurations,
    refine the most promising ones, and apply a final uniform scaling.
    """
    rng = np.random.default_rng(12345)   # deterministic seed for reproducibility
    margin = 0.01

    # ----- deterministic seeds -----
    candidates = _base_candidate_sets()

    # ----- additional random seeds for diversity -----
    for _ in range(16):
        rand = rng.uniform(margin, 1 - margin, size=(26, 2))
        candidates.append(rand)

    # ----- evaluate all seeds once -----
    scores = []
    radii_list = []
    for centers in candidates:
        radii = _optimal_radii(centers)
        scores.append(radii.sum())
        radii_list.append(radii)

    # keep top‑k distinct seeds for deep refinement
    top_k = 6
    top_idxs = np.argsort(scores)[-top_k:]

    best_sum = -1.0
    best_centers = None
    best_radii = None

    for idx in top_idxs:
        refined_centers, refined_radii, refined_sum = _refine_candidate(
            candidates[idx],
            rng=rng,
            margin=margin,
        )
        if refined_sum > best_sum:
            best_sum = refined_sum
            best_centers = refined_centers
            best_radii = refined_radii

    # Extra intensive polishing on the current best solution
    if best_centers is not None:
        extra_centers, extra_radii, extra_sum = _hill_climb(
            best_centers,
            iters=1200,
            step_start=0.018,
            margin=margin,
            rng=rng,
        )
        if extra_sum > best_sum:
            best_sum = extra_sum
            best_centers, best_radii = extra_centers, extra_radii

    # Fallback – in the unlikely event refinement fails, return the best raw candidate
    if best_centers is None:
        best_idx = np.argmax(scores)
        best_centers = candidates[best_idx]
        best_radii = radii_list[best_idx]
        best_sum = scores[best_idx]

    # ----- final uniform scaling to squeeze any remaining slack -----
    scaled_radii = _scale_radii(best_centers, best_radii)
    best_sum = scaled_radii.sum()
    best_radii = scaled_radii

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
