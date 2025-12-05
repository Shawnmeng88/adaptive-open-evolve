# EVOLVE-BLOCK-START
import numpy as np

def _max_radius_for_center(pt, centers, radii):
    """Maximum feasible radius for a single point given other circles."""
    x, y = pt
    # distance to walls
    r = min(x, y, 1.0 - x, 1.0 - y)
    if centers.shape[0]:
        d = np.sqrt(((centers - pt) ** 2).sum(axis=1)) - radii
        r = min(r, d.min())
    return max(r, 0.0)


def _compute_max_radii(centers):
    """Return maximal radii for all centers respecting walls and non‑overlap."""
    n = centers.shape[0]
    radii = np.empty(n, dtype=float)
    for i in range(n):
        # treat other circles as fixed when computing radius for i
        other_centers = np.delete(centers, i, axis=0)
        other_radii = np.delete(radii, i) if i < n else np.empty(0)
        radii[i] = _max_radius_for_center(centers[i], other_centers, other_radii)
    return radii


def _hex_grid(radius):
    """Generate hexagonal lattice points for a given radius."""
    dy = np.sqrt(3) * radius
    centers = []
    y = radius
    row = 0
    while y <= 1.0 - radius + 1e-12:
        offset = radius if row % 2 == 0 else 2 * radius
        x = offset
        while x <= 1.0 - radius + 1e-12:
            centers.append([x, y])
            x += 2 * radius
        row += 1
        y += dy
    return np.array(centers)


def _optimal_uniform_radius(target_n=26, iters=30):
    """Binary search for the largest uniform radius that yields at least target_n points."""
    low, high = 0.0, 0.5
    best = 0.0
    for _ in range(iters):
        mid = (low + high) / 2.0
        cnt = _hex_grid(mid).shape[0]
        if cnt >= target_n:
            best = mid
            low = mid
        else:
            high = mid
    return best


def construct_packing():
    """Dense packing of 26 circles using hex‑grid seeds and greedy local search."""
    n = 26
    rng = np.random.default_rng(seed=1)          # perturbations
    jitter_rng = np.random.default_rng(seed=0)    # seed jitter

    max_seeds = 4
    max_iters_per_seed = 3000
    step_scale = 0.03

    best_sum = -1.0
    best_centers = None
    best_radii = None

    uniform_r = _optimal_uniform_radius(target_n=n, iters=30)

    for seed_idx in range(max_seeds):
        centers = _hex_grid(uniform_r)
        if centers.shape[0] > n:
            centers = centers[:n]

        if seed_idx > 0:
            # small jitter to escape local minima
            offset = jitter_rng.uniform(-0.01, 0.01, size=centers.shape)
            centers = np.clip(centers + offset, 0.0, 1.0)

        radii = _compute_max_radii(centers)

        # greedy local search on center positions
        for _ in range(max_iters_per_seed):
            i = rng.integers(n)
            proposal = centers[i].copy()
            proposal += rng.normal(scale=step_scale, size=2)
            np.clip(proposal, 0.0, 1.0, out=proposal)

            new_centers = centers.copy()
            new_centers[i] = proposal
            new_radii = _compute_max_radii(new_centers)

            if np.any(new_radii <= 0):
                continue

            if new_radii.sum() > radii.sum():
                centers, radii = new_centers, new_radii

        total = radii.sum()
        if total > best_sum:
            best_sum = total
            best_centers = centers.copy()
            best_radii = radii.copy()

    # deterministic fine‑tuning after the best seed is selected
    for i in range(n):
        improved = True
        while improved:
            improved = False
            for dx, dy in [(step_scale, 0), (-step_scale, 0),
                           (0, step_scale), (0, -step_scale)]:
                cand = best_centers[i].copy()
                cand[0] = np.clip(cand[0] + dx, 0.0, 1.0)
                cand[1] = np.clip(cand[1] + dy, 0.0, 1.0)

                new_centers = best_centers.copy()
                new_centers[i] = cand
                new_radii = _compute_max_radii(new_centers)

                if new_radii.sum() > best_sum:
                    best_sum = new_radii.sum()
                    best_centers = new_centers
                    best_radii = new_radii
                    improved = True
                    break

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
