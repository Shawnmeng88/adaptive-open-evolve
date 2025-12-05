# EVOLVE-BLOCK-START
import numpy as np

def compute_max_radii(centers):
    """
    Compute a safe maximal radius for each circle:
        radius_i = min( distance to square border,
                        0.5 * min_{j!=i} distance(centers_i, centers_j) )
    Guarantees no overlap and containment.
    """
    border = np.minimum.reduce([
        centers[:, 0],
        centers[:, 1],
        1.0 - centers[:, 0],
        1.0 - centers[:, 1]
    ])

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists, axis=1)

    return np.minimum(border, min_dist / 2.0)


def construct_packing():
    """
    Initialise a hexagonal lattice of 26 points and then improve it
    using a stochastic hill‑climb followed by a deterministic
    local refinement. Returns (centers, radii, total_radius_sum).
    """
    # ---------- initial hexagonal lattice ----------
    spacing = 0.18
    margin = spacing / 2.0
    sqrt3 = np.sqrt(3.0)

    pts = []
    y = margin
    row = 0
    while y <= 1.0 - margin + 1e-12:
        offset = (row % 2) * spacing / 2.0
        x = margin + offset
        while x <= 1.0 - margin + 1e-12:
            pts.append([x, y])
            x += spacing
        row += 1
        y += spacing * sqrt3 / 2.0

    init_centers = np.array(pts)[:26]
    n = init_centers.shape[0]

    # ---------- stochastic hill‑climb ----------
    rng = np.random.RandomState(0)
    best_centers = init_centers.copy()
    best_radii = compute_max_radii(best_centers)
    best_score = float(best_radii.sum())

    iterations = 15000
    step_scale = 0.02
    decay_interval = 3000

    for it in range(iterations):
        if it and (it % decay_interval) == 0:
            step_scale *= 0.7

        i = rng.randint(n)
        proposal = best_centers[i] + rng.uniform(-step_scale, step_scale, size=2)
        proposal = np.clip(proposal, 0.0, 1.0)

        new_centers = best_centers.copy()
        new_centers[i] = proposal
        new_radii = compute_max_radii(new_centers)
        new_score = float(new_radii.sum())

        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers
            best_radii = new_radii

    # ---------- deterministic local refinement ----------
    refine_step = step_scale * 0.5
    for i in range(n):
        improved = True
        while improved:
            improved = False
            for dx, dy in ((refine_step, 0), (-refine_step, 0),
                           (0, refine_step), (0, -refine_step)):
                proposal = best_centers[i] + np.array([dx, dy])
                proposal = np.clip(proposal, 0.0, 1.0)
                new_centers = best_centers.copy()
                new_centers[i] = proposal
                new_radii = compute_max_radii(new_centers)
                new_score = float(new_radii.sum())
                if new_score > best_score + 1e-12:
                    best_score = new_score
                    best_centers = new_centers
                    best_radii = new_radii
                    improved = True
                    break

    return best_centers, best_radii, best_score
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
