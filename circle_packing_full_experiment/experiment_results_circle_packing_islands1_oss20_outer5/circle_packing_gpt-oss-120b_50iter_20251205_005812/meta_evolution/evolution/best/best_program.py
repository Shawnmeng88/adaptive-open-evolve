# EVOLVE-BLOCK-START
import numpy as np


def compute_max_radii(centers):
    """
    Vectorised admissible radii:
        radius = min(distance to square border,
                     half the distance to the nearest other centre)
    """
    if centers.size == 0:
        return np.array([])

    # distance to the four borders
    border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
    )

    # pairwise centre distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)  # ignore self‑distance

    nearest = np.min(dists, axis=1)
    radii = np.minimum(border, nearest / 2.0)
    return np.clip(radii, 1e-8, None)


def _generate_hex_lattice(step):
    """
    Produce a hexagonal lattice of points inside the unit square.
    """
    margin = step / 2.0
    v_spacing = step * np.sqrt(3) / 2.0
    points = []
    row = 0
    y = margin
    while y <= 1.0 - margin + 1e-12:
        offset = 0.0 if (row % 2 == 0) else step / 2.0
        x = margin + offset
        while x <= 1.0 - margin + 1e-12:
            points.append([x, y])
            x += step
        row += 1
        y += v_spacing
    return np.array(points)


def construct_packing():
    """
    Search over hexagonal lattice spacings, keep the 26 points with the
    largest total admissible radius, then refine with stochastic moves.
    """
    target_n = 26
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    best_step = None

    # ---- 1. coarse scan -------------------------------------------------
    for step in np.linspace(0.10, 0.26, 81):
        pts = _generate_hex_lattice(step)
        if pts.shape[0] < target_n:
            continue

        # radii for the full lattice
        radii_full = compute_max_radii(pts)

        # keep the 26 points with largest individual radii
        idx = np.argpartition(radii_full, -target_n)[-target_n:]
        cand = pts[idx]

        # recompute radii for the trimmed set
        cand_radii = compute_max_radii(cand)
        total = cand_radii.sum()

        if total > best_sum:
            best_sum = total
            best_centers = cand
            best_radii = cand_radii
            best_step = step

    # ---- 2. fine scan around the best step -----------------------------
    if best_step is not None:
        fine_steps = np.linspace(
            max(0.10, best_step - 0.015),
            min(0.26, best_step + 0.015),
            41,
        )
        for step in fine_steps:
            pts = _generate_hex_lattice(step)
            if pts.shape[0] < target_n:
                continue

            radii_full = compute_max_radii(pts)
            idx = np.argpartition(radii_full, -target_n)[-target_n:]
            cand = pts[idx]
            cand_radii = compute_max_radii(cand)
            total = cand_radii.sum()

            if total > best_sum:
                best_sum = total
                best_centers = cand
                best_radii = cand_radii
                best_step = step

    # ---- 3. fallback safety ---------------------------------------------
    if best_centers is None:
        pts = _generate_hex_lattice(0.18)[:target_n]
        best_centers = pts
        best_radii = compute_max_radii(pts)
        best_sum = best_radii.sum()

    # ---- 4. stochastic refinement ----------------------------------------
    rng = np.random.default_rng(42)
    n_iter = 4000
    step_scale = 0.03
    decay = 0.9995

    centers = best_centers.copy()
    radii = best_radii.copy()
    sum_radii = best_sum

    for _ in range(n_iter):
        i = rng.integers(target_n)
        proposal = centers[i] + rng.normal(scale=step_scale, size=2)
        proposal = np.clip(proposal, 0.0, 1.0)

        new_centers = centers.copy()
        new_centers[i] = proposal
        new_radii = compute_max_radii(new_centers)

        if np.any(new_radii <= 0):
            continue

        new_sum = new_radii.sum()
        if new_sum > sum_radii:
            centers = new_centers
            radii = new_radii
            sum_radii = new_sum

        step_scale *= decay

    return centers, radii, float(sum_radii)
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
