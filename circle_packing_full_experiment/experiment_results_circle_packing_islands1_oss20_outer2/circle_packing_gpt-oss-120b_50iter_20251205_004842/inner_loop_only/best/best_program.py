# EVOLVE-BLOCK-START
import numpy as np

def _generate_points(spacing: float) -> np.ndarray | None:
    """Generate up to 26 points on a hexagonal grid with given spacing.
    Returns None if fewer than 26 points can be placed."""
    pts = []
    y = spacing / 2
    row = 0
    while len(pts) < 26 and y <= 1 - spacing / 2:
        offset = 0 if row % 2 == 0 else spacing / 2
        x = spacing / 2 + offset
        while len(pts) < 26 and x <= 1 - spacing / 2:
            pts.append([x, y])
            x += spacing
        y += spacing * np.sqrt(3) / 2
        row += 1
    if len(pts) < 26:
        return None
    return np.array(pts[:26])

def compute_max_radii(centers: np.ndarray) -> np.ndarray:
    """Greedy radius adjustment: start with border limits then tighten pairwise constraints."""
    n = centers.shape[0]
    # border‑limited radii
    radii = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )
    # pairwise distances
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    i, j = np.triu_indices(n, k=1)
    order = np.argsort(dists[i, j])
    for idx in order:
        a, b = i[idx], j[idx]
        d = dists[a, b]
        if radii[a] + radii[b] > d:
            limit = d / 2.0
            if radii[a] > limit:
                radii[a] = limit
            if radii[b] > limit:
                radii[b] = limit
    return radii

def construct_packing():
    """Select the spacing that yields the largest total radius sum."""
    best_sum = -1.0
    best_centers = None
    best_radii = None
    # candidate spacings around the known good value
    for spacing in (0.16, 0.17, 0.175, 0.18, 0.185, 0.19, 0.195, 0.20):
        centers = _generate_points(spacing)
        if centers is None:
            continue
        radii = compute_max_radii(centers)
        total = radii.sum()
        if total > best_sum:
            best_sum, best_centers, best_radii = total, centers, radii
    # Fallback – should never happen
    if best_centers is None:
        best_centers = np.zeros((0, 2))
        best_radii = np.zeros(0)
        best_sum = 0.0
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
