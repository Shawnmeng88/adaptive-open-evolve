"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a packing of 26 circles inside the unit square using a
    hexagonal (triangular) lattice. The lattice spacing `s` and a small
    offset are searched deterministically to maximise the sum of radii.
    """
    import numpy as np

    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # Helper: generate hexagonal lattice points for given spacing and offset
    def hex_lattice(s, offset):
        dx, dy = offset
        points = []
        row = 0
        y = dy
        h = s * np.sqrt(3) / 2.0  # vertical distance between rows
        while y <= 1.0:
            x_start = dx + (s / 2.0 if row % 2 else 0.0)
            x = x_start
            while x <= 1.0:
                points.append([x, y])
                x += s
            row += 1
            y += h
        return np.asarray(points)

    # Search over lattice spacing
    # Upper bound for spacing is roughly 0.5 (two circles across the square)
    # Lower bound 0.05 gives many points; we use a modest grid.
    s_values = np.linspace(0.05, 0.5, 30)

    for s in s_values:
        # Offsets to try: 0 or half-step in each direction
        dx_steps = [0.0, s / 2.0]
        dy_steps = [0.0, (s * np.sqrt(3) / 2.0) / 2.0]

        for dx in dx_steps:
            for dy in dy_steps:
                centers = hex_lattice(s, (dx, dy))
                if centers.shape[0] == 0:
                    continue

                # Compute maximal radius for each centre:
                # limited by walls and by half the lattice spacing (to avoid overlap)
                wall_dist = np.minimum.reduce(
                    [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
                )
                radii = np.minimum(wall_dist, s / 2.0)

                # If we have more than n circles, keep the n with largest radii
                if radii.shape[0] > n:
                    idx = np.argpartition(-radii, n - 1)[:n]
                    selected_centers = centers[idx]
                    selected_radii = radii[idx]
                else:
                    selected_centers = centers
                    selected_radii = radii

                # Verify feasibility (pairwise distances >= sum of radii)
                # Small tolerance to avoid floating‑point issues
                ok = True
                if selected_centers.shape[0] > 1:
                    diff = selected_centers[:, None, :] - selected_centers[None, :, :]
                    dists = np.sqrt(np.sum(diff ** 2, axis=2))
                    np.fill_diagonal(dists, np.inf)
                    if np.any(dists < selected_radii[:, None] + selected_radii[None, :] - 1e-9):
                        ok = False

                if not ok:
                    continue

                total = np.sum(selected_radii)
                if total > best_sum:
                    best_sum = total
                    best_centers = selected_centers
                    best_radii = selected_radii

    # Fallback: if search failed (should not happen), use the original simple pattern
    if best_centers is None:
        # simple fallback pattern (same as previous version)
        centers = np.zeros((n, 2))
        centers[0] = [0.5, 0.5]
        for i in range(8):
            angle = 2 * np.pi * i / 8
            centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]
        for i in range(16):
            angle = 2 * np.pi * i / 16
            centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]
        centers = np.clip(centers, 0.01, 0.99)
        best_centers = centers
        best_radii = np.minimum.reduce(
            [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
        )
        # simple pairwise scaling as in original code
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                if best_radii[i] + best_radii[j] > d:
                    scale = d / (best_radii[i] + best_radii[j])
                    best_radii[i] *= scale
                    best_radii[j] *= scale

    sum_radii = float(np.sum(best_radii))
    return best_centers, best_radii, sum_radii
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
