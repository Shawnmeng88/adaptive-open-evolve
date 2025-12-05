# EVOLVE-BLOCK-START
"""Greedy max‑radius packing for 26 circles in the unit square."""
import numpy as np


def construct_packing():
    # deterministic generator – same layout every run
    rng = np.random.default_rng(12345)

    # many uniformly sampled candidate positions
    candidates = rng.random((20000, 2))
    used = np.zeros(len(candidates), dtype=bool)

    centers = []
    radii = []

    def feasible_radii(pts, cur_centers, cur_radii):
        """largest radius each point can have without leaving the square
        or overlapping existing circles."""
        # distance to the four sides of the unit square
        edge = np.minimum.reduce([pts[:, 0], pts[:, 1],
                                  1 - pts[:, 0], 1 - pts[:, 1]])

        if cur_centers.shape[0] == 0:
            return edge

        # distance to existing circles minus their radii
        diff = pts[:, None, :] - cur_centers[None, :, :]          # (m,k,2)
        d = np.linalg.norm(diff, axis=2) - cur_radii              # (m,k)
        mindist = np.min(d, axis=1)                               # (m,)
        return np.minimum(edge, mindist)

    for _ in range(26):
        # candidates that are still free
        free_idx = np.where(~used)[0]
        if free_idx.size == 0:
            break

        pts = candidates[free_idx]
        cur_c = np.array(centers) if centers else np.empty((0, 2))
        cur_r = np.array(radii)   if radii   else np.empty((0, ))

        rad = feasible_radii(pts, cur_c, cur_r)

        # choose the point that can accommodate the biggest circle
        best = np.argmax(rad)
        best_center = pts[best]
        best_radius = rad[best]

        # guard against numerical zeroes
        if best_radius <= 0:
            best_radius = 1e-8

        centers.append(best_center)
        radii.append(best_radius)
        used[free_idx[best]] = True

    centers = np.asarray(centers, dtype=float)
    radii   = np.asarray(radii,   dtype=float)
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
    # Uncomment to visualize:
    # visualize(centers, radii)