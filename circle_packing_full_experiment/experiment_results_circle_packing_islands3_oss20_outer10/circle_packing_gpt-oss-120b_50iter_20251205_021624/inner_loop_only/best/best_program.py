# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def construct_packing():
    pts = np.linspace(0.1, 0.9, 5)
    xv, yv = np.meshgrid(pts, pts)
    centers = np.column_stack((xv.ravel(), yv.ravel()))
    if centers.shape[0] < 26:
        rng = np.random.default_rng(0)
        extra = rng.random((26 - centers.shape[0], 2)) * 0.8 + 0.1
        centers = np.vstack((centers, extra))
    centers = centers[:26]
    radii = compute_max_radii_lp(centers)
    return centers, radii, radii.sum()

def compute_max_radii_lp(c):
    n = c.shape[0]
    border = np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))

    c_vec = -np.ones(n)
    A_blocks = [np.eye(n)]
    b_blocks = [border]

    rows, vals = [], []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = row[j] = 1
            rows.append(row)
            vals.append(dists[i, j])
    if rows:
        A_blocks.append(np.vstack(rows))
        b_blocks.append(np.array(vals))

    A_ub = np.vstack(A_blocks)
    b_ub = np.concatenate(b_blocks)

    res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, None)] * n, method='highs')
    if res.success:
        return res.x

    np.fill_diagonal(dists, np.inf)
    nearest = np.min(dists, axis=1) / 2.0
    return np.minimum(border, nearest)
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
