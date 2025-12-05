# EVOLVE-BLOCK-START
"""Improved deterministic hexagonal packing for n=26 circles.

The idea:
* Use a regular hexagonal (triangular) lattice.
* Choose the largest possible uniform spacing `s` such that at least 26 points
  fit inside the unit square **with a margin of s/2** from every side.
* This margin guarantees that each circle of radius `s/2` stays completely
  inside the square and that no two circles overlap (nearest‑neighbour distance
  is exactly `s`).

The algorithm is fully deterministic, uses only NumPy and pure Python,
and respects the required signatures.
"""
import numpy as np

def _hex_grid(spacing: float):
    """
    Generate a hexagonal lattice inside the square [spacing/2, 1‑spacing/2]².

    Parameters
    ----------
    spacing : float
        Desired distance between neighbouring lattice points.

    Returns
    -------
    centers : np.ndarray, shape (m, 2)
        All lattice points that lie completely inside the reduced square.
    """
    margin = spacing / 2.0
    # vertical step for a triangular lattice
    dy = spacing * np.sqrt(3) / 2.0

    # rows start at y = margin and increase by dy while staying inside
    rows_y = []
    y = margin
    while y <= 1.0 - margin + 1e-12:
        rows_y.append(y)
        y += dy

    centers = []
    for row_idx, y in enumerate(rows_y):
        # even rows start at x = margin, odd rows are shifted by spacing/2
        x_start = margin + (spacing / 2.0 if row_idx % 2 else 0.0)
        x = x_start
        while x <= 1.0 - margin + 1e-12:
            centers.append((x, y))
            x += spacing
    return np.array(centers, dtype=np.float64)


def _max_spacing_for_n(target_n: int, eps: float = 1e-4):
    """
    Binary‑search the largest spacing that yields at least `target_n` points.

    Returns
    -------
    spacing : float
        Largest feasible spacing (within `eps`).
    """
    lo, hi = 0.0, 1.0  # spacing cannot exceed the side length
    # Perform a fixed number of iterations to guarantee termination
    for _ in range(30):
        mid = (lo + hi) / 2.0
        pts = _hex_grid(mid)
        if pts.shape[0] >= target_n:
            lo = mid          # feasible – try larger spacing
        else:
            hi = mid          # too large – shrink
        if hi - lo < eps:
            break
    return lo


def construct_packing():
    """
    Construct a deterministic arrangement of 26 circles in the unit square.

    Returns
    -------
    centers : np.ndarray, shape (26, 2)
        (x, y) coordinates of the circle centres.
    radii   : np.ndarray, shape (26,)
        Radius of each circle (all equal).
    sum_of_radii : float
        Sum of all radii – the fitness‑relevant quantity.
    """
    n = 26

    # 1️⃣ Find the largest uniform spacing that can accommodate at least 26 points.
    spacing = _max_spacing_for_n(n, eps=1e-5)

    # 2️⃣ Generate the lattice and keep only the first 26 points (deterministic order).
    all_centers = _hex_grid(spacing)
    centers = all_centers[:n].copy()          # shape (26, 2)

    # 3️⃣ All circles have the same radius = spacing / 2.
    radius = spacing / 2.0
    radii = np.full(n, radius, dtype=np.float64)

    # 4️⃣ Compute the sum of radii (the fitness metric).
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Original helper retained for compatibility.
    It recomputes the radii respecting wall and pairwise constraints,
    but for the deterministic lattice the radii are already optimal.
    """
    n = centers.shape[0]
    # wall distances
    wall = np.minimum.reduce([centers[:, 0],
                              centers[:, 1],
                              1.0 - centers[:, 0],
                              1.0 - centers[:, 1]])
    radii = wall.copy()

    # pairwise reduction – simple iterative clipping
    # (converges quickly because the initial radii are already feasible)
    for _ in range(5):
        dx = centers[:, 0][:, None] - centers[:, 0]
        dy = centers[:, 1][:, None] - centers[:, 1]
        d = np.sqrt(dx ** 2 + dy ** 2) + np.eye(n)  # avoid zero on diagonal
        # enforce r_i + r_j <= d_ij
        excess = radii[:, None] + radii - d
        mask = excess > 0
        # reduce the larger radius of each offending pair
        reduce_i = np.where(mask, excess / 2.0, 0.0)
        radii = np.minimum(radii, radii - reduce_i.max(axis=1))
        radii = np.clip(radii, 0.0, wall)
    return radii


# EVOLVE-BLOCK-END


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
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centers, radii)):
        circ = Circle(c, r, alpha=0.5, edgecolor='k')
        ax.add_patch(circ)
        ax.text(c[0], c[1], str(i), ha="center", va="center", fontsize=8)

    plt.title(f"Hexagonal packing (n={len(centers)}, sum={np.sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # Uncomment to visualise the result:
    # visualize(centers, radii)