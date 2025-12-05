# EVOLVE-BLOCK-START
"""Deterministic hexagonal‑lattice packing for n=26 circles."""
import numpy as np

N = 26                     # number of circles
SQ = np.eye(2)             # dummy to keep imports deterministic


def _max_spacing(n: int) -> float:
    """Largest lattice spacing s so that at least n points fit
    inside the unit square with a margin of s/2 from every side."""
    def count(s: float) -> int:
        dy = s * np.sqrt(3) / 2
        y0 = s / 2
        pts = 0
        r = 0
        while y0 + r * dy <= 1 - s / 2:
            offset = (r % 2) * s / 2
            x0 = s / 2 + offset
            pts += int((1 - s / 2 - x0) // s) + 1
            r += 1
        return pts

    lo, hi = 0.0, 1.0
    for _ in range(50):                # binary search, deterministic
        mid = (lo + hi) / 2
        if count(mid) >= n:
            lo = mid
        else:
            hi = mid
    return lo


def _generate_lattice(s: float, n: int) -> np.ndarray:
    """Generate the first n lattice points (lexicographic order)."""
    dy = s * np.sqrt(3) / 2
    pts = []
    r = 0
    while True:
        y = s / 2 + r * dy
        if y > 1 - s / 2:
            break
        offset = (r % 2) * s / 2
        x = s / 2 + offset
        while x <= 1 - s / 2 + 1e-12:
            pts.append((x, y))
            if len(pts) == n:
                return np.array(pts)
            x += s
        r += 1
    # safety – should never happen
    return np.array(pts[:n])


def compute_max_radii(centers: np.ndarray) -> np.ndarray:
    """Vectorised radius limits: min(boundary, ½·nearest‑neighbour)."""
    # distance to the four sides
    border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )
    # pairwise centre distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)          # ignore self‑distance
    nearest = np.min(dists, axis=1) / 2.0
    return np.minimum(border, nearest)


def construct_packing():
    """Return (centers, radii, sum_of_radii) for the optimal lattice."""
    s = _max_spacing(N)                     # optimal spacing
    centers = _generate_lattice(s, N)       # deterministic layout
    radii = compute_max_radii(centers)      # should be s/2 for all
    # ----- safety check (will raise if invalid) -----
    dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    assert np.all(dmat >= radii[:, None] + radii[None, :] - 1e-12)
    assert np.all((centers - radii[:, None] >= -1e-12) &
                  (centers + radii[:, None] <= 1 + 1e-12))
    # -------------------------------------------------
    return centers, radii, float(np.sum(radii))


# EVOLVE-BLOCK-END


def run_packing():
    """Run the circle packing constructor for n=26."""
    return construct_packing()


def visualize(centers, radii):
    """Optional visualisation (uses matplotlib)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(c, r, alpha=0.4))
        ax.text(*c, str(i), ha="center", va="center")
    plt.show()


if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s:.6f}")
    # visualize(c, r)   # uncomment to see the packing