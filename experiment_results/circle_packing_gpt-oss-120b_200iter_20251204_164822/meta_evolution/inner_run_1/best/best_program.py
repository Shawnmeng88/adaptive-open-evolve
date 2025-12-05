# EVOLVE-BLOCK-START
"""Hex‑lattice circle packing for n=26 (deterministic, vectorised)"""
import numpy as np
from scipy.spatial import distance

N = 26
SEED = 12345
rng = np.random.default_rng(SEED)


def hex_lattice(n):
    """Generate ≥n points on a triangular lattice inside the unit square."""
    s = 0.2                         # lattice spacing (tuned)
    dy = s * np.sqrt(3) / 2
    pts = []
    r = 0
    while len(pts) < n:
        y = r * dy
        if y > 1:
            break
        offset = 0.5 * s * (r % 2)
        c = 0
        while True:
            x = offset + c * s
            if x > 1:
                break
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append([x, y])
                if len(pts) == n:
                    break
            c += 1
        r += 1
    # deterministic jitter to avoid perfect symmetry
    jitter = rng.uniform(-1e-4, 1e-4, size=(n, 2))
    pts = np.array(pts) + jitter
    return np.clip(pts, 0.0, 1.0)


def compute_radii(centers):
    """Maximum radii for fixed centres (pairwise ½‑distance rule)."""
    side = np.minimum.reduce([centers[:, 0],
                              centers[:, 1],
                              1 - centers[:, 0],
                              1 - centers[:, 1]])
    D = distance.cdist(centers, centers)
    np.fill_diagonal(D, np.inf)
    half_min = 0.5 * D.min(axis=1)
    return np.minimum(side, half_min)


def push_apart(centers, max_iter=200, eps=1e-7):
    """Iteratively separate overlapping circles."""
    alpha = 0.2
    for _ in range(max_iter):
        radii = compute_radii(centers)
        D = distance.cdist(centers, centers)
        np.fill_diagonal(D, np.inf)
        overlap = radii[:, None] + radii[None, :] - D
        mask = overlap > eps
        if not mask.any():
            break
        i_idx, j_idx = np.where(mask)
        # process each overlapping pair once
        processed = set()
        for i, j in zip(i_idx, j_idx):
            if (j, i) in processed:
                continue
            processed.add((i, j))
            ov = overlap[i, j]
            if ov <= 0:
                continue
            direction = centers[j] - centers[i]
            dist = np.linalg.norm(direction)
            if dist == 0:
                direction = rng.normal(size=2)
                dist = np.linalg.norm(direction)
            direction /= dist
            shift = alpha * ov / 2.0 * direction
            centers[i] -= shift
            centers[j] += shift
        centers = np.clip(centers, 0.0, 1.0)
        alpha *= 0.95
    return centers


def construct_packing():
    """Full pipeline – layout → push‑apart → radii → sum."""
    centers = hex_lattice(N)
    centers = push_apart(centers)
    radii = compute_radii(centers)
    sum_radii = radii.sum()
    return centers, radii, sum_radii


# EVOLVE-BLOCK-END


def run_packing():
    """Run the circle packing constructor for n=26."""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """Optional visualiser (unchanged)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(c, r, alpha=0.5))
        ax.text(c[0], c[1], str(i), ha="center", va="center")
    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s:.6f}")
    # visualize(c, r)   # uncomment to see the layout