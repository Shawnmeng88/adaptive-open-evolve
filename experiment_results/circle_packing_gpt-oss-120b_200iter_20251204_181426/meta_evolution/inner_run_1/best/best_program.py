# EVOLVE-BLOCK-START
"""Deterministic hex‑lattice + tiny local search for n=26 circles."""
import numpy as np

def compute_max_radii(centers: np.ndarray) -> np.ndarray:
    """Maximum radii limited by square edges and other circles (vectorised)."""
    edge = np.minimum(centers, 1 - centers).min(axis=1)
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    pair = d.min(axis=1) / 2.0
    radii = np.minimum(edge, pair)
    # strict validity checks
    assert np.all(radii >= 0), "negative radius"
    assert np.all(centers >= 0) and np.all(centers <= 1), "centre out of unit square"
    assert np.all(d >= (radii[:, None] + radii[None, :]) - 1e-12), "overlap detected"
    return radii

def construct_packing():
    """Create 26 centres, then improve sum‑of‑radii with a deterministic offset search."""
    n = 26
    rows = int(np.ceil(np.sqrt(2 * n / np.sqrt(3))))          # rows for a dense triangular lattice
    s = 1.0 / rows                                            # horizontal spacing
    pts = []
    for r in range(rows):
        y = (r + 0.5) * s * np.sqrt(3) / 2.0
        if y > 1 - 1e-12:
            break
        cols = rows + (r % 2)
        for c in range(cols):
            x = (c + 0.5 + 0.5 * (r % 2)) * s
            if x > 1 - 1e-12:
                continue
            pts.append([x, y])
            if len(pts) == n:
                break
        if len(pts) == n:
            break

    centers = np.clip(np.array(pts[:n]), 0.0, 1.0)
    radii = compute_max_radii(centers)
    cur_sum = radii.sum()

    # deterministic local search – fixed offsets, two full passes
    h = s / 20.0
    offsets = np.array([[h, 0], [-h, 0], [0, h], [0, -h]])
    for _ in range(2):                         # two passes
        for i in range(n):
            best_sum, best_center, best_radii = cur_sum, centers[i].copy(), radii
            for off in offsets:
                cand = centers[i] + off
                if np.any(cand < 0) or np.any(cand > 1):
                    continue
                cand_centers = centers.copy()
                cand_centers[i] = cand
                try:
                    cand_radii = compute_max_radii(cand_centers)
                except AssertionError:
                    continue
                cand_sum = cand_radii.sum()
                if cand_sum > best_sum + 1e-12:
                    best_sum, best_center, best_radii = cand_sum, cand, cand_radii
            if best_sum > cur_sum + 1e-12:
                centers[i] = best_center
                radii = best_radii
                cur_sum = best_sum

    # final safety recompute
    radii = compute_max_radii(centers)
    return centers, radii, radii.sum()

# EVOLVE-BLOCK-END

def run_packing():
    """Run the constructor for n=26."""
    return construct_packing()

def visualize(centers, radii):
    """Optional visualisation (requires matplotlib)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(c, r, alpha=0.5, edgecolor='k'))
        ax.text(*c, str(i), ha='center', va='center')
    plt.title(f"n={len(centers)}  sum radii={np.sum(radii):.6f}")
    plt.show()

if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s:.6f}")
    # visualize(c, r)   # uncomment to see the packing