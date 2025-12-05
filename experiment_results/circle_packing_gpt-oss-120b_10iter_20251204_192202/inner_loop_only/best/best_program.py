# EVOLVE-BLOCK-START
"""Stochastic optimisation of 26‑circle packing in a unit square."""
import numpy as np

def _max_radii(pts: np.ndarray) -> np.ndarray:
    """Largest feasible radii for given centres."""
    rad = np.minimum.reduce([pts[:, 0], pts[:, 1], 1 - pts[:, 0], 1 - pts[:, 1]])
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if rad[i] + rad[j] > d:
                s = d / (rad[i] + rad[j])
                rad[i] *= s
                rad[j] *= s
    return rad


def _initial_layout() -> np.ndarray:
    """Deterministic, well‑spread start (5×5 grid + centre)."""
    rng = np.random.default_rng(0)
    g = np.linspace(0.1, 0.9, 5)
    xv, yv = np.meshgrid(g, g)
    pts = np.column_stack([xv.ravel(), yv.ravel()])[:25]
    pts = np.vstack([pts, [0.5, 0.5]])
    pts += rng.uniform(-0.02, 0.02, pts.shape)   # tiny jitter
    return pts


def _hill_climb(pts: np.ndarray, iters: int = 6000) -> np.ndarray:
    """Stochastic hill‑climbing to maximise the sum of radii."""
    rng = np.random.default_rng()
    best = pts.copy()
    best_sum = _max_radii(best).sum()
    for _ in range(iters):
        i = rng.integers(len(best))
        cand = best.copy()
        cand[i] = np.clip(cand[i] + rng.normal(scale=0.03, size=2), 0.01, 0.99)
        s = _max_radii(cand).sum()
        if s > best_sum:
            best, best_sum = cand, s
    return best


def construct_packing():
    """Return centres, radii and total sum of radii for 26 circles."""
    centres = _hill_climb(_initial_layout())
    radii = _max_radii(centres)
    return centres, radii, float(radii.sum())

# EVOLVE-BLOCK-END


def run_packing():
    """Convenient wrapper used by the unchanged driver code."""
    return construct_packing()


def visualize(centers, radii):
    """Optional Matplotlib visualisation – unchanged from the original."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(c, r, alpha=0.5))
        ax.text(*c, str(i), ha="center", va="center")
    plt.title(f"Circle Packing (n={len(centers)}, sum={radii.sum():.6f})")
    plt.show()


if __name__ == "__main__":
    centres, radii, total = run_packing()
    print(f"Sum of radii: {total:.6f}")
    # Uncomment to see the packing:
    # visualize(centres, radii)