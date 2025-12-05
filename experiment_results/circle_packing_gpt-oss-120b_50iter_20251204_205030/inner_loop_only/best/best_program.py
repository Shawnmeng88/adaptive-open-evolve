"""
Deterministic circle packing for 26 circles (unit square).

Improvements over the previous version
------------------------------------
* A short deterministic “annealing” phase is added after the push‑apart
  optimisation.  It makes tiny random moves and accepts them either if they
  improve the total admissible radius sum or with a Metropolis probability.
* The random generator is seeded, so the result is reproducible.
* Minor clean‑ups keep the source short while preserving the original API.
"""

import numpy as np

# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
def _hex_lattice(n: int, m: float, s: float) -> np.ndarray:
    """Triangular lattice confined to [m,1‑m]², stopped after *n* points."""
    rows = int(np.ceil((1 - 2 * m) / (s * np.sqrt(3) / 2))) + 1
    pts = []
    for r in range(rows):
        y = m + r * s * np.sqrt(3) / 2
        if y > 1 - m:
            break
        off = 0.5 * (r % 2)
        cols = int(np.ceil((1 - 2 * m) / s)) + 1
        for c in range(cols):
            x = m + (c + off) * s
            if x > 1 - m:
                continue
            pts.append([x, y])
            if len(pts) == n:
                return np.array(pts, dtype=float)
    return np.array(pts[:n], dtype=float)


def _radii(cent: np.ndarray) -> np.ndarray:
    """Admissible radius for each centre."""
    # distance to the four sides
    wall = np.minimum.reduce([cent[:, 0], cent[:, 1], 1 - cent[:, 0], 1 - cent[:, 1]])

    # half the distance to the nearest neighbour
    d = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    neigh = d.min(axis=1) / 2.0

    return np.minimum(wall, neigh)


def _push_apart(cent: np.ndarray, step: float = 0.002, it: int = 12) -> np.ndarray:
    """Deterministic local optimisation – move each point away from its nearest neighbour."""
    for _ in range(it):
        d = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        nn = d.argmin(axis=1)
        vec = cent - cent[nn]
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        cand = np.clip(cent + step * vec / norm, 0.0, 1.0)
        if _radii(cand).sum() > _radii(cent).sum():
            cent = cand
    return cent


def _anneal(cent: np.ndarray, rng: np.random.Generator,
            steps: int = 1200, T0: float = 5e-3, Tf: float = 1e-4) -> np.ndarray:
    """Very short deterministic simulated‑annealing phase."""
    best, best_score = cent.copy(), _radii(cent).sum()
    cur, cur_score = best.copy(), best_score
    temps = np.geomspace(T0, Tf, steps)

    for T in temps:
        i = rng.integers(len(cur))
        delta = (rng.random(2) - 0.5) * 0.04          # ±0.02 in each direction
        cand = cur.copy()
        cand[i] = np.clip(cand[i] + delta, 0.0, 1.0)

        cand_score = _radii(cand).sum()
        if cand_score > cur_score or rng.random() < np.exp((cand_score - cur_score) / T):
            cur, cur_score = cand, cand_score
            if cur_score > best_score:
                best, best_score = cur.copy(), cur_score
    return best


# ----------------------------------------------------------------------
# Constructor
# ----------------------------------------------------------------------
def construct_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Return (centres, radii, sum_of_radii) for the best layout found."""
    n = 26
    margins = (0.04, 0.05, 0.06, 0.07, 0.08)
    spacings = (0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22)

    # 1️⃣ coarse grid → best deterministic layout
    best_cent, best_score = None, -1.0
    for m in margins:
        for s in spacings:
            pts = _push_apart(_hex_lattice(n, m, s))
            sc = _radii(pts).sum()
            if sc > best_score:
                best_cent, best_score = pts, sc

    # 2️⃣ short deterministic annealing from the best start
    rng = np.random.default_rng(42)          # reproducible seed
    best_cent = _anneal(best_cent, rng)

    rad = _radii(best_cent)
    return best_cent, rad, float(rad.sum())


def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Public API – returns (centres, radii, sum_of_radii)."""
    return construct_packing()


# ----------------------------------------------------------------------
# Optional visualisation
# ----------------------------------------------------------------------
def visualize(cent: np.ndarray, rad: np.ndarray) -> None:
    """Simple Matplotlib visualisation (optional)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    for i, (c, r) in enumerate(zip(cent, rad)):
        ax.add_patch(Circle(c, r, edgecolor="C0", facecolor="C0", alpha=0.4))
        ax.text(*c, str(i), ha="center", va="center", fontsize=8)

    plt.title(f"Sum of radii = {rad.sum():.4f}")
    plt.show()


# ----------------------------------------------------------------------
# Entry‑point for manual testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    centres, radii, total = run_packing()
    print(f"Sum of radii: {total:.6f}")
    # visualize(centres, radii)   # uncomment to see the layout