"""Compact 26‑circle packing with a cheap stochastic refinement.

Public API (unchanged):
    construct_packing() → (centers, radii, sum_of_radii)
"""

from __future__ import annotations
import random
import numpy as np
from itertools import combinations

# ----------------------------------------------------------------------
# Optional linear‑programming backend (SciPy)
# ----------------------------------------------------------------------
try:                                 # fast exact optimiser if present
    from scipy.optimize import linprog
    _HAS_LP = True
except Exception:                   # pragma: no cover
    _HAS_LP = False

# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
def _hex_grid(n: int, s: float, rot: float = 0.0) -> np.ndarray:
    """Triangular lattice (spacing *s*) optionally rotated by *rot* radians."""
    dy = s * np.sqrt(3) / 2
    pts, y, row = [], s / 2, 0
    while y < 1 and len(pts) < n:
        off = 0 if row % 2 == 0 else s / 2
        x = s / 2 + off
        while x < 1 and len(pts) < n:
            pts.append([x, y])
            x += s
        y += dy
        row += 1
    pts = np.array(pts[:n])
    # centre‑pad if the lattice is too short
    if pts.shape[0] < n:
        extra = np.full((n - pts.shape[0], 2), 0.5)
        pts = np.vstack([pts, extra])
    # rotate about centre (0.5,0.5)
    if rot:
        c = pts - 0.5
        cs, sn = np.cos(rot), np.sin(rot)
        rot_m = np.array([[cs, -sn], [sn, cs]])
        pts = (c @ rot_m) + 0.5
    # keep a safe distance from the walls
    return np.clip(pts, 0.01, 0.99)


def _border_limits(c: np.ndarray) -> np.ndarray:
    """Maximum admissible radius for each centre limited by the four walls."""
    return np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])


# ----------------------------------------------------------------------
# Radius optimisation
# ----------------------------------------------------------------------
def _lp_optimize(c: np.ndarray) -> np.ndarray:
    """Linear‑programming maximise Σr under border & pairwise constraints."""
    n = len(c)
    b = _border_limits(c)

    rows, rhs = [], []
    for i, j in combinations(range(n), 2):
        row = np.zeros(n)
        row[i] = row[j] = 1.0
        rows.append(row)
        rhs.append(np.linalg.norm(c[i] - c[j]))
    A, ub = (np.array(rows), np.array(rhs)) if rows else (None, None)

    res = linprog(-np.ones(n), A_ub=A, b_ub=ub,
                  bounds=[(0, bi) for bi in b],
                  method="highs", options={"presolve": True})
    return res.x if res.success else b


def _heur_optimize(c: np.ndarray) -> np.ndarray:
    """Deterministic pairwise scaling – always feasible and very fast."""
    r = _border_limits(c).copy()
    for i, j in combinations(range(len(c)), 2):
        d = np.linalg.norm(c[i] - c[j])
        if r[i] + r[j] > d:
            s = d / (r[i] + r[j])
            r[i] *= s
            r[j] *= s
    return r


def _opt_radii(c: np.ndarray) -> np.ndarray:
    """Pick the best available optimiser (LP if available, else heuristic)."""
    return _lp_optimize(c) if _HAS_LP else _heur_optimize(c)


# ----------------------------------------------------------------------
# Stochastic refinement
# ----------------------------------------------------------------------
def _stochastic_refine(start: np.ndarray,
                       iters: int = 1500,
                       step: float = 0.03,
                       seed: int | None = None) -> np.ndarray:
    """Random walk that keeps the layout with the highest Σr."""
    rng = random.Random(seed)
    best = start.copy()
    best_val = _opt_radii(best).sum()

    for _ in range(iters):
        # pick a circle and propose a small move
        i = rng.randrange(len(best))
        cand = best.copy()
        dx = (rng.random() * 2 - 1) * step
        dy = (rng.random() * 2 - 1) * step
        cand[i] += (dx, dy)
        cand[i] = np.clip(cand[i], 0.01, 0.99)

        val = _opt_radii(cand).sum()
        if val > best_val:
            best, best_val = cand, val
    return best


# ----------------------------------------------------------------------
# Layout search – coarse sweep + stochastic polishing
# ----------------------------------------------------------------------
def _best_layout(n: int) -> np.ndarray:
    """Return a centre layout with a large Σr after a cheap global search."""
    # sweep spacing and a few rotation angles
    spacings = np.linspace(0.14, 0.22, 9)
    rotations = np.linspace(0, np.pi / 6, 5)          # 0° … 30°
    best, best_sum = None, -1.0

    for s in spacings:
        for rot in rotations:
            pts = _hex_grid(n, s, rot)
            rad = _opt_radii(pts)
            total = rad.sum()
            if total > best_sum:
                best, best_sum = pts, total

    # a short stochastic polish (fast heuristic runs)
    best = _stochastic_refine(best, iters=1200, step=0.025)
    return best


# ----------------------------------------------------------------------
# Public constructor
# ----------------------------------------------------------------------
def construct_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Return (centres, radii, sum_of_radii) for the 26‑circle problem."""
    centres = _best_layout(26)
    radii = _opt_radii(centres)
    return centres, radii, float(radii.sum())


# ----------------------------------------------------------------------
# Helper / entry‑point (unchanged API)
# ----------------------------------------------------------------------
def run_packing():
    """Convenient entry‑point used by the evaluation harness."""
    return construct_packing()


def visualize(centers: np.ndarray, radii: np.ndarray):
    """Optional Matplotlib visualisation (not required for scoring)."""
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
    plt.title(f"Stochastic packing – sum = {radii.sum():.4f}")
    plt.show()


if __name__ == "__main__":
    cs, rs, total = run_packing()
    print(f"Sum of radii: {total:.6f}")
    # visualize(cs, rs)   # uncomment to see the layout