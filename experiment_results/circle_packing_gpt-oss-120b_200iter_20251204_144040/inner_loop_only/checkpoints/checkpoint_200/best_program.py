"""Improved 26‑circle packing – tighter search + three‑stage polish.

Public API (unchanged):
    construct_packing() → (centers, radii, sum_of_radii)
"""

from __future__ import annotations
import math, random
from itertools import combinations
import numpy as np

# ----------------------------------------------------------------------
# optional exact optimiser (SciPy)
# ----------------------------------------------------------------------
try:                       # fast LP if SciPy is available
    from scipy.optimize import linprog
    _USE_LP = True
except Exception:         # pragma: no cover
    _USE_LP = False

# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------
def _grid(n: int, s: float, rot: float = 0.0) -> np.ndarray:
    """Triangular lattice (spacing *s*) rotated by *rot* rad."""
    dy = s * math.sqrt(3) / 2
    pts, y, row = [], s / 2, 0
    while y < 1 and len(pts) < n:
        off = 0.0 if row % 2 == 0 else s / 2
        x = s / 2 + off
        while x < 1 and len(pts) < n:
            pts.append([x, y])
            x += s
        y += dy
        row += 1
    pts = np.array(pts[:n])
    if pts.shape[0] < n:                     # centre‑pad if short
        pts = np.vstack([pts, np.full((n - pts.shape[0], 2), 0.5)])
    if rot:
        c = pts - 0.5
        cs, sn = math.cos(rot), math.sin(rot)
        pts = (c @ np.array([[cs, -sn], [sn, cs]])) + 0.5
    return np.clip(pts, 0.01, 0.99)


def _border(c: np.ndarray) -> np.ndarray:
    """Maximum admissible radius for each centre limited by the walls."""
    return np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])

# ----------------------------------------------------------------------
# radius optimisation
# ----------------------------------------------------------------------
def _lp_opt(c: np.ndarray) -> np.ndarray:
    n = len(c)
    b = _border(c)
    rows, rhs = [], []
    for i, j in combinations(range(n), 2):
        row = np.zeros(n)
        row[i] = row[j] = 1.0
        rows.append(row)
        rhs.append(np.linalg.norm(c[i] - c[j]))
    A = np.array(rows) if rows else None
    ub = np.array(rhs) if rhs else None
    res = linprog(-np.ones(n), A_ub=A, b_ub=ub,
                  bounds=[(0, bi) for bi in b],
                  method="highs", options={"presolve": True})
    return res.x if res.success else b


def _heur_opt(c: np.ndarray) -> np.ndarray:
    r = _border(c).copy()
    for i, j in combinations(range(len(c)), 2):
        d = np.linalg.norm(c[i] - c[j])
        if r[i] + r[j] > d:
            s = d / (r[i] + r[j])
            r[i] *= s
            r[j] *= s
    return r


_opt_radii = _lp_opt if _USE_LP else _heur_opt

# ----------------------------------------------------------------------
# simulated‑annealing style refinement
# ----------------------------------------------------------------------
def _anneal(start: np.ndarray,
            iters: int,
            step: float,
            decay: float,
            seed: int | None = None) -> np.ndarray:
    rng = random.Random(seed)
    best = cur = start.copy()
    best_val = cur_val = _opt_radii(best).sum()
    for _ in range(iters):
        i = rng.randrange(len(cur))
        cand = cur.copy()
        cand[i] += (rng.random() * 2 - 1) * step, (rng.random() * 2 - 1) * step
        cand[i] = np.clip(cand[i], 0.01, 0.99)

        val = _opt_radii(cand).sum()
        # accept if better; occasional uphill move keeps diversity
        if val > cur_val or rng.random() < 0.001:
            cur, cur_val = cand, val
            if val > best_val:
                best, best_val = cand, val
        step *= decay
    return best


# ----------------------------------------------------------------------
# tiny deterministic hill‑climb – final polish
# ----------------------------------------------------------------------
def _local_refine(layout: np.ndarray,
                  delta: float = 0.008,
                  passes: int = 3) -> np.ndarray:
    """Move each centre by ±δ while it improves the radius sum."""
    best = layout.copy()
    best_val = _opt_radii(best).sum()
    for _ in range(passes):
        improved = False
        for i in range(len(best)):
            for dx, dy in ((delta, 0), (-delta, 0), (0, delta), (0, -delta)):
                cand = best.copy()
                cand[i] += (dx, dy)
                cand[i] = np.clip(cand[i], 0.01, 0.99)
                val = _opt_radii(cand).sum()
                if val > best_val + 1e-12:
                    best, best_val = cand, val
                    improved = True
        if not improved:
            break
    return best

# ----------------------------------------------------------------------
# layout search – dense sweep + three‑stage polish
# ----------------------------------------------------------------------
def _best_layout(n: int) -> np.ndarray:
    # very fine sweep (more points than previous versions)
    spacings = np.linspace(0.125, 0.245, 23)       # 23 spacings
    rotations = np.linspace(0.0, math.pi / 4, 15) # 15 rotations
    best, best_val = None, -1.0
    for s in spacings:
        for r in rotations:
            pts = _grid(n, s, r)
            val = _opt_radii(pts).sum()
            if val > best_val:
                best, best_val = pts, val

    # three‑stage annealing (coarse → medium → fine)
    best = _anneal(best, iters=1500, step=0.036, decay=0.998, seed=41)
    best = _anneal(best, iters=2000, step=0.018, decay=0.9995, seed=99)
    best = _anneal(best, iters=1200, step=0.009, decay=0.9999, seed=123)

    # final deterministic tweak
    best = _local_refine(best, delta=0.006, passes=2)
    return best

# ----------------------------------------------------------------------
# public constructor
# ----------------------------------------------------------------------
def construct_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Return (centres, radii, sum_of_radii) for the 26‑circle problem."""
    centres = _best_layout(26)
    radii = _opt_radii(centres)
    return centres, radii, float(radii.sum())


def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Entry‑point used by the evaluation harness."""
    return construct_packing()


if __name__ == "__main__":
    cs, rs, total = run_packing()
    print(f"Sum of radii: {total:.6f}")