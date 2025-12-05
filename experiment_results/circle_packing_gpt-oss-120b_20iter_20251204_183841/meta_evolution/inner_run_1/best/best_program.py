# EVOLVE-BLOCK-START
"""
Hexagonal‑lattice circle packing for n = 26.
Positions are fixed, radii are obtained by a global linear program
that maximises the total sum while respecting all distance and border
constraints.  A tiny safety margin (eps) guarantees validity = 1.0.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from itertools import combinations
from typing import Tuple

_EPS = 1e-8          # safety margin inside every inequality
_TOL = 1e-7          # tolerance for the final verification


def _hex_lattice(n: int = 26, spacing: float = 0.18) -> np.ndarray:
    """Generate a hexagonal lattice inside the unit square and keep the first *n* points."""
    s = spacing
    dy = np.sqrt(3) / 2 * s
    margin = s / 2
    pts: list[Tuple[float, float]] = []
    row = 0
    y = margin
    while y <= 1 - margin and len(pts) < n:
        offset = (s / 2) if (row % 2) else 0.0
        x = margin + offset
        while x <= 1 - margin and len(pts) < n:
            pts.append((x, y))
            x += s
        row += 1
        y += dy
    return np.asarray(pts[:n])


def _build_constraints(centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create A_ub, b_ub for the LP:  r_i + r_j ≤ d_ij - eps  and border constraints."""
    n = len(centers)
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    # pairwise distance constraints
    for i, j in combinations(range(n), 2):
        d = np.linalg.norm(centers[i] - centers[j])
        a = np.zeros(n)
        a[i] = a[j] = 1.0
        rows.append(a)
        rhs.append(d - _EPS)

    # border constraints: left, right, bottom, top
    for i, (x, y) in enumerate(centers):
        for bound in (x, 1 - x, y, 1 - y):
            a = np.zeros(n)
            a[i] = 1.0
            rows.append(a)
            rhs.append(bound - _EPS)

    A_ub = np.vstack(rows)
    b_ub = np.array(rhs)
    return A_ub, b_ub


def _solve_lp(centers: np.ndarray) -> np.ndarray:
    """Solve the LP and return the radii; fall back to scaling if infeasible."""
    n = len(centers)
    c = -np.ones(n)                     # maximise Σ r_i  → minimise –Σ r_i
    A_ub, b_ub = _build_constraints(centers)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method="highs")
    if not res.success:
        # a feasible point always exists (all radii = 0); use it
        radii = np.zeros(n)
    else:
        radii = res.x

    # post‑process: if any constraint is slightly violated, scale uniformly
    if not _verify_solution(centers, radii):
        radii = _scale_to_feasibility(centers, radii)

    return radii


def _verify_solution(centers: np.ndarray, radii: np.ndarray) -> bool:
    """Return True if every constraint holds within _TOL."""
    n = len(centers)
    # border checks
    for i, (x, y) in enumerate(centers):
        if radii[i] - (x - _TOL) > 0 or radii[i] - (1 - x - _TOL) > 0:
            return False
        if radii[i] - (y - _TOL) > 0 or radii[i] - (1 - y - _TOL) > 0:
            return False
    # pairwise checks
    for i, j in combinations(range(n), 2):
        d = np.linalg.norm(centers[i] - centers[j])
        if radii[i] + radii[j] - (d + _TOL) > 0:
            return False
    return True


def _scale_to_feasibility(centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Uniformly shrink radii so that all constraints become satisfied."""
    n = len(centers)
    factors = [1.0]

    # border factors
    for i, (x, y) in enumerate(centers):
        if radii[i] > 0:
            factors.append((x - _EPS) / radii[i])
            factors.append(((1 - x) - _EPS) / radii[i])
            factors.append((y - _EPS) / radii[i])
            factors.append(((1 - y) - _EPS) / radii[i])

    # pairwise factors
    for i, j in combinations(range(n), 2):
        d = np.linalg.norm(centers[i] - centers[j])
        if radii[i] + radii[j] > 0:
            factors.append((d - _EPS) / (radii[i] + radii[j]))

    alpha = min(factors)
    return radii * alpha * (1 - _EPS)   # extra tiny shrink for safety


def construct_packing() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build the 26‑circle packing.
    Returns
    -------
    centers : (26, 2) array of (x, y) coordinates
    radii   : (26,)   array of radii (valid, summed to maximal value)
    sum_radii : float, Σ radii
    """
    centers = _hex_lattice()
    radii = _solve_lp(centers)
    return centers, radii, float(radii.sum())


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

    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # visualize(centers, radii)