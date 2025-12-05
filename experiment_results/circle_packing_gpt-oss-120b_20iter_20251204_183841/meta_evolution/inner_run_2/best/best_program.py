"""Hexagonal‑lattice circle packing for n=26 circles (LP‑optimised radii)"""

from __future__ import annotations

import sys
from typing import Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


def _hex_lattice_points(n: int = 26, spacing: float = 0.16) -> np.ndarray:
    """
    Generate the first ``n`` points of a triangular (hexagonal) lattice
    that lie inside the unit square.

    The lattice uses a horizontal spacing ``spacing`` and a vertical
    spacing ``spacing * sqrt(3) / 2``.  Points are taken row‑by‑row
    until ``n`` points are collected.

    Returns
    -------
    centers : np.ndarray, shape (n, 2)
        (x, y) coordinates of the circles.
    """
    pts: list[Tuple[float, float]] = []
    dy = spacing * np.sqrt(3) / 2
    y = spacing / 2
    row = 0
    while len(pts) < n and y <= 1 - spacing / 2 + 1e-12:
        # offset every other row by half a horizontal step
        offset = 0.0 if row % 2 == 0 else spacing / 2
        x = offset + spacing / 2
        while x <= 1 - spacing / 2 + 1e-12:
            pts.append((x, y))
            if len(pts) >= n:
                break
            x += spacing
        row += 1
        y += dy
    return np.array(pts[:n], dtype=float)


def _build_pairwise_constraints(centers: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
    """
    Build the sparse matrix ``A_ub`` and vector ``b_ub`` that encode
    the non‑overlap constraints   r_i + r_j ≤ dist(i, j)   for all i < j.

    Parameters
    ----------
    centers : np.ndarray, shape (n, 2)

    Returns
    -------
    A_ub : scipy.sparse.csr_matrix, shape (m, n)
    b_ub : np.ndarray, shape (m,)
    """
    n = centers.shape[0]
    # number of pairwise constraints
    m = n * (n - 1) // 2
    data = np.ones(2 * m, dtype=float)          # each row has two ones
    row_ind = np.empty(2 * m, dtype=int)
    col_ind = np.empty(2 * m, dtype=int)
    b = np.empty(m, dtype=float)

    idx = 0
    for i in range(n):
        ci = centers[i]
        for j in range(i + 1, n):
            # distance between centres
            d = np.linalg.norm(ci - centers[j])
            b[idx] = d
            # row ``idx`` has coefficient 1 for i and 1 for j
            row_ind[2 * idx] = idx
            col_ind[2 * idx] = i
            row_ind[2 * idx + 1] = idx
            col_ind[2 * idx + 1] = j
            idx += 1

    A_ub = csr_matrix((data, (row_ind, col_ind)), shape=(m, n))
    return A_ub, b


def _border_limits(centers: np.ndarray) -> np.ndarray:
    """
    Maximum radius of a circle centred at ``centers`` limited only by the
    unit‑square borders.
    """
    x = centers[:, 0]
    y = centers[:, 1]
    return np.minimum.reduce([x, y, 1 - x, 1 - y])


def _verify_solution(
    radii: np.ndarray,
    centers: np.ndarray,
    eps: float = 1e-12,
) -> bool:
    """
    Check that a set of radii respects all geometric constraints.
    Returns ``True`` if the solution is feasible, otherwise ``False``.
    """
    # border check
    border = _border_limits(centers)
    if not np.all(radii <= border + eps):
        return False

    # pairwise check
    diff = centers[:, None, :] - centers[None, :, :]          # (n, n, 2)
    dists = np.sqrt(np.sum(diff ** 2, axis=2))               # (n, n)
    # only upper triangle needed
    i_upper, j_upper = np.triu_indices_from(dists, k=1)
    if not np.all(radii[i_upper] + radii[j_upper] <= dists[i_upper, j_upper] + eps):
        return False

    return True


def construct_packing() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build a 26‑circle packing inside the unit square.

    Returns
    -------
    centers : np.ndarray, shape (26, 2)
        Circle centres.
    radii   : np.ndarray, shape (26,)
        Optimised radii (guaranteed feasible).
    sum_of_radii : float
        Sum of all radii (the optimisation objective).
    """
    n = 26
    centers = _hex_lattice_points(n)

    # ---- linear‑programming formulation ------------------------------------
    # maximise sum(r)  <=> minimise -sum(r)
    c = -np.ones(n, dtype=float)

    # pairwise non‑overlap constraints
    A_ub, b_ub = _build_pairwise_constraints(centers)

    # upper bounds from the square borders (tighten the feasible region)
    border = _border_limits(centers)
    bounds = [(0.0, float(b)) for b in border]

    # solve
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
        options={"presolve": True, "time_limit": 30},
    )
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    radii = res.x

    # tiny uniform down‑scale to guarantee strict feasibility under floating‑point noise
    delta = 1e-12
    radii *= (1.0 - delta)

    # ---- verification -------------------------------------------------------
    if not _verify_solution(radii, centers, eps=1e-12):
        raise RuntimeError("Verification failed – solution is not feasible.")

    sum_of_radii = float(np.sum(radii))
    return centers, radii, sum_of_radii


# ---------------------------------------------------------------------------
# The following helper functions are unchanged from the original template.
# ---------------------------------------------------------------------------

def run_packing() -> Tuple[np.ndarray, np.ndarray, float]:
    """Run the circle packing constructor for n=26."""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers: np.ndarray, radii: np.ndarray) -> None:
    """
    Visualise the circle packing.

    Parameters
    ----------
    centers : np.ndarray, shape (n, 2)
    radii   : np.ndarray, shape (n,)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centers, radii)):
        circ = Circle(c, r, alpha=0.5, edgecolor="k")
        ax.add_patch(circ)
        ax.text(c[0], c[1], str(i), ha="center", va="center", fontsize=8)

    plt.title(f"Circle Packing (n={len(centers)}, sum={np.sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    # Simple sanity check when the module is executed directly.
    try:
        centers, radii, total = run_packing()
        print(f"Sum of radii: {total:.6f}")
    except Exception as exc:
        print(f"Error during packing: {exc}", file=sys.stderr)
        sys.exit(1)

    # Uncomment to see a plot:
    # visualize(centers, radii)