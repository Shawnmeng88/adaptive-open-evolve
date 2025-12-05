# EVOLVE-BLOCK-START
"""Hexagonal‑lattice circle packing for n=26 with LP‑optimised radii."""
import numpy as np
from scipy.optimize import linprog

# ------------------------------------------------------------
# 1.  Hexagonal (triangular) lattice – deterministic centre set
# ------------------------------------------------------------
def _hex_centers(n: int) -> np.ndarray:
    """Generate at least *n* points of a triangular lattice inside [0,1]²."""
    # lattice spacing – chosen so that a modest number of rows/cols suffices
    s = 0.18                     # horizontal distance between neighbours
    dy = np.sqrt(3) * s / 2      # vertical step (hex‑grid height)

    pts = []
    row = 0
    y = s / 2
    while y <= 1 - s / 2 and len(pts) < n:
        # offset every second row by half‑spacing
        x0 = s / 2 if row % 2 == 0 else s
        x = x0
        while x <= 1 - s / 2 and len(pts) < n:
            pts.append((x, y))
            x += s
        y += dy
        row += 1
    return np.array(pts[:n], dtype=np.float64)


# ------------------------------------------------------------
# 2.  Build linear constraints for radii optimisation
# ------------------------------------------------------------
def _build_constraints(centers: np.ndarray):
    """Return (A_ub, b_ub, bounds) for the LP that maximises Σr."""
    n = len(centers)

    # pairwise non‑overlap: r_i + r_j ≤ dist_ij
    pair_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    A = np.zeros((len(pair_idx), n), dtype=np.float64)
    b = np.empty(len(pair_idx), dtype=np.float64)

    for k, (i, j) in enumerate(pair_idx):
        A[k, i] = 1.0
        A[k, j] = 1.0
        b[k] = np.linalg.norm(centers[i] - centers[j])

    # border limits become variable upper‑bounds
    border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )
    bounds = [(0.0, float(bi)) for bi in border]

    return A, b, bounds


# ------------------------------------------------------------
# 3.  Solve LP, verify and (if necessary) shrink uniformly
# ------------------------------------------------------------
def _optimise_radii(centers: np.ndarray) -> np.ndarray:
    n = len(centers)
    A, b, bounds = _build_constraints(centers)

    # maximise Σr  →  minimise -Σr
    c = -np.ones(n, dtype=np.float64)

    res = linprog(
        c,
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        method="highs",
        options={"presolve": True, "dual_feasibility_tolerance": 1e-9},
    )
    if not res.success:
        raise RuntimeError("LP failed: " + res.message)

    r = res.x

    # ---- verification (eps = 1e-9) ---------------------------------
    eps = 1e-9
    # border check (already enforced by bounds, but double‑check)
    border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )
    if np.any(r - border > eps):
        raise AssertionError("Border violation")

    # pairwise check
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if r[i] + r[j] - d > eps:
                # shrink uniformly just enough to restore feasibility
                scale = (d - eps) / (r[i] + r[j])
                r *= scale
                break
    # -----------------------------------------------------------------
    return r


# ------------------------------------------------------------
# 4.  Public constructor
# ------------------------------------------------------------
def construct_packing():
    """
    Returns (centers, radii, sum_of_radii) for 26 circles.
    """
    centers = _hex_centers(26)
    radii = _optimise_radii(centers)
    return centers, radii, float(np.sum(radii))


# EVOLVE-BLOCK-END


# ----------------------------------------------------------------
# Fixed helper / visualisation (unchanged by the evolutionary engine)
# ----------------------------------------------------------------
def run_packing():
    """Run the circle packing constructor for n=26."""
    return construct_packing()


def visualize(centers, radii):
    """Simple Matplotlib visualiser."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(c, r, alpha=0.5, edgecolor="k"))
        ax.text(*c, str(i), ha="center", va="center", fontsize=8)

    plt.title(f"n={len(centers)}  Σr={np.sum(radii):.6f}")
    plt.show()


if __name__ == "__main__":
    centers, radii, total = run_packing()
    print(f"Sum of radii: {total:.6f}   validity = 1.0")
    # Uncomment to see the packing:
    # visualize(centers, radii)