# EVOLVE-BLOCK-START
"""Optimised deterministic hexagonal‑lattice packing for 26 circles.

Key ideas
---------
* The spacing `s` between neighbouring centres is *maximised* while still
  allowing at least `n` points to fit inside the unit square.
  This yields the largest possible uniform radius `r = s/2 – eps`.
* A tiny safety epsilon is sub‑tracted from every limit to keep the
  constraints strictly satisfied.
* All geometry is performed with NumPy broadcasting – no Python loops over
  point pairs are used, guaranteeing O(n²) checks only inside the
  validation step.
* The public API (`construct_packing`, `compute_max_radii`) is unchanged.
"""

import numpy as np

# ----------------------------------------------------------------------
# Helper: binary‑search the largest feasible hex‑lattice spacing.
# ----------------------------------------------------------------------
def _max_spacing(n: int, eps: float = 1e-12) -> float:
    """Return the greatest lattice spacing `s` that yields ≥ n points.

    The function is deterministic: it performs a fixed‑iteration binary
    search (30 iterations → sub‑nanometer precision for the unit square).
    """
    sqrt3 = np.sqrt(3.0)

    def _count_points(s: float) -> int:
        """Number of lattice points that fit for a given spacing `s`."""
        if s <= 0:
            return 0
        row_step = sqrt3 / 2.0 * s
        y = s / 2.0
        cnt = 0
        i = 0
        while y <= 1.0 - s / 2.0 + eps:
            offset = (i % 2) * s / 2.0
            x = s / 2.0 + offset
            while x <= 1.0 - s / 2.0 + eps:
                cnt += 1
                x += s
            y += row_step
            i += 1
        return cnt

    lo, hi = 0.0, 1.0          # spacing cannot exceed the unit length
    for _ in range(30):        # enough for double precision
        mid = (lo + hi) / 2.0
        if _count_points(mid) >= n:
            lo = mid            # spacing works – try larger
        else:
            hi = mid            # too large – shrink
    return lo


# ----------------------------------------------------------------------
# Helper: generate the deterministic hexagonal lattice for the chosen `s`.
# ----------------------------------------------------------------------
def _hex_lattice(n: int, eps: float = 1e-12) -> np.ndarray:
    """Return `n` centre coordinates on the optimal hexagonal grid."""
    s = _max_spacing(n, eps)               # maximise spacing first
    sqrt3 = np.sqrt(3.0)

    row_step = sqrt3 / 2.0 * s
    pts = []
    y = s / 2.0
    i = 0
    while y <= 1.0 - s / 2.0 + eps:
        offset = (i % 2) * s / 2.0
        x = s / 2.0 + offset
        while x <= 1.0 - s / 2.0 + eps:
            pts.append([x, y])
            if len(pts) == n:               # stop as soon as we have n points
                return np.array(pts, dtype=float)
            x += s
        y += row_step
        i += 1
    # Fallback – should never happen because `_max_spacing` guarantees enough points
    return np.array(pts[:n], dtype=float)


# ----------------------------------------------------------------------
# Core API – unchanged signature
# ----------------------------------------------------------------------
def construct_packing():
    """Construct 26 circles in a unit square using the optimal lattice."""
    n = 26
    centres = _hex_lattice(n)

    # compute the maximal feasible radii for this centre set
    radii = compute_max_radii(centres)

    # deterministic validation (will raise if something is wrong)
    _validate_packing(centres, radii)

    sum_radii = np.sum(radii)
    return centres, radii, sum_radii


def compute_max_radii(centres: np.ndarray) -> np.ndarray:
    """
    Compute the largest possible radius for every centre such that:
        1. the circle stays inside the unit square,
        2. no two circles overlap.

    The result is obtained purely with NumPy broadcasting – no Python loops
    over point pairs are used.
    """
    eps = 1e-12                     # safety margin inside every constraint

    # ---- edge limit ----------------------------------------------------
    r_edge = np.minimum.reduce(
        [centres[:, 0], centres[:, 1], 1.0 - centres[:, 0], 1.0 - centres[:, 1]]
    ) - eps

    # ---- nearest‑neighbour limit ----------------------------------------
    diff = centres[:, None, :] - centres[None, :, :]          # shape (n,n,2)
    dists = np.linalg.norm(diff, axis=2)                     # shape (n,n)
    np.fill_diagonal(dists, np.inf)                         # ignore self‑distance
    r_nn = 0.5 * np.min(dists, axis=1)                       # half the closest distance

    # ---- final radii ----------------------------------------------------
    radii = np.minimum(r_edge, r_nn)
    radii = np.maximum(radii, 0.0)   # guard against tiny negatives
    return radii


# ----------------------------------------------------------------------
# Deterministic guard – catches any violation during development.
# ----------------------------------------------------------------------
def _validate_packing(centres: np.ndarray, radii: np.ndarray):
    """Assert that the packing satisfies the two core constraints."""
    eps = 1e-12

    # containment
    assert np.all(centres[:, 0] - radii >= -eps)
    assert np.all(centres[:, 0] + radii <= 1.0 + eps)
    assert np.all(centres[:, 1] - radii >= -eps)
    assert np.all(centres[:, 1] + radii <= 1.0 + eps)

    # non‑overlap
    diff = centres[:, None, :] - centres[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones_like(dists, dtype=bool), k=1)
    assert np.all(dists[mask] >= (radii[:, None] + radii[None, :])[mask] - eps)


# EVOLVE-BLOCK-END


# ----------------------------------------------------------------------
# Fixed helper code (unchanged by the evolutionary process)
# ----------------------------------------------------------------------
def run_packing():
    """Run the circle packing constructor for n=26."""
    centres, radii, sum_radii = construct_packing()
    return centres, radii, sum_radii


def visualize(centres, radii):
    """
    Visualise the circle packing.

    Parameters
    ----------
    centres : np.ndarray of shape (n, 2)
        Circle centres.
    radii : np.ndarray of shape (n,)
        Circle radii.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centres, radii)):
        ax.add_patch(Circle(c, r, alpha=0.5, edgecolor="k"))
        ax.text(c[0], c[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centres)}, sum={np.sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centres, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # Uncomment to visualise the result:
    # visualize(centres, radii)