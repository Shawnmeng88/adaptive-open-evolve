# EVOLVE-BLOCK-START
"""
Improved 26‑circle packing.

The 25 base centres form a regular 5×5 grid.  Instead of fixing the
26th centre, we examine a modest deterministic set of candidate
positions (a 0.05‑spaced grid inside the unit square) and keep the one
that yields the largest possible sum of radii.  Radii are obtained by a
tiny linear programme (maximise Σ r) that respects the square borders and
pairwise non‑overlap constraints; a simple geometric fallback is used
when SciPy is unavailable.

This extra‑point search raises the total radius sum compared with the
previous fixed‑point version while remaining fast and fully deterministic.
"""
import numpy as np

# ----------------------------------------------------------------------
# Optional SciPy linear‑programming backend (fallback to a heuristic).
try:
    from scipy.optimize import linprog
    _SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    _SCIPY_AVAILABLE = False


def _optimal_radii(centres: np.ndarray) -> np.ndarray:
    """
    Compute radii that maximise the total sum for a fixed set of centres.

    Parameters
    ----------
    centres : np.ndarray, shape (n, 2)

    Returns
    -------
    radii : np.ndarray, shape (n,)
    """
    n = centres.shape[0]

    # 1. distance to the four borders
    border = np.minimum.reduce(
        [centres[:, 0], centres[:, 1], 1.0 - centres[:, 0], 1.0 - centres[:, 1]]
    )  # (n,)

    # 2. pairwise centre distances
    diff = centres[:, None, :] - centres[None, :, :]          # (n, n, 2)
    pair = np.sqrt(np.sum(diff ** 2, axis=2))                # (n, n)

    if _SCIPY_AVAILABLE:
        # maximise Σ r  → minimise -Σ r
        cobj = -np.ones(n)

        # a) r_i ≤ border_i
        A = np.eye(n)
        b = border.copy()

        # b) r_i + r_j ≤ d_ij  (i < j)
        rows = []
        rhs = []
        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i] = row[j] = 1.0
                rows.append(row)
                rhs.append(pair[i, j])
        if rows:
            A = np.vstack([A, rows])
            b = np.concatenate([b, rhs])

        res = linprog(
            cobj,
            A_ub=A,
            b_ub=b,
            bounds=[(0, None)] * n,
            method="highs",
            options={"presolve": True},
        )
        if res.success:
            return np.clip(res.x, 0.0, None)

    # ------------------------------------------------------------------
    # Fallback heuristic: radius_i = min(border_i, nearest/2)
    np.fill_diagonal(pair, np.inf)
    nearest = pair.min(axis=1)
    return np.minimum(border, nearest / 2.0)


def construct_packing():
    """
    Build a deterministic layout of 26 circles and return the centres,
    their radii and the sum of radii.
    """
    # ----- 1. 5×5 grid -------------------------------------------------
    grid = np.linspace(0.1, 0.9, 5)               # [0.1,0.3,0.5,0.7,0.9]
    base = np.array([(x, y) for x in grid for y in grid], float)  # (25,2)

    # ----- 2. candidate extra points -----------------------------------
    # 0.05‑spaced grid covering the interior, avoiding points that are
    # already occupied (within 0.02 of a base centre).
    cand_vals = np.arange(0.05, 0.96, 0.05)
    candidates = np.array(
        [(x, y) for x in cand_vals for y in cand_vals], float
    )  # (361,2)

    # discard candidates that are too close to any base centre
    diff = candidates[:, None, :] - base[None, :, :]          # (361,25,2)
    dists = np.sqrt(np.sum(diff ** 2, axis=2))               # (361,25)
    min_dist = dists.min(axis=1)
    candidates = candidates[min_dist > 0.02]

    # ----- 3. evaluate each candidate ----------------------------------
    best_sum = -np.inf
    best_centres = None
    best_radii = None

    for extra in candidates:
        centres = np.vstack([base, extra])
        radii = _optimal_radii(centres)
        total = radii.sum()
        if total > best_sum:
            best_sum = total
            best_centres = centres
            best_radii = radii

    # In the extremely unlikely event that no candidate survived (should
    # never happen), fall back to the original fixed extra point.
    if best_centres is None:
        best_centres = np.vstack([base, [0.85, 0.85]])
        best_radii = _optimal_radii(best_centres)
        best_sum = best_radii.sum()

    return best_centres, best_radii, float(best_sum)
# EVOLVE-BLOCK-END


def run_packing():
    """Run the circle packing constructor for n=26."""
    centres, radii, total = construct_packing()
    return centres, radii, total


def visualize(centres, radii):
    """Optional visualisation."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centres, radii)):
        ax.add_patch(Circle(c, r, alpha=0.5, edgecolor='k'))
        ax.text(c[0], c[1], str(i), ha='center', va='center')

    plt.title(f'Circle Packing (n={len(centres)}, sum={radii.sum():.6f})')
    plt.show()


if __name__ == '__main__':
    centres, radii, total = run_packing()
    print(f'Sum of radii: {total:.6f}')
    # visualize(centres, radii)   # uncomment to see the layout