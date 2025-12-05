# EVOLVE-BLOCK-START
"""Deterministic hexagonal‑lattice circle packing for n = 26.

Improvements over the previous version
--------------------------------------
1. **Spacing optimisation** – we try many lattice spacings (still
   deterministic) and keep the one that yields the largest total
   radius sum while still providing at least 26 points.
2. **Centres translation** – after the lattice is generated we shift it
   so that its bounding box is centred inside the unit square.  This
   maximises the distance of every centre to the walls and therefore
   enlarges the admissible radii.
3. **Same robust geometry code** – the helper `compute_max_radii`
   (unchanged signature) still computes the exact feasible radii for a
   given centre set, using only NumPy vectorised operations.
4. **Deterministic** – no randomness, no external state; the same input
   always produces the same packing.

The public API (`construct_packing`, `run_packing`, `visualize`) is
identical to the original program.
"""

import numpy as np

# ----------------------------------------------------------------------
# Public constructor
# ----------------------------------------------------------------------
def construct_packing():
    """
    Returns
    -------
    centres : np.ndarray, shape (26, 2)
        (x, y) coordinates of the circle centres.
    radii  : np.ndarray, shape (26,)
        Radius of each circle, guaranteed to satisfy containment and
        non‑overlap constraints.
    sum_radii : float
        Sum of all radii.
    """
    n_required = 26

    # ------------------------------------------------------------
    # 1️⃣  Optimise the lattice spacing (deterministic grid search)
    # ------------------------------------------------------------
    best_sum = -1.0
    best_centres = None

    # search range – the spacing cannot be larger than 0.5 (otherwise
    # at most one point per row would fit) and not smaller than 0.05
    # (far more than enough points).  200 steps give a fine enough
    # resolution while staying deterministic.
    steps = 200
    for i in range(steps):
        spacing = 0.05 + (0.45) * i / (steps - 1)          # 0.05 … 0.50
        centres = _generate_lattice(spacing, n_required)
        if centres is None:
            continue                                        # not enough points
        centres = _centre_translation(centres)              # centre inside the square
        radii = compute_max_radii(centres)
        total = np.sum(radii)
        if total > best_sum:
            best_sum = total
            best_centres = centres
            best_radii = radii

    # The search is guaranteed to find at least one feasible layout
    # (the original spacing = 0.20 works), so `best_centres` cannot be
    # None.
    _validate_packing(best_centres, best_radii)

    sum_radii = float(best_sum)
    return best_centres, best_radii, sum_radii


# ----------------------------------------------------------------------
# Helper: deterministic hexagonal lattice generation
# ----------------------------------------------------------------------
def _generate_lattice(spacing: float, n_required: int):
    """
    Build a regular triangular (hexagonal) lattice with the given
    spacing and return the first `n_required` points.
    If the lattice does not contain enough points, ``None`` is returned.
    """
    vert_step = spacing * np.sqrt(3.0) / 2.0

    # maximal number of rows that could possibly fit (including a tiny
    # epsilon to avoid floating‑point edge effects)
    max_rows = int(np.floor(1.0 / vert_step + 1e-9)) + 1

    pts = []
    for row in range(max_rows):
        y = row * vert_step
        if y > 1.0:
            break

        # even rows start at x = 0, odd rows are offset by spacing/2
        offset = 0.0 if (row % 2 == 0) else spacing / 2.0
        max_cols = int(np.floor((1.0 - offset) / spacing + 1e-9)) + 1

        for col in range(max_cols):
            x = offset + col * spacing
            if x > 1.0:
                break
            pts.append([x, y])
            if len(pts) == n_required:          # early stop – deterministic order
                return np.asarray(pts, dtype=float)

    # Not enough points for the required count
    return None


# ----------------------------------------------------------------------
# Helper: translate points so that their bounding box is centred
# ----------------------------------------------------------------------
def _centre_translation(centres: np.ndarray) -> np.ndarray:
    """
    Translate the whole set of centres so that the smallest axis‑aligned
    bounding box is centred inside the unit square.
    The operation is deterministic and never moves a point outside [0,1].
    """
    mins = centres.min(axis=0)
    maxs = centres.max(axis=0)
    # free space on each side
    free = 1.0 - (maxs - mins)
    shift = free / 2.0 - mins          # amount to add to every coordinate
    # clipping is not needed – by construction shift keeps all points
    # inside the unit square, but we guard against numerical noise.
    shifted = centres + shift
    shifted = np.clip(shifted, 0.0, 1.0)
    return shifted


# ----------------------------------------------------------------------
# Core geometry – unchanged signature (allowed to be edited internally)
# ----------------------------------------------------------------------
def compute_max_radii(centres: np.ndarray) -> np.ndarray:
    """
    Determine the largest possible radius for each centre while
    respecting the two core constraints:
      * the circle must stay inside the unit square,
      * circles must not overlap.

    Parameters
    ----------
    centres : np.ndarray, shape (n, 2)
        Coordinates of the circle centres.

    Returns
    -------
    radii : np.ndarray, shape (n,)
        Feasible radii (non‑negative, float64).
    """
    eps = 1e-9                     # safety margin for floating‑point

    # ---- wall distances -------------------------------------------------
    # distance from each centre to the four sides of the unit square
    wall_dist = np.minimum(centres, 1.0 - centres)          # (n, 2)
    wall_limit = np.min(wall_dist, axis=1)                 # (n,)

    # ---- neighbour distances --------------------------------------------
    # pairwise Euclidean distance matrix
    diff = centres[:, None, :] - centres[None, :, :]        # (n, n, 2)
    dists = np.linalg.norm(diff, axis=2)                   # (n, n)
    np.fill_diagonal(dists, np.inf)                       # ignore self‑distance
    nearest = np.min(dists, axis=1)                        # distance to closest neighbour

    # ---- radius is the tighter of the two limits ------------------------
    radii = np.minimum(wall_limit, nearest / 2.0)

    # enforce non‑negativity and safety margin
    radii = np.maximum(radii - eps, 0.0)
    return radii


# ----------------------------------------------------------------------
# Validation – unchanged (keeps validity == 1.0)
# ----------------------------------------------------------------------
def _validate_packing(centres: np.ndarray, radii: np.ndarray):
    """
    Re‑check the two core constraints with a tiny epsilon.
    Raises AssertionError if any violation is detected.
    """
    eps = 1e-9

    # containment
    assert np.all(radii >= 0.0), "Negative radius encountered."
    assert np.all(centres - radii[:, None] >= -eps), "Circle exceeds left/bottom wall."
    assert np.all(centres + radii[:, None] <= 1.0 + eps), "Circle exceeds right/top wall."

    # non‑overlap
    diff = centres[:, None, :] - centres[None, :, :]          # (n, n, 2)
    dists = np.linalg.norm(diff, axis=2)                     # (n, n)
    np.fill_diagonal(dists, np.inf)
    min_allowed = radii[:, None] + radii[None, :] + eps
    assert np.all(dists >= min_allowed), "Circles overlap."


# ----------------------------------------------------------------------
# Fixed helper – unchanged by the evolution process
# ----------------------------------------------------------------------
def run_packing():
    """Run the circle packing constructor for n=26."""
    centres, radii, sum_radii = construct_packing()
    return centres, radii, sum_radii


def visualize(centres, radii):
    """
    Visualise the packing (optional).

    Parameters
    ----------
    centres : np.ndarray, shape (n, 2)
    radii   : np.ndarray, shape (n,)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centres, radii)):
        ax.add_patch(Circle(c, r, alpha=0.5, edgecolor="black"))
        ax.text(c[0], c[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centres)}, sum={np.sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centres, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # Uncomment to visualise the result:
    # visualize(centres, radii)

# EVOLVE-BLOCK-END