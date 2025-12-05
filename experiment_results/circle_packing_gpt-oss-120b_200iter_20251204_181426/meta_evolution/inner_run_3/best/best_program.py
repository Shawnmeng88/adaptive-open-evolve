# EVOLVE-BLOCK-START
"""
Deterministic 5×5 square‑grid packing for n=26 circles.

A uniform square lattice with 5 points per side fits 25 circles of radius
r = 0.1 inside the unit square (centres at 0.1, 0.3, 0.5, 0.7, 0.9).
To satisfy the required count of 26 circles we add a single zero‑radius
circle at the corner (0, 0).  Zero radius does not affect the total sum
but keeps the packing valid and deterministic.

The implementation is pure NumPy, deterministic and includes a final
validation step that raises an exception if any geometric constraint is
violated.
"""

import numpy as np


def construct_packing():
    """Create centres, radii and the sum of radii for n=26 circles."""
    n = 26                     # required number of circles
    k = 5                      # points per side of the square grid
    radius = 0.1               # (1‑2·r)/(k‑1) = 0.2  ⇒  r = 0.1

    # ---- 1. generate the 5×5 grid (25 circles) ----------------------------
    coords = np.linspace(radius, 1.0 - radius, k)
    xv, yv = np.meshgrid(coords, coords)
    centres_grid = np.column_stack([xv.ravel(), yv.ravel()])

    # ---- 2. add a dummy zero‑radius circle to reach n = 26 ----------------
    dummy_centre = np.array([[0.0, 0.0]])          # on the corner
    dummy_radius = np.array([0.0])

    centres = np.vstack([centres_grid, dummy_centre])
    radii = np.concatenate([np.full(centres_grid.shape[0], radius), dummy_radius])

    # ---- 3. safety validation (pure NumPy) --------------------------------
    _validate_packing(centres, radii)

    # ---- 4. return results -------------------------------------------------
    sum_radii = np.sum(radii)
    return centres, radii, sum_radii


# -------------------------------------------------------------------------
# Helper functions (deterministic, NumPy‑only)
# -------------------------------------------------------------------------

def compute_max_radii(centres: np.ndarray) -> np.ndarray:
    """
    Re‑compute the exact maximal admissible radii for a given set of centres.
    This function is kept for compatibility; the constructor already supplies
    optimal radii for the generated layout.
    """
    # distance to the four sides of the unit square
    border = np.minimum.reduce(
        [centres[:, 0], centres[:, 1],
         1.0 - centres[:, 0], 1.0 - centres[:, 1]]
    )

    # pairwise centre distances
    diff = centres[:, None, :] - centres[None, :, :]          # (n, n, 2)
    dists = np.linalg.norm(diff, axis=2)                     # (n, n)
    np.fill_diagonal(dists, np.inf)

    # nearest neighbour distance for each centre
    nearest = np.min(dists, axis=1)

    # maximal admissible radius is the smaller of the border margin and
    # half the nearest‑neighbour distance
    radii = np.minimum(border, nearest / 2.0)
    return radii


def _validate_packing(centres: np.ndarray, radii: np.ndarray):
    """Pure NumPy assertions – raises if any geometric constraint is violated."""
    # 1) circles stay inside the unit square
    if not np.all((centres - radii[:, None]) >= -1e-12) or not np.all(
            (centres + radii[:, None]) <= 1.0 + 1e-12):
        raise AssertionError("Some circles exceed the unit‑square boundaries.")

    # 2) circles do not overlap
    diff = centres[:, None, :] - centres[None, :, :]          # (n, n, 2)
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_allowed = radii[:, None] + radii[None, :]
    if not np.all(dists >= min_allowed - 1e-12):
        raise AssertionError("Circles overlap.")


# EVOLVE-BLOCK-END


# -------------------------------------------------------------------------
# Fixed helper code (unchanged by the evolutionary process)
# -------------------------------------------------------------------------

def run_packing():
    """Run the circle packing constructor for n=26."""
    centres, radii, sum_radii = construct_packing()
    return centres, radii, sum_radii


def visualize(centres, radii):
    """Simple Matplotlib visualisation (optional)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centres, radii)):
        ax.add_patch(Circle(c, r, alpha=0.5, edgecolor="k"))
        ax.text(c[0], c[1], str(i), ha="center", va="center", fontsize=8)

    plt.title(f"Square‑grid packing (n={len(centres)}, sum={np.sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centres, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # Uncomment to see the layout:
    # visualize(centres, radii)