# EVOLVE-BLOCK-START
"""
Deterministic dense hex‑lattice packing for n=26 circles.

The algorithm:
1. Build a fine triangular (hex) lattice that comfortably covers the unit square.
2. Keep the 26 points that are closest to the centre (0.5, 0.5).  
   This yields the most compact cluster and therefore the largest possible
   uniform radius after scaling.
3. Compute a single scaling factor *a* that expands the cluster just enough
   that the circles touch the square borders **and** their nearest neighbours.
   Because the lattice is regular, all circles obtain the same radius
   `r = a * spacing / 2`.
4. Return the scaled centre coordinates and the uniform radii.

All operations are fully vectorised, deterministic and respect the
unit‑square and non‑overlap constraints.  The helper `compute_max_radii`
remains available for validation (it reproduces the same radii for this
construction)."""

import numpy as np


def _hex_lattice_fine(spacing: float) -> np.ndarray:
    """
    Generate a fine triangular lattice (spacing `spacing`) that covers the
    unit square.  The lattice is centred at (0.5, 0.5) for easier scaling.
    """
    dy = spacing * np.sqrt(3) / 2.0

    # create a grid large enough to contain the unit square after centering
    # we generate rows from -1 to +2 to be safe
    rows = np.arange(-1, 3, dy)
    pts = []

    for iy, y in enumerate(rows):
        offset = (iy % 2) * spacing / 2.0
        cols = np.arange(-1, 3, spacing) + offset
        xs = cols
        ys = np.full_like(xs, y)
        pts.append(np.stack([xs, ys], axis=1))

    pts = np.concatenate(pts, axis=0)
    # centre the lattice at (0.5, 0.5)
    pts += 0.5 - np.array([0.5, 0.5])
    return pts


def compute_max_radii(centres: np.ndarray) -> np.ndarray:
    """
    Return the largest radii that keep all circles inside the unit square
    and non‑overlapping.  The computation is completely vectorised.

    For each centre we take the minimum of:
      * distance to the four square sides,
      * half the distance to the nearest other centre.
    """
    # distance to the four borders
    border = np.minimum.reduce(
        [centres[:, 0], 1 - centres[:, 0], centres[:, 1], 1 - centres[:, 1]]
    )

    # pairwise centre distances (inf on diagonal to ignore self)
    diff = centres[:, None, :] - centres[None, :, :]
    D = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(D, np.inf)

    # half the distance to the closest neighbour
    nearest = D.min(axis=1) / 2.0

    return np.minimum(border, nearest)


def _assert_valid(centres: np.ndarray, radii: np.ndarray) -> None:
    """Deterministic sanity check – raises if a constraint is broken."""
    eps = 1e-12
    # border constraints
    assert np.all(radii <= centres[:, 0] + eps)
    assert np.all(radii <= centres[:, 1] + eps)
    assert np.all(radii <= 1 - centres[:, 0] + eps)
    assert np.all(radii <= 1 - centres[:, 1] + eps)

    # pairwise non‑overlap
    diff = centres[:, None, :] - centres[None, :, :]
    D = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(D, np.inf)
    assert np.all(radii[:, None] + radii[None, :] <= D + eps)


def construct_packing():
    """
    Build the packing for n = 26 circles.

    Returns
    -------
    centres : np.ndarray shape (26, 2)
        (x, y) positions of the circle centres.
    radii   : np.ndarray shape (26,)
        Uniform radii respecting all constraints.
    sum_radii : float
        Sum of the radii (the optimisation objective).
    """
    n = 26
    # 1. fine lattice – a small spacing guarantees enough points
    fine_spacing = 0.04          # chosen deterministically
    lattice = _hex_lattice_fine(fine_spacing)

    # 2. keep the 26 points closest to the centre (0.5, 0.5)
    centre = np.array([0.5, 0.5])
    d2 = np.sum((lattice - centre) ** 2, axis=1)
    order = np.argsort(d2)[:n]
    selected = lattice[order]

    # 3. compute scaling factor a so that circles touch borders and neighbours
    #    let s = fine_spacing (original neighbour distance)
    #    after scaling by a, neighbour distance = a * s
    #    radius r = a * s / 2
    #    border constraint for each point i:
    #        r <= 0.5 - a * max(|dx_i|, |dy_i|)
    #    => a <= 0.5 / (max_abs_i + s/2)
    dx = selected[:, 0] - 0.5
    dy = selected[:, 1] - 0.5
    max_abs = np.max(np.maximum(np.abs(dx), np.abs(dy)))
    a = 0.5 / (max_abs + fine_spacing / 2.0)

    # 4. apply scaling about the centre
    centres = 0.5 + a * (selected - 0.5)

    # 5. uniform radius
    radii = np.full(n, a * fine_spacing / 2.0)

    # final deterministic validation – will raise on any bug
    _assert_valid(centres, radii)

    return centres, radii, float(radii.sum())


# EVOLVE-BLOCK-END


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
        ax.text(*c, str(i), ha="center", va="center", fontsize=8)

    plt.title(f"n={len(centres)}  sum radii={radii.sum():.6f}")
    plt.show()


if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s:.6f}")
    # visualize(c, r)   # uncomment to see the packing