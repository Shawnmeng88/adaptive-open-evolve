# EVOLVE-BLOCK-START
"""
Deterministic 26‑circle packing with a tiny step‑search.

A hexagonal lattice is generated for many candidate spacings.
For each spacing the maximal admissible radii (border & neighbour limits)
are computed and the layout with the largest total radius sum is kept.
The public API (run_packing → centres, radii, sum) is unchanged.
"""
import numpy as np

N = 26                     # required number of circles
STEP_MIN, STEP_MAX = 0.15, 0.25   # search interval for the lattice step
STEP_COUNT = 101                # how many candidates to test (≈0.001 resolution)


def _max_radii(pts: np.ndarray) -> np.ndarray:
    """Largest radii that keep circles inside the unit square and non‑overlapping."""
    # distance to the four borders
    border = np.minimum.reduce([pts[:, 0], pts[:, 1],
                               1 - pts[:, 0], 1 - pts[:, 1]])

    # pairwise centre distances (ignore self‑distance)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)

    # half the nearest neighbour distance
    neigh = d.min(axis=1) / 2.0
    return np.minimum(border, neigh)


def _hex_lattice(step: float) -> np.ndarray:
    """
    Produce up to N points on a hexagonal lattice for a given step.
    A margin equal to half the step guarantees that all generated points
    lie at least `margin` away from the square border.
    """
    margin = step / 2.0
    pts = []
    y = margin
    row = 0
    v_step = np.sqrt(3) / 2 * step          # vertical spacing of a hex grid
    while y < 1 - margin and len(pts) < N:
        # every second row is shifted by half a step
        x_start = margin + (step / 2 if row % 2 else 0.0)
        x = x_start
        while x < 1 - margin and len(pts) < N:
            pts.append([x, y])
            x += step
        y += v_step
        row += 1
    return np.array(pts[:N])


def construct_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Search the step interval for the layout with the largest sum of radii.
    Returns the corresponding centres, radii and the sum.
    """
    best_sum = -1.0
    best_centres = best_radii = None

    # inclusive linspace – ensures the endpoints are examined
    steps = np.linspace(STEP_MIN, STEP_MAX, STEP_COUNT)

    for s in steps:
        centres = _hex_lattice(s)
        # In the rare case the lattice yields fewer than N points, skip it
        if centres.shape[0] < N:
            continue
        radii = _max_radii(centres)
        total = radii.sum()
        if total > best_sum:
            best_sum, best_centres, best_radii = total, centres, radii

    # Fallback – should never happen, but keeps the function total‑safe
    if best_centres is None:
        best_centres = _hex_lattice(STEP_MIN)
        best_radii = _max_radii(best_centres)
        best_sum = best_radii.sum()

    return best_centres, best_radii, float(best_sum)


# EVOLVE-BLOCK-END


def run_packing():
    """Entry point – identical to the original API."""
    return construct_packing()


def visualize(centres, radii):
    """Optional quick visualisation with Matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centres, radii)):
        ax.add_patch(Circle(c, r, alpha=0.4))
        ax.text(*c, str(i), ha="center", va="center", fontsize=8)

    plt.title(f"Sum of radii = {radii.sum():.4f}")
    plt.show()


if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s:.6f}")
    # visualize(c, r)   # uncomment to see the layout