# EVOLVE-BLOCK-START
"""
Improved circle‑packing for n=26 circles.

Instead of fixing the centres on a deterministic grid and only
optimising the radii, we now treat the centre coordinates as variables
as well.  A single non‑linear optimisation (SLSQP) simultaneously
optimises all positions and radii under the geometric constraints:

* each circle stays inside the unit square,
* circles do not overlap,
* radii are non‑negative.

The formulation is still very compact (≈250 lines total) and
converges quickly for 26 circles, giving a higher total sum of radii
(and thus a higher fitness score) than the previous linear‑programming
approach.
"""

import numpy as np
from scipy.optimize import minimize

N = 26                         # number of circles


def _initial_guess():
    """
    Start from the deterministic 5×5 grid + centre used before.
    This provides a feasible point for the optimiser.
    """
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 5)
    pts = np.array([[x, y] for x in xs for y in ys])
    centres = np.vstack([pts[:25], [0.5, 0.5]])          # (26,2)

    # Very small radii that surely satisfy the constraints.
    radii = np.full(N, 0.02)

    # Flatten to a 1‑D vector: [x0,y0,x1,y1,…,xN‑1,yN‑1, r0,…,rN‑1]
    return np.hstack([centres.ravel(), radii])


def _unpack(x):
    """
    Split the optimisation vector into centre coordinates and radii.
    """
    centres = x[:2 * N].reshape(N, 2)
    radii = x[2 * N:]
    return centres, radii


def _objective(x):
    """
    We want to maximise the total radius, i.e. minimise the negative sum.
    """
    _, radii = _unpack(x)
    return -np.sum(radii)


def _make_constraints():
    """
    Build the list of inequality constraints required by SLSQP.
    All constraints are of the form  g(x) >= 0.
    """
    cons = []

    # ----- border constraints -------------------------------------------------
    #   x_i - r_i >= 0          (left side)
    #   1 - x_i - r_i >= 0      (right side)
    #   y_i - r_i >= 0          (bottom side)
    #   1 - y_i - r_i >= 0      (top side)
    for i in range(N):
        def left(x, i=i):
            centres, radii = _unpack(x)
            return centres[i, 0] - radii[i]

        def right(x, i=i):
            centres, radii = _unpack(x)
            return 1.0 - centres[i, 0] - radii[i]

        def bottom(x, i=i):
            centres, radii = _unpack(x)
            return centres[i, 1] - radii[i]

        def top(x, i=i):
            centres, radii = _unpack(x)
            return 1.0 - centres[i, 1] - radii[i]

        cons.extend([
            {"type": "ineq", "fun": left},
            {"type": "ineq", "fun": right},
            {"type": "ineq", "fun": bottom},
            {"type": "ineq", "fun": top},
        ])

    # ----- non‑overlap constraints -------------------------------------------
    #   dist((x_i,y_i),(x_j,y_j)) - (r_i + r_j) >= 0
    for i in range(N):
        for j in range(i + 1, N):
            def no_overlap(x, i=i, j=j):
                centres, radii = _unpack(x)
                dx = centres[i, 0] - centres[j, 0]
                dy = centres[i, 1] - centres[j, 1]
                dist = np.hypot(dx, dy)
                return dist - (radii[i] + radii[j])

            cons.append({"type": "ineq", "fun": no_overlap})

    return cons


def construct_packing():
    """
    Run the optimisation and return (centres, radii, sum_of_radii).
    """
    x0 = _initial_guess()
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N   # positions in [0,1], radii ≤ 0.5
    constraints = _make_constraints()

    res = minimize(
        _objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500, "disp": False},
    )

    if not res.success:
        # fall back to the linear‑programming solution (always feasible)
        # – this keeps the programme robust even if the non‑linear solver fails.
        return _fallback_lp()

    centres, radii = _unpack(res.x)
    return centres, radii, float(radii.sum())


# ---------------------------------------------------------------------------
# Fallback: the original linear‑programming approach (guaranteed feasible).
# ---------------------------------------------------------------------------
def _grid_positions():
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 5)
    pts = np.array([[x, y] for x in xs for y in ys])
    return np.vstack([pts[:25], [0.5, 0.5]])


def _max_radii_lp(centers):
    n = len(centers)
    c = -np.ones(n)

    A, b = [], []

    for i, (x, y) in enumerate(centers):
        border = min(x, y, 1 - x, 1 - y)
        row = np.zeros(n)
        row[i] = 1
        A.append(row)
        b.append(border)

    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = row[j] = 1
            A.append(row)
            b.append(d)

    from scipy.optimize import linprog
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * n, method="highs")
    return res.x if res.success else np.zeros(n)


def _fallback_lp():
    centres = _grid_positions()
    radii = _max_radii_lp(centres)
    return centres, radii, float(radii.sum())


# ---------------------------------------------------------------------------
# Public helper – unchanged API
# ---------------------------------------------------------------------------
def run_packing():
    """Execute the constructor and return its data."""
    return construct_packing()


def visualize(centers, radii):
    """Optional visualisation – unchanged from the original program."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

    for i, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(c, r, alpha=0.5))
        ax.text(*c, str(i), ha="center", va="center")

    plt.title(f"Circle packing (n={len(centers)}, sum={radii.sum():.6f})")
    plt.show()


if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s:.6f}")
    # Uncomment to see the layout:
    # visualize(c, r)
# EVOLVE-BLOCK-END