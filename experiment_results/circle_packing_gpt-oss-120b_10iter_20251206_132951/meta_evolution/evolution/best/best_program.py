"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a packing of 26 circles in the unit square using a
    hexagonal lattice as a scaffold and then solving a linear program
    to obtain the optimal (maximal sum) radii for the chosen centres.
    """
    import numpy as np
    from scipy.optimize import linprog

    n_circles = 26

    # ---------- hexagonal lattice scaffold ----------
    def hex_grid(r):
        """Return list of centre coordinates for a hexagonal grid with
        equal radius r (spacing 2r horizontally, sqrt(3)*r vertically)."""
        s = 2 * r
        dy = np.sqrt(3) * r
        centres = []
        row = 0
        y = r
        while y <= 1 - r + 1e-12:
            offset = (row % 2) * r
            x = r + offset
            while x <= 1 - r + 1e-12:
                centres.append((x, y))
                x += s
            row += 1
            y += dy
        return centres

    # binary search the largest uniform radius that yields at least 26 points
    low, high = 0.0, 0.5
    for _ in range(30):
        mid = (low + high) / 2
        if len(hex_grid(mid)) >= n_circles:
            low = mid
        else:
            high = mid
    scaffold_r = low
    scaffold_centres = np.array(hex_grid(scaffold_r)[:n_circles])

    # ---------- linear programming to maximise sum of radii ----------
    n = scaffold_centres.shape[0]

    # objective: maximise sum(r)  -> minimise -sum(r)
    c = -np.ones(n)

    # inequality matrix A_ub @ r <= b_ub
    A_ub = []
    b_ub = []

    # border constraints: r_i <= distance to nearest wall
    for i in range(n):
        x, y = scaffold_centres[i]
        border = min(x, y, 1 - x, 1 - y)
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(border)

    # pairwise non‑overlap constraints: r_i + r_j <= distance between centres
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(scaffold_centres[i] - scaffold_centres[j])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    bounds = [(0.0, None)] * n

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not res.success:
        # fallback: use the uniform radius from the scaffold
        radii = np.full(n, scaffold_r)
    else:
        radii = res.x

    sum_radii = radii.sum()
    return scaffold_centres, radii, sum_radii
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
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

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)
