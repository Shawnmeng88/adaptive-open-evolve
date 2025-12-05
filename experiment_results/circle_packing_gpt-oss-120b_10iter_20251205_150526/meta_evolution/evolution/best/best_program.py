"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def _solve_lp(centers):
    """
    Solve a linear program to maximize the sum of radii for fixed centers.
    Returns an array of radii (length n). If the LP fails, falls back to
    border‑limited uniform radii.
    """
    import numpy as np
    from scipy.optimize import linprog

    n = centers.shape[0]
    # Objective: maximize sum r_i  -> minimize -sum r_i
    c = -np.ones(n)

    # Build inequality constraints A_ub @ r <= b_ub
    A_rows = []
    b_vals = []

    # Border constraints
    for i in range(n):
        x, y = centers[i]
        e = np.zeros(n)
        e[i] = 1.0
        # left, right, bottom, top
        A_rows.append(e); b_vals.append(x)          # r_i <= x
        A_rows.append(e); b_vals.append(1.0 - x)    # r_i <= 1 - x
        A_rows.append(e); b_vals.append(y)          # r_i <= y
        A_rows.append(e); b_vals.append(1.0 - y)    # r_i <= 1 - y

    # Pairwise non‑overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_vals.append(dist)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)

    bounds = [(0.0, None)] * n

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return np.maximum(res.x, 0.0)
    # fallback: use the most restrictive border distance for each circle
    border_r = np.minimum.reduce([
        centers[:, 0],
        centers[:, 1],
        1.0 - centers[:, 0],
        1.0 - centers[:, 1]
    ])
    return border_r


def construct_packing():
    """
    Construct a packing of 26 circles in the unit square.
    Starts from a hexagonal lattice, optimizes radii via a linear program,
    then performs a short Monte‑Carlo refinement of the centres.
    Returns:
        centers (np.ndarray): shape (26, 2)
        radii   (np.ndarray): shape (26,)
        sum_radii (float)
    """
    import numpy as np

    n = 26
    sqrt3 = np.sqrt(3.0)
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # ------------------------------------------------------------------
    # 1) Enumerate modest hexagonal lattices and keep the best LP‑optimized one
    # ------------------------------------------------------------------
    for rows in range(2, 11):
        for cols in range(2, 11):
            # Maximum number of circles this pattern can hold
            total_possible = rows * cols - (rows // 2)
            if total_possible < n:
                continue

            # Lattice spacing limited by the square borders
            s_horiz = 1.0 / cols
            s_vert = 1.0 / ((rows - 1) * (sqrt3 / 2.0) + 1.0)
            s = min(s_horiz, s_vert)

            # Generate up to n centre points
            centers = []
            radius_guess = s / 2.0
            for r in range(rows):
                y = radius_guess + r * (sqrt3 / 2.0) * s
                offset = 0.0 if (r % 2 == 0) else radius_guess
                num_in_row = cols if (r % 2 == 0) else max(cols - 1, 0)
                for c in range(num_in_row):
                    x = radius_guess + offset + c * s
                    if x > 1.0 - radius_guess + 1e-12 or y > 1.0 - radius_guess + 1e-12:
                        continue
                    centers.append([x, y])
                    if len(centers) == n:
                        break
                if len(centers) == n:
                    break

            centers = np.array(centers, dtype=float)

            # Optimize radii for this centre set
            radii = _solve_lp(centers)
            sum_r = radii.sum()
            if sum_r > best_sum:
                best_sum = sum_r
                best_centers = centers.copy()
                best_radii = radii.copy()

    # ------------------------------------------------------------------
    # 2) Fallback – should never happen, but keep a deterministic grid
    # ------------------------------------------------------------------
    if best_centers is None:
        # Simple 5×6 grid as a safe default
        cols, rows = 6, 5
        xs = (np.arange(cols) + 0.5) / (cols + 1)
        ys = (np.arange(rows) + 0.5) / (rows + 1)
        xv, yv = np.meshgrid(xs, ys)
        pts = np.column_stack([xv.ravel(), yv.ravel()])[:n]
        best_centers = pts
        best_radii = _solve_lp(best_centers)
        best_sum = best_radii.sum()

    # ------------------------------------------------------------------
    # 3) Short Monte‑Carlo refinement of centre positions
    # ------------------------------------------------------------------
    rng = np.random.default_rng(12345)
    centers = best_centers
    radii = best_radii
    total_sum = best_sum

    for _ in range(30):
        i = rng.integers(n)
        delta = rng.uniform(-0.05, 0.05, size=2)
        new_center = np.clip(centers[i] + delta, 0.0, 1.0)

        # keep other centres unchanged
        trial_centers = centers.copy()
        trial_centers[i] = new_center

        trial_radii = _solve_lp(trial_centers)
        trial_sum = trial_radii.sum()

        if trial_sum > total_sum + 1e-9:
            centers = trial_centers
            radii = trial_radii
            total_sum = trial_sum

    return centers, radii, float(total_sum)
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
