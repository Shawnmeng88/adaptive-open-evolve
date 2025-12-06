"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Construct a packing of 26 circles in the unit square.
    Starts from a dense hexagonal lattice and refines the centers
    with a simple hill‑climbing loop that repeatedly re‑optimises the radii
    via linear programming.
    Returns:
        centers (np.ndarray): shape (26,2)
        radii   (np.ndarray): shape (26,)
        sum_radii (float)
    """
    import numpy as np
    from scipy.optimize import linprog

    N = 26
    rng = np.random.default_rng()

    # --------------------------------------------------------------
    # Helper: solve LP for fixed centers, maximising sum of radii
    # --------------------------------------------------------------
    def optimal_radii(centers: np.ndarray) -> np.ndarray:
        m = centers.shape[0]
        # maximise sum r_i  -> minimise -sum r_i
        c = -np.ones(m)

        rows = []
        rhs = []

        # distance to the square borders
        border = np.minimum.reduce(
            [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
        )
        rows.extend(np.eye(m))
        rhs.extend(border)

        # pairwise non‑overlap constraints
        for i in range(m):
            for j in range(i + 1, m):
                d = np.linalg.norm(centers[i] - centers[j])
                row = np.zeros(m)
                row[i] = 1
                row[j] = 1
                rows.append(row)
                rhs.append(d)

        A_ub = np.vstack(rows)
        b_ub = np.array(rhs)

        bounds = [(0, None)] * m
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )
        return res.x if res.success else np.zeros(m)

    # --------------------------------------------------------------
    # Helper: generate a dense hexagonal lattice and pick N points
    # --------------------------------------------------------------
    def generate_initial_centers(num_points: int) -> np.ndarray:
        spacing = 0.12  # denser than the original 0.18
        dy = spacing * np.sqrt(3) / 2
        pts = []
        y = spacing / 2
        row = 0
        while y < 1 - spacing / 2:
            offset = spacing / 2 if (row % 2) else 0.0
            x = spacing / 2 + offset
            while x < 1 - spacing / 2:
                pts.append([x, y])
                x += spacing
            y += dy
            row += 1
        pts = np.array(pts)

        # If we have fewer than needed (unlikely), pad with random points
        if pts.shape[0] < num_points:
            extra = rng.uniform(0, 1, size=(num_points - pts.shape[0], 2))
            pts = np.vstack([pts, extra])

        # Randomly select the required number of points
        idx = rng.choice(pts.shape[0], size=num_points, replace=False)
        return pts[idx]

    # --------------------------------------------------------------
    # Initial placement
    # --------------------------------------------------------------
    centers = generate_initial_centers(N)

    # --------------------------------------------------------------
    # Hill‑climbing optimisation loop
    # --------------------------------------------------------------
    best_centers = centers.copy()
    best_radii = optimal_radii(best_centers)
    best_sum = best_radii.sum()

    max_iter = 2000
    step_scale = 0.04          # initial perturbation magnitude
    stagnation_limit = 200     # iterations without improvement before reducing step

    no_improve = 0

    for _ in range(max_iter):
        # pick a single circle to move
        i = rng.integers(N)
        proposal = best_centers.copy()
        delta = rng.normal(scale=step_scale, size=2)
        proposal[i] += delta

        # keep the moved centre inside the unit square (tiny margin)
        proposal[i] = np.clip(proposal[i], 0.001, 0.999)

        radii = optimal_radii(proposal)
        s = radii.sum()

        if s > best_sum:
            best_sum = s
            best_centers = proposal
            best_radii = radii
            no_improve = 0
        else:
            no_improve += 1

        # adapt step size if we are stuck
        if no_improve >= stagnation_limit:
            step_scale *= 0.7
            step_scale = max(step_scale, 0.005)  # don't go to zero
            no_improve = 0

    return best_centers, best_radii, best_sum
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
