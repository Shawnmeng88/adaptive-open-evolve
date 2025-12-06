"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Generate a set of 26 circle centers and compute the maximal feasible radii
    using a linear program. Several heuristic layouts are tried and the best
    (largest total radius) is returned.
    Returns:
        centers (np.ndarray): shape (26, 2)
        radii   (np.ndarray): shape (26,)
        sum_radii (float)
    """
    import numpy as np

    def generate_hex_grid(num_points, jitter=0.0):
        """
        Produce a hexagonal lattice inside the unit square.
        If jitter>0, a small random displacement is added to each point.
        """
        spacing = 0.18                     # approximate distance between neighboring centers
        vert = spacing * np.sqrt(3) / 2.0  # vertical distance between rows
        margin = spacing / 2.0             # keep a margin from the borders

        points = []
        row = 0
        y = margin
        while y <= 1 - margin and len(points) < num_points:
            offset = (spacing / 2.0) if (row % 2 == 1) else 0.0
            x = margin + offset
            while x <= 1 - margin and len(points) < num_points:
                pt = np.array([x, y])
                if jitter > 0.0:
                    pt += (np.random.rand(2) - 0.5) * jitter
                    pt = np.clip(pt, margin, 1 - margin)
                points.append(pt)
                x += spacing
            row += 1
            y += vert

        # If we still have fewer points, fill the remaining slots with random interior points
        while len(points) < num_points:
            pt = np.random.uniform(margin, 1 - margin, size=2)
            points.append(pt)

        return np.array(points[:num_points])

    def optimal_radii(centers):
        """
        Solve a linear program to maximize sum of radii given fixed centers.
        Constraints:
            - radius_i <= distance to each of the four walls
            - radius_i + radius_j <= distance between centers i and j
        """
        n = centers.shape[0]
        # Objective: maximize sum r_i  -> minimize -sum r_i
        c = -np.ones(n)

        # Pairwise constraints
        A = []
        b = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A.append(row)
                b.append(d)

        # Convert to numpy arrays for linprog
        A = np.array(A, dtype=np.float64)
        b = np.array(b, dtype=np.float64)

        # Wall distance bounds
        bounds = []
        for (x, y) in centers:
            wall_dist = min(x, y, 1.0 - x, 1.0 - y)
            bounds.append((0.0, wall_dist))

        # Solve LP
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if not res.success:
            # Fallback: very small radii to keep feasibility
            return np.zeros(n)
        radii = res.x
        # Numerical cleanup
        radii = np.maximum(radii, 0.0)
        return radii

    best_sum = -1.0
    best_centers = None
    best_radii = None

    # Try a few deterministic and random layouts
    for attempt in range(30):
        jitter = 0.0 if attempt < 5 else 0.02 * (attempt - 4)  # increase jitter later
        centers = generate_hex_grid(26, jitter=jitter)
        radii = optimal_radii(centers)
        total = radii.sum()
        if total > best_sum:
            best_sum = total
            best_centers = centers.copy()
            best_radii = radii.copy()

    return best_centers, best_radii, best_sum

def compute_max_radii(centers):
    """
    Compatibility wrapper – retained for external calls.
    Delegates to the LP‑based optimizer.
    """
    return optimal_radii(centers)
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
