"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """Create a packing for n=26 using a hexagonal lattice for better density."""
    n = 26

    def hex_centers(num):
        """Generate up to `num` points on a hexagonal lattice inside the unit square."""
        # Start with a relatively fine spacing; will be trimmed down to required count.
        spacing = 0.18
        points = []
        row = 0
        while True:
            y = row * (np.sqrt(3) / 2) * spacing
            if y > 1:
                break
            x_offset = (spacing / 2) if (row % 2) else 0.0
            col = 0
            while True:
                x = x_offset + col * spacing
                if x > 1:
                    break
                points.append((x, y))
                col += 1
            row += 1

        points = np.array(points)

        # If we didn't generate enough points, fall back to a simple grid.
        if len(points) < num:
            xs = np.linspace(0.05, 0.95, int(np.ceil(np.sqrt(num))))
            xv, yv = np.meshgrid(xs, xs)
            return np.column_stack((xv.ravel(), yv.ravel()))[:num]

        # Prefer interior points (farther from the border) to maximise radii.
        border = np.minimum.reduce(
            [points[:, 0], points[:, 1], 1 - points[:, 0], 1 - points[:, 1]]
        )
        idx = np.argsort(-border)  # descending distance to border
        return points[idx[:num]]

    centers = hex_centers(n)
    radii = compute_max_radii(centers)
    return centers, radii, radii.sum()


def compute_max_radii(centers):
    """Linear‑programming step: maximise sum of radii given fixed centres."""
    n = centers.shape[0]

    # Maximum radius limited by distance to the unit‑square border.
    border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )

    # Pairwise centre distances.
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)

    # Inequalities: r_i + r_j ≤ d_ij for i < j.
    rows = []
    rhs = []
    for i in range(n):
        for j in range(i + 1, n):
            coeff = np.zeros(n)
            coeff[i] = coeff[j] = 1.0
            rows.append(coeff)
            rhs.append(dists[i, j])

    # Combine border constraints (r_i ≤ border_i) with pair constraints.
    A_ub = np.vstack([np.eye(n), rows]) if rows else np.eye(n)
    b_ub = np.concatenate([border, rhs]) if rows else border

    # Objective: maximise sum(r_i) → minimise -sum(r_i).
    c = -np.ones(n)

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=(0, None),
        method="highs",
        options={"presolve": True},
    )
    return np.maximum(res.x, 0) if res.success else border
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
