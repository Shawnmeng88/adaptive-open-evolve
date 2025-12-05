"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import linprog  # Pre-imported for LP approaches

# EVOLVE-BLOCK-START
def construct_packing():
    """
    Deterministic construction for 26 circles.
    Uses two complementary seed layouts (regular grid and a hexagonal lattice
    tuned by linear programming) and refines each with a deterministic
    hill‑climbing loop.  The best result (largest sum of radii) is returned.
    """
    import numpy as np
    from scipy.optimize import linprog

    n = 26
    eps = 1e-9
    rng = np.random.default_rng(0)  # deterministic RNG for all hill‑climbing steps

    # ------------------------------------------------------------------
    # Helper: linear‑program optimiser for a fixed centre set
    # ------------------------------------------------------------------
    def solve_lp(centers: np.ndarray):
        m = centers.shape[0]

        # wall limits
        wall = np.minimum.reduce(
            [centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]]
        )

        # pairwise distances
        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.linalg.norm(diff, axis=2)

        # inequality matrix: r_i <= wall_i
        A = np.eye(m)
        b = wall.copy()

        # r_i + r_j <= d_ij
        rows = []
        vals = []
        for i in range(m):
            for j in range(i + 1, m):
                row = np.zeros(m)
                row[i] = 1.0
                row[j] = 1.0
                rows.append(row)
                vals.append(dists[i, j])
        if rows:
            A = np.vstack([A, np.array(rows)])
            b = np.concatenate([b, np.array(vals)])

        # maximise sum(r)  → minimise -sum(r)
        c = -np.ones(m)

        res = linprog(
            c,
            A_ub=A,
            b_ub=b,
            bounds=[(0.0, None)] * m,
            method="highs",
            options={"presolve": True},
        )
        if res.success:
            radii = res.x
        else:
            radii = compute_max_radii(centers)
        return radii, float(np.sum(radii))

    # ------------------------------------------------------------------
    # Helper: greedy fallback (used only if LP fails)
    # ------------------------------------------------------------------
    def compute_max_radii(centers: np.ndarray):
        m = centers.shape[0]
        r = np.ones(m)
        for i in range(m):
            x, y = centers[i]
            r[i] = min(x, y, 1.0 - x, 1.0 - y)
        for i in range(m):
            for j in range(i + 1, m):
                d = np.linalg.norm(centers[i] - centers[j])
                if r[i] + r[j] > d:
                    scale = d / (r[i] + r[j])
                    r[i] *= scale
                    r[j] *= scale
        return r

    # ------------------------------------------------------------------
    # Layout 1: regular 5×5 grid + one extra corner point
    # ------------------------------------------------------------------
    def grid_layout():
        step = 0.2
        offset = 0.1
        pts = []
        for i in range(5):
            for j in range(5):
                pts.append([offset + i * step, offset + j * step])
        pts.append([0.95, 0.05])
        return np.array(pts[:n], dtype=float)

    # ------------------------------------------------------------------
    # Layout 2: hexagonal lattice tuned by LP
    # ------------------------------------------------------------------
    def generate_hex_lattice(dx: float) -> np.ndarray:
        dy = np.sqrt(3.0) / 2.0 * dx
        pts = []
        row = 0
        y = dy / 2.0
        while y < 1.0 - dy / 2.0 + eps:
            offset = 0.0 if row % 2 == 0 else dx / 2.0
            x = dx / 2.0 + offset
            while x < 1.0 - dx / 2.0 + eps:
                pts.append([x, y])
                x += dx
            y += dy
            row += 1
        return np.array(pts, dtype=float)

    def fps_select(points: np.ndarray, k: int) -> np.ndarray:
        selected = [points[0]]
        remaining = points[1:].copy()
        while len(selected) < k:
            dists = np.min(
                np.linalg.norm(
                    remaining[:, None, :] - np.array(selected)[None, :, :], axis=2
                ),
                axis=1,
            )
            idx = np.argmax(dists)
            selected.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)
        return np.array(selected, dtype=float)

    def best_hex_layout():
        best_centers = None
        best_score = -1.0
        for dx in (0.18, 0.16, 0.14):
            lattice = generate_hex_lattice(dx)
            if lattice.shape[0] < n:
                continue
            centers = fps_select(lattice, n)
            _, score = solve_lp(centers)
            if score > best_score:
                best_score = score
                best_centers = centers
        return best_centers

    # ------------------------------------------------------------------
    # Hill‑climbing optimiser (deterministic, same RNG for reproducibility)
    # ------------------------------------------------------------------
    def hill_climb(start_pts, iters=3000, sigma_start=0.02):
        pts = start_pts.copy()
        radii, best_val = solve_lp(pts)

        for k in range(iters):
            sigma = sigma_start * (1.0 - k / iters)  # linear cooling
            idx = rng.integers(n)

            proposal = pts[idx] + rng.normal(scale=sigma, size=2)
            proposal = np.clip(proposal, eps, 1.0 - eps)

            new_pts = pts.copy()
            new_pts[idx] = proposal

            new_radii, new_val = solve_lp(new_pts)

            if new_val > best_val + 1e-12:
                pts = new_pts
                radii = new_radii
                best_val = new_val

        return pts, radii, best_val

    # ------------------------------------------------------------------
    # Evaluate both initial layouts and keep the best final result
    # ------------------------------------------------------------------
    candidates = []

    # 1) Grid layout
    grid_pts = grid_layout()
    candidates.append(grid_pts)

    # 2) Hex layout (may be None if all spacings fail)
    hex_pts = best_hex_layout()
    if hex_pts is not None:
        candidates.append(hex_pts)

    best_final = None
    best_score = -1.0
    for init in candidates:
        final_pts, final_radii, final_score = hill_climb(init)
        if final_score > best_score:
            best_score = final_score
            best_final = (final_pts, final_radii, final_score)

    # Return the best packing found
    return best_final[0], best_final[1], float(best_final[2])
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
