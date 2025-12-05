# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import linprog


def _max_radii_lp(centers: np.ndarray) -> np.ndarray:
    """Linear program that maximises the sum of radii for fixed centres."""
    n = centers.shape[0]

    # distance to each side of the unit square
    border = np.minimum.reduce(
        [centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]]
    )

    # pairwise distance constraints: r_i + r_j <= d_ij
    m = n * (n - 1) // 2
    A = np.zeros((m, n))
    b = np.empty(m)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            b[idx] = np.linalg.norm(centers[i] - centers[j]) - 1e-12
            idx += 1

    bounds = [(0.0, border[i]) for i in range(n)]

    res = linprog(
        c=-np.ones(n),
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not res.success:
        return border.copy()
    # clamp to border with a tiny epsilon to avoid numerical slip
    return np.clip(res.x, 0.0, border)


def _deterministic_layout() -> np.ndarray:
    """Classic three‑ring deterministic centre layout."""
    n = 26
    centers = np.zeros((n, 2))
    centers[0] = [0.5, 0.5]                     # central circle
    for i in range(8):                         # inner ring
        ang = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.30 * np.cos(ang),
                         0.5 + 0.30 * np.sin(ang)]
    for i in range(16):                        # outer ring
        ang = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.70 * np.cos(ang),
                         0.5 + 0.70 * np.sin(ang)]
    return np.clip(centers, 0.01, 0.99)


def _random_layout(rng: np.random.Generator) -> np.ndarray:
    """Generate a random layout safely away from the borders."""
    return rng.uniform(0.05, 0.95, size=(26, 2))


def _verify(centers: np.ndarray, radii: np.ndarray, eps: float = 1e-9) -> bool:
    """Return True iff all constraints are satisfied (with tiny tolerance)."""
    n = len(radii)
    # border constraints
    if np.any(centers[:, 0] - radii < -eps):
        return False
    if np.any(centers[:, 0] + radii > 1 + eps):
        return False
    if np.any(centers[:, 1] - radii < -eps):
        return False
    if np.any(centers[:, 1] + radii > 1 + eps):
        return False
    # pairwise non‑overlap
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > d + eps:
                return False
    return True


def _local_search(
    start_centers: np.ndarray,
    rng: np.random.Generator,
    max_iters: int = 12000,
    init_step: float = 0.10,
    decay: float = 0.9993,
    temp_start: float = 5e-3,
    temp_decay: float = 0.9997,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulated‑annealing hill‑climb on centre positions."""
    n = start_centers.shape[0]
    best_centers = start_centers.copy()
    best_radii = _max_radii_lp(best_centers)
    best_score = best_radii.sum()

    step = init_step
    temp = temp_start
    no_improve = 0

    for _ in range(max_iters):
        i = rng.integers(n)
        proposal = best_centers[i] + rng.uniform(-step, step, size=2)
        proposal = np.clip(proposal, 0.01, 0.99)

        trial_centers = best_centers.copy()
        trial_centers[i] = proposal

        trial_radii = _max_radii_lp(trial_centers)
        trial_score = trial_radii.sum()

        delta = trial_score - best_score
        accept = False
        if delta > 1e-12:
            accept = True
        else:
            if temp > 0:
                prob = math.exp(delta / temp)
                if rng.random() < prob:
                    accept = True

        if accept:
            best_centers = trial_centers
            best_radii = trial_radii
            best_score = trial_score
            no_improve = 0
        else:
            no_improve += 1

        step *= decay
        temp *= temp_decay
        if step < 1e-4 or no_improve > 2000:
            break

    return best_centers, best_radii, best_score


def _refine_centers(
    centers: np.ndarray,
    rng: np.random.Generator,
    iterations: int = 2000,
    delta: float = 0.001,
) -> np.ndarray:
    """Tiny random nudges kept only if they improve the LP sum."""
    current_radii = _max_radii_lp(centers)
    current_score = current_radii.sum()
    n = centers.shape[0]

    for _ in range(iterations):
        i = rng.integers(n)
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction) + 1e-12
        proposal = centers[i] + delta * direction
        proposal = np.clip(proposal, 0.01, 0.99)

        new_centers = centers.copy()
        new_centers[i] = proposal
        new_radii = _max_radii_lp(new_centers)
        new_score = new_radii.sum()

        if new_score > current_score + 1e-12:
            centers = new_centers
            current_score = new_score
            current_radii = new_radii

    return centers


def _refine_multi(
    centers: np.ndarray,
    rng: np.random.Generator,
    passes: int = 4,
    init_delta: float = 0.002,
) -> np.ndarray:
    """Apply several passes of _refine_centers with decreasing step size."""
    delta = init_delta
    for _ in range(passes):
        centers = _refine_centers(centers, rng, iterations=1500, delta=delta)
        delta *= 0.5
    return centers


def construct_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Build a 26‑circle packing using multiple restarts, two‑phase annealing and deep refinement."""
    rng = np.random.default_rng(12345)

    # generate a richer pool of starting layouts
    starts = [_deterministic_layout()] + [_random_layout(rng) for _ in range(7)]

    best_centers, best_radii, best_score = None, None, -np.inf

    for start in starts:
        # ----- coarse annealing phase -----
        c, _, _ = _local_search(
            start,
            rng,
            max_iters=8000,
            init_step=0.12,
            decay=0.9995,
            temp_start=5e-3,
            temp_decay=0.9997,
        )

        # ----- progressive multi‑pass refinement -----
        c = _refine_multi(c, rng, passes=6, init_delta=0.002)

        # ----- fine‑grained annealing phase -----
        c, _, _ = _local_search(
            c,
            rng,
            max_iters=4000,
            init_step=0.02,
            decay=0.9999,
            temp_start=1e-4,
            temp_decay=0.9999,
        )

        # final LP solve
        r = _max_radii_lp(c)
        score = r.sum()

        # safety fallback
        if not _verify(c, r):
            c = _deterministic_layout()
            r = _max_radii_lp(c)
            score = r.sum()

        if score > best_score:
            best_centers, best_radii, best_score = c, r, score

    # ----- final global polishing on the best solution -----
    best_centers = _refine_multi(best_centers, rng, passes=8, init_delta=0.001)
    best_radii = _max_radii_lp(best_centers)
    best_score = best_radii.sum()
    if not _verify(best_centers, best_radii):
        best_centers = _deterministic_layout()
        best_radii = _max_radii_lp(best_centers)
        best_score = best_radii.sum()

    return best_centers, best_radii, best_score
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
