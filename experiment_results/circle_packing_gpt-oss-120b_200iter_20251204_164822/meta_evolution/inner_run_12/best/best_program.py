# EVOLVE-BLOCK-START
"""Vectorised push‑apart circle packing for n=26 (deterministic)."""
import numpy as np
from scipy.spatial import distance

SEED = 42                     # fixed seed – never changed
np.random.seed(SEED)

N = 26
MAX_ITER = 500
STEP0 = 0.12                  # initial displacement scale
DECAY = 0.95                  # step decay factor
DECAY_EVERY = 50             # iterations between decays
TOL_REL = 1e-6               # relative improvement tolerance


def _hex_lattice(n):
    """Generate ≥n points in a hexagonal lattice inside the unit square."""
    # spacing chosen so lattice comfortably fits 26 points
    a = 0.18
    rows = int(np.ceil(1 / (a * np.sqrt(3) / 2)))
    pts = []
    for r in range(rows):
        y = a * np.sqrt(3) / 2 * r + a / 2
        offset = (r % 2) * a / 2
        xs = np.arange(offset + a / 2, 1 - a / 2 + 1e-12, a)
        for x in xs:
            if 0 < x < 1 and 0 < y < 1:
                pts.append([x, y])
                if len(pts) == n:
                    return np.array(pts)
    return np.array(pts[:n])


def _max_radii(centers):
    """Exact maximal feasible radii for given centre positions."""
    side = np.minimum.reduce([centers[:, 0], 1 - centers[:, 0],
                              centers[:, 1], 1 - centers[:, 1]])
    d = distance.cdist(centers, centers, "euclidean")
    np.fill_diagonal(d, np.inf)
    rad = np.minimum(side, d.min(axis=1) / 2.0)
    return rad, d


def _push_apart(centers):
    """Iterative vectorised repulsion – returns converged centres & radii."""
    step = STEP0
    old_score = -np.inf
    for it in range(1, MAX_ITER + 1):
        rad, d = _max_radii(centers)

        # ---- compute repulsive forces where circles overlap ----
        overlap = rad[:, None] + rad[None, :] - d          # >0 ⇒ overlap
        np.fill_diagonal(overlap, 0.0)
        mask = overlap > 0
        if mask.any():
            # direction vectors between centres
            diff = centers[:, None, :] - centers[None, :, :]          # (n,n,2)
            dist = d[..., None] + 1e-12
            dir_vec = diff / dist
            # force magnitude proportional to overlap
            force = dir_vec * (overlap[..., None] * mask[..., None])
            total_force = force.sum(axis=1)                           # (n,2)
            centers += step * total_force
            # keep centres inside the square (tiny clipping)
            np.clip(centers, 0.0, 1.0, out=centers)

        # ---- evaluate improvement ----
        new_score = rad.sum()
        rel_imp = (new_score - old_score) / (old_score + 1e-12)
        if rel_imp < TOL_REL:
            break
        old_score = new_score

        # ---- decay step size periodically ----
        if it % DECAY_EVERY == 0:
            step *= DECAY
    return centers, rad


def construct_packing():
    """Build the packing, apply multi‑start perturbations and keep the best."""
    base = _hex_lattice(N)

    # three deterministic perturbations (including the unperturbed case)
    offsets = np.array([[0, 0],
                        [0.005, -0.003],
                        [-0.004, 0.006]])

    best_score = -np.inf
    best_centers = None
    best_radii = None

    for off in offsets:
        centres = base + off                     # deterministic shift
        centres = np.clip(centres, 0.0, 1.0)      # stay inside the square
        centres, radii = _push_apart(centres)
        score = radii.sum()
        if score > best_score:
            best_score, best_centers, best_radii = score, centres, radii

    return best_centers, best_radii, best_score


# EVOLVE-BLOCK-END


def run_packing():
    """Run the circle packing constructor for n=26."""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


# (visualisation code unchanged – omitted for brevity)
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s:.6f}")