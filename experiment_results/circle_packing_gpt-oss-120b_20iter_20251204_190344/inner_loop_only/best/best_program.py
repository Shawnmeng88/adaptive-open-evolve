# EVOLVE-BLOCK-START
"""26‑circle packing – hill‑climb on a grid seed with LP radii."""
import itertools, numpy as np, scipy.optimize as opt
from random import random, gauss

def _lp_radii(cent):
    n = len(cent)
    c = -np.ones(n)                     # maximise Σr  → minimise –Σr
    A, b = [], []

    # border constraints
    for i, (x, y) in enumerate(cent):
        A.append(np.eye(1, n, k=i)[0])
        b.append(min(x, y, 1 - x, 1 - y))

    # pair‑wise non‑overlap
    for i, j in itertools.combinations(range(n), 2):
        d = np.linalg.norm(cent[i] - cent[j])
        row = np.zeros(n); row[i] = row[j] = 1
        A.append(row); b.append(d)

    res = opt.linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')
    return res.x if res.success else np.zeros(n)

def _initial_grid():
    g = np.linspace(0.1, 0.9, 5)               # 5×5 lattice
    pts = np.column_stack([p.ravel() for p in np.meshgrid(g, g)])[:26]
    if pts.shape[0] < 26:                      # add centre if needed
        pts = np.vstack([pts, [0.5, 0.5]])
    return pts

def construct_packing(iters=2500, step=0.07):
    """Hill‑climb from a grid seed; returns best centres, radii, sum."""
    best_c = _initial_grid()
    best_r = _lp_radii(best_c)
    best_val = best_r.sum()

    for _ in range(iters):
        i = np.random.randint(26)                       # pick a circle
        cand = best_c.copy()
        # propose a small jitter, stay inside [0,1]²
        cand[i] += [gauss(0, step), gauss(0, step)]
        cand[i] = np.clip(cand[i], 0, 1)

        rad = _lp_radii(cand)
        s = rad.sum()
        if s > best_val:                               # accept improvement
            best_c, best_r, best_val = cand, rad, s
    return best_c, best_r, best_val
# EVOLVE-BLOCK-END

def run_packing():
    """Entry point – returns centres, radii and total radius sum."""
    return construct_packing()

def visualize(centers, radii):
    """Simple Matplotlib visualisation."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots()
    ax.set_aspect('equal'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for (x, y), r in zip(centers, radii):
        ax.add_patch(Circle((x, y), r, alpha=0.5, edgecolor='k'))
    plt.title(f'Sum of radii = {radii.sum():.4f}')
    plt.show()

if __name__ == '__main__':
    C, R, S = run_packing()
    print(f'Sum of radii: {S:.6f}')
    # visualize(C, R)   # uncomment to see the packing