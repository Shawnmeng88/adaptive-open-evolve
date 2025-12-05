# EVOLVE-BLOCK-START
"""Fast 26‑circle packing in a unit square.
Hex‑grid seed → hill‑climb centres → LP for optimal radii."""
import numpy as np, itertools, math
from scipy.optimize import linprog

def _hex(step=0.22):
    pts, dy, y, r = [], step*math.sqrt(3)/2, 0.0, 0
    while y <= 1 and len(pts) < 26:
        off = 0 if r%2==0 else step/2
        x = off
        while x <= 1 and len(pts) < 26:
            pts.append([x, y]); x += step
        y += dy; r += 1
    return np.array(pts[:26])

def _lp(cent):
    n = len(cent)
    A, b = [], []
    for i,(x,y) in enumerate(cent):
        A.append(np.eye(n)[i]); b.append(min(x,y,1-x,1-y))
    for i,j in itertools.combinations(range(n),2):
        row=np.zeros(n); row[i]=row[j]=1
        A.append(row); b.append(np.linalg.norm(cent[i]-cent[j]))
    res=linprog(-np.ones(n),A_ub=A,b_ub=b,bounds=[(0,None)]*n,method='highs')
    return res.x if res.success else np.zeros(n)

def _climb(base, it=2000, step=0.02, decay=0.999):
    best_c, best_r = base.copy(), _lp(base)
    best_s = best_r.sum()
    for _ in range(it):
        i = np.random.randint(26)
        cand = best_c.copy()
        cand[i] += np.random.uniform(-step,step,2)
        np.clip(cand,0,1,out=cand)
        rad = _lp(cand)
        s = rad.sum()
        if s>best_s:
            best_c,best_r,best_s = cand,rad,s
        step*=decay
    return best_c,best_r,best_s

def construct_packing(trials=3):
    best = (None,None,-1.0)
    for _ in range(trials):
        c,r,s = _climb(_hex())
        if s>best[2]: best=(c,r,s)
    return best

def run_packing():
    """Public entry point – returns centres, radii and their total length."""
    return construct_packing()

# EVOLVE-BLOCK-END

def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig,ax=plt.subplots(figsize=(6,6))
    ax.set_aspect('equal');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(True)
    for (x,y),r in zip(centers,radii):
        ax.add_patch(Circle((x,y),r,alpha=0.4))
    plt.title(f'Sum = {radii.sum():.4f}')
    plt.show()

if __name__=='__main__':
    c,r,s=run_packing()
    print(f'Sum of radii: {s:.6f}')
    # visualize(c,r)   # uncomment to see the packing