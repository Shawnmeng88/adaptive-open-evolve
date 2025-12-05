# EVOLVE-BLOCK-START
import numpy as np

def compute_max_radii(centers):
    n = len(centers)
    r = np.full(n, np.inf)
    r = np.minimum(r, np.minimum.reduce([centers[:,0], centers[:,1],
                                        1-centers[:,0], 1-centers[:,1]]))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i]-centers[j]) / 2.0
            if d < r[i]: r[i] = d
            if d < r[j]: r[j] = d
    return r

def hex_init():
    n, cols, rows = 26, 5, 6
    rh = 1/(2*cols)
    rv = 1/(2+(rows-1)*np.sqrt(3))
    r = min(rh, rv)
    pts = []
    for row in range(rows):
        y = r + row*np.sqrt(3)*r
        xs = (r + np.arange(cols)*2*r) if row%2==0 else (2*r + np.arange(cols-1)*2*r)
        for x in xs:
            pts.append((x,y))
            if len(pts)==n: break
        if len(pts)==n: break
    return np.array(pts)

def construct_packing():
    best_sum = -1.0
    best_centers = None
    rng = np.random.default_rng()
    for _ in range(3):                     # three independent runs
        centers = hex_init()
        cur_sum = compute_max_radii(centers).sum()
        step = 0.03
        for i in range(2000):
            idx = rng.integers(len(centers))
            delta = rng.uniform(-step, step, 2)
            new_c = np.clip(centers[idx]+delta,0,1)
            cand = centers.copy()
            cand[idx] = new_c
            s = compute_max_radii(cand).sum()
            if s > cur_sum:
                centers, cur_sum = cand, s
            step *= 0.9999
        if cur_sum > best_sum:
            best_sum, best_centers = cur_sum, centers.copy()
    radii = compute_max_radii(best_centers)
    return best_centers, radii, best_sum
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
