# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    pts = []
    ys = [0.1, 0.3, 0.5, 0.7, 0.9]
    for i, y in enumerate(ys):
        cnt = 5 if i % 2 == 0 else 4
        xs = np.linspace(0.1 if i % 2 == 0 else 0.2, 0.9, cnt)
        for x in xs:
            pts.append([x, y])
    pts += [[0.05, 0.05], [0.95, 0.05], [0.5, 0.95]]
    centers = np.array(pts[:26])
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)

def compute_max_radii(c):
    n = len(c)
    r = np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])
    for _ in range(10):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(c[i] - c[j])
                if r[i] + r[j] > d:
                    excess = r[i] + r[j] - d
                    if r[i] >= r[j]:
                        r[i] -= excess
                    else:
                        r[j] -= excess
                    changed = True
        if not changed:
            break
    return np.clip(r, 0, None)
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
