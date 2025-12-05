# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    """
    Hexagonal packing of 26 circles in a unit square.
    Rows pattern: [6,5,6,5,4]  (total 26)
    Returns centers, radii, sum_of_radii.
    """
    rows = [6, 5, 6, 5, 4]
    n = sum(rows)

    # binary search for maximal radius r
    lo, hi = 0.0, 0.5
    for _ in range(40):
        r = (lo + hi) / 2.0
        ok = True
        # vertical extent check
        height = 2 * r + (len(rows) - 1) * np.sqrt(3) * r
        if height > 1.0:
            ok = False
        # horizontal extent per row
        for i, cnt in enumerate(rows):
            width = 2 * r * cnt
            if width > 1.0:
                ok = False
                break
        if ok:
            lo = r
        else:
            hi = r

    r = lo
    # generate centers
    centers = []
    y = r
    for i, cnt in enumerate(rows):
        offset = r if i % 2 else 0.0  # shift odd rows by half spacing
        x_start = r + offset
        for j in range(cnt):
            x = x_start + j * 2 * r
            if x > 1 - r + 1e-12:  # safety clamp
                continue
            centers.append([x, y])
        y += np.sqrt(3) * r

    centers = np.array(centers[:n])  # ensure exactly n points
    radii = np.full(n, r)
    sum_radii = radii.sum()
    return centers, radii, sum_radii
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
