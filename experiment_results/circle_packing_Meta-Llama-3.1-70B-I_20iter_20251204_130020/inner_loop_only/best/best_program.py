# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))

    # Place circles in a structured pattern
    # This is a simple pattern - evolution will improve this

    # First, place a large circle in the center
    centers[0] = [0.5, 0.5]

    # Place 8 circles around it in a ring
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for i, angle in enumerate(angles):
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    # Place 16 more circles in an outer ring
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    for i, angle in enumerate(angles):
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]

    # Additional positioning adjustment to make sure all circles
    # are inside the square and don't overlap
    # Clip to ensure everything is inside the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    # Improve fitness by adjusting circle positions and radii
    for _ in range(100):
        # Randomly select two circles
        i, j = np.random.choice(n, 2, replace=False)

        # Calculate the distance between the two circles
        dist = np.linalg.norm(centers[i] - centers[j])

        # If the circles are too close, adjust their radii
        if radii[i] + radii[j] > dist:
            # Scale both radii proportionally
            scale = dist / (radii[i] + radii[j])
            radii[i] *= scale
            radii[j] *= scale

        # Randomly adjust the position of one circle
        k = np.random.choice([i, j])
        centers[k] += np.random.uniform(-0.01, 0.01, 2)

        # Clip to ensure everything is inside the unit square
        centers = np.clip(centers, 0.01, 0.99)

        # Recompute maximum valid radii
        radii = compute_max_radii(centers)

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # First, limit by distance to square borders
    radii = np.minimum(radii, centers[:, 0])
    radii = np.minimum(radii, centers[:, 1])
    radii = np.minimum(radii, 1 - centers[:, 0])
    radii = np.minimum(radii, 1 - centers[:, 1])

    # Then, limit by distance to other circles
    # Each pair of circles with centers at distance d can have
    # sum of radii at most d to avoid overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii


def genetic_optimization(centers, radii, population_size=100, generations=100):
    """
    Perform genetic optimization to improve the circle packing.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
        population_size: int, size of the population
        generations: int, number of generations

    Returns:
        np.array of shape (n, 2) with optimized (x, y) coordinates
        np.array of shape (n) with optimized radius of each circle
    """
    n = centers.shape[0]

    # Initialize population
    population = np.random.uniform(0, 1, size=(population_size, n, 2))

    for generation in range(generations):
        # Evaluate fitness
        fitness = np.zeros(population_size)
        for i in range(population_size):
            new_centers = population[i]
            new_radii = compute_max_radii(new_centers)
            fitness[i] = np.sum(new_radii)

        # Select parents
        parents = np.random.choice(population_size, size=2, p=fitness / np.sum(fitness))

        # Crossover
        offspring = (population[parents[0]] + population[parents[1]]) / 2

        # Mutation
        offspring += np.random.uniform(-0.01, 0.01, size=(n, 2))

        # Replace least fit individual
        population[np.argmin(fitness)] = offspring

    # Return best individual
    best_individual = population[np.argmax(fitness)]
    best_radii = compute_max_radii(best_individual)
    return best_individual, best_radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    optimized_centers, optimized_radii = genetic_optimization(centers, radii)
    return optimized_centers, optimized_radii, np.sum(optimized_radii)


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

    # Uncomment to visualize:
    visualize(centers, radii)