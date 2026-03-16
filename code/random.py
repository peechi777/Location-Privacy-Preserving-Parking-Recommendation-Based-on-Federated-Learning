import numpy as np
import matplotlib.pyplot as plt

#設定半徑
def plot_random_points(center, radii, num_samples):
    plt.figure(figsize=(8, 8))
    plt.title("Random Positions within Various Radii")
    colors = ['blue', 'green', 'red']
    labels = ['2 km', '5 km', '7 km']
    
 
    for radius, color, label in zip(radii, colors, labels):
        angles = np.random.uniform(0, 2 * np.pi, num_samples)
        r = np.random.uniform(0, radius, num_samples)
        x = center[0] + r * np.cos(angles)
        y = center[1] + r * np.sin(angles)
        plt.scatter(x, y, color=color, alpha=0.4, label=f"Radius = {label}")

    plt.legend()
    plt.grid(True)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.axis('equal')  # Set equal scaling by changing axis limits
    plt.show()


center_point = (0, 0)
radii = [2, 5, 7]
num_samples = 1000


plot_random_points(center_point, radii, num_samples)