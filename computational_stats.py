import time
import numpy as np
import matplotlib.pyplot as plt
from region_growing import region_growing
from show_how_it_works import generate_continuous_tentacles

def measure_and_plot_performance(output_file="performance_plot.png", num_repeats=5):
    """
    Measures and plots the computation time of the region_growing algorithm 
    as a function of image size and saves the plot as an image file.
    
    Args:
        output_file (str): Path to save the output plot image.
        num_repeats (int): Number of times to repeat each measurement for averaging.
    """
    image_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]  # Different image sizes
    computation_times = []

    for size in image_sizes:
        # Generate image
        image = generate_continuous_tentacles(random_state=42, img_size=size)
        
        # Convert to NumPy array for region growing
        image_array = np.array(image)
        
        # Measure computation time (average over num_repeats)
        times = []
        for _ in range(num_repeats):
            start_time = time.time()
            region = region_growing(image_array, threshold=10)
            times.append(time.time() - start_time)
        
        # Ignore the first run (warm-up effect) and compute the average
        average_time = np.mean(times[1:])  # Skip the first measurement
        computation_times.append(average_time)
    
    # Normalize the Y=X line for comparison
    max_time = max(computation_times)
    y_equal_x = [size / max(image_sizes) * max_time for size in image_sizes]

    # Plot results
    plt.figure(figsize=(7, 4))
    plt.plot(image_sizes, computation_times, marker='o', label="Measured Times")
    plt.plot(image_sizes, y_equal_x, linestyle="--", color="red", label="Reference Line (Y=x)")
    plt.title("Computation Time vs Image Size (Averaged)", fontsize=7)
    plt.xlabel("Image Size (pixels)", fontsize=5)
    plt.ylabel("Computation Time (seconds)", fontsize=5)
    plt.grid(True)
    plt.legend(fontsize=5)

    # Save the plot
    plt.savefig(output_file)
    print(f"Performance plot saved to {output_file}")

if __name__ == "__main__":
    measure_and_plot_performance(output_file="performance_plot.png", num_repeats=5)