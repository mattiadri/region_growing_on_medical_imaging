import time
import numpy as np
import matplotlib.pyplot as plt
from region_growing import region_growing
from show_how_it_works import generate_continuous_tentacles

def measure_and_plot_performance(output_file="performance_log_plot.png", num_repeats=5):
    """
    Measures and plots the computation time of the region_growing algorithm 
    as a function of image size on a log-log scale and saves the plot as an image file.
    """
    image_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    computation_times = []
    num_pixels = [size ** 2 for size in image_sizes]

    for size in image_sizes:
        image = generate_continuous_tentacles(random_state=42, img_size=size)
        image_array = np.array(image)
        
        times = []
        for _ in range(num_repeats):
            start_time = time.time()
            region = region_growing(image_array, threshold=10)
            times.append(time.time() - start_time)
        
        average_time = np.mean(times[1:])  # Skip the first measurement
        computation_times.append(average_time)

    # Plot on a log-log scale
    plt.figure(figsize=(7, 4))
    plt.loglog(num_pixels, computation_times, marker='o', label="Measured Times")
    plt.loglog(num_pixels, np.array(num_pixels) * computation_times[0] / num_pixels[0],
               linestyle="--", color="red", label="O(N) Reference Line")
    plt.title("Computation Time vs Number of Pixels (Log-Log Scale)", fontsize=7)
    plt.xlabel("Number of Pixels (log scale)", fontsize=5)
    plt.ylabel("Computation Time (log scale)", fontsize=5)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=5)
    plt.savefig(output_file)
    print(f"Log-log performance plot saved to {output_file}")

if __name__ == "__main__":
    measure_and_plot_performance(output_file="performance_plot.png", num_repeats=5)