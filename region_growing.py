import matplotlib.pyplot as plt
import numpy as np
from filters import apply_filter_pipeline
from data_loader import load_images_and_segmentations
from numba import njit
from collections import deque

@njit
def region_growing(image, seed=None, threshold=10):
    """
    Performs region growing using BFS with iterative mean update.

    Args:
        image (ndarray): Grayscale image.
        seed (tuple, optional): Seed (x, y). Defaults to the center of the image.
        threshold (int): Maximum allowed intensity difference from the current region mean.

    Returns:
        ndarray: Boolean array of the grown region.
    """
    h, w = image.shape

    # Default seed is the center of the image
    if seed is None:
        seed_x = h // 2
        seed_y = w // 2
    else:
        seed_x, seed_y = seed

    # Initialize region stats
    region_sum = float(image[seed_x, seed_y])
    region_count = 1

    visited = np.zeros((h, w), dtype=np.bool_)
    region = np.zeros((h, w), dtype=np.bool_)

    # Arrays to implement a BFS queue
    queue_x = np.empty(h * w, dtype=np.int32)
    queue_y = np.empty(h * w, dtype=np.int32)
    front, back = 0, 0

    # Enqueue seed and mark visited
    visited[seed_x, seed_y] = True
    queue_x[back] = seed_x
    queue_y[back] = seed_y
    back += 1

    # 8-connectivity
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while front < back:
        x = queue_x[front]
        y = queue_y[front]
        front += 1

        current_mean = region_sum / region_count
        # Check pixel homogeneity with respect to current region mean
        if abs(image[x, y] - current_mean) <= threshold:
            region[x, y] = True
            region_sum += image[x, y]
            region_count += 1

            # Enqueue neighbors
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w:
                    if not visited[nx, ny]:
                        visited[nx, ny] = True
                        queue_x[back] = nx
                        queue_y[back] = ny
                        back += 1
                        # Optional safety check:
                        # if back >= h * w:
                        #     break

    return region

def region_growing_step_by_step(image, threshold=10, seed=None):
    """Perform adaptive region growing using BFS, yielding intermediate regions.

    Args:
        image (ndarray): Grayscale input image.
        threshold (int): Allowed intensity difference from the current mean.
        seed (tuple, optional): (x, y) starting point. Defaults to image center.

    Yields:
        ndarray: Boolean array representing the region at each step.
    """
    h, w = image.shape

    # If no seed is provided, use image center
    if seed is None:
        seed_x, seed_y = h // 2, w // 2
    else:
        seed_x, seed_y = seed

    # Keeps track of whether each pixel was checked
    visited = np.zeros((h, w), dtype=bool)
    # Marks which pixels belong to the region
    region = np.zeros((h, w), dtype=bool)

    # We track the region's cumulative sum and how many pixels it has
    region_sum = 0.0
    region_count = 0

    # Initialize BFS
    queue = deque()
    queue.append((seed_x, seed_y))

    # 8-connectivity directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        x, y = queue.popleft()

        # If already visited, skip
        if visited[x, y]:
            continue
        visited[x, y] = True

        # For the very first pixel, we automatically accept it
        # and initialize our running mean
        if region_count == 0:
            region[x, y] = True
            region_sum += image[x, y]
            region_count += 1
            # Yield the region as it is after including this pixel
            yield region.copy()

            # Enqueue neighbors
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                    queue.append((nx, ny))

        else:
            current_mean = region_sum / region_count

            # If the new pixel is within threshold of the current mean, accept it
            if abs(image[x, y] - current_mean) <= threshold:
                region[x, y] = True
                region_sum += image[x, y]
                region_count += 1
                yield region.copy()

                # Enqueue neighbors
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                        queue.append((nx, ny))

    return region

def calculate_iou(region, segmentation):
    """
    Calculate the Intersection over Union (IoU) between a region and segmentation.

    Parameters:
        region (ndarray): Boolean array of the region.
        segmentation (ndarray): Boolean array of the ground truth segmentation.

    Returns:
        float: IoU value.
    """
    intersection = np.logical_and(region, segmentation).sum()
    union = np.logical_or(region, segmentation).sum()
    return intersection / union if union > 0 else 0

def inspect_algorithm(image_path, segmentation_path, slice_index=10, pipeline='pipeline_1', threshold=10, save_figures=False):
    """
    Inspect the region growing algorithm step-by-step on a single image.

    Parameters:
        image_path (str): Path to the input `.mhd` image file.
        segmentation_path (str): Path to the ground truth `.mhd` segmentation file.
        slice_index (int): Index of the image slice to process.
        pipeline (str): Name of the filtering pipeline to apply.
        threshold (int): Intensity difference threshold for region growing.
        save_figures (bool): Whether to save the generated plots.

    Returns:
        ndarray: Boolean array of the final grown region.
    """
    # Load the image and corresponding segmentation
    image, segmentation = load_images_and_segmentations(image_path, segmentation_path, slice_index)

    # Apply the chosen filtering pipeline
    filtered_image = apply_filter_pipeline(image, pipeline)
    # Perform region growing
    seed = (filtered_image.shape[0] // 2, filtered_image.shape[1] // 2)
    mask = region_growing(filtered_image, seed=seed, threshold=threshold)

    # Visualize results
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(filtered_image, cmap='gray')
    axes[1].set_title("Filtered Image")
    axes[1].axis("off")

    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title("Region Growing Mask")
    axes[2].axis("off")

    axes[3].imshow(segmentation, cmap='gray')
    axes[3].set_title("Ground Truth")
    axes[3].axis("off")

    axes[4].imshow(segmentation, cmap='gray', alpha=0.5)
    axes[4].imshow(mask, cmap='jet', alpha=0.5)
    axes[4].set_title("Overlap (Mask vs GT)")
    axes[4].axis("off")

    plt.tight_layout()
    plt.show()

    # Optionally save the visualization
    if save_figures:
        fig.savefig("inspection_results.png")

    return mask