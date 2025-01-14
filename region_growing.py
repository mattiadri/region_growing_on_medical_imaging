import matplotlib.pyplot as plt
import numpy as np
from filters import apply_filter_pipeline
from data_loader import load_images_and_segmentations
from numba import njit
from collections import deque

@njit
def region_growing(image, seed=None, threshold=10):
    """
    Performs region growing using BFS.

    Args:
        image (ndarray): Grayscale image.
        seed (tuple, optional): Seed (x, y). Defaults to the center of the image.
        threshold (int): Intensity difference threshold.

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
    
    seed_value = image[seed_x, seed_y]
    
    # Precompute which pixels are within threshold
    mask = np.abs(image - seed_value) <= threshold
    
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
        
        # If within threshold, add to region
        if mask[x, y]:
            region[x, y] = True
            
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
    """
    Perform region growing step-by-step using BFS and yield intermediate regions.

    Parameters:
        image (ndarray): Grayscale input image.
        seed (tuple, optional): Starting point for region growing (x, y). Defaults to the center of the image.
        threshold (int): Intensity difference threshold for region inclusion.

    Yields:
        ndarray: Boolean array of the region at each step.
    """
    h, w = image.shape

    # Default to the center of the image if no seed is provided
    if seed is None:
        seed_x, seed_y = h // 2, w // 2
    else:
        seed_x, seed_y = seed

    seed_value = image[seed_x, seed_y]

    # Precompute valid region based on the threshold
    mask = np.abs(image - seed_value) <= threshold

    # Initialize visited and region arrays
    visited = np.zeros_like(mask, dtype=bool)
    region = np.zeros_like(mask, dtype=bool)
    queue = deque([(seed_x, seed_y)])

    # 8-connectivity directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        x, y = queue.popleft()

        if visited[x, y]:
            continue
        visited[x, y] = True

        if mask[x, y]:
            region[x, y] = True
            yield region.copy()  # Yield the current region

            # Enqueue valid neighbors
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                    queue.append((nx, ny))

    return region  # Final region after BFS

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