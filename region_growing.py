import matplotlib.pyplot as plt
import numpy as np
from filters import apply_filter_pipeline
from data_loader import load_images_and_segmentations

def region_growing(image, seed, threshold):
    visited = np.zeros_like(image, dtype=bool)
    region = np.zeros_like(image, dtype=bool)
    stack = [seed]
    seed_value = image[seed]

    while stack:
        x, y = stack.pop()
        if visited[x, y]:
            continue
        visited[x, y] = True

        if abs(image[x, y] - seed_value) <= threshold:
            region[x, y] = True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                    stack.append((nx, ny))
    return region

def calculate_iou(region, segmentation):
    intersection = np.logical_and(region, segmentation).sum()
    union = np.logical_or(region, segmentation).sum()
    return intersection / union if union > 0 else 0

def inspect_algorithm(image_path, segmentation_path, slice_index=10, pipeline='pipeline_1', threshold=10, save_figures=False):
    """
    Inspect the steps of the algorithm on a single image.

    Args:
        image_path (str): Path to the `.mhd` image file.
        segmentation_path (str): Path to the `.mhd` segmentation file.
        slice_index (int): The index of the slice to process.
        pipeline (str): The filtering pipeline to use.
        threshold (float): The threshold for region growing.
        save_figures (bool): Whether to save the figures as images.

    Returns:
        None: Displays the plots for inspection.
    """
    # Load image and segmentation
    image, segmentation = load_images_and_segmentations(image_path, segmentation_path, slice_index)

    # Apply filtering pipeline
    filtered_image = apply_filter_pipeline(image, pipeline)

    # Perform region growing
    seed = (filtered_image.shape[0] // 2, filtered_image.shape[1] // 2)
    mask = region_growing(filtered_image, seed, threshold)

    # Plot the results
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

    # Save figures if requested
    if save_figures:
        fig.savefig("inspection_results.png")