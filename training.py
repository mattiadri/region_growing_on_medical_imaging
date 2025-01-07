from tqdm import tqdm
import os
import numpy as np
from region_growing import region_growing, calculate_iou
from filters import apply_filter_pipeline
from data_loader import load_images_and_segmentations
from scipy.ndimage import center_of_mass

def estimate_threshold(training_folder, slice_index=10, pipeline='pipeline_1'):
    thresholds = []

    files = [file for file in os.listdir(training_folder) if file.endswith(".mhd") and not file.endswith("_segmentation.mhd")]

    for file in tqdm(files, desc="Estimating Threshold"):
        image_path = os.path.join(training_folder, file)
        segmentation_path = image_path.replace(".mhd", "_segmentation.mhd")

        # Load images and apply filters
        image, segmentation = load_images_and_segmentations(image_path, segmentation_path, slice_index)
        filtered_image = apply_filter_pipeline(image, pipeline)

        # Calculate the centroid or fallback to center
        if np.any(segmentation):
            seed = tuple(map(int, center_of_mass(segmentation)))
        else:
            seed = (filtered_image.shape[0] // 2, filtered_image.shape[1] // 2)

        # Test different thresholds and find the best IoU
        best_threshold = None
        best_iou = 0
        for threshold in range(1, 500, 50):
            region = region_growing(filtered_image, seed, threshold)
            iou = calculate_iou(region, segmentation)
            if iou > best_iou:
                best_iou = iou
                best_threshold = threshold

        if best_threshold is not None:
            thresholds.append(best_threshold)

    # Return the average threshold
    return np.mean(thresholds)