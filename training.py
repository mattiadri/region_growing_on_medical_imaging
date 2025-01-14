from tqdm import tqdm
import os
import numpy as np
from region_growing import region_growing, calculate_iou
from filters import apply_filter_pipeline
from data_loader import load_images_and_segmentations
from scipy.ndimage import center_of_mass

def estimate_threshold(training_folder, slice_index=10, pipeline='pipeline_1', coarse_steps=5, fine_steps=10):
    thresholds = []

    files = [file for file in os.listdir(training_folder) 
             if file.endswith(".mhd") and not file.endswith("_segmentation.mhd")]

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

        # Coarse search for threshold
        best_threshold = None
        best_iou = 0
        coarse_range = np.linspace(0.1, 2, num=coarse_steps)
        for threshold in coarse_range:
            region = region_growing(filtered_image, seed, threshold)
            iou = calculate_iou(region, segmentation)
            if iou > best_iou:
                best_iou = iou
                best_threshold = threshold

        # Refine search around the best threshold
        if best_threshold is not None:
            fine_range = np.linspace(
                max(0.1, best_threshold - (2 / coarse_steps)),
                min(2, best_threshold + (2 / coarse_steps)),
                num=fine_steps
            )
            for threshold in fine_range:
                region = region_growing(filtered_image, seed, threshold)
                iou = calculate_iou(region, segmentation)
                if iou > best_iou:
                    best_iou = iou
                    best_threshold = threshold

            thresholds.append(best_threshold)

    # Return the median threshold
    if thresholds:
        return np.median(thresholds)
    else:
        print(f"Warning: No thresholds found for folder {training_folder} with pipeline {pipeline}.")
        return None