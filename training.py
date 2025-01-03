import os
import numpy as np
from region_growing import region_growing, calculate_iou
from filters import apply_filter_pipeline
from data_loader import load_images_and_segmentations

def estimate_threshold(training_folder, slice_index=10, pipeline='pipeline_1'):
    thresholds = []

    for file in os.listdir(training_folder):
        # Consider only .mhd files that are not segmentation files
        if file.endswith(".mhd") and not file.endswith("_segmentation.mhd"):
            image_path = os.path.join(training_folder, file)
            segmentation_path = image_path.replace(".mhd", "_segmentation.mhd")

            # Load images and apply filters
            image, segmentation = load_images_and_segmentations(image_path, segmentation_path, slice_index)
            filtered_image = apply_filter_pipeline(image, pipeline)

            # Perform region growing and calculate IoU
            seed = (filtered_image.shape[0] // 2, filtered_image.shape[1] // 2)
            for threshold in range(1, 500, 5):
                region = region_growing(filtered_image, seed, threshold)
                iou = calculate_iou(region, segmentation)
                if iou > 0:
                    thresholds.append(threshold)
                    break

    return np.mean(thresholds)