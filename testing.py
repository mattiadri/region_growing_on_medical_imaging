import os
from region_growing import region_growing, calculate_iou
from filters import apply_filter_pipeline
from data_loader import load_images_and_segmentations

def test_model(test_folder, threshold, slice_index=10, pipeline='pipeline_1'):
    results = []

    files = [file for file in os.listdir(test_folder) 
             if file.endswith(".mhd") and not file.endswith("_segmentation.mhd")]

    for file in files:
        image_path = os.path.join(test_folder, file)
        segmentation_path = image_path.replace(".mhd", "_segmentation.mhd")

        # Load images and apply filters
        image, segmentation = load_images_and_segmentations(image_path, segmentation_path, slice_index)
        filtered_image = apply_filter_pipeline(image, pipeline)

        # Perform region growing and calculate IoU
        seed = (filtered_image.shape[0] // 2, filtered_image.shape[1] // 2)
        region = region_growing(filtered_image, seed, threshold)
        iou = calculate_iou(region, segmentation)

        # Append results
        results.append({"file": file, "IoU": iou})

    return results