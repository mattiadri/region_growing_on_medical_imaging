import SimpleITK as sitk
import numpy as np

def load_images_and_segmentations(image_path, segmentation_path, slice_index=10):
    """
    Load an image and its corresponding segmentation and return the specified slice.
    
    Args:
        image_path (str): Path to the `.mhd` image file.
        segmentation_path (str): Path to the `.mhd` segmentation file.
        slice_index (int): The index of the slice to extract.
        
    Returns:
        tuple: (image_slice, segmentation_slice) as NumPy arrays.
    """
    # Load image
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # Load segmentation
    segmentation = sitk.ReadImage(segmentation_path)
    segmentation_array = sitk.GetArrayFromImage(segmentation)

    # Extract the specified slice
    image_slice = image_array[slice_index]
    segmentation_slice = segmentation_array[slice_index]

    return image_slice, segmentation_slice