import numpy as np
import SimpleITK as sitk
from skimage import exposure, filters, restoration

def apply_filter_pipeline(image, pipeline='pipeline_1'):
    """
    Applies different preprocessing pipelines before region growing or segmentation.
    
    Args:
        image (np.ndarray): Input image.
        pipeline (str): Pipeline identifier.
    
    Returns:
        np.ndarray: Processed image.
    """

    if pipeline == 'pipeline_1':
        # N4 bias field correction
        corrected_image = n4_bias_field_correction(image)
        # Denoising
        denoised_image = denoise_image(corrected_image)
        # Intensity normalization
        normalized_image = z_score_normalization(denoised_image)
        return normalized_image

    elif pipeline == 'pipeline_2':
        # Median or Gaussian filtering
        smoothed_image = filters.gaussian(image, sigma=1)  # You can switch to median filtering
        # Histogram equalization
        equalized_image = exposure.equalize_hist(smoothed_image)
        # Intensity standardization
        standardized_image = intensity_standardization(equalized_image)
        return standardized_image

    elif pipeline == 'pipeline_3':
        # Bias field correction
        corrected_image = n4_bias_field_correction(image)
        # CLAHE
        normalized_for_clahe = normalize_image(corrected_image)
        clahe_image = exposure.equalize_adapthist(normalized_for_clahe, clip_limit=0.03)
        # Smoothing
        smoothed_image = filters.gaussian(clahe_image, sigma=0.5)
        return smoothed_image

    else:
        print('no filter applied')
        return image

# Helper functions for the different steps in the pipelines

def n4_bias_field_correction(image):
    itk_image = sitk.GetImageFromArray(image.astype(np.float32))
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_itk_image = corrector.Execute(itk_image)
    return sitk.GetArrayFromImage(corrected_itk_image)

def denoise_image(image):
    # Assume image is grayscale, so no channel axis is needed
    return restoration.denoise_wavelet(image, channel_axis=None)

def z_score_normalization(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std

def intensity_standardization(image):
    # This function needs a pre-defined standardization strategy
    new_min = 0
    new_max = 1
    standardized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    standardized_image = standardized_image * (new_max - new_min) + new_min
    return standardized_image

def normalize_image(image):
    # Normalize the image to be in the range [0, 1] for float images
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image