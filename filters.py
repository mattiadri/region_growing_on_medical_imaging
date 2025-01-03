import cv2
import numpy as np
from skimage import exposure
from scipy.ndimage import gaussian_filter, median_filter

def apply_gaussian_filter(image, sigma=1):
    return gaussian_filter(image, sigma=sigma)

def apply_median_filter(image, size=3):
    return median_filter(image, size=size)

def apply_bilateral_filter(image, sigma_color=0.1, sigma_spatial=15):
    return cv2.bilateralFilter(image.astype(np.float32), d=-1, sigmaColor=sigma_color, sigmaSpace=sigma_spatial)

def apply_anisotropic_diffusion(image, weight=0.1, max_iter=100):
    img = image.astype(np.float32)
    for _ in range(max_iter):
        grad = np.gradient(img)
        flux = [g * np.exp(-(g / weight) ** 2) for g in grad]
        img += sum([np.gradient(f)[0] for f in flux])
    return img

def apply_histogram_equalization(image):
    return exposure.equalize_hist(image)

def apply_filter_pipeline(image, pipeline='pipeline_1'):
    if pipeline == 'pipeline_1':
        filtered = apply_gaussian_filter(image, sigma=1)
        filtered = apply_median_filter(filtered, size=3)
    elif pipeline == 'pipeline_2':
        filtered = apply_bilateral_filter(image, sigma_color=0.1, sigma_spatial=15)
        filtered = apply_anisotropic_diffusion(filtered, weight=0.1, max_iter=50)
    elif pipeline == 'pipeline_3':
        filtered = apply_histogram_equalization(image)
        filtered = apply_gaussian_filter(filtered, sigma=1)
        filtered = apply_bilateral_filter(filtered, sigma_color=0.1, sigma_spatial=15)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")
    return filtered