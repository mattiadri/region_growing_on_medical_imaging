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

def apply_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F)

def apply_edge_detection(image, method='sobel'):
    if method == 'sobel':
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobel_x, sobel_y)
    elif method == 'canny':
        edges = cv2.Canny(image, 100, 200)
    else:
        raise ValueError("Unknown edge detection method: Choose 'sobel' or 'canny'")
    return edges

def apply_filter_pipeline(image, pipeline='pipeline_1'):
    if pipeline == 'pipeline_1':  # Preprocessing + Rilevamento dei Bordi
        # Ridurre il rumore e migliorare i bordi
        filtered = apply_gaussian_filter(image, sigma=1)
        filtered = apply_median_filter(filtered, size=3)
        edges = apply_edge_detection(filtered, method='sobel')  # o 'canny' per contorni più netti
        return edges

    elif pipeline == 'pipeline_2':  # Segmentazione + Contrasto
        # Rimuovere il rumore preservando i bordi, migliorare il contrasto e segmentare
        filtered = apply_bilateral_filter(image, sigma_color=0.1, sigma_spatial=15)
        filtered = apply_anisotropic_diffusion(filtered, weight=0.1, max_iter=50)
        filtered = apply_histogram_equalization(filtered)
        
        # Assicurarsi che l'immagine sia nell'intervallo corretto [0, 255]
        filtered = np.uint8(255 * (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered)))
        
        # Thresholding con Otsu
        _, thresholded = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded

    elif pipeline == 'pipeline_3':  # Dettaglio + Contorni Avanzati
        # Miglioramento dei dettagli e rilevamento dei bordi
        filtered = apply_gaussian_filter(image, sigma=1)
        filtered = apply_median_filter(filtered, size=3)
        laplacian = apply_laplacian(filtered)
        edges = apply_edge_detection(laplacian, method='sobel')
        return edges

    elif pipeline == 'pipeline_4':  # Contrasto avanzato + Segmentazione
        # Migliorare il contrasto e segmentare la prostata
        filtered = apply_bilateral_filter(image, sigma_color=0.1, sigma_spatial=15)
        filtered = apply_anisotropic_diffusion(filtered, weight=0.1, max_iter=50)
        filtered = apply_histogram_equalization(filtered)
        
        # Assicurarsi che l'immagine sia nell'intervallo corretto [0, 255]
        filtered = np.uint8(255 * (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered)))
        
        # Thresholding con Otsu
        _, thresholded = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded

    elif pipeline == 'pipeline_5':  # Minimalista (Nessun Filtro)
        return image

    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")