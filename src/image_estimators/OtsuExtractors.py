# This module provides Otsu thresholding extractors for density and fiber count estimation.

import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import label

from .BaseFeatureExtractor import BaseFeatureExtractor
class OtsuExtractor(BaseFeatureExtractor):
    """Extract density and fiber count features using Otsu thresholding."""
    
    def __init__(self, *args, **kwargs):
        """Initialize Otsu extractor."""
        super().__init__(*args, **kwargs)
        
        self.extractor_name = "OtsuExtractor"
        self.feature_names = ["Otsu Density", "Otsu Fibre Count"]
    
    def get_features(self, image):
        """Extract Otsu density and fiber count as feature maps."""
        
        # image shape: (H, W, 1)
        img2d = image[:, :, 0]
        valid_mask = np.isfinite(img2d)
        # Otsu threshold only on valid pixels
        thresh = threshold_otsu(img2d[valid_mask]) if np.any(valid_mask) else 0
        binary = np.zeros_like(img2d, dtype=bool)
        binary[valid_mask] = img2d[valid_mask] > thresh
        # Density: mean over valid pixels only
        otsu_density = np.mean(binary[valid_mask]) if np.any(valid_mask) else np.nan
        # Fibre count: count connected components only in valid region
        # (label ignores NaNs since binary is False there)
        otsu_fibre_count = label(binary)[1]
        # Return shape (H, W, 2) filled with [otsu_density, otsu_fibre_count]
        a = np.full((image.shape[0], image.shape[1], 2), np.array([otsu_density, otsu_fibre_count]))
        return a