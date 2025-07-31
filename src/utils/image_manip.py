# This module provides functions for image manipulation including Gaussian blur with NaN handling.

import numpy as np
import scipy.ndimage
import cv2

def gaussian_blur_with_nan(image, tile_size=64):
    """Apply Gaussian blur to image channels while preserving NaN values."""
    # Apply a Gaussian blur to all dimensions of features
    # sigma=1 for all axes, can be adjusted as needed
    sigma = tile_size / 2
    ksize = int(2 * round(sigma) + 1)
    for i in range(image.shape[2]):
        channel = image[:, :, i]
        nan_mask = np.isnan(channel)
        if nan_mask.any():
            mean_val = np.nanmean(channel)
            channel_filled = np.where(nan_mask, mean_val, channel).astype(np.float32, copy=False)
            # Use OpenCV for fast Gaussian blur
            filtered = cv2.GaussianBlur(channel_filled, (ksize, ksize), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
            filtered[nan_mask] = np.nan
            image[:, :, i] = filtered
        else:
            image[:, :, i] = cv2.GaussianBlur(channel, (ksize, ksize), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    return image