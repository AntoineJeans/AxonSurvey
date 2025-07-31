# This module provides functions for reading, writing, and processing image files including TIFF format and mask generation.

import os
import numpy as np
import imageio.v3 as iio
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import flood

def tif_to_numpy(image_path, output_dims=3, channel_number=None):
    """Load a TIFF image and convert it to a numpy array with specified dimensions."""
    # Type assertions
    if not isinstance(image_path, str):
        raise TypeError(f"image_path must be a string, got {type(image_path).__name__}")
    if channel_number is not None and not isinstance(channel_number, int):
        raise TypeError(f"channel_number must be an integer or None, got {type(channel_number).__name__}")
    
    # File existence check
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = iio.imread(image_path).astype(np.float32) 
    
    # hardcoded to swap the channel dimension into last position
    
    
    if channel_number is None and output_dims < len(img.shape): raise ValueError(f"output_dims has fewer dimensions than image, specify channel or increase output_dims")
    
    # Handle channel selection
    if channel_number is not None:
        # Check if image has channels (at least 3D)
        if img.ndim < 3: raise ValueError(f"Cannot select channel {channel_number} from {img.ndim}D image")
        
        # Check channel bounds
        num_channels = img.shape[2]
        if channel_number < 0: raise ValueError(f"channel_number must be non-negative, got {channel_number}")
        if channel_number >= num_channels: raise ValueError(f"channel_number {channel_number} out of bounds for image with {num_channels} channels")
        
        img = img[:, :, channel_number].squeeze()
    else:
        if len(img.shape) == 3:
            img = img.transpose(1,2,0)

    if len(img.shape) == 2 and output_dims == 3:
        img = np.expand_dims(img, axis=-1)
        
    assert isinstance(img, np.ndarray), "Result must be a numpy array"
    assert len(img.shape) == output_dims, f"Resulting image array must be {output_dims}D"
    return img
     
def numpy_to_tif(image, file):
    """Save a numpy array as a TIFF image file."""
    # Always overwrite if file exists
    if os.path.exists(file):
        os.remove(file)
    iio.imwrite(file, image)


def generate_image_outer_mask(image):
    """Generate a mask by flood filling from corners to detect background regions."""
    # If image has channels, use the first channel
    if image.ndim == 3:
        image = image[:, :, 0]

    h, w = image.shape
    mask = np.zeros_like(image, dtype=bool)
    corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]

    for y, x in corners:
        if image[y, x] == 0 and not mask[y, x]:
            mask |= flood(image, (y, x), connectivity=1, tolerance=0)

    return ~mask