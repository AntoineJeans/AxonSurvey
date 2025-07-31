# This module provides a base class for implementing different tracing algorithms with caching capabilities.

import numpy as np
import os

from abc import abstractmethod
from ..utils.imageio import numpy_to_tif, tif_to_numpy

class BaseTracer():
    """Base class for implementing tracing algorithms with caching and validation."""
    
    def __init__(self, tracer_name=None):
        """Initialize tracer with optional name for caching."""
        self.tracer_name = tracer_name
        
    @abstractmethod
    def make_tracing(self, image):
        """Abstract method to be implemented by subclasses for creating tracing."""
        pass
    
    def trace(self, image, mask = None, tracing_cache_folder = None):
        """Trace image with optional mask and caching support."""
        if tracing_cache_folder is not None:
            assert isinstance(tracing_cache_folder, str), "tracing_cache_folder must be None or a string"
            file_path = os.path.join(tracing_cache_folder, self.tracer_name + ".tif")
        # shape is (H,W,C)

        # Type checks
        assert isinstance(image, np.ndarray), "image must be a numpy array"
        assert image.dtype == np.float32, "image must be of dtype np.float32"
        assert image.ndim == 3, "Invalid Image: input image must have 3 dimensions (H, W, C)"

        if self.img_input_size is not None:
            assert image.shape[0] >= self.img_input_size, "Invalid Image: Dimension 0 of image smaller than input size"
            assert image.shape[1] >= self.img_input_size, "Invalid Image: Dimension 1 of image smaller than input size"
 
        if mask is not None:
            assert isinstance(mask, np.ndarray), "mask must be a numpy array"
            assert mask.dtype == bool or mask.dtype == np.bool_, "mask must be of boolean dtype"
            assert mask.ndim == 2, "Invalid Mask: mask must have 2 dimensions (H, W)"
            assert image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1], \
                "Image and mask spatial dimensions must match"
            
        if tracing_cache_folder is not None and os.path.exists(file_path):
            tracing = (tif_to_numpy(file_path, channel_number=0, output_dims=2)[:, :, np.newaxis] > 0)
        else:
            tracing = self.make_tracing(image)
            

        # Assert tracing shape and dtype
        assert tracing.ndim == 3, "Invalid tracing from make_tracing: output tracing must have 3 dimensions (H, W, 1)"
        assert isinstance(tracing, np.ndarray), "output tracing must be a numpy array, problem with make_tracing"
        assert tracing.shape[:2] == image.shape[0:2], "tracing and mask spatial dimensions must match"
        assert tracing.dtype == bool or tracing.dtype == np.bool_, "tracing must be of boolean dtype"

        # Apply mask if provided
        if mask is not None:
            tracing = tracing & mask[:, :, np.newaxis] if tracing.ndim == 3 else tracing & mask

        # Save tracing if path is provided
        if tracing_cache_folder is not None:
            # Pre-save assert
            
            os.makedirs(tracing_cache_folder, exist_ok=True)
            # save if doesn't exist
            if not os.path.exists(file_path) :
                numpy_to_tif((tracing.astype(np.uint8)) * 255, file_path)

            # Post-save assert
            assert os.path.exists(tracing_cache_folder), f"Tracing file was not saved at {tracing_cache_folder}"

        return tracing
    
    def trace_batch(self, images):
        """Trace multiple images in batch."""
        return [self.trace(img) for img in images]









