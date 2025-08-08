# Module for training models on rat brain image data with ground truth property extraction.

from ..utils.ImageLoader import ImageLoader
import copy
import numpy as np
from ..utils.traceProps import get_trace_density

class Trainer:
    """Handles model training on rat brain images with ground truth property extraction."""
    
    def __init__(self, image_paths, center_sample_size=128, ground_truth_functions=[get_trace_density]):
        # image_paths can be a list of .tif files or the root folder of a dataset created by sampling
        
        self.loader = ImageLoader(image_paths, split_step=None, final_img_size=center_sample_size)
        self.loader.load()

        self.images = self.loader.get_images()
        self.masks = self.loader.get_masks()

        self.ground_truth_functions = ground_truth_functions
        
    def fit_model(self, model, plot_correlation=False, property_names=None):
        """Trains a model on loaded images using extracted ground truth properties."""
        ground_truths = [[func(mask) for func in self.ground_truth_functions] for mask in copy.deepcopy(self.masks)]
        model.fit(copy.deepcopy(self.images), np.array(ground_truths).astype(np.float32), 
                  plot_correlation=plot_correlation, property_names=property_names)

