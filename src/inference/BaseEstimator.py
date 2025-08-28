# This module provides a base class for implementing different density estimation algorithms.

from ..utils.imageio import tif_to_numpy
import numpy as np


from src.utils.imageio import generate_image_outer_mask

class BaseEstimator:
    """Base class for implementing density estimation algorithms with sample data and model support."""
    
    def __init__(self, sample_data, image_paths, sample_weights=None, model=None):
        """Initialize base estimator with sample data and image paths."""
        self.sample_data = sample_data # I guess can be a list of y only ?
        self.population = image_paths # idek
        self.sample_weights = None
        self.model = model

    def estimate(self, confidence_interval=0.95, weighted_by_size=False):
        """Abstract method to be implemented by subclasses for density estimation."""
        pass

    def predict_points(self):
        """Predict density points for all images in the population."""
        # We only use density here by indexing [:, :, -1]
        points = [self.predict_image_density(path) for path in self.population]
        return points
        
    def predict_image_density(self, image_path):
        """Predict density for a single image using the model."""
        image = tif_to_numpy(image_path)
        mask = generate_image_outer_mask(image)
        density_map = self.model.predict(image, image_path=image_path, mask=mask)[:, :, -1]
        predicted_properties = np.nanmean(density_map)
        return predicted_properties
        