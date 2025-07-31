# This module provides a temporary estimator that uses a model to predict density with confidence intervals.
# For testing purposes, more work has to be done to establish those intervals correctly

from .BaseEstimator import BaseEstimator
import numpy as np
from ..utils.imageio import tif_to_numpy
from scipy.stats import norm

class TemporaryEstimator(BaseEstimator):
    """Temporary estimator that uses a model to predict density with confidence intervals."""
    
    def __init__(self, model, sample_data, image_paths):
        """Initialize temporary estimator with model and data."""
        self.model = model
        self.sub_image_size = 128
        super().__init__(sample_data, image_paths)

    def estimate(self, expected_rmse, confidence_interval=0.95, weighted_by_size=False):
        """Estimate density with confidence intervals using model predictions."""
        # get single expected density and bounds in the density of a population of images (RMSE must be on that whole population and not only part of it)
        # get average density on the whole population

        n_in_images = self.get_n_per_image(self.population)
        total_n = np.sum(n_in_images)
        exps = [self.infer_density_on_image(img_path) for img_path in self.population]
        
        if weighted_by_size:
            proportion = n_in_images / total_n
            exp = np.dot(exps, proportion)
        else:
            exp = np.mean(exps) 
        
        margin = np.sqrt(self.get_variance_margin(total_n, expected_rmse, confidence_interval)**2)
        margin = margin + 0.001 # it's fake anyway, make it more weird
        return exp, exp + margin, exp - margin

    def infer_density_on_image(self, image_path):
        """Infer density on a single image using the model."""
        image = tif_to_numpy(image_path)
        density_map = self.model.predict(image)[:, :, -1]
        predicted_properties = np.nanmean(density_map)
        return predicted_properties

    def get_n_in_image(self, image_path):
        """Calculate number of sub-images in a single image."""
        img = tif_to_numpy(image_path)
        return int(img.shape[0] / self.sub_image_size * img.shape[1] / self.sub_image_size)
    
    def get_n_per_image(self, image_paths):
        """Calculate number of sub-images for each image in the population."""
        return [self.get_n_in_image(image_path) for image_path in image_paths]

    def get_variance_margin(self, n, expected_rmse, confidence):
        """Calculate variance margin for confidence intervals."""
        sample_mean_std = expected_rmse / np.sqrt(n)
        z_score = norm.ppf(1 - (1 - confidence) / 2)
        margin = z_score * sample_mean_std
        return margin