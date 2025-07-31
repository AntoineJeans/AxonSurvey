# This module provides baseline mean extractors for simple feature extraction from images.

import numpy as np
from .BaseFeatureExtractor import BaseFeatureExtractor   

# kind of makes no sense with the new PropertyModel, at least useful for debugging 
class PopulationMeanExtractor(BaseFeatureExtractor):
    """Extract population mean as a baseline feature for all images."""
    
    def __init__(self, *args, **kwargs):
        """Initialize population mean extractor."""
        super().__init__(*args, **kwargs)
        self.population_property_mean = None
        
        self.extractor_name = "PopulationMeanExtractor"
        self.feature_names = ["Population Pixel Mean"]

    def fit(self, _, properties):
        """Fit extractor by calculating population mean from properties."""
        self.population_property_mean = np.mean(properties)

    def get_features(self, image):
        """Extract population mean as feature map."""
        return np.full(image.shape, np.array([self.population_property_mean]))


class ImageMeanExtractor(BaseFeatureExtractor):
    """Extract image mean as a baseline feature for each image."""
    
    def __init__(self, *args, **kwargs):
        """Initialize image mean extractor."""
        super().__init__(*args, **kwargs)
        
        self.extractor_name = "ImageMeanExtractor"
        self.feature_names = ["Image Pixel Mean"]
    
    def get_features(self, image):
        """Extract image mean as feature map."""
        # Ignore np.nan for mean calculation
        if np.isnan(image).all():
            mean_in_region = np.nan
        else:
            mean_in_region = np.nanmean(image) # equals np.nan if all in images are nan
        a = np.full(image.shape, np.array([mean_in_region]))
        return a



