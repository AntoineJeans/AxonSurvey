# This module provides threshold-based density extractors with local and global thresholding options.

import numpy as np
from scipy.stats import pearsonr

from ..utils.Splitter import Splitter
from ..imgproc.Transforms import RemoveThreshold

from ..utils.viz import display_grayscale

from .BaseFeatureExtractor import BaseFeatureExtractor
class ThresholdDensityExtractor(BaseFeatureExtractor):
    """Extract density features using threshold-based methods with optional local/global processing."""
    
    def __init__(self, n_stds = 1, local=True, *args, **kwargs):
        """Initialize threshold density extractor with local or global thresholding."""
        self.local = local
        # A little bit special. if local is False we do global tresholding over the whole image
        # n_std is used (there can't be a fit, that would be more difficult to implement)
        super().__init__(*args, **kwargs)
        self.treshold_func = RemoveThreshold(n_stds=n_stds)
        
        self.extractor_name = "ThresholdDensityExtractor"
        self.feature_names = ["Local Treshold Density"] if local else ["Global Treshold Density"] 
        
    def fit(self, images, properties):
        """Fit extractor by optimizing threshold parameters for best correlation."""
        if self.local:
            # # set up fit iteration
            n_stds_ls = np.arange(1, 7, 0.5)
            # finds and sets the number of std above the mean creates the tightest linear fit between X and Y
            pearsons_for_n_stds = [self.pearson_with_n_stds(n_std, images, properties) for n_std in n_stds_ls]
            best_n_std = n_stds_ls[np.argmax(pearsons_for_n_stds)]
            self.treshold_func = RemoveThreshold(n_stds=best_n_std)

    def pearson_with_n_stds(self, n_stds, images, properties):
        """Calculate Pearson correlation for a specific number of standard deviations."""
        self.treshold_func = RemoveThreshold(n_stds=n_stds)
        extracted = np.array(self.extract_images(images))
        extracted_features = np.nanmean(extracted,  axis=(1,2)).squeeze()
        return self.pearson(extracted_features, properties)
        
    def pearson(self, features, properties):
        """Calculate average Pearson correlation between features and properties."""
        correlations = []
        for i in range(properties.shape[1]):
            props = properties[:, i]
            corr, _ = pearsonr(features, props)
            correlations.append(corr)
        return np.mean(correlations)
    
    def preprocessing(self, image, mask, _):
        """Apply threshold-based preprocessing to image."""
        # Call super without passing self explicitly
        image = super().preprocessing(image, mask, _)

        if self.local:
            mask_func = lambda sub_image : (self.treshold_func(sub_image) > 0)
            join_func = lambda x, y: x | y
            splitter = Splitter(self.feature_extraction_tile_size, mask_function=mask_func, mask_dtype=bool, invalid_fill_value=False, join_function=join_func)
            new_image = splitter.process(image)
        else: 
            new_image = (self.treshold_func(image) > 0)

        return new_image

    def get_features(self, image):
        """Extract density as feature map based on thresholded pixel count."""
        # outputs a single value representing the count of pixels in the image n_stds above the image mean
        density_for_image = [np.sum(image) / (image.shape[0] * image.shape[1])]
        a = np.full(image.shape,  np.array([density_for_image]))
        return a



