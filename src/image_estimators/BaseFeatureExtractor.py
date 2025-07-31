# This module provides a base class for implementing feature extraction algorithms from images with caching support.

import numpy as np
import scipy.ndimage
from sklearn.linear_model import LinearRegression
from ..utils.Splitter import Splitter              
import copy
import os

from ..utils.imageio import numpy_to_tif, tif_to_numpy

from ..utils.image_manip import gaussian_blur_with_nan

import time

from ..utils.graphs import plot_single_variable_fitness

from abc import abstractmethod
class BaseFeatureExtractor:
    """Base class for implementing feature extraction algorithms with caching and splitting support."""
    
    def __init__(self, feature_extraction_tile_size=None, add_noise=False):
        """Initialize feature extractor with optional tiling and noise settings."""
        
        self.feature_extraction_tile_size = feature_extraction_tile_size
        self.extractor_name = None
        self.feature_names = []
        
        if self.feature_extraction_tile_size is not None:
            mask_func = self.get_features
            # join_func = lambda x, y: np.where(np.isnan(x) | np.isnan(y), np.where(np.isnan(x), y, x), (x + y) / 2)
            join_func = lambda x, y: np.where(np.isnan(x), y, x)
            self.splitter = Splitter(feature_extraction_tile_size, mask_func, join_func)
            
        self.is_fittable = self.fit.__func__ != BaseFeatureExtractor.fit
        self.n_features = 1 # by default

        self.add_noise = add_noise
            
        
    def extract(self, image, mask=None, cache_folder=None):
        """Extract feature maps from image with optional masking and caching."""
        assert self.extractor_name is not None, f"Extractor has no name"
        assert len(self.feature_names) > 0, f"Extractor has no feature names"
        assert image.ndim == 3, f"Input image must have 3 dimensions, got {image.ndim} dimensions with shape {image.shape}"
        assert image.shape[0] >= 16, f"First dimension of the image must be at least 16, got {image.shape[0]}"
        assert image.shape[1] >= 16, f"Second dimension of the image must be at least 16, got {image.shape[1]}"

        image_shape = image.shape

        # check if features already exist
        if cache_folder is None:
            use_cached_features = False
        else: 
            feature_paths = [os.path.join(cache_folder, ft_name + ".tif") for ft_name in self.feature_names]
            use_cached_features = all([os.path.exists(path) for path in feature_paths]) 

        if use_cached_features:
            features = []
            for i, ft_name in enumerate(self.feature_names): 
                file_path = os.path.join(cache_folder, ft_name + ".tif") 
                features.append(tif_to_numpy(file_path, output_dims=2))
            features = np.array(features)
            features = features.transpose(1,2,0)
        else:
            # if not, compute them
            image = copy.deepcopy(image)

            image = self.preprocessing(image, mask, cache_folder)
            
            assert image.shape == image_shape, f"Issue with {self.extractor_name} preprocessing: changed image shape from {image_shape} to {image.shape}"
        
            if self.feature_extraction_tile_size is None:
                features = self.get_features(image)
            else:
                features = self.splitter.process(image)
            
            assert features.shape[2] > 0, 'Created feature map must be of shape at least (a,b,1) but 3rd dim is under 1'

            features = self.postprocessing(features)

            # Then cache them
            if cache_folder is not None:
                for i, ft_name in enumerate(self.feature_names):
                    ft_map = features[:, :, i]
                    file_path = os.path.join(cache_folder, ft_name + ".tif")
                    if not os.path.exists(file_path):
                        os.makedirs(cache_folder, exist_ok=True)
                        # numpy_to_tif((ft_map.astype(np.uint8)) * 255, file_path)
                        numpy_to_tif(ft_map, file_path)
        
        assert features.shape[2] > 0, 'Created feature map must be of shape at least (a,b,1) but 3rd dim is under 1'
        # Assert features have 3 dimensions
        assert features.ndim == 3, f"Features must have 3 dimensions, got {features.ndim} dimensions with shape {features.shape}"

        # Mask out features at mask locations
        if mask is not None:
            assert mask.shape == image_shape[:2], "mask must match the first two dimensions of the image"
            # Set all feature channels to np.nan where mask is True            
            features[~mask, :] = np.nan

        return features
    

    def extract_images(self, images):
        """Extract features from multiple images."""
        return [self.extract(img) for img in images]
    
    def plot_prediction(self, images, properties, property_names = None):
        """Plot prediction fitness between extracted features and target properties."""
        if property_names is None: property_names = [f'prop{i}' for i in range(properties.shape[1])]

        features = []
        # for image, mask in zip(images, masks):
        for image in images:
            ft_maps = self.extract(image)
            ft = np.nanmean(ft_maps, axis=(0,1))
            features.append(ft)
        features = np.array(features)

        for i, ft_name in enumerate(self.feature_names):
            for j, prop_name in enumerate(property_names):
                feature_list = features[:, i]
                property_list = properties[:, j]

                plot_single_variable_fitness(feature_list, property_list, 
                                             estimated_name=ft_name, 
                                             real_name=prop_name)
                

    def fit(self, images, properties):
        """Fit extractor to better predict properties from extracted features."""
        # Override if extractor requires fitting
        pass    


    def preprocessing(self, image, mask, cache_file):
        """Apply preprocessing to image before feature extraction."""
        # Override if preprocessing needs to be done before tiling for feature extraction
        if mask is not None:
            image[~mask, :] = np.nan
        return image
       
    def postprocessing(self, features):
        """Apply postprocessing to extracted features."""
        if self.add_noise:
            return gaussian_blur_with_nan(features, self.feature_extraction_tile_size)
        else:
            return features
    
    @abstractmethod  
    def get_features(self, sub_image):
        """Abstract method to be implemented by subclasses for feature extraction."""
        pass


