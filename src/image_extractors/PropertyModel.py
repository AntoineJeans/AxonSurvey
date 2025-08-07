# This module provides a property model that combines feature extractors with linear regression for property estimation.

import numpy as np
from sklearn.linear_model import LinearRegression
from ..utils.Splitter import Splitter
import os

from ..utils.viz import show_feature_importance

from ..utils.image_manip import gaussian_blur_with_nan


from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import time

class PropertyModel:
    """
    Class for estimating image properties. 
    This model can take as input an image of some size and output a prediction for one or multiple 'properties' (if fitted).
    Since these models must perform well on little training data (sets of around 100 images), they are made of 2 components : 
    - feature extractors; pretrained models/Harshly regularized models/deterministic processes/ that take as input an images
        and output low-dimensional vectors representing important features of this image (each extractor may output many)
    - feature_to_prop_model : A simple model (by default linear) to map extracted feature to the properties we want to estimate.
    Args:
        extractor_list: Feature extractor instance. Can't be empty
        model : An instanciated (not fitted) simple model that should convert features to properties
    """

    
    def __init__(self, extractors, model, batch_size_inference=False, cache_folder = r'.\data\project_tracing_and_features', 
                 add_noise = False):
        """
        Initialize the estimator with input shape and inference settings.

        Args:
            input_shape (tuple): Shape of the input images.
            extractor (optional): Feature extractor instance. If None, estimates are computed directly.
            batch_size_inference (bool): Flag to enable batch processing during inference.
        """

        self.extractors = extractors
        self.model = model
        self.batch_size_inference = batch_size_inference
        self.feature_to_prop_model = LinearRegression()
        
        self.m_features = sum([extractor.n_features for extractor in self.extractors])
        self.n_properties = 0 # must be set during fit

        self.cache_folder = cache_folder

        self.sub_image_size = 128

        self.add_noise = add_noise

        self.scaler = StandardScaler()
        
    def fit(self, images, properties, plot_correlation=True, property_names=None):
        """
        Fit the PropertyModel with corresponding images and properties. For extractors that admit a fitting method, 
        also fits that extractor. Fitting mecanics are flexible, they depend on the extractor

        Args:
            images (list or numpy.ndarray): Images for fitting.
            properties (list or numpy.ndarray): Ground truth property values.
        """
        
        for extractor in self.extractors:
            if extractor.is_fittable: extractor.fit(images, properties)

        if plot_correlation:
            for extractor in self.extractors:
                extractor.plot_prediction(images, properties, property_names=property_names)
            
        features = self.predict_features(images, mean=True)
        
        # print(features)
        features = self.scaler.fit_transform(features)
        # print(features)

        self.model.fit(features, properties)

        self.n_properties = properties.shape[1]
    
    def predict_features(self, images, mask=None, mean=True, cache_folder = None):
        """Creates an array of shape by default (n_images, n_features, img_shape0, img_shape1) containing the obtained features
        if mean, img_shape0, img_shape1 are averaged over, and output is (n_images, n_features)

        Args:
            images ([2D array]): images
        """
        features = []
        for image in images:
            features_for_image = []
            for extractor in self.extractors:
                # for an image of shape (a,b) extractors output a list of feature maps [n_features * (a,b)]
                ft_maps = extractor.extract(image, mask=mask, cache_folder=cache_folder)
                features_for_image.append(ft_maps)
            features_for_image = np.concatenate(features_for_image, axis=2)
            features.append(features_for_image)
            
        features = np.array(features) 
        if mean: features = np.nanmean(features, axis=(1,2))
        return features
    
    def predict(self, image, tile_size = 64, image_path = None, mask=None):
        """Predict properties for a single image using tiled processing."""
        if image_path is None: cache_folder = None
        else: cache_folder = self.get_cache_folder_from_image_path(image_path)

        features = self.predict_features([image], mask=mask, mean=False, cache_folder=cache_folder)[0]

        # print(features)
        # features is (1, n_features, img_shape0, img_shape1)
        # we want to create an output of shape (n_properties, img_shape0, img_shape1)
        # To do this, the image's properties are tiled than for each tile, a vector of size n_features is created.
        # Than, the model outputs for each vector, a vector of size n_properties
        # is used to create a full np array of size (n_properties, img_shape0, img_shape1)
        # could be switched to a kernel approach if not too compute intensive

        
        mask_func = self.feature_map_to_property_map

        # mean but count out the 0s
        ### COULD CAUSE BUGS IDC
        join_func = lambda x, y: np.where(np.isnan(x), y, x)
        
        # batch size has no effect unless the underlying extractor has batch processing implemented, in which case it should accelerate everything
        self.splitter = Splitter(tile_size, batch_size=1, mask_function=mask_func, join_function=join_func)

        properties = self.splitter.process(features)

        # Set all indices that are nan in the first channel of the input image to nan in all property channels
        nan_mask = np.isnan(features[:, :, 0])

        for i in range(properties.shape[2]):
            properties[:, :, i][nan_mask] = np.nan

        

        properties = self.postprocessing(properties)

        return properties
        
    def feature_map_to_property_map(self, feature_map):
        """Convert feature map to property map using trained model."""
        # of size (n_features, a, b)
        a, b = feature_map.shape[0], feature_map.shape[1]
        if np.isnan(feature_map).all():
            features = np.full((feature_map.shape[2],), np.nan)
        else:
            with np.errstate(all='ignore'):
                features = np.nanmean(feature_map, axis=(0,1))

        if not any(np.isnan(features)):
            
            # print(features)
            features = self.scaler.transform(np.array([features]))
            # print(features)
            properties = self.model.predict(features)[0]
            # print(properties)
        else:
            properties = np.full((self.n_properties), np.nan)

        property_map = np.array([np.full((a, b), property) for property in properties])
        # outputs size (n_properties, a, b)
        return property_map.transpose(1,2,0)


    def get_cache_folder_from_image_path(self, image_path):
        """Generate cache folder path from image path."""
        # Normalize and split the path
        parts = os.path.normpath(image_path).split(os.sep)
        # Get last 3 folders and file name
        if len(parts) < 4:
            raise ValueError("image_path must have at least 3 folders and a file name")
        last_parts = parts[-4:-1]
        cache_path = os.path.join(self.cache_folder, *last_parts)
        return cache_path
    

    def postprocessing(self, properties, tile_size=64):
        """Apply postprocessing to predicted properties."""
        if self.add_noise:
            return gaussian_blur_with_nan(properties, tile_size)
        else:
            return properties 
        
    def list_features(self):
        """List all feature names from all extractors."""
        feature_names = []
        for extractor in self.extractors:
            feature_names += extractor.feature_names
        return feature_names
        
    def feature_weights(self, prop_number):
        """Get feature weights for a specific property."""
        return self.model.coef_[prop_number, :]
    
    def show_feature_weights(self, property_names=None):
        """Display feature importance weights for all properties."""
        if property_names is None: property_names = [f"prop_{i}" for i in range(self.n_properties)]
        feature_names = self.list_features()
        for i, property in enumerate(property_names):
            ft_weights = self.feature_weights(i)
            show_feature_importance(ft_weights, feature_names, property)
            
    


