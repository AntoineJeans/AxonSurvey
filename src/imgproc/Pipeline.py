# On pourrait faire une classe pipeline pour enchaîner les transformations plus facilement
import numpy as np
from ..utils.viz import display_edges, display_grayscale

from abc import abstractmethod
class BaseProcessingPipeline:
    def __init__(self, transforms, display_before=False, display_during=False):
        self.transforms = transforms
        self.display_before = display_before
        self.display_during = display_during
    
    def __call__(self, image):
        
        if self.display_before: self.display(image)
        
        message, valid_image = self.is_valid(image)
        if valid_image:
            for transform in self.transforms:
                image = self.transform_image(transform, image)
                if self.display_during: self.display(image)
            return image
        else:
            raise ValueError(f"Image invalide: {message}")
    
    @abstractmethod
    def transform_image(self, transform, image):
        pass
    
    @abstractmethod
    def is_valid(self, image):
        pass 
    
    @abstractmethod
    def display(self, image):
        pass 
    
    
# assumes input is a numpy array, of shape (x,y), np.uint8. Outputs array of same type
class GrayscalePipeline(BaseProcessingPipeline):
    def __init__(self, transforms, display_before=False, display_during=False):
        super().__init__(transforms, display_before=display_before, display_during=display_during)
    
    def is_valid(self, image):
        return "" , image.dtype == np.uint8 and len(image.shape) == 2
        
    def transform_image(self, transform, image):
        return transform(image)
    
    def display(self, image): display_grayscale(image)
    
class MultiChannelGrayscalePipeline:
    def __init__(self, transforms, n_channels, display_before=False):
        super().__init__(transforms, display_before=display_before)
        self.n_channels = n_channels
        self.allowed_types = (np.uint8,)

    def is_valid(self, image):
        return "" , image.dtype == np.uint8 and len(image.shape) == 3 and image.shape[2] == self.n_channels

    def transform_image(self, transform, image):
        transformed_channels = [
            transform(image[:, :, c]) for c in range(self.n_channels)
        ]
        return np.stack(transformed_channels, axis=2)
    
    def display(self, image): 
        for c in range(self.n_channels):
            display_grayscale(image[:, :, c])
  
    

    
# assumes input is a numpy array, of shape (x,y), np.uint8. Outputs array of same type
class BinaryPipeline(BaseProcessingPipeline):
    def __init__(self, transforms, display_before=False, display_during=False):
        super().__init__(transforms, display_before=display_before, display_during=display_during)
        self.allowed_types = (np.bool_, bool)
    
    def is_valid(self, image):
        return "" , image.dtype in self.allowed_types and len(image.shape) == 2
        
    def transform_image(self, transform, image):
        return transform(image)
    
    def display(self, image): display_edges(image)
    
class MultiChannelBinaryPipeline(BaseProcessingPipeline):
    def __init__(self, transforms, n_channels, display_before=False):
        super().__init__(transforms, display_before=display_before)
        self.n_channels = n_channels
        self.allowed_types = (np.bool_, bool)

    def is_valid(self, image):
        return "" , image.dtype in self.allowed_types and len(image.shape) == 3 and image.shape[2] == self.n_channels
    
    def transform_image(self, transform, image):
        transformed_channels = [
            transform(image[:, :, c]) for c in range(self.n_channels)
        ]
        return np.stack(transformed_channels, axis=2)
    
    def display(self, image): 
        for c in range(self.n_channels):
            display_edges(image[:, :, c])

    
class GrayscaleToBinaryPipeline(BaseProcessingPipeline):
    def __init__(self, gs_pipeline, bin_pipeline, bin_transform=None):
        self.gs_pipeline=gs_pipeline
        self.bin_pipeline=bin_pipeline
        self.bin_transform=bin_transform
        
    def __call__(self, image):
        output_gray = self.gs_pipeline(image)
        input_bin = self.bin_transform(output_gray)
        output_bin = self.bin_pipeline(input_bin)
        return output_bin
    

