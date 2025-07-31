import cv2
import numpy as np

# grayscale
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        # Assuming image is a NumPy array
        return (image - self.mean) / self.std
    
class AdjustBrightness:
    def __init__(self, factor):
        self.factor = factor  # factor > 1 brightens, < 1 darkens

    def __call__(self, image):
        return np.clip(image * self.factor, 0, 255).astype(np.uint8)

class AdjustContrast:
    def __init__(self, factor):
        self.factor = factor  # > 1 increases contrast, < 1 decreases

    def __call__(self, image):
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        return np.clip((image - mean) * self.factor + mean, 0, 255).astype(np.uint8)

class AdjustSaturation:
    def __init__(self, factor):
        self.factor = factor  # > 1 more saturated, < 1 less

    def __call__(self, image):
        return np.clip(image.astype(np.float32) * self.factor, 0, 255).astype(np.uint8)

class AdjustSharpness:
    def __init__(self, factor):
        self.factor = factor  # 0 = blurry, 1 = original, >1 = sharper

    def __call__(self, image):
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        return np.clip(image * self.factor + blurred * (1 - self.factor), 0, 255).astype(np.uint8)

class LocalNormalization:
    def __init__(self, kernel_size=9, epsilon=1e-5):
        self.kernel_size = kernel_size
        self.epsilon = epsilon

    def __call__(self, image):
        image = image.astype(np.float32)
        local_mean = cv2.blur(image, (self.kernel_size, self.kernel_size))
        local_sq_mean = cv2.blur(image**2, (self.kernel_size, self.kernel_size))
        local_std = np.sqrt(local_sq_mean - local_mean**2 + self.epsilon)
        normalized = (image - local_mean) / (local_std + self.epsilon)
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)


    
# grayscale
class RemoveThreshold:
    def __init__(self, n_stds):
        self.n_stds = n_stds

    def __call__(self, image):
        if np.isnan(image).all(): return np.zeros(image.shape, image.dtype) 

        # Handle np.nan by ignoring them in mean/std and not modifying their value
        mask_valid = ~np.isnan(image)
        intensity = np.nanmean(image)
        intensity_std = np.nanstd(image)
        threshold = intensity + self.n_stds * intensity_std

        # Create a copy to avoid modifying input in-place
        result = image.copy()
        below_thresh = (result < threshold) & mask_valid
        result[below_thresh | ~mask_valid] = 0
        return result

from scipy import ndimage
class RemoveChunks:
    def __init__(self, min_area):
        self.min_area = min_area

    def __call__(self, image):
        binary_image = image.astype(bool)
        
        # Label connected components
        labeled_array, num_features = ndimage.label(binary_image, structure=np.ones((3,3)))
        
        if num_features == 0:
            return np.zeros_like(binary_image, dtype=bool)
        
        # Count pixels in each component
        component_sizes = np.bincount(labeled_array.ravel())
        
        # Create mask for components to keep (skip background at index 0)
        keep_mask = np.concatenate([[False], component_sizes[1:] >= self.min_area])
        
        # Apply mask
        return keep_mask[labeled_array]


'''class ConnectSegments:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
        edges = cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        return edges'''
    


class ConnectSegments:
    """Connect close fiber components using morphological closing"""
    
    def __init__(self, kernel_size=10, kernel_shape='ellipse'):
        """
        Args:
            connection_distance: Maximum distance to connect components
            kernel_shape: 'ellipse', 'rectangle', or 'cross'
        """
        self.connection_distance = kernel_size
        self.kernel_shape = kernel_shape

    def __call__(self, mask):
        """Connect close components using morphological closing"""
        mask = mask.astype(bool)
        
        # Create structuring element
        kernel_size = self.connection_distance * 2 + 1
        
        if self.kernel_shape == 'ellipse':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (kernel_size, kernel_size))
        elif self.kernel_shape == 'rectangle':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                             (kernel_size, kernel_size))
        else:  # cross
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, 
                                             (kernel_size, kernel_size))
        
        # Apply morphological closing to connect nearby components
        connected_mask = cv2.morphologyEx(mask.astype(np.uint8), 
                                        cv2.MORPH_CLOSE, kernel)
        
        return connected_mask.astype(bool)


from skimage.morphology import skeletonize
class Skeletonize:
    def __call__(self, image):
        return skeletonize(image)



# grayscale to binary
class Canny:
    def __init__(self, bottom=80, top=150):
        self.bottom = bottom
        self.top = top
        
    def __call__(self, image):
        edges = cv2.Canny(image, self.bottom, self.top)
        return edges > 0


