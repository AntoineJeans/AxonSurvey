# This module contains functions to create nice graphs for model performances and predictions

import numpy as np
import matplotlib.pyplot as plt
from ..imgproc.utils import contrast_img

# Convert hex to normalized RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4)])

def display_grayscale(img, title="Axon image"):
    # Use hardcoded hex colors for min and max
    color_max_hex = "#b5e3e3" 
    color_min_hex = "#004d4d"

    color_min = hex_to_rgb(color_min_hex)
    color_max = hex_to_rgb(color_max_hex)

    # If img has multiple channels, use the first channel as the 2D image
    if img.ndim == 3:
        img = img[:, :, 0]

    # If img is boolean, convert to float before normalization
    if img.dtype == bool or img.dtype == np.bool_:
        img = img.astype(np.float32)

    img_norm = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img) + 1e-8)
    gradient_img = np.zeros((*img.shape, 3), dtype=np.float32)
    for i in range(3):
        gradient_img[..., i] = img_norm * color_max[i] + (1 - img_norm) * color_min[i]
    # Set np.nan locations to pure white
    nan_mask = np.isnan(img)
    gradient_img[nan_mask] = [1.0, 1.0, 1.0]
    plt.imshow(gradient_img)

    fontsize = 15 if len(title) < 50 else 12

    plt.title(title, fontsize=fontsize)
    plt.show()
    
def display_edges(img, title="Axon mask"):     
    plt.imshow(img)
    plt.title(title)
    plt.show()
    
def highlight_detected_axons(original_image, edges, alpha=0.5, constrast=True, original_coef=1):
    
    if constrast: original_image = contrast_img(original_image, contrast_factor=10)

    rgb_image = np.zeros((original_image.shape[0], original_image.shape[1], 3), dtype=np.uint8)
    rgb_image[..., 2] = original_coef * original_image
    rgb_image[..., 0] = (edges.astype(np.float32) * 255 * alpha).astype(np.uint8)

    plt.imshow(rgb_image)
    plt.title('Detected axons highlighted in red over original image')
    plt.show()

def show_feature_importance(weights, labels, property_name):
    plt.bar(labels, weights)
    plt.xticks(rotation=30)
    plt.ylabel("Feature Importance")
    plt.axhline(y=0, color='black', linewidth=0.8)
    plt.title(f"Learned feature correlation for {property_name} property")
    plt.tight_layout()
    
    plt.show()