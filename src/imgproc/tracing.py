from PIL import Image
import os 
import numpy as np

from ..utils.viz import display_edges, display_grayscale
from .denoising import remove_by_std_treshold

import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize



def canny_edge_detection(np_image, bottom=80, top=150, display=False):
    temp_path = 'data/modified_images/temp.jpg'
    Image.fromarray(np_image).save(temp_path)
    img = cv2.imread(temp_path)
    os.remove(temp_path)
    
    # Apply Canny edge detector
    edges = cv2.Canny(img, bottom, top)
    if display: display_edges(edges, title="Edges during iterative canny")
    return edges
    
    # pas utilisé mais il est là
    """
    def iterative_canny(image_np, n_iter=10, bottom=80, top=150, sigma=0.3, n_stds=1, display_iter=None):
    # doit être grayscale
    if display_iter is not None: display_edges(image_np, title='Image avant iterative_canny')
    for i in range(1, n_iter + 1):
        image_np = remove_by_std_treshold(image_np, n_stds=n_stds)
        image_np = gaussian_filter(image_np, sigma=sigma)
        display_this_iteration = (display_iter is not None) and (i%display_iter == 0) and (i != n_iter + 1)
        image_np = canny_edge_detection(image_np, display=display_this_iteration)

    if display_iter is not None: display_edges(image_np, title='Image après iterative_canny')
    return image_np
    """



def remove_small_chunks(edges, min_area=30, display = True):
    # doit être binaire
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    filtered_edges = np.zeros_like(edges, dtype=bool)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_edges[labels == i] = True
            
    if display: display_edges(filtered_edges, title="edges after small chunks removed")
    
    return filtered_edges


def connect_line_segments(edges, kernel_size=15, display=False):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    edges = cv2.morphologyEx(edges.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    if display: display_edges(edges, title="edges after edge connections")
    return edges



def skeletonize_edges(edges, display=True):
    skeleton = skeletonize(edges)
    if display: 
        plt.imshow(skeleton, cmap='gray')
        plt.title("Skeletonized detected edges")
        plt.show()
    return skeleton
