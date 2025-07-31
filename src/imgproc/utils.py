import numpy as np

def reframe_np_img(np_img, x_s=0, x_e=None, y_s=0, y_e=None):
    # img de forme (x, y, n_composite)
    #if x_s is None: x_s = 0
    if x_e is None: x_e = np_img.shape[0]   
    #if y_s is None: y_s = 0
    if y_e is None: y_e = np_img.shape[1]
    return np_img[x_s:x_e, y_s:y_e] 

def contrast_img(original_image, contrast_factor=10):
    mean_value = np.mean(original_image)
    return np.clip((original_image - mean_value) * contrast_factor + mean_value, 0, 255)