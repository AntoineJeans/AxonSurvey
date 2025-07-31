import numpy as np
import os

from imgproc.utils import reframe_np_img

def dump_info(image):
    """A handy function to print details of an image object."""
    name = image.name if hasattr(image, 'name') else None # xarray
    if name is None and hasattr(image, 'getName'): name = image.getName() # Dataset
    if name is None and hasattr(image, 'getTitle'): name = image.getTitle() # ImagePlus
    print(f" name: {name or 'N/A'}")
    print(f" type: {type(image)}")
    print(f"dtype: {image.dtype if hasattr(image, 'dtype') else 'N/A'}")
    print(f"shape: {image.shape}")
    print(f" dims: {image.dims if hasattr(image, 'dims') else 'N/A'}")


def compress_img(ij_gateway, img, save_path = None, dimensions = (0, 100, 0, 100), display=False, return_image=False):
    if save_path is None: save_path = "data/modified_images/temp_image.tif"
    if os.path.exists(save_path):
        os.remove(save_path)
    
    x_s, x_e, y_s, y_e = dimensions
    
    image_np = ij_gateway.py.from_java(img)
    image_np_c = reframe_np_img(image_np, x_s = x_s, x_e = x_e, y_s = y_s, y_e = y_e)
    print(image_np_c.name)
    # data = xr.DataArray(image_np_c, dims=('1','2','3'))
    image_ij_c = ij_gateway.py.to_dataset(image_np_c)
    
    # image_ij_c = ij_gateway.py.to_java(image_ij_c)
    # image_ij_c = ij_gateway.py.to_imagej(image_np_c.values)
    ij_gateway.io().save(image_ij_c, save_path)
    
    if (display):
        print(f"before, shape = {image_np.shape}")
        ij_gateway.ui().show(img)
        print(f"after, shape = {image_np_c.shape}")
        ij_gateway.ui().show(image_ij_c)
    
    if return_image: return image_ij_c
    
