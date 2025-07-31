import numpy as np
from datetime import datetime
import os
import imageio.v3 as iio
import cv2

def mask_to_binary(mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    return mask

from skimage.util import view_as_windows
def split_in_smaller_imgs(img, img_size=128, step_size=64):
    patches = view_as_windows(img, window_shape=(img_size, img_size), step=(step_size, step_size))
    patches = patches.reshape(-1, img_size, img_size)
    return patches

from skimage.io import imsave
def save_to_png(patches, output_dir):
    for i, patch in enumerate(patches):
        iio.imwrite(f'{output_dir}/patch_{i:04d}.png', patch)


def create_new_folders_for_dataset(root_dir):
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(root_dir + "/all_images/", exist_ok=True)
    os.makedirs(root_dir + "/all_masks/", exist_ok=True)
     
def get_new_dataset_root(eval_dataset=False):
    if eval_dataset: root_dir = "data/axon_data/evaluation_datasets/"
    else: root_dir = "data/axon_data/training_datasets/"
    
    folders = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
    new_folder_name = f"dataset_{len(folders)}"
    root_dir = root_dir + new_folder_name
    return root_dir

def log_info(info, root_dir):
    info_path = os.path.join(root_dir, "info.txt")
    with open(info_path, "w") as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")

def create_training_dataset(image_path, pipeline=None, mask_path=None, img_size=128, step_size=64, trace_source=None, eval_dataset=False):
    # set up directories
    root = get_new_dataset_root(eval_dataset=eval_dataset)
    create_new_folders_for_dataset(root)
    image_save_folder = root + "/all_images/"
    mask_save_folder = root + "/all_masks/"
    
    # read split, preprocess and save image
    #image_j = ij.io().open(image_path)
    #image = ij.py.from_java(image_j).values
    image = iio.imread(image_path)
    
    if pipeline is not None: image = pipeline(image)
    img_patches = split_in_smaller_imgs(image, img_size=img_size, step_size=step_size)
    save_to_png(img_patches, output_dir=image_save_folder)
    
    if mask_path is not None:
        #mask_j = ij.io().open(mask_path)
        #mask = ij.py.from_java(mask_j).values
        
        mask = iio.imread(mask_path)
        mask = mask_to_binary(mask)
        mask_patches = split_in_smaller_imgs(mask, img_size=img_size, step_size=step_size)
        save_to_png(mask_patches, output_dir=mask_save_folder)

    now = datetime.now()
    creation_date = now.strftime("%Y-%m-%d")
    creation_time = now.strftime("%H:%M:%S")
    info = {
        "creation_date": creation_date,
        "creation_time": creation_time,
        "original_image_path": image_path,
        "original_image_dimensions": (image.shape[0], image.shape[1]),
        "number_of_images" : len(img_patches),
        "size_of_images": (img_size,img_size),
        "step_size": step_size,
        "has_traces": (mask_path is not None),
        "trace_source": trace_source
    }
    
    log_info(info, root)