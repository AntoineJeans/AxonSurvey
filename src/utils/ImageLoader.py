import numpy as np
import os 
import ast
from skimage.util import view_as_windows

from .imageio import tif_to_numpy
from ..dataprep.TracingChecker import TracingChecker

class ImageLoader:
    # Reads images from input path
    # input can be : 1 - A single dataset path or 2 - a list of image folders
    def __init__(self, image_paths, split_step=None, final_img_size=None):
        # image_paths can be a list of .tif files or the root folder of a dataset created by sampling
        
        self.img_file_name = 'img.tif'
        self.mask_file_name = 'tracings.tif'
        self.outer_mask_file_name = "outer_mask.tif"
        self.info_file_name = 'info.txt'
        
        self.final_img_size = final_img_size
        self.split_step = split_step

        self.from_dataset = not isinstance(image_paths, list)
        
        if self.from_dataset:
            assert os.path.isdir(image_paths), "If image paths are not a list, must be a dataset"
            checker = TracingChecker(image_paths)
            assert checker.is_valid(), "Passed dataset is not valid"
            assert checker.get_labelled_ratio() > 0, "Passed has no labaled datapoints"

            # gets the images in lists from the existing dataset           
            image_paths = checker.get_img_folders_with_tracings()
        

        assert len(image_paths) > 0, "Must have at least one image to evaluate"
        assert self.all_folders_exist(image_paths), "All given image folders must exist"

        self.image_paths = [os.path.join(img, self.img_file_name) for img in image_paths]
        self.mask_paths = [os.path.join(img, self.mask_file_name) for img in image_paths]
        self.info_paths = [os.path.join(img, self.info_file_name) for img in image_paths]
        
        assert self.all_files_exist(self.image_paths) and self.all_files_exist(self.mask_paths), "All given images and masks must exist"
        
    def check_imgs_shape(self):
        
        self.img_shape = self.images[0].shape
        self.mask_shape = self.masks[0].shape
        assert len(self.img_shape) == 3, "images must be 3D"
        assert len(self.mask_shape) == 2, "masks must be 2D"
        assert all([img.shape == self.img_shape for img in self.images]), "All input images don't have the same shape!"
        assert all([mask.shape == self.mask_shape for mask in self.masks]), "All input masks don't have the same shape!"

    def get_paths(self) : 
        return self.image_paths, self.mask_paths

    def load(self):

        self.images =[tif_to_numpy(img) for img in self.image_paths]
        self.masks = [tif_to_numpy(mask, output_dims=2, channel_number=0).astype(bool) for mask in self.mask_paths]        
        
        # all images must be same shape
        self.check_imgs_shape()
        

        if self.img_shape != self.final_img_size:
            assert self.final_img_size < self.img_shape[0] and self.final_img_size < self.img_shape[1], "center_sample_size must be smaller than the img shape"

            images = []
            masks = []

            if self.split_step is not None:
                for img, mask in zip(self.images, self.masks):
                    
                    a = view_as_windows(img, window_shape=(self.final_img_size,self.final_img_size, self.img_shape[2]), step=(self.split_step,self.split_step, 1))
                    b = view_as_windows(mask, window_shape=(self.final_img_size,self.final_img_size), step=(self.split_step,self.split_step))

                    images += [im for im in a.reshape(-1,self.final_img_size,self.final_img_size, self.img_shape[2])]
                    masks += [im for im in b.reshape(-1,self.final_img_size,self.final_img_size)]

                self.images = images
                self.masks = masks


            else:
                if self.final_img_size is not None:
                    middle_x, middle_y = self.img_shape[0] // 2, self.img_shape[0] // 2
                    ofs = self.final_img_size // 2 
                    neg_x, pos_x = ofs, ofs + (self.final_img_size % 2)
                    neg_y, pos_y = ofs, ofs + (self.final_img_size % 2)
                    
                    self.images =[ img[middle_x - neg_x:middle_x + pos_x, middle_y - neg_y:middle_y + pos_y] for img in self.images]
                    self.masks =[ mask[middle_x - neg_x:middle_x + pos_x, middle_y - neg_y:middle_y + pos_y] for mask in self.masks]
                
        self.check_imgs_shape()


    def all_folders_exist(self, folders):
        return all(os.path.isdir(f) for f in folders)
    
    def all_files_exist(self, files):
        return all(os.path.exists(f) for f in files)
    
    def get_images(self) : return self.images

    def get_masks(self) : return self.masks

    def get_image_and_tracing_subset(self, n, seed):
        """
        Returns a random subset of n images and masks using the given seed.
        """
        rng = np.random.default_rng(seed)
        total = len(self.images)
        assert n <= total, "Requested subset size exceeds available images"
        indices = rng.choice(total, size=n, replace=True)
        images_subset = [self.images[i] for i in indices]
        masks_subset = [self.masks[i] for i in indices]
        return images_subset, masks_subset
    
    def get_ROI_paths(self):
        files = []
        for info_file in self.info_paths:
            with open(info_file, "r") as f: lines = f.readlines()  
            if len(lines) == 0: raise ValueError(f"ERROR : Empty info file {info_file}")
            else: files.append(lines[1].strip())
        for file_path in files: assert os.path.exists(file_path), f"original file does not exist: {file_path}"
        return files
    
    def get_starting_points(self):
        points = []
        for info_file in self.info_paths:
            with open(info_file, "r") as f: lines = f.readlines()  
            if len(lines) == 0: raise ValueError("ERROR : Empty info file")
            else: points.append(ast.literal_eval(lines[0].strip()))
        return points
        
        
    



