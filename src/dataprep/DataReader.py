# Module for reading and traversing rat brain dataset directory structures.

import os
import pathlib
import numpy as np
from ..utils.imageio import tif_to_numpy, numpy_to_tif

class DataReader:
    """
    DataReader provides utilities for traversing and extracting information from a dataset directory structure.
    The structure must be one in the same format as those created with FileStructuror and must contain ROIs in leaf directories.

    This class is intended to facilitate reading datasets organized in nested folders, where each leaf folder
    contains image files (e.g., .tif or .tiff). It can:
      - Recursively find all leaf folders containing image data.
      - Validate the dataset structure.
      - Compute region masks and areas for subregions where the cropped ROI is not perfectly rectangular (doesn't take the whole image).
      - Extract region names from paths.
      - Filter paths by region group.

    Args:
        root_read_dir (str): Root directory of the dataset to read.
    """
    def __init__(self, root_read_dir):
        """Initializes with the root directory to read from."""
        self.root_read_dir = root_read_dir

    def read_dir_is_valid(self):
        """Returns True if there are any valid leaf folders in the dataset."""
        return len(self.get_paths()) > 0

    def get_paths(self):
        """Returns a list of all leaf folders (containing files, no subdirs) under root."""
        leaf_folders = []

        for current_dir, subdirs, files in os.walk(self.root_read_dir):
            if files and not subdirs:
                leaf_folders.append(current_dir)
        return leaf_folders
        
    # assumes all_subregion_folders is already computed
    # memory param because keeping all area masks in memory may be too large for some projects
    def get_area_per_path(self, memory=True):
        """Returns array of areas for each subregion folder, optionally caching masks."""
        areas = []
        if self.all_subregion_folders is None or self.all_subregion_folders == []:
            raise ValueError("all_subregion_folders has not been calculated")
            
        else:
            for subr_path in self.all_subregion_folders:
                tif_files = [f for f in os.listdir(subr_path) if f.lower().endswith(('.tif', '.tiff'))]
                if len(tif_files) == 0 or len(tif_files) > 1:
                    raise FileNotFoundError(f"There is 0 or more than one .tif/.tiff files found in {subr_path}")
                else:
                    region_image = tif_to_numpy(tif_files[0]).astype(np.uint8)
                    mask = self.get_outer_mask(region_image)
                    areas.append(np.sum(mask))
                    if memory: self.mask_disks[subr_path] = mask
        return np.array(areas)
    
    def get_outer_mask(self, img):
        """Returns a boolean mask where image is nonzero."""
        return (img != 0)
    
    def get_region(self, path):
        """Extracts and returns the region name from a given path."""
        path = pathlib.Path(path[len(self.root_read_dir) + 1:])
        return path.parts[2]

    def get_all_paths_for_regions(self, region_group):
        """Returns all leaf folder paths belonging to the specified region group."""
        all_paths = self.get_paths()
        return [path for path in all_paths if self.get_region(path) in region_group]
