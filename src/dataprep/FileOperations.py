# Module for file structure management and multi-channel image processing operations.

import os
import itertools

class GenericFileStructuror:
    """Creates generic folder structures based on hierarchical subregion combinations."""
    
    def __init__(self, subregion_hierarchy=[[]], base_path="./data/axon_data/project_scans/"):
        """Initialize GenericFileStructuror with subregion hierarchy and base path."""
        # subregion_lists is a list of lists where at each index i are the possible options at depth i
        # In this project, index=0 are individual rats, index=1 are individual slices, index=2 are regions in that slide
        # subregion_hierarchy items are strings that must contain the necessary info to indentify the slice.
        
        self.subregion_hierarchy = subregion_hierarchy
        self.base_path = base_path
        
    def make_folders(self): 
        """Creates folder structure for all combinations in subregion hierarchy."""
        for path_parts in itertools.product(*self.subregion_hierarchy):
            folder_path = os.path.join(self.base_path, *path_parts)
            os.makedirs(folder_path, exist_ok=True)
            
    
class SpecFileStructuror:
    """Creates specific folder structures for rat brain data with rats, bregmas, and subregions."""
    
    def __init__(self, rat_list, bregma_dict, subregion_list, base_path="./data/axon_data/project_scans/"):
        """Initialize SpecFileStructuror with rat, bregma, and subregion info."""
        # subregion_lists is a list of lists where at each index i are the possible options at depth i
        # In this project, index=0 are individual rats, index=1 are individual slices, index=2 are regions in that slide
        # subregion_hierarchy items are strings that must contain the necessary info to indentify the slice.
        
        self.base_path = base_path
        self.rat_list = rat_list
        self.bregma_dict = bregma_dict
        self.subregion_list = subregion_list
        
        if self.bregma_dict is None:
            self.bregma_dict = {None:None,}
        
        
    def make_folders(self): 
        """Creates folder structure for rats, bregmas, and subregions."""
        for rat in self.rat_list:
            rat_path = os.path.join(self.base_path, rat)
            os.makedirs(rat_path, exist_ok=True)
            for bregma in self.bregma_dict[rat]:
                bregma_path = os.path.join(rat_path, bregma)
                os.makedirs(bregma_path, exist_ok=True)
                for subr in self.subregion_list:
                    subr_path = os.path.join(bregma_path, subr)
                    os.makedirs(subr_path, exist_ok=True)


from ..utils.imageio import tif_to_numpy, numpy_to_tif
import os
from ..dataprep.DataReader import DataReader

class ChannelSplitter:
    """Splits multi-channel .tif images into separate single-channel files."""
    
    def __init__(self, unlabeled_dir):
        """Initializes with directory containing unlabeled data."""
        self.unlabeled_dir = unlabeled_dir
        self.dr = DataReader(unlabeled_dir)
        
    def split_all_images(self, channel_names = ["th", "dbh"]):
        """Splits all multi-channel images into separate channel files."""
        for image in self.get_all_composite_image_paths():
            self.split_image(image, channel_names)
        
    def get_all_composite_image_paths(self):
        """Returns paths to all multi-channel .tif images in dataset folders."""
        folders = self.dr.get_paths()
        
        img_paths = []
        for folder in folders: 
            multi_channel_images = [path for path in self.get_tif_paths(folder) if self.image_has_many_channels(path)]
            img_paths = img_paths + multi_channel_images
        
        return img_paths
    

    def split_image(self, image_path, channels):
        """Splits a multi-channel image into separate files for each channel."""
        image = tif_to_numpy(image_path)
        folder_path = self.get_folder_from_path(image_path)
        for i, ch in enumerate(channels):
            channel_path = os.path.join(folder_path, ch + ".tif")
            numpy_to_tif(image[i, :,:], channel_path)
        
        
    def get_folder_from_path(self, path):
        """Returns the folder containing the given file path."""
        return os.path.dirname(path)
        

    def image_has_many_channels(self, image_path):
        """Returns True if image has more than 2 dimensions (multi-channel)."""
        return len(tif_to_numpy(image_path).shape) > 2
    
    def get_tif_paths(self, folder_path):    
        """Returns list of .tif/.tiff file paths in a folder."""
        tif_files = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.tif', '.tiff')):
                tif_files.append(os.path.join(folder_path, filename))
        return tif_files