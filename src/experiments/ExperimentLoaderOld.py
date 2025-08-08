# This module provides an experiment loader for managing different levels of experimental data organization and validation.

from ..dataprep.TracingChecker import TracingChecker
from ..dataprep.DataReader import DataReader

import glob
import os
import pathlib 
from collections import Counter


MINIMUM_ACCEPTABLE_IMAGES = 25

class ExperimentLoader:
    """Load and validate experimental data for different levels of analysis."""
    
    def __init__(self, level, groups, train_path, test_path, unlabeled_dir, val_path=None):
        """Initialize experiment loader with data paths and grouping configuration."""
        # LEVELS : broad -> granular
        # 0 : estimate density across whole population (maybe compare with rat features like sex), example: what is average density 
        # groups : [[some set of rats], [some set of other rats], ...]
        # 1 : estimate density for whole rats, example: Is rat 301 more dense than 302?
        # groups : [[rat1], [rat2], ....]
        # 2 : estimate density for sections of brains: Are contrallesionnal sections more dense than ipsi?
        # groups : [[sets of regions], [sets of other regions], ...]
        # 3 : estimate density for sections of brains of indiviual rats: 
        # groups : [[rat1, section1], [rat1, section2], [rat3, section3], ...]

        self.level = level
        self.groups = groups
        
        self.train_datasets = self.get_dataset_folders(train_path)
        self.val_datasets = None if val_path is None else self.get_dataset_folders(val_path)
        self.test_datasets = self.get_dataset_folders(test_path)
        self.unlabeled_dir = unlabeled_dir
        
    def get_dataset_folders(self, path):
        """Get valid dataset folders from a given path."""
        if TracingChecker(path).is_valid(): return [path]
        elif isinstance(self.dataset_folder, list):
            all_folders = self.dataset_folder
        else: 
            all_folders = [os.path.join(self.test_path, f) for f in os.listdir(self.test_path)]
        return [fldr for fldr in all_folders if TracingChecker(fldr).is_valid()]
    
    
    def find_all_image_paths_by_region_group(self, datasets):
        """Find image paths for each region group across datasets."""
        img_paths = []
        for region_group in self.groups: 
            paths_for_group = []
            for dataset in datasets:
                tr = TracingChecker(dataset)
                paths_for_group = paths_for_group + tr.get_img_paths_from_group(region_group)
            img_paths.append(paths_for_group)
        return img_paths


    def imbalance_score(self, sample, potential_values):
        """Calculate imbalance score for dataset sampling."""
        counts = Counter(sample)
        total = len(sample)
        freqs = [counts[val] / total if val in counts else 0 for val in potential_values]
        mean_freq = 1 / len(potential_values)
        # Variance from uniform distribution
        variance = sum((f - mean_freq) ** 2 for f in freqs) / len(potential_values)
        # Normalize by max variance (worst case: all in one class)
        max_variance = (1 - mean_freq)**2 * (1 - 1/len(potential_values))
        score = variance / max_variance if max_variance != 0 else 0
        return score  # 0 = perfectly balanced, 1 = maximally imbalanced


    def get_original_file_from_img_folder(self, img_path):
        """Extract original file path from image folder info."""
        info_file_path = os.path.join(img_path, 'info.txt')
        with open(info_file_path, "r") as f: lines = f.readlines()        
        path = lines[1].strip() if len(lines) > 1 else ""
        return str(pathlib.Path(path).parent)
    
    def get_experiment_train_data(self):
        """Get training data for the experiment."""
        return self.get_experiment_data(self.train_datasets)
    
    def get_experiment_val_data(self):
        """Get validation data for the experiment."""
        return self.get_experiment_data(self.val_datasets)
    
    def get_experiment_test_data(self):
        """Get test data for the experiment."""
        return self.get_experiment_data(self.test_datasets)
    
        
    def get_experiment_data(self, datasets):
        """Get experiment data based on analysis level."""
        dr = DataReader(self.unlabeled_dir)
        match self.level:
            case 0:
                print(f'level {self.level} not implemented yet')
            case 1:
                print(f'level {self.level} not implemented yet')
            case 2:
                img_paths_for_groups = self.find_all_image_paths_by_region_group(datasets)
                all_possible_og_files = [dr.get_all_paths_for_regions(grp) for grp in self.groups]
            case 3:
                print(f'level {self.level} not implemented yet')
            case _:
                print(f'level {self.level} not implemented yet')


        for grp, paths, possible_og in zip(self.groups, img_paths_for_groups, all_possible_og_files):
            if len(paths) < MINIMUM_ACCEPTABLE_IMAGES: raise ValueError(f"region group {grp} only has {len(paths)} tracings... make more!")
            original_file_for_imgs = [self.get_original_file_from_img_folder(p) for p in paths]
            imb_score = self.imbalance_score(original_file_for_imgs, possible_og)
            if imb_score > 0.25: print(f"WARNING: dataset sampling is unbalanced for testing, imb_score = {imb_score}")

        return img_paths_for_groups
    
    def get_files_from_folders(self, folders, channel):
        """Get files from folders with optional channel filtering."""
        all_files = []
        if channel is None:
            for og_folder in folders:
                all_files = all_files + glob.glob(og_folder + r"\*.tif") + glob.glob(og_folder + r"\{channel}.tiff")
        else: 
            for og_folder in folders:
                file_path = os.path.join(og_folder, f"{channel}.tif")
                if os.path.exists(file_path):
                    all_files.append(file_path)
        return all_files
    
    def get_inference_data(self, channel=None):
        """Get inference data based on analysis level."""
        # groups : [[sets of regions], [sets of other regions], ...]

        dr = DataReader(self.unlabeled_dir)
        match self.level:
            case 0:
                print(f'level {self.level} not implemented yet')
            case 1:
                print(f'level {self.level} not implemented yet')
            case 2:
                all_possible_og_files = [dr.get_all_paths_for_regions(grp) for grp in self.groups]
            case 3:
                print(f'level {self.level} not implemented yet')
            case _:
                print(f'level {self.level} not implemented yet')

        return [self.get_files_from_folders(folder, channel) for folder in all_possible_og_files]

    

    


    