# This module provides an experiment loader for managing different levels of experimental data organization and validation.

from ..dataprep.TracingChecker import TracingChecker
from ..utils.imbalance import imbalance_score

import os
import pathlib 
from pathlib import Path

MINIMUM_ACCEPTABLE_IMAGES = 25


class ExperimentLoader:
    """Load and validate experimental data for different levels of analysis."""
    
    def __init__(self, groups, train_path, test_path, raw_images_dir, val_path=None):
        """Initialize experiment loader with data paths and grouping configuration."""
        # groups is a list of RatGroup that define what to separate for quantification

        self.groups = groups
        
        self.train_dataset = self.get_dataset_folders(train_path)
        self.val_dataset = None if val_path is None else self.get_dataset_folders(val_path)
        self.test_dataset = self.get_dataset_folders(test_path)

        self.unlabeled_dir = raw_images_dir

        self.inference_paths_for_groups = self.get_inference_paths()


        self.info_file_name = "info.txt"
                
    def get_dataset_folders(self, path):
        """Get valid image folders from a given "dataset" path. Image folders contain strictly annotated image patches"""
        if TracingChecker(path).is_valid():
            all_folders = [os.path.join(path, f) for f in os.listdir(path)]
        else: all_folders = []
        return all_folders
    

    def get_inference_paths(self):
        all_groups_paths = []
        
        for group in self.groups:
            group_paths = []
            
            # Walk through the unlabeled directory structure
            for rat_id in os.listdir(self.unlabeled_dir):
                rat_path = os.path.join(self.unlabeled_dir, rat_id)
                
                # Check if this is a directory and if rat should be included
                if os.path.isdir(rat_path) and group.include_rat(rat_id):
                    
                    # Walk through slice folders
                    for slice_name in os.listdir(rat_path):
                        slice_path = os.path.join(rat_path, slice_name)
                        
                        if os.path.isdir(slice_path):
                            # Walk through region folders
                            for region_name in os.listdir(slice_path):
                                region_path = os.path.join(slice_path, region_name)
                                
                                # Check if this is a directory and if region should be included
                                if os.path.isdir(region_path) and group.include_region(region_name):
                                    # Check if this region folder contains .tif images
                                    tif_files = [f for f in os.listdir(region_path) if f.lower().endswith(('.tif', '.tiff'))]
                                    if tif_files:
                                        group_paths.append(region_path)
            
            all_groups_paths.append(group_paths)
        return all_groups_paths

    def get_original_file_from_img_folder(self, img_path):
        """Extract original file path from image folder info."""
        info_file_path = os.path.join(img_path, 'info.txt')
        with open(info_file_path, "r") as f: lines = f.readlines()        
        path = lines[1].strip() if len(lines) > 1 else ""
        return str(pathlib.Path(path).parent)
    
    def get_experiment_train_data(self):
        """Get training data for the experiment."""
        data = self.get_experiment_data(self.train_dataset)
        self.verify_group_data(data)
        return data
    
    def get_experiment_val_data(self):
        """Get validation data for the experiment."""
        data = self.get_experiment_data(self.val_dataset)
        self.verify_group_data(data)
        return data
    
    def get_experiment_test_data(self):
        """Get test data for the experiment."""
        data =  self.get_experiment_data(self.test_dataset)
        self.verify_group_data(data)
        return data
        
    
    def verify_group_data(self, data):
        for grp, data, possible_og in zip(self.groups, data, self.inference_paths_for_groups):
            # just a couple of verifications
            file_for_imgs = [self.get_original_file_from_img_folder(fldr) for fldr in data]
            region_folder_for_imgs =  [Path(image_path).parent for image_path in file_for_imgs]
            imb_score = imbalance_score(region_folder_for_imgs, possible_og)
            
            if imb_score > 0.25: print(f"WARNING: dataset sampling is unbalanced, imb_score = {imb_score}")
            if len(data) < MINIMUM_ACCEPTABLE_IMAGES: print(f" WARNING : region group {grp} only has {len(data)} tracings... make more!")
        return data

        
    def get_experiment_data(self, dataset):
        """Get experiment data based on rat groups."""

        rats = self.get_rats_from_folders(dataset)
        regions = self.get_regions_from_folders(dataset)

        all_groups_data = []
        for group in self.groups:
            group_data = []
            for image_folder, rat, region in zip(dataset, rats, regions):
                if group.include_rat(rat) and group.include_region(region):
                    group_data.append(image_folder)
            all_groups_data.append(group_data)
        return all_groups_data


    def get_rats_from_folders(self, folders):
        rats = []
        for fldr in folders:
            info_file_path = os.path.join(fldr, self.info_file_name)
            with open(str(info_file_path), "r") as f: lines = f.readlines()         
            rat = lines[3].strip() if len(lines) > 1 else ""
            rats.append(rat)
        return rats
    
    def get_regions_from_folders(self, folders):
        regions = []
        for fldr in folders:
            info_file_path = os.path.join(fldr, self.info_file_name)
            with open(str(info_file_path), "r") as f: lines = f.readlines()         
            region_line = lines[5].strip() if len(lines) > 1 else ""
            regions.append(region_line)

        return regions
            
    
    def get_inference_data(self, channel):
        """Get inference data."""
        all_inference_images = []
        for inference_data_group in self.inference_paths_for_groups:
            group_inference_images = []
            for region_folder in inference_data_group:
                file_path = os.path.join(region_folder, f"{channel}.tif")
                if os.path.exists(file_path):
                    group_inference_images.append(file_path)
            all_inference_images.append(group_inference_images)
        return all_inference_images

    

    

    