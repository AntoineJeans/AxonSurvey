# Module for validating and analyzing rat brain tracing dataset structure and content.

import os
import numpy as np
from scipy.stats import mannwhitneyu


from ..utils.traceProps import get_trace_density
from ..utils.imageio import tif_to_numpy

from ..utils.graphs import display_inference_bounds



import matplotlib.pyplot as plt
import seaborn as sns

class TracingChecker:
    """Validates dataset structure and provides statistical analysis of rat brain tracing data."""
    
    def __init__(self, root_read_dir, expected_channels = []):
        self.root_read_dir = root_read_dir
        self.expected_channels = expected_channels

        self.image_sample_file_name = "img.tif"
        self.tracing_file_name = "tracings.tif"
        self.mask_file_name = "outer_mask.tif"
        self.info_file_name = "info.txt"

        self.expected_files = self.expected_channels + \
            [self.image_sample_file_name, self.tracing_file_name, 
             self.mask_file_name, self.info_file_name, "tracings", "img.ndf"]

        self.n_images = self.count_images()

    def count_images(self):
        """Counts the number of image folders in the root directory."""
        return len(self.get_folders_in_root())

    def files_in_root(self):
        """Checks if there are files (not folders) in the root directory."""
        return not ([f for f in os.listdir(self.root_read_dir) if os.path.isfile(os.path.join(self.root_read_dir, f))] == [])
    
    def get_folders_in_root(self):
        """Gets a list of folder names in the root directory."""
        return [f for f in os.listdir(self.root_read_dir) if os.path.isdir(os.path.join(self.root_read_dir, f))]

    def get_problems(self):
        """Identifies missing or extra files in the dataset structure."""
        missing_images, missing_tracings, missing_channels, missing_masks, missing_info, extra_stuff= [], [], {}, [], [], []

        files_in_root = self.files_in_root()
        folders = self.get_folders_in_root()


        for fldr in folders:
            has_extra = False
            files = [f for f in os.listdir(os.path.join(self.root_read_dir, fldr)) if os.path.isfile(os.path.join(self.root_read_dir, fldr, f))]
            folders = [f for f in os.listdir(os.path.join(self.root_read_dir, fldr)) if os.path.isdir(os.path.join(self.root_read_dir, fldr, f))]
            has_extra = has_extra or len(folders) > 0

            if self.image_sample_file_name not in files: missing_images.append(fldr)
            if self.tracing_file_name not in files: missing_tracings.append(fldr)
            if self.mask_file_name not in files: missing_masks.append(fldr)
            if self.info_file_name not in files: missing_info.append(fldr)

            missing_c = [chan for chan in self.expected_channels if chan not in files]
            if len(missing_c) > 0: missing_channels[fldr] = missing_c
            
            extra = [file for file in files if file not in self.expected_files]
            has_extra = has_extra or len(extra) > 0

            if has_extra: extra_stuff.append(fldr)

        return files_in_root, missing_images, missing_tracings, missing_channels, missing_masks, missing_info, extra_stuff
    
    def display_problems(self, problems):
        """Displays a summary of dataset problems."""
        files_in_root, missing_images, missing_tracings, missing_channels, missing_masks, missing_info, extra_stuff = problems

        print(f"{'-' * 40}")
        print(f"Checking labeled dataset at path {self.root_read_dir}")
        print(f"1: {len(missing_images)} missing image samples | {missing_images}")
        print(f"2: {len(missing_tracings)} missing tracings | {missing_tracings}")
        print(f"3: {len(missing_channels.keys())} missing channels | {missing_channels}")
        print(f"4: {len(missing_masks)} missing masks | {missing_masks}")
        print(f"5: {len(missing_info)} missing info.txt files | {missing_info}")

        if extra_stuff != []: print(f'warning: {len(extra_stuff)} image folders contain extra stuff that shouldnt be there {extra_stuff}')
        if files_in_root: print(f'warning: non-folders in {self.root_read_dir}')

    def check(self):
        """Checks and displays dataset problems."""
        problems = self.get_problems()
        self.display_problems(problems)
        
    # definition : The dataset is valid if there are no other problems but missing labels
    def is_valid(self, display_probs=False):
        """Returns True if dataset structure is valid."""
        problems = self.get_problems()
        if display_probs: self.display_problems(problems)
        files_in_root, missing_images, _, missing_channels, missing_masks, missing_info, extra_stuff = problems
        
        valid = not files_in_root and (self.n_images > 0)
        for prob_list in [missing_images, missing_channels, missing_masks, missing_info, extra_stuff]:
            valid = valid and len(prob_list) == 0
        return valid
        

    def get_labelled_ratio(self):
        """Returns the ratio of labeled images in the dataset."""
        problems = self.get_problems()
        _, _, unlabeled_imgs, _, _, _, _ = problems
        
        return 1 - (len(unlabeled_imgs) / self.n_images)
    
    def filter_imgs_with_condition(self, condition):
        """Filters image folders by a given condition."""
        return [fldr for fldr in self.get_folders_in_root() if condition(fldr)]
    
    def get_img_from_file_condition(self, file):
        """Returns a function to check if an image folder originates from a specific file."""
        def func(fldr):
            info_file_path = os.path.join(self.root_read_dir, fldr, self.info_file_name)
            with open(info_file_path, "r") as f: lines = f.readlines()      
            second_line = lines[1].strip() if len(lines) > 1 else ""
            return second_line == file
        return func
    
    def get_img_from_region_condition(self, regions):
        """Returns a function to check if an image folder matches a region."""
        def func(fldr):
            info_file_path = os.path.join(self.root_read_dir, fldr, self.info_file_name)
            with open(str(info_file_path), "r") as f: lines = f.readlines()         
            region_line = lines[5].strip() if len(lines) > 1 else ""
            return region_line in regions
        return func
    
    def get_imgs_has_tracing_condition(self):
        """Returns a function to check if an image folder has a tracing file."""
        def func(fldr):
            img_path = os.path.join(self.root_read_dir, fldr, self.image_sample_file_name)
            tracing_pth = os.path.join(self.root_read_dir, fldr, self.tracing_file_name)
            return os.path.exists(img_path) and os.path.exists(tracing_pth)
        
        return func

    def get_img_paths_from_original(self, original_file):
        """Gets image folder paths from the original file name."""
        fldrs = self.filter_imgs_with_condition(self.get_img_from_file_condition(original_file))
        return [os.path.join(self.root_read_dir, f) for f in fldrs]
    
    def get_img_paths_from_region(self, region):
        """Gets image folder paths for a single region."""
        return self.get_img_paths_from_region_group([region])
    
    def get_img_paths_from_region_group(self, regions):
        """Gets image folder paths for a group of regions."""
        fldrs = self.filter_imgs_with_condition(self.get_img_from_region_condition(regions))
        return [os.path.join(self.root_read_dir, f) for f in fldrs]
    
    def get_img_folders_with_tracings(self):    
        """Gets image folder paths that have tracing files."""
        fldrs = self.filter_imgs_with_condition(self.get_imgs_has_tracing_condition())
        full_foldr_paths = [os.path.join(self.root_read_dir, f) for f in fldrs]
        return full_foldr_paths
    
    def get_imgs_tracing_paths(self):
        """Gets paths to images and their tracing files."""
        full_foldr_paths = self.get_img_folders_with_tracings()
        return [os.path.join(f, self.image_sample_file_name) for f in full_foldr_paths], \
                    [os.path.join(f, self.tracing_file_name) for f in full_foldr_paths]
                    
    def get_original_file(self, folder_path):
        """Gets the original file name from an image folder."""
        info_file_path = os.path.join(self.root_read_dir, folder_path, self.info_file_name)
        with open(str(info_file_path), "r") as f: lines = f.readlines()         
        return lines[1].strip() if len(lines) > 1 else ""
        
    def get_all_original_folders(self):
        """Gets all original file names for folders with tracings."""
        return [self.get_original_file(fldr) for fldr in self.get_img_folders_with_tracings()]
            
    def get_ground_truth_for_region_group(self, region_group, ground_truth_function):
        """Gets axon densities for a group of regions."""
        paths = self.get_img_paths_from_region_group(region_group)
        tracing_files = [os.path.join(fldr, self.tracing_file_name) for fldr in paths]
        tracing_images = [tif_to_numpy(trace_file, channel_number=0, output_dims=2).astype(bool) for trace_file in tracing_files]
        return [ground_truth_function(tr) for tr in tracing_images]
        
    def get_p(self, region_groups):
        """Computes p-value for density difference between two region groups."""
        densities_for_groups = [self.get_densities_for_region_group(region_group) for region_group in region_groups]
        statistic, p_value = mannwhitneyu(densities_for_groups[0], densities_for_groups[1], alternative='two-sided')

        return p_value


    def display_densities_for_groups(self, region_groups, group_labels=None):
        """Displays histograms of axon densities for region groups."""
        
        if group_labels is None: group_labels = [str(grp) for grp in region_groups]
        
        densities_for_groups = [self.get_densities_for_region_group(region_group) for region_group in region_groups]
        n_counts = len(densities_for_groups[0])
        
        colors = sns.color_palette("Set2", len(densities_for_groups))
        
        bins = 10
        plt.hist(densities_for_groups, bins=bins, label=group_labels, alpha=0.7, histtype='stepfilled', color=colors)


        plt.legend()
        plt.xlabel("Axon density")
        plt.ylabel("Counts")

        plt.yticks(range(0, n_counts//2, 5))
        plt.title(f"Real density distribution on {n_counts} images of each region")
        plt.grid(True)
        plt.show()
                    
    def statistical_test_between_groups(self, region_groups, ground_truth_function, alpha = 0.05, 
                                        group_labels=None, display_confidence=True, 
                                        n_bootstrap = 1000):
        """Performs statistical test and displays confidence intervals between region groups."""
        
        if group_labels is None: group_labels = [str(grp) for grp in region_groups]
        
        densities_for_groups = [self.get_ground_truth_for_region_group(region_group, ground_truth_function) for region_group in region_groups]
        
        for densities, label in zip(densities_for_groups, group_labels):
            print(f"Real density mean in group {label} : {np.mean(densities)}, Std for group is : {np.std(densities)}")

        if display_confidence:
            confidence_bounds = []
            for dens in densities_for_groups:
                m = []
                for i in range(n_bootstrap):
                    sampled_densities = np.random.choice(dens, size=len(dens), replace=True)

                    m.append(np.mean(sampled_densities))
                l = np.percentile(m, 100*alpha/2)
                u = np.percentile(m, 100*(1-alpha/2))
                confidence_bounds.append([(np.mean(m), u, l)])

            title = f"Expected axon density from human tracings by direct statistical test, confidence={(1-alpha)}"      
            display_inference_bounds(confidence_bounds, group_labels, title)

            
        if len(region_groups) == 2:
            statistic, p_value = mannwhitneyu(densities_for_groups[0], densities_for_groups[1], alternative='two-sided')

            # Output result
            print(f"U statistic: {statistic}")
            print(f"p-value: {p_value}")

            # Conclusion at 95% confidence level
            if p_value < alpha:
                print("Means (or central tendencies) are significantly different.")
            else:
                print("No significant difference in means.")

            return p_value
        return None





