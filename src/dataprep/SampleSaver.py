# Work in progress

import os
import pathlib
import numpy as np

from abc import abstractmethod

from ..dataprep.DataReader import DataReader
from ..dataprep.TracingChecker import TracingChecker

class SamplingStrategy:
    """
    Base class for sampling strategies.
    Subclasses should implement the sample_indices method.
    """
    def __init__(self, project_raw_image_dir, channel, sample_dimensions, stratify_regions, stratify_group=True, groups=None):

        """
        population must be able to represent enough information to do every kind of sampling. 
        
        We do this by passing a list where each element corresponds to a group we want to sample from
        groups can be the same as experiment groups.
        Each groups is also a list of elements. Each of those elements is a list of images and represents a ROI (one or many split larger images)

        By default, for each group, the same amount of images (n/group_count) are sampled. There can also be one group
        Also by default, all regions are stratified

        example [ # groups
        [ # regions
        [img1, img2, img3], [img1, img2, img3]
        ]
        [ # regions
        [img1, img2, img3], [img1, img2, img3]
        ]
        ]

        not sure if stratify_group should be a param

        Sampling strategies may or may not require the image information, but it"s always passed and it's used if needed.
        """

        if not (os.path.exists(project_raw_image_dir)): raise ValueError(f"read directory {project_raw_image_dir} doesn't exist")
        self.dr = DataReader(project_raw_image_dir)
        if not (self.dr.read_dir_is_valid()): raise ValueError(f"read directory {project_raw_image_dir} not valid")

        self.root_read_dir = project_raw_image_dir
        self.channel = channel

        self.sample_dimensions = sample_dimensions
        self.sample_area = sample_dimensions[0] * sample_dimensions[1]

        self.groups = groups
        self.stratify_regions = stratify_regions
        self.stratify_group = stratify_group


        self.all_subregion_folders = self.dr.get_paths()

    def get_groups(self):
        """Return the groups used for sampling."""
        return self.groups

    def get_group_count(self): 
        """Return the number of groups."""
        if self.groups is None:
            return 1
        else: return len(self.groups)

    def set_population(self, population):
        """Set the population and compute group and region counts."""
        self.population = population
        self.group_counts, self.region_count = self.get_group_and_region_counts_counts()
        
    def get_group_and_region_counts_counts(self):
        """Return group counts and region counts for the population."""
        region_counts = []
        for group in self.population: region_counts.append([len(region) for region in group])
        group_counts = [sum(group) for group in region_counts]
        return group_counts, region_counts


    @abstractmethod
    def sample_indices(self):
        """
        Return indices of the sampled items.

        Args:
            population_size (int): Total number of items to sample from.
            sample_size (int): Number of items to sample.
            **kwargs: Additional arguments for specific strategies.

        Returns:
            np.ndarray: Array of selected indices.
        """
        raise NotImplementedError("Subclasses must implement sample_indices.")
    
    def suggest_n(self):
        """Suggest sample size n (default -1)."""
        return -1
    


class SampleSaver:
    """
    SampleSaver generates labeled sample datasets from a larger pool of unlabeled images.
    It supports lots of sampling methods, SRS, stratified or model-based

    This class supports:
      - Uniform or area-proportional sampling from subregions.
      - Stratification by region.
      - Reusing existing labeled samples if available.
      - Sampling random patches from large images, with configurable patch size.
      - Saving sampled images, masks, and metadata to a new dataset directory.

    Args:
        sample_dimensions (tuple): (height, width) of each sample patch.
        new_dataset_dir (str): Directory to save the new labeled dataset.
        unlabeled_dir (str): Directory containing the source images.
        channel (str): Channel name to sample from each image.
        existing_labeled_dataset (str or None): Path to an existing labeled dataset to reuse samples from.
        uniform_sampling (bool): If True, sample equally from each subregion; else, sample by area.
        stratify_by_region (bool): If True, stratify sampling by region.
    """

    # sampling_strategy has the information for unlabeled_dir, sample_dimensions, channel and groups for stratification
    # # uniform_sampling = True, stratify_by_region = True

    # In other words SampleSaver only handles the saving logic, not the sampling logic

    # Will ignore existing_labeled_dataset for now, a simple way of 
    # combining this with sampling_strategy might look like only allowing sampling_strategy to be SRS 

    
    def __init__(self, new_dataset_dir, sampling_strategy, existing_labeled_dataset=None):

        
        if (os.listdir(new_dataset_dir)): raise ValueError(f"new dataset directory {new_dataset_dir} is not empty. use new name or remove existing one manually")
        if not (os.path.exists(new_dataset_dir)): raise ValueError(f"new dataset directory {new_dataset_dir} doesn't exist")

        self.existing_labeled_dataset = existing_labeled_dataset
        if existing_labeled_dataset is not None:
            if not (os.path.exists(existing_labeled_dataset)): raise ValueError(f"existing dataset {existing_labeled_dataset} doesn't exist")
            self.tracing_checker = TracingChecker(existing_labeled_dataset)
            if not (self.tracing_checker.is_valid()): raise ValueError(f"existing dataset {existing_labeled_dataset} isn't valid")
            if (self.tracing_checker.get_labelled_ratio() == 0.0): raise ValueError(f"existing dataset {existing_labeled_dataset} has no labelled data")
        
        self.root_write_dir = new_dataset_dir


    
        self.sampling_strategy = sampling_strategy
        self.population_paths = self.get_population_path_structure()
        print(self.population_paths)
        self.population, self.masks, self.starting_points = self.load_population(self.population_paths)
        self.sampling_strategy.set_population(self.population)

        self.n_regions = len(self.all_subregion_folders)
        if not self.uniform_sampling: 
            self.areas = self.dr.get_area_per_path()
            self.reg_probs = self.get_region_sample_prob()
            
        self.mask_disks = {}

        # creates really basic tracings just to test things out
        self.test_fake_tracings = False