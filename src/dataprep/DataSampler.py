# Module for sampling and creating labeled datasets from rat brain image populations.

import os
import pathlib
import numpy as np
from ..utils.imageio import tif_to_numpy, numpy_to_tif
from ..dataprep.DataReader import DataReader
from ..dataprep.TracingChecker import TracingChecker

from ..utils.viz import display_grayscale

from skimage.util import view_as_windows


class DataSampler:
    """
    DataSampler generates labeled sample datasets from a larger pool of unlabeled images.
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
    
    def __init__(self, sample_dimensions, new_dataset_dir, unlabeled_dir, sampling_strategy,
                 channel="th", existing_labeled_dataset=None, uniform_sampling = True, stratify_by_region = True):
        
        # uniform_sampling : determines if images are taken at equal amounts in each subregion. 
        #                   if False, sample proportionnal to the area of each subregion
        
        # existing_labeled_dataset:a path to a directory already created by datasampler. It must not be empty.
        #                   The number of images from each ROI is not affected, but for each new image to label,
        #                   If a traced image from that ROI exists in the existing dataset, it is taken instead, 
        #                   until all images from that ROI are used, then samples as usual. If none, all new samples.
        #                   supposes the sample dimensions for both datasets are the same
        
        if not (os.path.exists(unlabeled_dir)): raise ValueError(f"read directory {unlabeled_dir} doesn't exist")
        if (os.listdir(new_dataset_dir)): raise ValueError(f"new dataset directory {new_dataset_dir} is not empty. use new name or remove existing one manually")
        if not (os.path.exists(new_dataset_dir)): raise ValueError(f"new dataset directory {new_dataset_dir} doesn't exist")

        self.dr = DataReader(unlabeled_dir)
        if not (self.dr.read_dir_is_valid()): raise ValueError(f"read directory {unlabeled_dir} not valid")
        
        if existing_labeled_dataset is not None:
            if not (os.path.exists(existing_labeled_dataset)): raise ValueError(f"existing dataset {existing_labeled_dataset} doesn't exist")
            self.tracing_checker = TracingChecker(existing_labeled_dataset)
            if not (self.tracing_checker.is_valid()): raise ValueError(f"existing dataset {existing_labeled_dataset} isn't valid")
            if (self.tracing_checker.get_labelled_ratio() == 0.0): raise ValueError(f"existing dataset {existing_labeled_dataset} has no labelled data")
        
        
        self.root_read_dir = unlabeled_dir
        self.root_write_dir = new_dataset_dir
        self.existing_labeled_dataset = existing_labeled_dataset
        
        self.sample_dimensions = sample_dimensions
        self.sample_area = sample_dimensions[0] * sample_dimensions[1]
    
        self.uniform_sampling = uniform_sampling
        self.stratify_by_region = stratify_by_region
        
        self.channel = channel
        
        self.all_subregion_folders = self.dr.get_paths()


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
        


    def get_population_path_structure(self):
        """Gets population path structure based on sampling strategy groups."""
        groups = self.sampling_strategy.get_groups()
        if groups is None:
            return [self.all_subregion_folders] # no groups means all images treated as one big group for sampling
        else: 
            population_paths = []
            for group in groups:
                population_paths.append([path for path in self.all_subregion_folders if self.dr.get_region(path) in group])
            return population_paths
                
    def load_population(self, population_paths):
        """Loads population images, masks, and starting points from paths."""
        outer_acceptable_ratio = 0.5
        population = []
        pop_masks = []
        pop_points = []

        for group in population_paths:
            regions_im, regions_ma, regions_po = [], [], []
            for path in group:
                img_path = os.path.join(path, f'{self.channel}.tif')
                print(f"opening image {img_path}")
                
                # creates a np array of all images
                
                images_window = view_as_windows(tif_to_numpy(img_path), window_shape=self.sample_dimensions, step=self.sample_dimensions)
                images = images_window.reshape(-1, self.sample_dimensions[0], self.sample_dimensions[1])
                
                masks = np.array([self.dr.get_outer_mask(im) for im in images])
                points = np.empty(images_window.shape[:2], dtype=object)
                for i in range(points.shape[0]): 
                    for j in range(points.shape[1]) :
                        points[i,j] = (i*self.sample_dimensions[0], j*self.sample_dimensions[1])

                points = points.reshape(-1)
                
                func = lambda mask: (np.sum(mask) / self.sample_area) > outer_acceptable_ratio

                valid_indices = [i for i, im in enumerate(masks) if func(im)]

                images = [a for a in images[valid_indices]]
                masks = [a for a in masks[valid_indices]]
                points = [a for a in points[valid_indices]]

                print(f"found {len(images)} {len(masks)} {len(points)} valid images, of shape {self.sample_dimensions}")
 
                regions_im.append(images)
                regions_ma.append(masks)
                regions_po.append(points)

            population.append(regions_im)
            pop_masks.append(regions_ma)
            pop_points.append(regions_po)

        return population, pop_masks, pop_points

    def suggest_n(self):
        """Suggests statistically informed sample size from sampling strategy."""
        return self.sampling_strategy.suggest_n()

    def sample(self, n):
        """Samples n patches from population using the sampling strategy."""
        sampled_population_indices = self.sampling_strategy.sample_indices(n)
        return self.get_sample_from_groups_indices(sampled_population_indices)
        
                    
    def get_sample_from_groups_indices(self, indices):
        """Gets samples from group indices and returns formatted sample data."""
        sample  = ([], [], [], [])
        for grp_indices, grp_ROI, grp_masks, grp_points, grp_paths in zip(indices, self.population, self.masks, self.starting_points, self.population_paths):
            for reg_indices, reg_ROI, reg_masks, reg_points, reg_path in zip(grp_indices, grp_ROI, reg_masks, reg_points, grp_paths):
                for sampled_index in reg_indices:
                    print(grp_points)
                    sample[0].append(reg_ROI[sampled_index])
                    sample[1].append(reg_masks[sampled_index])
                    sample[2].append(reg_points[sampled_index])
                    sample[3].append(reg_path)
        return sample



    # supposing areas are already calculated
    def get_region_sample_prob(self):
        """Returns sampling probability for each region, proportional to area."""
        total_unsampled_area = np.sum(self.areas)
        return self.areas / total_unsampled_area
    
    def get_prop_path_counts(self, n):
        """Returns sample counts per region for area-proportional sampling."""
        if self.stratify_by_region:
            counts = np.random.multinomial(n, self.reg_probs)
        
        else : 
            sampled_indices = np.random.choice(self.n_regions, size=n, p=self.reg_probs)
            counts = np.bincount(sampled_indices, minlength=self.n_regions)
            
        return counts
    
    def get_uniform_paths_counts(self, n):
        """Returns sample counts per region for uniform sampling."""
        if self.stratify_by_region:
            samples_per_class = n // self.n_regions
            remainder = n % self.n_regions
            counts = np.full(self.n_regions, samples_per_class)
            counts[:remainder] += 1
        else: 
            sampled_indices = np.random.choice(self.n_regions, size=n)
            counts = np.bincount(sampled_indices, minlength=self.n_regions)
        return counts
            
    def get_sample_path_counts(self, n):
        """Returns sample counts per region based on sampling strategy."""
        # returns a list of how many times each image at the corresponding index should be sampled from for the dataset 
        if self.uniform_sampling:
            counts = self.get_uniform_paths_counts(n)
        else: 
            counts = self.get_prop_path_counts(n)
        return counts
    
    def sample_old(self, n, outer_acceptable_ratio = 0.5):
        """Samples n image patches, optionally using existing labeled data."""
        samples  = ([], [], [], [])
          
        counts = self.get_sample_path_counts(n)
        
        if self.existing_labeled_dataset is not None:
            counts, samples = self.add_samples_from_existing_dataset(counts, samples)
        
        for i in range(self.n_regions):
            if counts[i] > 0:
                img_folder = self.all_subregion_folders[i]
                img_path = os.path.join(img_folder, f'{self.channel}.tif')
                self.add_samples_from_image(counts[i], img_path, samples, outer_acceptable_ratio,)
                
        return samples
    
    def add_samples_from_existing_dataset(self, counts, samples):
        """Adds samples from an existing labeled dataset if available."""
        # add as many samples as possible from existing dataset and updates counts
        images, masks, starting_points, original_files = samples
        
        for i in range(self.n_regions):
            if counts[i] > 0:
                labeled_sample_paths = self.list_existing_label_paths_for_region(self.all_subregion_folders[i])
                
                if len(labeled_sample_paths) > 0:
                    n_to_add = min(counts[i], len(labeled_sample_paths))
                    counts[i] -= n_to_add
                    
                    for sample_path in labeled_sample_paths[:n_to_add]:
                        
                        # load and append values from the directory
                        img = tif_to_numpy(os.path.join(sample_path, self.tracing_checker.image_sample_file_name)).astype(np.uint8)
                        images.append(img)
                        
                        mask = tif_to_numpy(os.path.join(sample_path, self.tracing_checker.mask_file_name)).astype(bool)
                        masks.append(mask)
                        
                        with open(os.path.join(sample_path, self.tracing_checker.info_file_name), "r") as f:
                            lines = f.readlines()

                        first_line = lines[0].strip() if len(lines) > 0 else ""
                        tuple_value = tuple(map(int, first_line.strip("()").split(",")))
                        starting_points.append(tuple_value)
                        
                        second_line = lines[1].strip() if len(lines) > 1 else ""
                        original_files.append(second_line)
                
        
        return counts, (images, masks, starting_points, original_files)
        
    def list_existing_label_paths_for_region(self, region_path):
        """Returns paths to labeled samples for a given region."""
        return self.tracing_checker.get_img_paths_from_original(region_path)
    
    def get_random_point_in_image(self, img):
        """Returns a random (row, col) for sampling a patch from img."""
        row = np.random.randint(0, img.shape[0] - self.sample_dimensions[0])
        col = np.random.randint(0, img.shape[1] - self.sample_dimensions[1])
        return row, col
    
    def add_samples_from_image(self, n, img_path, samples, outer_acceptable_ratio):
        """Adds n valid samples from a single image to the dataset."""
        images, masks, starting_points, original_files = samples
        sampled = 0
        
        
        
        img = tif_to_numpy(img_path)
        
        if (self.sample_dimensions[0] > img.shape[0] or self.sample_dimensions[1] > img.shape[1]):
            print(f"WARNING: skipping img at {img_path} because too small for sample dimensions")
            return
        
        if img_path in self.mask_disks: img_mask = self.mask_disks[img_path]
        else: 
            img_mask = self.dr.get_outer_mask(img)
         
                
        for _ in range(n * 10):
            
            row, col = self.get_random_point_in_image(img)
            img_sample = img[row:row + self.sample_dimensions[0], col:col + self.sample_dimensions[1]]
            mask_sample = img_mask[row:row + self.sample_dimensions[0], col:col + self.sample_dimensions[1]]
            # BLACK_RATIO = 0.5
            if np.sum(mask_sample) / self.sample_area > outer_acceptable_ratio:
                
                images.append(img_sample)   
                masks.append(mask_sample) 
                starting_points.append((row, col))    
                original_files.append(img_path)
                sampled+=1
        
            if (sampled == n): break
            
        if (sampled < n): raise ValueError(f"Couldn't find {n} regions to sample in {img_path}")


    def create_dataset(self, size):
        """Creates a labeled dataset of the given size and saves to disk."""
        sample_number = 0
        samples = self.sample(size)

        for img, mask, point, file in zip(*samples):
            sample_number += 1
            save_path = self.root_write_dir + f"/img_{sample_number:04}/"
            
            os.makedirs(save_path, exist_ok=True)
            numpy_to_tif(img, save_path + "img.tif")
            numpy_to_tif(mask, save_path + "outer_mask.tif")

            if self.test_fake_tracings:
                numpy_to_tif(np.random.rand(img.shape[0], img.shape[1]) > 0.95, save_path + "tracings.tif")
            
            pth = pathlib.Path(file[len(self.root_read_dir):])
            file_parts = list(pth.parts)

            with open(save_path + "info.txt", "w") as f:
                f.write(str(point))
                f.write('\n')
                f.write(file)
                f.write('\n')
                for part in file_parts: 
                    f.write(part)
                    f.write('\n')

class TrainSampler(DataSampler):
    """Specialized DataSampler for creating training datasets."""
    def __init__(self, *args, **kwargs):
        """Initializes TrainSampler."""
        super().__init__(*args, **kwargs)     
            
class TestSampler(DataSampler):
    """Specialized DataSampler for creating test datasets."""
    def __init__(self,  *args, **kwargs):
        """Initializes TestSampler."""
        super().__init__(*args, **kwargs)