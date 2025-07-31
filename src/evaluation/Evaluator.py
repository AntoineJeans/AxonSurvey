# Base class for evaluating model performance on rat brain image data with ground truth comparison.

import numpy as np
import copy
import os
from pathlib import Path

from .ImageLoader import ImageLoader
import matplotlib.pyplot as plt

import random
from ..utils.viz import display_grayscale
from ..utils.imageio import tif_to_numpy

from src.utils.imageio import generate_image_outer_mask



from abc import abstractmethod
class Evaluator:
    """Base class for evaluating model predictions against ground truth properties."""
    
    def __init__(self, image_paths, center_sample_size=128, tracings_cache_folder=None):
        # image_paths can be a list of .tif files or the root folder of a dataset created by sampling
            
        self.loader = ImageLoader(image_paths, split_step=None, final_img_size=center_sample_size)
        self.image_paths, self.mask_paths = self.loader.get_paths()
        self.ROI_paths = self.loader.get_ROI_paths()
        self.starting_points = self.loader.get_starting_points()
        
        self.loader.load()
        self.images = self.loader.get_images()
        self.masks = self.loader.get_masks()
        
        self.mask_shape = self.masks[0].shape
        
        # self.ground_truth_functions must be specified in child classes, defined before calling base class constructor
        self.m_properties = len(self.ground_truth_functions)
        self.n_images = len(self.images)
        
        # cache params, for permanent tracings in file system
        self.tracings_cache_folder = tracings_cache_folder
        self.ground_truth_properties = self.get_ground_truths()

        self.set_up_ROI_masks()
        self.set_up_small_img_masks()

    def set_up_ROI_masks(self):
        """Creates outer masks for each ROI image."""
        ROIs = np.unique(self.ROI_paths)
        self.roi_to_mask = {}
        for roi in ROIs:
            self.roi_to_mask[roi] = generate_image_outer_mask(tif_to_numpy(roi))

    def starting_point_to_image(self, starting_point, image):
        """Extracts a sub-image centered on the starting point."""
        (a, b) = starting_point
        (a, b) = (a + 64, b + 64)
        return image[a:a + self.mask_shape[0], b:b + self.mask_shape[1]]



    def set_up_small_img_masks(self):
        """Creates small masks for each image based on ROI and starting points."""
        self.small_masks = []
        for (a,b), roi in zip(self.starting_points, self.ROI_paths):
            small_mask = self.starting_point_to_image((a,b), self.roi_to_mask[roi])
            self.small_masks.append(small_mask)
        
        
    def get_ground_truths(self):
        """Calculates ground truth properties for all masks."""
        properties = []
        for mask in self.masks:
            properties_for_this_image = [prop_func(mask) for prop_func in self.ground_truth_functions]
            properties.append(properties_for_this_image)
        return np.array(properties)  
            
    def predict(self, model, image, image_path, mask, map=False):
        """Makes predictions using the model on a single image."""
        property_array = model.predict(copy.deepcopy(image), image_path=image_path, mask=mask)
        if map: return property_array
        else: return np.array([np.nanmean(prop_arr, axis=(0,1)) for prop_arr in property_array.transpose(2,0,1)])
        
    
    def _predict_on(self, model, imgs, image_paths=None, masks=None, map=False):
        """Makes predictions on multiple images."""
        if image_paths is None: image_paths = [None for _ in range(len(imgs))]
        if masks is None: masks = [None for _ in range(len(imgs))]
        return [self.predict(model, im, path, mask, map=map) for im, path, mask in zip(imgs, image_paths, masks)]
            
    
    def evaluate(self, model, display_fitness=False):
        """Evaluates model performance on all images."""
        # Model name is simply used to identify the cached tracing. If no model name is passed, no cache is created 
        reals = self.ground_truth_properties
        ests = np.array(self._predict_on(model, self.images, image_paths=None, masks=self.small_masks, map=False))
        if display_fitness: 
            self.plot_fitness(ests, reals)
        return ests, reals
             
             
    def evaluate_ROIs(self, model, display_fitness=False):
        """Evaluates model performance on full ROI images."""
        reals = self.ground_truth_properties
        
        ROIs = np.unique(self.ROI_paths)
        whole_imgs = [tif_to_numpy(roi) for roi in ROIs]
        masks = [self.roi_to_mask[roi] for roi in ROIs]
        property_arrays = self._predict_on(model, whole_imgs, image_paths=ROIs, masks=masks, map=True)
        
        roi_to_props = {}
        for roi, props in zip(ROIs, property_arrays):
            roi_to_props[roi] = props
        
        estimated_properties = []
        for (a,b), roi in zip(self.starting_points, self.ROI_paths):
            props = self.starting_point_to_image((a, b), roi_to_props[roi])
            estimated_properties.append(np.nanmean(props, axis=(0,1)))

        estimated_properties = np.array(estimated_properties)
        if display_fitness: 
            self.plot_fitness(estimated_properties, reals)
            model.show_feature_weights(self.estimated_names)
            
        return estimated_properties, reals


    def _bootstrap(self, model, n_samples, score_function, param_name, display_hist=True, model_name = None):
        """Performs bootstrap evaluation with confidence intervals."""
        ests, reals = self.evaluate_ROIs(model, display_fitness=False) ### probably change display_fitness=True when working
        
        scores_for_properties = []
        for i in range(self.m_properties):
            prop_reals = [real[i] for real in reals]
            pro_ests = [est[i] for est in ests]
            
            scores = []
            for _ in range(n_samples):
                
                indices = np.random.choice(range(self.n_images), size=n_samples, replace=True)
                ests_subset = [prop_reals[i] for i in indices]
                reals_subset = [pro_ests[i] for i in indices]
                score = score_function(ests_subset, reals_subset)
                scores.append(score)
                
            if display_hist: self.plot_hist(scores, param=param_name)
            scores_for_properties.append(scores)
        return scores_for_properties
    
    def plot_hist(self, data, param="Pearson coefficient"):
        """Plots histogram of bootstrap scores."""
        plt.hist(data, bins='auto', edgecolor='black')
        plt.xlabel(f'{param}')
        plt.ylabel('N-bootstrapped trials')
        plt.title(f'N-bootstrapped trials by {param}')
        plt.show()

    def display_random_images_and_predictions(self, model, n_images=10):
        """Displays random images with their predictions and ground truth."""
        random_indices = random.sample(range(len(self.images)), n_images)
        

        sampled_ground_truth = [self.ground_truth_properties[i] for i in random_indices]
        sampled_images = [self.images[i] for i in random_indices]
        sampled_masks = [self.small_masks[i] for i in random_indices]
        sampled_tracings =  [self.masks[i] for i in random_indices]
        image_folders = [self.image_paths[i] for i in random_indices]


        predicted_properties = self._predict_on(model, sampled_images, image_paths=None, masks=sampled_masks, map=False)
        
        for ground_truth, pred, image, tracing, mask, fldr in zip(sampled_ground_truth, predicted_properties, sampled_images, sampled_tracings, sampled_masks, image_folders):
            gt_rounded = np.round(ground_truth.astype(np.float64), 2)
            pred_rounded = np.round(np.array(pred, dtype=np.float64), 2)
            np.set_printoptions(suppress=True)
            print(f"Real properties : {gt_rounded}, Predicted properties : {pred_rounded}")
            print(f"From path : {fldr}")

            display_grayscale(image, title="Original grayscale image")
            display_grayscale(mask, title="Image Outer mask")
            display_grayscale(tracing, title="Human Axon Tracing for that image")


    def display_random_ROIS(self, n_rois=3):
        """Displays random ROI images with their masks."""
        unique_rois = np.unique(self.ROI_paths)
        random_indices = random.sample(range(len(unique_rois)), n_rois)

        for i in random_indices:
            roi = unique_rois[i]
            print(f"ROI : {roi}")
            display_grayscale(tif_to_numpy(roi), title="ROI Whole Image")
            display_grayscale(self.roi_to_mask[roi], title="ROI Whole Image")
            



    def get_image_and_tracing_subset(self, n, seed):
        """Gets a subset of images and tracings for evaluation."""
        return self.loader.get_image_and_tracing_subset(n, seed)
    
    @abstractmethod
    def plot_fitness(self, ests, reals):
        """Plots fitness comparison between estimated and real values."""
        pass








