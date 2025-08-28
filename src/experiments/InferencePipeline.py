# This module provides a complete inference pipeline for training models, evaluating performance, and making predictions with confidence intervals.

import numpy as np
import copy
from scipy import stats

import json
from pathlib import Path
from datetime import datetime

from .ExperimentLoader import ExperimentLoader

from ..configs import make_models

from ..evaluation.Trainer import Trainer
from ..evaluation.RegressionEvaluator import RegressionEvaluator
from ..utils.traceProps import get_trace_density, get_mean_axon_length, get_axon_count

from ..utils.graphs import display_model_bounds, display_model_rmse, \
                                        display_test_bias_for_group, display_inference_points
from ..utils.graphs import display_inference_bounds

from ..inference.TempEstimator import TemporaryEstimator
from ..inference.BaseEstimator import BaseEstimator

import matplotlib.pyplot as plt

class InferencePipeline():
    """Complete pipeline for training models, evaluating performance, and making predictions with confidence intervals."""
    
    def __init__(self, groups, img_input_size, 
                 og_path, train_path, test_path, val_path=None, 
                 rmse_confidence=0.95, inference_confidence = 0.95, group_labels = None,
                 ground_truth_functions=[get_trace_density], propery_names=["Axon innervation density"],
                 debug_mode=False):
        """Initialize inference pipeline with experiment parameters and configuration."""
        plt.style.use('classic')


        # STEP 1 - SET EXPERIMENT PARAMETERS
        self.groups = groups
        self.n_groups = len(groups)
        
        self.group_labels = group_labels if group_labels is not None else [str(grp) for grp in self.groups]

        self.propery_names = propery_names
        self.ground_truth_functions = ground_truth_functions
        
        self.og_path = og_path
        self.train_path = train_path
        self.test_path = test_path
        self.val_path= val_path
        
        self.rmse_confidence = rmse_confidence
        self.inference_confidence = inference_confidence

        self.img_input_size = img_input_size
        self.input_shape = (img_input_size,img_input_size)
        
        self.debug_mode = debug_mode
        
        
        # Data to be loaded
        self.trainers = []
        self.val_evaluators = []
        self.test_evaluators = []
        self.inference_data = []
        
        # models to be created
        self.models = []
        self.model_names = []
        self.model_types = None
        
        # performances to be evaluated
        self.model_rmses = []
        self.best_models = []
        self.best_model_names = []
        self.expected_rmses = []

        self.experiment_id = self._get_next_experiment_id()
        self.experiment_date = datetime.now().strftime("%Y-%m-%d")
        self.experiment_id_string = str(self.experiment_id)
        
        # Experiment metadata
        self.experiment_name = None
        self.experimenter_name = None
        
    def _get_next_experiment_id(self):
        """Get the next experiment ID by counting existing experiments."""
        experiments_dir = Path("./experiments")
        if not experiments_dir.exists():
            return "0001"
        
        # Count existing experiment directories
        existing_experiments = [d for d in experiments_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        next_id = len(existing_experiments) + 1
        return f"{next_id:04d}"

    # STEP 2 - LOAD EXPERIMENT DATA
    # loads image paths to use from the experiment datasets and saves as attribute as trainers and evaluators for models
    def load_data(self):
        """Load training, validation, test, and inference data for the experiment."""
        self.loader = ExperimentLoader(self.groups, self.train_path, self.test_path, self.og_path, val_path=self.val_path)
        
        # Load training data
        training_data_for_groups = self.loader.get_experiment_train_data()
        
        self.trainers = [Trainer(train_paths, ground_truth_functions=self.ground_truth_functions) for train_paths in training_data_for_groups]

        # Load validation/testing data
        test_data = self.loader.get_experiment_test_data()
        self.test_evaluators = [
            RegressionEvaluator(image_paths=paths, tracings_cache_folder=None, 
                                estimated_names=self.propery_names, ground_truth_functions=self.ground_truth_functions) 
                                for paths in test_data
        ]

        
        if self.val_path is not None:
            val_data = self.loader.get_experiment_val_data()
            self.val_evaluators = [RegressionEvaluator(image_paths=paths, estimated_name=self.trace_prop_of_interest, ground_truth_function=self.prop_function) for paths in val_data]
            
        # Load inference data (large images whose density we want to estimate)
        self.inference_data = self.loader.get_inference_data(channel="th")

        
        
    # STEP 3.1 - Create models for each group
    def make_models(self, model_list=None, name_list=None):
        """Create models for each group using configuration or provided model list."""
        if model_list is None:
            self.models, self.model_names, self.model_types = make_models(self.n_groups)
        else:
            if name_list is None:
                name_list = [f"model_{i}" for i in range(len(model_list))] 
                
            self.models, self.model_names = [copy.deepcopy(model_list) for _ in range(self.n_groups)], [copy.deepcopy(name_list) for _ in range(self.n_groups)]
            self.model_types = None
      
    # STEP 3.2 - Train all the models :)      
    def train_models(self):
        """Train all models using the loaded training data."""
        for trainer, model_list in zip(self.trainers, self.models):
            for model in model_list: 
                if self.debug_mode: print(f"Training model {model}")
                trainer.fit_model(model, plot_correlation=self.debug_mode, property_names=self.propery_names)
        if self.debug_mode: self.show_images_per_model(1)
            
            
    def show_images_per_model(self, n_images):
        """Display sample images and predictions for each model."""
        for i, (group_label, evaluator) in enumerate(zip(self.group_labels, self.test_evaluators)):
            print(f"Images and predictions for models trained on group {group_label}")
            for iter_model, iter_name in zip(self.models[i], self.model_names[i]):
                print(f"Model = {iter_name}")
                evaluator.display_random_images_and_predictions(iter_model, n_images=n_images)

           
    # STEP 4 - EVALUATE EACH MODEL IN EACH GROUP ON ITS TEST DATASET FOR MODEL SELECTION
    # Ideally smaller bootstrapping than step 5 for fast model selection
    def evaluate_models(self, n_bootstraps_trials=100, display_model_performances=True):
        """Evaluate model performance using bootstrap trials."""
        self.evaluate_model_rmse(n_bootstraps_trials, display_model_performances)
        # self.evaluate_model_bias(n_bootstraps_trials, display_model_performances)

    def evaluate_model_rmse(self, n_bootstraps_trials, display_model_performances=True): 
        """Evaluate model RMSE using bootstrap trials."""
        # evaluators = self.val_evaluators if self.val_path is not None else self.test_evaluators
        bootstrap_func = lambda evaluator, model : evaluator.bootstrap_rmse(model, n_bootstraps_trials) 
        self.model_rmses = self.bootstrap_metric(bootstrap_func=bootstrap_func, display_model_performances=display_model_performances, metric_name="RMSE")
        # if self.debug_mode: 
        # self.display_all_models_predictions()
        
    def evaluate_model_bias(self, n_bootstraps_trials, display_model_performances=True):
        """Evaluate model bias using bootstrap trials."""
        bootstrap_func = lambda evaluator, model : evaluator.bootstrap_bias(model, n_bootstraps_trials) 
        self.model_biases = self.bootstrap_metric(bootstrap_func=bootstrap_func, display_model_performances=display_model_performances, metric_name="BIAS")

    
    def select_best_models(self):
        """Select best models based on performance metrics."""
        ### IMPORTANT : Should consider self.model_rmses for model selection in the future
        # but right now, bias is the biggest source of error
        best_model_ids = [np.argmin(group_perf) for group_perf in self.model_rmses]
        # For debugging, select first model
        # best_model_ids = [0 for _ in range(self.n_groups)]
        
        self.best_models = [self.models[i][best_model_ids[i]] for i in range(self.n_groups)]
        self.best_model_names = [self.model_names[i][best_model_ids[i]] for i in range(self.n_groups)]
        
        if self.debug_mode:
            self.display_best_models_predictions()
            for evaluator, model, name, label in zip(self.test_evaluators, self.best_models, self.best_model_names, self.group_labels):
                evaluator.real_and_predicted_distribution_in_dataset(model, name, label)

    def bootstrap_metric(self, bootstrap_func, display_model_performances=True, metric_name=""):
        """Calculate bootstrap metrics for all models."""
        evaluators = self.test_evaluators

        model_rmses = []
        for evaluator, model_list, name_list, model_type_list, group_label in zip(evaluators, self.models, self.model_names, self.model_types, self.group_labels):
            property_bounds = self.bootstrap_model_list(model_list, evaluator, bootstrap_func)
            error_points = []
            for (expec, lower, upper), property_name in zip(property_bounds, self.propery_names):
                if display_model_performances: 
                    title = f"{metric_name} scores for {property_name} predictions in {group_label} regions"
                    save_path = f"./figures/experiment_figures/{self.experiment_id_string}/model_performances"
                    display_model_bounds(expec, lower, upper, name_list, title=title, 
                                         metric_name=metric_name, save_path=save_path, model_types=model_type_list)
                error_points.append(upper)

            error_points = np.array(error_points).transpose(1, 0) # bring the model instances at the first dimension
            rmses = [evaluator.combine_rmses(errors) for errors in error_points]
            
            model_rmses.append(rmses)
            # print("ERROR FOR EACH PROPERTY:", error_points)
            # print("WEIGHTED RMSEs: ", rmses)
        return model_rmses
    
    def bootstrap_model_list(self, model_list, evaluators, bootstrap_func):
        """Bootstrap metrics for a list of models."""
        if not isinstance(evaluators, list): evaluators = [evaluators for _ in range(len(model_list))]

        # this assumes all models have the same property outputs
        n_properties = evaluators[0].m_properties

        all_metrics = []
        for model, evaluator in zip(model_list, evaluators):                
            all_metrics.append(bootstrap_func(evaluator, model))
        all_metrics = np.array(all_metrics)
        
        all_bounds = []
        for i in range(n_properties):
            expec, lower, upper = [], [], []
            for metrics in all_metrics:   
                this_property_metric_for_this_model = metrics[i]      
                l, u = evaluator.get_bounds(this_property_metric_for_this_model, confidence=self.rmse_confidence)
                expec.append(np.mean(this_property_metric_for_this_model))
                lower.append(l)
                upper.append(u)
            all_bounds.append((expec, lower, upper))

        return all_bounds
        

    # STEP 5 - DEDUCE RMSE ON MODELS 
    def calculate_model_uncertainty(self, n_bootstrap, use_upper_bound=False):
        """Calculate model uncertainty using bootstrap trials."""
        self.expected_rmses = self.get_best_models_rmses(n_bootstrap, use_upper_bound)
        # self.expected_biases = self.get_best_models_biases(n_bootstrap, use_upper_bound)

    def get_best_models_rmses(self, n_bootstraps_trials, use_upper_bound=True):
        """Get RMSE for best models using bootstrap trials."""
        bootstrap_func = lambda evaluator, model : evaluator.bootstrap_rmse(model, n_bootstraps_trials) 
        all_bounds = self.bootstrap_model_list(self.best_models, self.test_evaluators, bootstrap_func) 

        error_points = []
        for (expec, lower, upper), prop_name in zip(all_bounds, self.propery_names):           
            
            labels = [f" {name} on {label}" for name,label in zip(self.best_model_names, self.group_labels) ]
            title=f"RMSE for {prop_name} by best model in each region type, confidence={self.rmse_confidence}"
            save_path = f"./figures/experiment_figures/{self.experiment_id_string}/model_performances/"
            display_model_bounds(expec, lower, upper, labels, save_path=save_path,
                                title=title, metric_name="RMSE")
        
            if use_upper_bound:
                error_points.append(upper)
            else:
                error_points.append(expec)

        error_points = np.array(error_points).transpose(1, 0) # bring the model instances at the first dimension
        rmses = [evaluator.combine_rmses(errors) for errors, evaluator in zip(error_points, self.test_evaluators)]
        return rmses

    # def get_best_models_biases(self, n_bootstraps_trials, use_upper_bound=True):
    #     """Get bias for best models using bootstrap trials."""
    #     bootstrap_func = lambda evaluator, model : evaluator.bootstrap_bias(model, n_bootstraps_trials) 
    #     expec, lower, upper = self.bootstrap_model_list(self.best_models, self.test_evaluators, bootstrap_func)            
    #     labels = [f" {name} on {label}" for name,label in zip(self.best_model_names, self.group_labels) ]
    #     display_model_bounds(expec, lower, upper, labels, title=f"BIAS by best model in each region type, confidence={self.rmse_confidence}", metric_name="BIAS")
    #     return upper if use_upper_bound else expec
    

    # STEP 6 - INFER ON THE REGIONS WE CARE ABOUT
    def infer_mean_region_density_in_groups(self):
        """Infer mean region density for each group with confidence intervals."""
        # Here we leave open the possibility to predict a statistical estimator based on 
        # sample / model performance. 
        # This falls in the statistical field of model-based estimation.

        # This object should  have all information available for any estimator,
        # it's just a matter of implementation. 

        # For now, we simply get fixed point prediction for each ROI, and plot it
        # bisous

        use_fixed_point_predictions = True

        # inference data already loaded
        # For each group, we can identify any number of regions from the og files from which evaluation datasets are based on
        # And infer the mean density with confidence interval. The bigger the image is, the longer it will take, but the smaller the intervals will be (which is good)
        
        group_data = []
        for og_files, model, rmse in zip(self.inference_data, self.best_models, self.expected_rmses):
            if use_fixed_point_predictions:
                predictor = BaseEstimator(None, og_files, model=model)
                points = predictor.predict_points()
                group_data.append(points)

            else: 
                # left to implement -  can follow some of this next approach, 
                # maybe display_inference_bounds
                
                # predictors should be given sample data, population features, and sampling weights
                # and provide expected population mean and variance
                # 
                # predictors = [TempEstimator(model, None, og_files)] 
                # estimator_names = ["Est1", "Est2", "Est3"]
                # group_interval = []
                # for predictor in predictors:
                #        
                #     # e,u,l = predictor.estimate(rmse, confidence_interval=self.inference_confidence)
                #     # n = inf_man.get_n_per_image(og_files)
                #     # e,u,l = inf_man.get_bounds_for_image_group() 
                #     group_interval.append((e,u,l))
                #     # cheating for example graph
                #     group_interval.append((e, e + 0.8*(u - e), l + 0.2*(e - l)))
                #     group_interval.append((e, e + 0.6*(u - e), l + 0.4*(e - l)))
                # group_data.append(group_interval)
                pass


        inference_figure_save_path = f"./figures/experiment_figures/{self.experiment_id_string}/inference_results"
        # Now display
        if use_fixed_point_predictions:
            title = f"Predicted densities in sampled regions"
            display_inference_points(group_data, labels=self.group_labels, title=title, 
                                        save_path=inference_figure_save_path)
            
            
        else:
            estimator_names = ["not sure"]
            title = f"Expected axon density in sampled regions, confidence={self.inference_confidence}"
            display_inference_bounds(group_data, labels=self.group_labels, 
                                        title=title, predictor_names = estimator_names, 
                                        save_path=inference_figure_save_path)

        # self.plot_density_distribution_for_groups(inf_man)
        # if self.debug_mode:
        #     self.plot_density_distribution_for_images(inf_man)

        self.group_ROI_points = group_data


    def _convert_numpy_types(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def save_experiment_data(self):
        """Save experiment data to JSON file with all available information."""
        
        # Get dataset information
        n_training_samples = sum(len(trainer.loader.image_paths) for trainer in self.trainers) if self.trainers else 0
        n_testing_samples = sum(len(evaluator.loader.image_paths) for evaluator in self.test_evaluators) if self.test_evaluators else 0
        n_validation_samples = sum(len(evaluator.loader.image_paths) for evaluator in self.val_evaluators) if self.val_evaluators else 0
        

        # Extract and deduplicate studied data using list comprehensions
        
        # Check if any group has ALL_RATS or ALL_REGIONS
        has_all_rats = any(group.rats == "ALL_RATS" for group in self.groups)
        has_all_regions = any(group.regions == "ALL_REGIONS" for group in self.groups)
        
        if has_all_rats:
            studied_rats = np.array(["ALL_RATS"])
        else:
            studied_rats = np.unique([rat for group in self.groups for rat in group.rats])
            
        if has_all_regions:
            studied_regions = np.array(["ALL_REGIONS"])
        else:
            studied_regions = np.unique([region for group in self.groups for region in group.regions])
            
        studied_ROI_paths = np.unique([path for paths in self.loader.inference_paths_for_groups for path in paths])


        # Build groups data
        groups_data = []
        for i, (group, group_label, best_model_name, expected_rmse, 
                model_names, model_types, model_rmses,
                inference_data, inferred_ROI_points) in enumerate(zip(
            self.groups, self.group_labels, self.best_model_names, self.expected_rmses,
            self.model_names, self.model_types, self.model_rmses, self.loader.inference_paths_for_groups, self.group_ROI_points
        )):
            
            # Get average density from inference if available
            average_density = np.nanmean(inferred_ROI_points)
            roi_associations = {}
            
            for file, point in zip(inference_data, inferred_ROI_points):
                roi_associations[file] = point


            # Build models data for this group
            models_data = []
            
            for j, (model_name, model_rmse) in enumerate(zip(model_names, model_rmses)):
                model_type = model_types[j] if model_types is not None else None
                
                models_data.append({
                    "model_name": model_name,
                    "model_type": model_type,
                    "model_rmse": float(model_rmse) if model_rmse is not None else None,
                })
            
            group_data = {
                "name": group_label,
                "rats": ["ALL_RATS"] if group.rats == "ALL_RATS" else group.rats,
                "regions": ["ALL_REGIONS"] if group.regions == "ALL_REGIONS" else group.regions,
                "best_model_name": best_model_name,
                "expected_rmse": float(expected_rmse) if expected_rmse is not None else None,
                "average_density": average_density,
                "roi_associations": roi_associations,
                "models": models_data
            }
            groups_data.append(group_data)
        
        comparison_results_data = {
            "experiment_id": self.experiment_id,
            "experiment_date": self.experiment_date,
            "experiment_name" : self.experiment_name,
            "experimenter_name": self.experimenter_name,

            "datasets": {
                "training_dataset_path": str(self.train_path) if self.train_path else None,
                "n_training_samples": n_training_samples,
                "testing_dataset_path": str(self.test_path) if self.test_path else None,
                "n_testing_samples": n_testing_samples,
                "validation_dataset_path": str(self.val_path) if self.val_path else None,
                "n_validation_samples": n_validation_samples,
            },

            "studied_regions": {
                "all_rats": studied_rats.tolist(),  # Convert numpy arrays to lists for JSON serialization
                "all_region_types": studied_regions.tolist(),
                "all_ROI_paths": studied_ROI_paths.tolist()
            },

            "groups": groups_data,
        }

        # Convert all numpy types to JSON-serializable types
        comparison_results_data = self._convert_numpy_types(comparison_results_data)

        file_path = f"./experiments/{self.experiment_id}/data.json"
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(comparison_results_data, f, indent=2, ensure_ascii=False)




    # utils functions for displaying stuff 
    def display_best_models_predictions(self):
        """Display predictions for best models."""
        for model, model_name, evaluator,label in zip(self.best_models, self.best_model_names, self.test_evaluators, self.group_labels):
            print(f"real and predicted values for {model_name} (best) in {label}")
            evaluator.evaluate(model, display_fitness=True)
        
    def display_all_models_predictions(self):
        """Display predictions for all models."""
        for model_list, name_list, evaluator,label in zip(self.models, self.model_names, self.test_evaluators, self.group_labels):
            for model, model_name in zip(model_list, name_list):
                print(f"real and predicted values for {model_name}) in {label}")
                evaluator.evaluate(model, display_fitness=True)

                
    def plot_density_distribution_for_groups(self, inference_manager):
        """Plot density distribution for each group."""
        for og_files, model, name, label in zip(self.inference_data, self.best_models, self.best_model_names, self.group_labels):
            title = f'Predicted density distribution by {name} in {label} regions'
            inference_manager.density_distribution_in_img_list(model, og_files, title)

    def plot_density_distribution_for_images(self, inference_manager):
        """Plot density distribution for individual images."""
        for og_files, model in zip(self.inference_data, self.best_models):
            for image_path in og_files:
                inference_manager.density_distribution_in_img(model, image_path)

    def calculate_t_bias(self, est, real):
        """Calculate t-test bias between estimated and real values."""
        residuals = np.array(est) - np.array(real)
        # One-sample t-test
        _, p_value = stats.ttest_1samp(residuals, popmean=0)
        return p_value
            
                
    


