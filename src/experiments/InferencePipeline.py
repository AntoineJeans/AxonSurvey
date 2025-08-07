# This module provides a complete inference pipeline for training models, evaluating performance, and making predictions with confidence intervals.

import numpy as np
import copy
from scipy import stats

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
    
    def __init__(self, level, groups, img_input_size, 
                 og_path, train_path, test_path, val_path=None, 
                 rmse_confidence=0.95, inference_confidence = 0.95, group_labels = None,
                 ground_truth_functions=[get_trace_density], propery_names=["Axon innervation density"],
                 debug_mode=False):
        """Initialize inference pipeline with experiment parameters and configuration."""
        plt.style.use('classic')

        # Options for trace_prop_of_interest are : "Axon Density", "Axon Mean Length", "Axon Count"
        prop_functions = {
            "Axon Density" : get_trace_density,
            "Axon Mean Length" : get_mean_axon_length,
            "Axon Count" : get_axon_count,
        }

        # STEP 1 - SET EXPERIMENT PARAMETERS
        self.level = level
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


    # STEP 2 - LOAD EXPERIMENT DATA
    # loads image paths to use from the experiment datasets and saves as attribute as trainers and evaluators for models
    def load_data(self):
        """Load training, validation, test, and inference data for the experiment."""
        loader = ExperimentLoader(self.level, self.groups, self.train_path, self.test_path, self.og_path, val_path=self.val_path)
        
        # Load training data
        training_data_for_groups = loader.get_experiment_train_data()
        
        self.trainers = [Trainer(train_paths, ground_truth_functions=self.ground_truth_functions) for train_paths in training_data_for_groups]

        # Load validation/testing data
        test_data = loader.get_experiment_test_data()
        self.test_evaluators = [
            RegressionEvaluator(image_paths=paths, tracings_cache_folder=None, 
                                estimated_names=self.propery_names, ground_truth_functions=self.ground_truth_functions) 
                                for paths in test_data
        ]

        
        if self.val_path is not None:
            val_data = loader.get_experiment_val_data()
            self.val_evaluators = [RegressionEvaluator(image_paths=paths, estimated_name=self.trace_prop_of_interest, ground_truth_function=self.prop_function) for paths in val_data]
            
        # Load inference data (large images whose density we want to estimate)
        self.inference_data = loader.get_inference_data(channel="th")

        
        
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
                    save_path = "./figures/model_performances"
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
        

    # STEP 5 - DEDUCE RMSE AND BIAS ON MODELS 
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
            display_model_bounds(expec, lower, upper, labels, title=f"RMSE for {prop_name} by best model in each region type, confidence={self.rmse_confidence}", metric_name="RMSE")
        
            if use_upper_bound:
                error_points.append(upper)
            else:
                error_points.append(expec)

        error_points = np.array(error_points).transpose(1, 0) # bring the model instances at the first dimension
        rmses = [evaluator.combine_rmses(errors) for errors, evaluator in zip(error_points, self.test_evaluators)]
        return rmses

    def get_best_models_biases(self, n_bootstraps_trials, use_upper_bound=True):
        """Get bias for best models using bootstrap trials."""
        bootstrap_func = lambda evaluator, model : evaluator.bootstrap_bias(model, n_bootstraps_trials) 
        expec, lower, upper = self.bootstrap_model_list(self.best_models, self.test_evaluators, bootstrap_func)            
        labels = [f" {name} on {label}" for name,label in zip(self.best_model_names, self.group_labels) ]
        display_model_bounds(expec, lower, upper, labels, title=f"BIAS by best model in each region type, confidence={self.rmse_confidence}", metric_name="BIAS")
        return upper if use_upper_bound else expec
    

    
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

        # Now display
        if use_fixed_point_predictions:
            title = f"Predicted densities in sampled regions"
            display_inference_points(group_data, labels=self.group_labels, title=title, 
                                        save_path="./figures/prediction_intervals/")
            
        else:
            estimator_names = ["not sure"]
            title = f"Expected axon density in sampled regions, confidence={self.inference_confidence}"
            display_inference_bounds(group_data, labels=self.group_labels, 
                                        title=title, predictor_names = estimator_names, 
                                        save_path="./figures/prediction_intervals/")

        # self.plot_density_distribution_for_groups(inf_man)
        # if self.debug_mode:
        #     self.plot_density_distribution_for_images(inf_man)


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
        for og_files, model, name, laWbel in zip(self.inference_data, self.best_models, self.best_model_names, self.group_labels):
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
            
                
    


