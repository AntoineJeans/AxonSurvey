# Regression evaluator for analyzing model performance with correlation and error metrics.

import os 
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error

from ..utils.traceProps import get_trace_density
from .Evaluator import Evaluator
class RegressionEvaluator(Evaluator):
    """Evaluates regression models using correlation coefficients, RMSE, and bias metrics."""
    
    def __init__(self, estimated_names = ["Axon Density"], ground_truth_functions = [get_trace_density], *args, **kwargs):
        self.ground_truth_functions = ground_truth_functions
        self.estimated_names = estimated_names
        super().__init__(*args, **kwargs)

        self.property_scales = self.compute_property_scales()
        

    def bootstrap_pearson(self, model, n_samples, display_hist=False):
        """Performs bootstrap evaluation using Pearson correlation coefficient."""
        return self._bootstrap(model, n_samples, score_function=self.pearson, param_name="Pearson coefficient",  display_hist=display_hist)

    def bootstrap_rmse(self, model, n_samples, display_hist=False):
        """Performs bootstrap evaluation using RMSE metric."""
        return self._bootstrap(model, n_samples, score_function=self.rmse, param_name="RMSE", display_hist=display_hist)
    
    def bootstrap_bias(self, model, n_samples, display_hist=False):
        """Performs bootstrap evaluation using bias metric."""
        return self._bootstrap(model, n_samples, score_function=self.bias, param_name="BIAS", display_hist=display_hist)
    
    def pearson(self, ests, reals):
        """Calculates Pearson correlation coefficient between estimates and real values."""
        r, _ = pearsonr(ests, reals)
        return r

    def rmse(self, ests, reals):
        """Calculates root mean squared error between estimates and real values."""
        return root_mean_squared_error(ests, reals)
    
    def bias(self, ests, reals):
        """Calculates absolute bias between estimates and real values."""
        return abs(np.mean(np.array(ests) - np.array(reals)))
    
    def compute_property_scales(self):
        """Computes scaling factors for property normalization."""
        scales = []
        for prop in self.ground_truth_properties:
            np.mean(prop)
            scales.append(prop)
        return scales


    def get_weighted_rmse(self, predicted_properties, real_properties):
        """Calculates weighted RMSE using property-specific scaling factors."""
        rmses = []
        for scale, pred_prop, real_prop in zip(self.property_scales, predicted_properties, real_properties):
            norm_rmse = scale * self.rmse(pred_prop, real_prop)
            rmses.append(norm_rmse)
        return np.mean(rmses)
    
    def combine_rmses(self, rmses_for_each_property):
        """Combines RMSE values across properties using scaling factors."""
        scaled_rmses = [a * b for a,b in zip(rmses_for_each_property, self.property_scales)]
        return np.mean(scaled_rmses)

    

    def get_bounds(self, bootstrapped_scores, confidence = 0.95):
        """Calculates confidence interval bounds from bootstrapped scores."""
        lower_bound = sorted(bootstrapped_scores)[int((1-confidence) * len(bootstrapped_scores))]
        upper_bound = sorted(bootstrapped_scores)[-int((1-confidence) * len(bootstrapped_scores))]
        return lower_bound, upper_bound

    

    def plot_fitness(self, ests, reals):
        """Plots scatter plots comparing estimated vs real values with correlation statistics."""
        plt.style.use('seaborn-v0_8-white')
        assert len(ests) == len(reals), 'estimated and predicted value lists not same length'
        for est, real, estimated_name in zip(ests.transpose(1,0), reals.transpose(1,0), self.estimated_names):
            # Compute Pearson correlation coefficient
            r, p_value = pearsonr(est, real)
            # Create scatter plot
            plt.figure(figsize=(6, 4))
            plt.scatter(est, real, alpha=0.7, edgecolors='k')
            plt.xlabel(f"Estimated {estimated_name}")
            plt.ylabel(f"Real {estimated_name}")
            title = f"Expert vs Automatic {estimated_name}: R={r:.2f} ; p-value={p_value:.2e}"
            plt.title(title, pad=20)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        plt.style.use('classic')
        
    

    def real_and_predicted_distribution_in_dataset(self, model, name, label):
        """Plots distribution histograms for real and predicted values across the dataset."""
        plt.style.use('seaborn-v0_8-white')
        ests, reals = self.evaluate(model)

        for est, real, estimated_name in zip(ests.transpose(1,0), reals.transpose(1,0), self.estimated_names):
            bin_limit = max(max(est), max(real))
            bin_width = bin_limit / 20
            bins = np.arange(0, bin_limit, bin_width)
            
            plt.hist(est, bins=bins, edgecolor='black')
            plt.xlabel(f'{estimated_name}')
            plt.ylabel('Square count')
            plt.title(f'Predicted {estimated_name} distribution by {name} in {label} test images')
            plt.show()
            
            plt.hist(real, bins=bins, edgecolor='black')
            plt.xlabel(f'{estimated_name}')
            plt.ylabel('Square count')
            plt.title(f'Real {estimated_name} distribution in {label} test images')
            plt.show()
        plt.style.use('classic')
    
