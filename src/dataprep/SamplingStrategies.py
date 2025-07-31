# Work in progress

import numpy as np
from abc import abstractmethod
class SamplingStrategy:
    """
    Base class for sampling strategies.
    Subclasses should implement the sample_indices method.
    """
    def __init__(self, stratify_regions, stratify_group, groups=None):

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

        self.groups = groups
        self.stratify_regions = stratify_regions
        self.stratify_group = stratify_group

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

class SRS(SamplingStrategy):
    """
    Simple Random Sampling (SRS): randomly selects sample_size unique indices.
    """
    def __init__(stratify_regions=True, *args, **kwargs):
        """Initialize SRS sampling strategy."""
        super().__init__(stratify_regions, stratify_group=True, *args, **kwargs)

    def sample_indices(self, n):
        """Sample indices for each group and region."""
        n_per_group = int(n / self.get_group_count())
        selected = []
        for group, region_counts in zip(self.population, self.region_count):

            if self.stratify_regions: sampling_proportions = np.full(len(group,), 1/len(group))
            else : sampling_proportions = (np.array(region_counts) / np.sum(region_counts)) # proportionnal to img size

            sampling_quantities = (n_per_group * sampling_proportions).astype(int)
            selected_indices = [self.get_n_random_indices_in_list(quantity, region) for region, quantity in zip(group, sampling_quantities)]
            selected.append(selected_indices)
        return selected


    def get_n_random_indices_in_list(self, n, elements):
        """Return n unique random indices from the elements list."""
        if n > len(elements):
            raise ValueError("n cannot be greater than the number of elements.")
        rng = np.random.default_rng()
        a = rng.choice(len(elements), size=n, replace=False)
        return a

# Example: add more strategies as needed
class StratifiedSamplingStrategy(SamplingStrategy):
    """
    Stratified Sampling: samples from each stratum according to provided sizes.
    """
    def sample_indices(self, strata_sizes, sample_sizes_per_stratum, seed=None):
        """Sample indices for each stratum."""
        rng = np.random.default_rng(seed)
        indices = []
        for stratum_size, n in zip(strata_sizes, sample_sizes_per_stratum):
            idx = rng.choice(stratum_size, size=n, replace=False)
            indices.append(idx)
        return indices
