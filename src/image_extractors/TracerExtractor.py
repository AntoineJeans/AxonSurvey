# This module provides a trace extractor that uses tracers to generate features from images.

import numpy as np

from ..imgproc.Transforms import Skeletonize

from .BaseFeatureExtractor import BaseFeatureExtractor

import time 

class TraceExtractor(BaseFeatureExtractor):
    """Extract features from images using tracers and ground truth functions."""
    
    def __init__(self, tracer, ground_truth_functions, skeletonize_trace=True, feature_names=None, *args, **kwargs):
        """Initialize trace extractor with tracer and ground truth functions."""
        super().__init__(*args, **kwargs)

        self.tracer = tracer
        self.ground_truth_functions = ground_truth_functions
        self.skeletonize_trace = skeletonize_trace
        
        self.n_features = len(ground_truth_functions)
        
        self.extractor_name = self.tracer.tracer_name + "Extractor"
        if feature_names: self.feature_names = feature_names
        else: self.feature_names = [func.__name__ for func in ground_truth_functions]

        self.feature_names = [f"{ft} for {self.tracer.tracer_name} trace" for ft in self.feature_names]
        
        if skeletonize_trace: self.skel = Skeletonize()
        
    def preprocessing(self, image, mask, cache_file):
        """Apply tracer-based preprocessing including tracing and optional skeletonization."""
        # add more pre-trace processing steps if needed

        trace = self.tracer.trace(image, mask, cache_file)


        if self.skeletonize_trace: 
            trace = self.skel(trace)
            
        # add more post-trace processing steps if needed
        trace = super().preprocessing(trace.astype(np.float32), mask, cache_file)

        return trace
    
    def get_features(self, image):
        """Extract features from traced image using ground truth functions."""
        trace = (image > 0 & ~np.isnan(image)).squeeze()
        
        features = np.array([func(trace) for func in self.ground_truth_functions])

        trace_features = np.full(tuple(list(image.shape)[:2] + [self.n_features]), features.copy())
        return trace_features
