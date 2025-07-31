# Module for evaluating tracer algorithms against human-annotated ground truth traces.

import numpy as np
from scipy.ndimage import binary_dilation

from .Evaluator import Evaluator
class TracerEval(Evaluator):
    """Evaluates tracer algorithm performance against human-annotated ground truth."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def eval(self, tracer):
        """Evaluates a tracer algorithm on the current image."""
        predicted_trace = tracer.trace(self.image)
        return self.eval_trace(predicted_trace, self.mask)
    
    def eval_trace(self, predicted_trace, human_trace):
        """Calculates overlap ratio between predicted and human-annotated traces."""
        radius = 3
        structure = np.ones((2*radius + 1, 2*radius + 1), dtype=np.uint8)
        wide_human_trace = binary_dilation(human_trace, structure=structure).astype(np.uint8)
        return np.sum(predicted_trace) / (np.sum(predicted_trace & wide_human_trace)) 
        
    def eval_on_n_splits(self, tracer, n_splits):
        """Evaluates tracer algorithm on image splits and returns predicted traces."""
        images, masks = self.split(n_splits)
        est_traces = []
        for im in images:
            est_traces.append(tracer.trace(im))
            
        return est_traces
        
        
        