# This module provides functions for calculating trace properties including density, axon count, and mean axon length.

import numpy as np
from scipy.ndimage import label

def get_trace_density(trace, pixel_length = 1.0):
    """Calculate density from a trace mask by averaging pixel values."""
    check_trace_input(trace)
    if np.isnan(trace).all(): return np.nan
    else: return np.nanmean(trace) / pixel_length**2

def get_mean_axon_length(trace, pixel_length = 1.0):
    """Calculate mean axon length by dividing density by axon count."""
    count = get_axon_count(trace)
    dens = get_trace_density(trace, pixel_length)

    if count == 0:
        return 0.0
    else: 
        mal = dens / count
        return mal

from .trace_manips import thicken_trace
def get_axon_count(trace):
    """Count the number of connected components (axons) in the trace mask."""
    check_trace_input(trace)
    trace = thicken_trace(trace, 2)
    _, num = label(trace > 0)
    return num

def check_trace_input(trace):
    """Validate trace input format and data type."""
    assert isinstance(trace, np.ndarray), "trace must be a numpy array"
    assert trace.ndim == 2, "trace must be a 2D array"
    assert trace.dtype == bool or trace.dtype == np.bool_, "trace must be of boolean dtype"

