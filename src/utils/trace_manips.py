# This module provides functions for manipulating trace masks including thickening operations.

import numpy as np
from copy import deepcopy
from scipy.ndimage import binary_dilation

def thicken_trace(trace, thickness):
    """Thicken trace fibers using binary dilation with specified thickness."""
    # inputs boolean np array and outputs one of the same shape with more thick fibres (positive class)
    structure = np.ones((thickness,thickness), dtype=bool)
    trace = binary_dilation(trace, structure=structure)
    return trace
