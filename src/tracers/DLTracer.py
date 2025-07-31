# This module provides a deep learning tracer that uses neural networks for image tracing with splitting capabilities.

import torch
import numpy as np

from ..NNs.Unet import UNetModel
from ..utils.Splitter import Splitter

from ..imgproc.Transforms import RemoveChunks, ConnectSegments, Skeletonize
from ..utils.trace_manips import thicken_trace

from .BaseTracer import BaseTracer
class DLTracer(BaseTracer):
    """Deep learning tracer that uses neural networks for image tracing with splitting and postprocessing."""
    
    def __init__(self, model_path, model_type, img_input_size, *args, **kwargs):
        """Initialize deep learning tracer with trained model and splitting configuration."""
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"

        self.trained_model = model_type(in_channels=1, num_classes=1).to(self.device)
        self.trained_model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device), weights_only=False))
        
        self.img_input_size = img_input_size
        
        mask_function = self.make_small_tracing
        batch_mask_function = self.trace_batch
        join_func = lambda x, y: x | y
        self.splitter = Splitter(img_input_size, mask_function=mask_function, join_function=join_func, 
                                 mask_dtype=bool, batch_size=32, batch_mask_function=batch_mask_function,
                                 invalid_fill_value=False)
        self.img_input_size = img_input_size
        
        
        
    def get_split_mask(self, sub_images):
        """Apply neural network to batch of sub-images to generate masks."""
        # assumes shape (B, H, W, C)
        # swaps channels into 2nd dimension, more standard in torch models
        sub_images = sub_images.transpose(0, 3, 1, 2)
        images_tensor = torch.from_numpy(sub_images).float().to(self.device)
        with torch.no_grad():
            a = self.trained_model(images_tensor) > 0.5
            
        # returns shape (B, H, W, 1)
        output = a.cpu().numpy().transpose(0, 2, 3, 1)
        
        return output
    
    def make_small_tracing(self, image):
        """Generate tracing for small images using neural network."""
        image = np.expand_dims(image.copy(), axis=0)
        tracing = self.get_split_mask(image).squeeze(axis=0) # returns shape (H, W, 1)
        return tracing 
    
    def make_tracing(self, image): 
        """Generate tracing for images of any size using splitting or direct processing."""
        given_size_x = image.shape[0]
        given_size_y = image.shape[1]
        
        if given_size_x == self.img_input_size and given_size_y == self.img_input_size:
            tracing = self.make_small_tracing(image)
        
        elif given_size_x >= self.img_input_size and given_size_y >= self.img_input_size:
            tracing = self.splitter.process(image)
            # tracing = self.postprocessing(tracing)

        assert len(tracing.shape) == 3, "Invalid tracing in DLTracer: output tracing must be 3 dimensions"

        return tracing

    def postprocessing(self, trace):
        """Apply postprocessing steps including skeletonization and chunk removal."""
        skel = Skeletonize()
        trace = skel(trace)

        remove_chunks = RemoveChunks(min_area=7)
        trace = remove_chunks(trace.squeeze())
        trace = thicken_trace(trace, 5)[:, :, np.newaxis]

        connect_segs = ConnectSegments(kernel_size=2)
        trace = connect_segs(trace)

        return skel(trace)[:, :, np.newaxis]
 
    def trace_batch(self, images):
        """Process batch of images using neural network."""
        # In this project, image shapes are generally in the form of (H (height), W (width), C (channels))#
        # Therefore, inputs to get_split_mask are (B, H, W, C)
        assert len(images.shape) == 4, "Invalid Batch: input image batch must have only 4 dimensions"
        assert images.shape[1] >= self.img_input_size, "Invalid Image: Dimension 1 of image smaller than input size"
        assert images.shape[2] >= self.img_input_size, "Invalid Image: Dimension 2 of image smaller than input size"
        return self.get_split_mask(images) 