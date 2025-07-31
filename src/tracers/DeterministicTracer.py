# This module provides a deterministic tracer that uses a predefined pipeline for tracing.

from .BaseTracer import BaseTracer
class DeterministicTracer(BaseTracer):
    """Deterministic tracer that applies a predefined pipeline to generate tracing."""
    
    def __init__(self, ns_pipeline, *args, **kwargs):
        """Initialize deterministic tracer with a predefined pipeline."""
        self.img_input_size = None
        super().__init__(*args, **kwargs)
        self.ns_pipeline = ns_pipeline
        

    def make_tracing(self, image): 
        """Apply the predefined pipeline to generate tracing."""
        return self.ns_pipeline(image) # .astype(bool)
