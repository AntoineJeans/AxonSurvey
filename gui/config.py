from pathlib import Path

class Config:
    """Configuration constants for the application"""
    # Directory paths
    DATA_DIR = Path("../data/project_scans")
    CACHE_DIR = Path("../data/project_tracing_and_features/")

    USED_SEGMENTATION_MODEL_NAME = "DLTr_GIGAAug" 
    USED_SEGMENTATION_MODEL_PATH = Path("../trained_models/50ep_aug_128_lr-3.pth")

    STATIC_DIR = Path("static")
    TEMPLATES_DIR = Path("templates")
    
    MAX_DISPLAY_SIZE = (800, 600)
    DEFAULT_COLORMAP = "viridis"


    # Not sure if the following are useful: 
    IMAGE_PATTERNS = ["image.tif", "image.tiff", "*.tif", "*.tiff"]
    SEGMENTATION_SUFFIX = "_seg.TIF"
    FEATURE_MAP_SUFFIX = "_features.TIF"
    METADATA_FILE = "meta.json"

    # Available regions for comparison
    PREDEFINED_REGIONS = ["cortex", "hippocampus", "striatum", "thalamus", "cerebellum"]
    ###
    
    