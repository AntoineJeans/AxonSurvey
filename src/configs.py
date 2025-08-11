"""
This file serves as a central location for storing useful variables, constants, and configuration settings
that may be needed throughout the axon_density package or by external scripts. By placing commonly used
variables here, they can be easily imported and reused in other modules or scripts without duplication.

Typical use cases for this file include:
- Defining global constants (e.g., thresholds, default parameters)
- Storing configuration options that control package behavior
- Providing variables that need to be accessed from outside the package

To use variables defined in this file, simply import them as needed:
    from axon_density.survey_estimation.configs import VARIABLE_NAME
"""

from .image_extractors.TracerExtractor import TraceExtractor
from .tracers.DLTracer import DLTracer
from .NNs.Unet import UNetModel
import copy
import os


use_DL_models = False
use_ensemble_extractor_models = False

# img_input_size has to be 128 for those models
img_input_size = 128

trained_models_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../trained_models/basic_trainings")
)

DL_tracer_dicts = [
    # {
    #     "name": "DLTr_Simple",
    #     "category": "standard_DL",
    #     "path": os.path.join(trained_models_dir, "2025-06-06_SimpleDataset.pth"),
    #     "model_type": UNetModel,
    # },
    # {
    #     "name": "DLTr_Tiled",
    #     "category": "standard_DL",
    #     "path": os.path.join(trained_models_dir, "2025-06-06_TiledDataset.pth"),
    #     "model_type": UNetModel,
    # },
    {
        "name": "DLTr_Aug",
        "category": "standard_DL",
        "path": os.path.join(trained_models_dir, "2025-06-06_AugDataset.pth"),
        "model_type": UNetModel,
    },
    {
        "name": "DLTr_GIGAAug",
        "category": "standard_DL",
        "path": os.path.join(trained_models_dir, "2025-06-06_GIGAAugmented.pth"),
        "model_type": UNetModel,
    },

    {
        "name": "DLTr_topo1",
        "category": "topology_aware_models",
        "path": os.path.join(os.path.dirname(__file__), "../trained_models/topology_aware_models", "2025-07-21_AugDataset_thick_axons_connected_loss_0.4_0.0.pth"),
        "model_type": UNetModel,
    },
    
]

if use_DL_models:
    dl_tracers = [DLTracer(tr["path"], tr["model_type"], img_input_size, tracer_name=tr["name"]) for tr in DL_tracer_dicts]

from .tracers.DeterministicTracer import DeterministicTracer
from .imgproc.pipeline_instances import edge_skeleton_pipeline, xTreme_skeleton_pipeline

other_tracers_extractor_names = ["NS_edge", "NS_Xedge"]
other_tracers = [
    DeterministicTracer(edge_skeleton_pipeline, tracer_name="NS_edge"), 
    DeterministicTracer(xTreme_skeleton_pipeline, tracer_name="NS_Xedge"), 
]

from src.utils.traceProps import get_axon_count, get_mean_axon_length, get_trace_density
ground_truth_functions = [get_axon_count, get_mean_axon_length, get_trace_density]
feature_names = ["Fibre Count", "Mean Fibre Length", "Foreground Density"]


from .image_extractors.BaselineMeanExtractors import PopulationMeanExtractor, ImageMeanExtractor
from .image_extractors.ThresholdDensityExtractor import ThresholdDensityExtractor
from .image_extractors.OtsuExtractors import OtsuExtractor
from src.image_extractors.PropertyModel import PropertyModel

from sklearn.linear_model import LinearRegression

def make_models(n_groups):

    # Here are all feature extractors defined in this project, they can be used alone or in any combinations through the PropertyModel!
    baseline_extractors = [
        PopulationMeanExtractor(),
        ImageMeanExtractor(feature_extraction_tile_size=64)
    ]
    baseline_extractor_names = ["Population_mean", "Image_mean"]
    baseline_extractor_types = ["baseline", "baseline"]
    
    treshold_extractors = [
        ThresholdDensityExtractor(feature_extraction_tile_size=64, local=True),
        ThresholdDensityExtractor(feature_extraction_tile_size=64, local=False),
        OtsuExtractor(feature_extraction_tile_size=64)
    ]
    treshold_extractor_names = ["Local_threshold", "Global_threshold", "Otsu_threshold"]
    treshold_extractor_types = ["threshold", "threshold", "threshold"]
    
    if use_DL_models:
        dl_extractors = [
            TraceExtractor(tracer, ground_truth_functions=ground_truth_functions, 
                            feature_names=feature_names, feature_extraction_tile_size=64)   for tracer in dl_tracers
        ]
        dl_extractor_names = [tr["name"] for tr in DL_tracer_dicts]
        dl_extractor_types = ["DL" for _ in DL_tracer_dicts]

    other_tracers_extractors = [
        TraceExtractor(tracer, ground_truth_functions=ground_truth_functions, 
                        feature_names=feature_names, feature_extraction_tile_size=64)   for tracer in other_tracers
    ]
    other_tracers_extractor_names = ["NS_edge", "NS_Xedge"]
    other_tracers_extractor_types = ["other", "other"]


    base_cache_path = r".\\data\\project_tracing_and_features\\"
    
    # First, for every extractor, we create a prop model for only it
    all_extractors = baseline_extractors + treshold_extractors   #+ other_tracers_extractors
    all_names = baseline_extractor_names + treshold_extractor_names  #+ other_tracers_extractor_names
    all_types = baseline_extractor_types + treshold_extractor_types #+ other_tracers_extractor_types
    
    if use_DL_models:
        all_extractors += dl_extractors
        all_names += dl_extractor_names
        all_types += dl_extractor_types

    # prop_model_names = [f"only_{extr}" for extr in all_names]
    prop_model_names = [f"{extr}" for extr in all_names]
    prop_models = [ PropertyModel(extractors=[extr], model=LinearRegression(), 
                                  cache_folder=os.path.join(base_cache_path, name)) for extr,name in zip(all_extractors, prop_model_names)]

    # Then we try some combinations!
    if use_ensemble_extractor_models:
        many_simple = "Multiple_simple"
        prop_model_names.append(many_simple)
        prop_models.append(PropertyModel(extractors=baseline_extractors + treshold_extractors, model=LinearRegression(), 
                                    cache_folder=os.path.join(base_cache_path, many_simple)))
        all_types.append("threshold")

    if use_DL_models and use_ensemble_extractor_models:
        many_tracer_name = "Multiple_tracers"
        prop_model_names.append(many_tracer_name)
        prop_models.append(PropertyModel(extractors=dl_extractors, model=LinearRegression(), 
                                    cache_folder=os.path.join(base_cache_path, many_tracer_name)))
        all_types.append("DL")
        
        mixed_name = "Mixed_extractors"
        prop_model_names.append(mixed_name)
        prop_models.append(PropertyModel(extractors=dl_extractors+baseline_extractors + treshold_extractors, model=LinearRegression(), 
                                    cache_folder=os.path.join(base_cache_path, mixed_name)))
        all_types.append("DL")
    
    return [copy.deepcopy(prop_models[:]) for _ in range(n_groups)], \
            [copy.deepcopy(prop_model_names[:]) for _ in range(n_groups)], \
            [copy.deepcopy(all_types[:]) for _ in range(n_groups)]

    # return [copy.deepcopy(prop_models[:1]) for _ in range(n_groups)], \
    #         [copy.deepcopy(prop_model_names[:1]) for _ in range(n_groups)], \
    #         [copy.deepcopy(all_types[:1]) for _ in range(n_groups)]