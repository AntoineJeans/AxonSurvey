from src.utils.traceProps import get_axon_count, get_mean_axon_length, get_trace_density
from src.experiments.InferencePipeline import InferencePipeline

from config import Config



def run_quantification(rat_groups, experiment_name=None, experimenter_name=None):

    ## Init params
    img_input_size = 128

    ## These can be integrated into the gui as user selection
    group_labels = [group.group_name for group in rat_groups]
    ground_truth_functions = [get_trace_density]
    propery_names = ["Axon innervation density"]
    # --- runs a standard flow for creating, "training/callibrating" and testing models, then generating quantification
    pipe = InferencePipeline(   rat_groups, img_input_size,
                            og_path = Config.DATA_DIR,
                            train_path=Config.TRAINING_DIR,
                            test_path=Config.TESTING_DIR, 
                            group_labels = group_labels, 
                            ground_truth_functions=ground_truth_functions, 
                            propery_names=propery_names,
                            debug_mode=False
                            )
    print("Loading data and training models ...")
    pipe.load_data()
    pipe.make_models()
    pipe.train_models()

    print("Evaluating models on test set ...")
    pipe.evaluate_models(display_model_performances=True, n_bootstraps_trials=50)
    pipe.select_best_models()
    pipe.calculate_model_uncertainty(n_bootstrap=500)

    print("Predicting on inference data ...")
    pipe.infer_mean_region_density_in_groups()

    print("Almost done! Saving experiment data ...")
    # Set experiment metadata before saving
    pipe.experiment_name = experiment_name
    pipe.experimenter_name = experimenter_name
    pipe.save_experiment_data()
    # ----- #

    # All data of interest is saved as a regular experiment so this doesn't have to return data

    return pipe.experiment_id