import numpy as np

def estimate_axon_density(skeleton, pixel_to_micron=0.1, display=True):
    neuron_length_in_image = np.sum(skeleton) * pixel_to_micron
    image_width = 400 * pixel_to_micron
    image_length = 400 * pixel_to_micron
    density = neuron_length_in_image/(image_width * image_length)

    if display: print(f"Density estimation = {density} axon micron/micron^2")
    
    return density