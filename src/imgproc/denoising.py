def remove_by_std_treshold(grayscale_image, n_stds):
    intensity = grayscale_image.mean()
    intensity_std = grayscale_image.std()
    mask = grayscale_image < (intensity + n_stds * intensity_std)
    grayscale_image[mask] = 0
    return grayscale_image

