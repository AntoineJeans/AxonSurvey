"""
Image Extractors
================

This package contains a variety of extractors and a single PropertyModel. The PropertyModel functions at a higher level 
and can have any amount of extractors to help with the final prediction. 
Extractors create features which are thereafter regressed to properties using annotated samples.

For extractors, these objects can take as input an image of any size (and shape through a mask) and output a prediction for a 'property'.
Since these models must perform well on little training data (sets of around 100 images).

'properties' can be any continuous variable that can be deduced by a person or any labelling technique. 
Currently, properties are defined by a function that inputs a ground truth binary mask and outputs a floating point value. 

Examples:
    - Axon Density - from a segmentation mask where distinctly long thin fibres are traced by a person - Function sums 1-pixel divided by all pixels in image
    - Counts of cells - from a mask where each disconnected point in the mask is a different cell in the original image - Function counts disconnected points

When fitting, ImageExtractors take as input the original image and the human segmentation. Properties are calculated from the segmentation, 
then features are extracted from the original image. Then the feature_to_prop_model finds the best fit to map features to properties.

Some important extractors in this package:

PopulationMeanExtractor and ImageMeanExtractor:
    Weak feature extractors that simply extract one relatively uninformative feature from the original image.
    They can be used as baselines to give an intuitive upperbound on model error. PopulationMeanExtractor uses the mean property value across all training data, while ImageMeanExtractor uses the mean pixel value of each individual image.

OtsuExtractor:
    A flexible model that is constructed from a processing pipeline that can't be fitted. The pipeline is a sequence of image processing steps 
    (for instance: thresholding, edge detection, skeletonization) that should generally extract a single more informative feature from the image.
    (Could be improved by accepting a list of pipelines that result in a list of 2-10 important features)

ThresholdDensityExtractor:
    A model that extracts density features using threshold-based methods with optional local/global processing. It can optimize threshold parameters during fitting for best correlation with properties.

TraceExtractor:
    A model that directly takes as input an already fitted tracer (see tracers) and uses its output directly (without fitting the tracer) to extract features using the property function.
    The reasoning for using a prefitted tracing is 2-fold:
    1 - The best tracing models will generally be based on neural networks or algorithms that are expensive to train. Using saved pretrained models makes the process simpler and faster.
    2 - Training algorithms can be trained from less specific data (from other projects or augmented images) but the feature_to_prop_model requires fewer and high quality datapoints.
"""

