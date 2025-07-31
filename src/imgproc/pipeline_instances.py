    
from .tracing import canny_edge_detection, remove_small_chunks, connect_line_segments, skeletonize_edges
# recieves a grayscale image of some axons, detects axons and returns the skeleton of the detected axons
def first_axon_segmentation_pipeline(grayscale_img, display=False):
    if display: display_iter=5
    else: display_iter=None
    edges = canny_edge_detection(grayscale_img, display=display)
    edges = remove_small_chunks(edges, display=display, min_area=20)
    return connect_line_segments(edges, display=display)

def first_axon_skeleton_pipeline(grayscale_img, display=False):
    edges = first_axon_segmentation_pipeline(grayscale_img, display=display)
    return skeletonize_edges(edges, display=display)

from .Pipeline import GrayscalePipeline, BinaryPipeline, GrayscaleToBinaryPipeline
from .Transforms import *

gray_transforms = [
    # LocalNormalization(kernel_size=boots_with_the_fur),
]
gp = GrayscalePipeline(gray_transforms)  
gray_to_bin = Canny(bottom=80, top=160)
 



bin_transforms = [
    ConnectSegments(kernel_size=2),
    RemoveChunks(min_area=20),
    ConnectSegments(kernel_size=20),
    Skeletonize()
] 
bp = BinaryPipeline(bin_transforms)
edge_skeleton_pipeline = GrayscaleToBinaryPipeline(gs_pipeline=gp, bin_pipeline=bp, bin_transform=gray_to_bin)



bin_transforms = [
    ConnectSegments(kernel_size=10),
    RemoveChunks(min_area=10),
    ConnectSegments(kernel_size=30),
    RemoveChunks(min_area=30),
    Skeletonize()
] 
bp = BinaryPipeline(bin_transforms)
xTreme_skeleton_pipeline = GrayscaleToBinaryPipeline(gs_pipeline=gp, bin_pipeline=bp, bin_transform=gray_to_bin)