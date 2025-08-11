from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageEnhance
from pathlib import Path

from config import Config

import logging
import glob
import shutil
import json
logger = logging.getLogger(__name__)


def convert_tif_to_jpg_and_save(tif_path: Path, output_filename: str, lumin_scale=5.0, use_raw_path=False) -> Optional[str]:
    """
    Convert a TIF file to JPG and save it in static/images directory
    Resizes images to fit within 1080p (1920x1080) while maintaining aspect ratio
   
    Args:
        tif_path: Path to the TIF file
        output_filename: Name for the output JPG file
       
    Returns:
        Filename of the saved JPG file or None if conversion failed
    """
    try:
        if not tif_path.exists():
            logger.warning(f"TIF file not found: {tif_path}")
            return None
           
        # Open the TIF image
        with Image.open(tif_path) as img:
            # Convert to RGB if necessary (TIF might be grayscale or other format)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            elif img.mode == 'L':
                img = img.convert('RGB')
            
            # Resize to fit within 1080p bounds while maintaining aspect ratio
            max_width = 1920
            max_height = 1080
            
            # Calculate current dimensions
            width, height = img.size
            
            # Only resize if image exceeds 1080p bounds
            if width > max_width or height > max_height:
                # Calculate scaling ratios
                width_ratio = max_width / width
                height_ratio = max_height / height
                
                # Use the smaller ratio to ensure both dimensions fit
                scale_ratio = min(width_ratio, height_ratio)
                
                # Calculate new dimensions
                new_width = int(width * scale_ratio)
                new_height = int(height * scale_ratio)
                
                # Resize the image using high-quality resampling
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)


            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(lumin_scale)
           
            if not use_raw_path:
                # Create static/images directory if it doesn't exist
                static_images_dir = Path("static/images")
                static_images_dir.mkdir(parents=True, exist_ok=True)
            
                # Save as JPG
                image_path = static_images_dir / output_filename
                img.save(image_path, 'JPEG', quality=85)
            else:
                img.save(output_filename, 'JPEG', quality=85)
           
            # Return just the filename for url_for() usage
            print(output_filename)
            return output_filename
           
    except Exception as e:
        logger.error(f"Error converting TIF to JPG: {e}")
        return None


def validate_safe_path(base_path: Path, *path_components: str) -> Optional[Path]:
    """
    Validate that a path is safe and within the base directory
    
    Args:
        base_path: The base directory that paths must be within
        *path_components: Path components to join
        
    Returns:
        Path if safe, None if unsafe
    """
    try:
        # Join the path components
        full_path = base_path.joinpath(*path_components)
        
        # Resolve to absolute path to prevent path traversal
        resolved_path = full_path.resolve()
        base_resolved = base_path.resolve()
        
        # Check if the resolved path is within the base directory
        if not str(resolved_path).startswith(str(base_resolved)):
            logger.warning(f"Path traversal attempt detected: {full_path}")
            return None
            
        return full_path
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Invalid path components: {path_components}, error: {e}")
        return None

def get_region_cached_path(rat_id: str, slice_name: str, region: str) -> Optional[Path]:
    """Get cached path with validation"""
    return validate_safe_path(Config.CACHE_DIR, Config.USED_SEGMENTATION_MODEL_NAME, rat_id, slice_name, region)

def get_cached_segmentation_path(rat_id: str, slice_name: str, region: str) -> Optional[Path]:
    """Get the path where segmentation should be cached"""
    base_path = get_region_cached_path(rat_id, slice_name, region)
    if not base_path:
        return None
    return base_path / f"{Config.USED_SEGMENTATION_MODEL_NAME}.tif"

def get_cached_feature_map_path(rat_id: str, slice_name: str, region: str) -> Optional[Path]:
    """Get the path where feature map should be cached"""
    base_path = get_region_cached_path(rat_id, slice_name, region)
    if not base_path:
        return None
    feature_file_name = f"Fibre Count for {Config.USED_SEGMENTATION_MODEL_NAME} trace.tif"
    return base_path / feature_file_name



def _setup_comparison_directory() -> Path:
    """Create and return the comparison directory path"""
    comparison_dir = Path("./gui/static/comparison")
    comparison_dir.mkdir(parents=True, exist_ok=True)
    return comparison_dir

def _copy_inference_image(experiment_id: str, figures_path: str, comparison_dir: Path) -> Optional[str]:
    """Copy inference image and return the static path"""
    inference_image_path = f"{figures_path}/inference_results/*.png"
    inference_images = glob.glob(inference_image_path)
    
    if not inference_images:
        return None
        
    source_inference = Path(inference_images[0])
    dest_inference = comparison_dir / f"inference_{experiment_id}.png"
    
    try:
        shutil.copy2(source_inference, dest_inference)
        main_inference_image = f"comparison/inference_{experiment_id}.png"
        logger.info(f"Copied inference image: {source_inference} -> {dest_inference}")
        return main_inference_image
    except Exception as e:
        logger.error(f"Failed to copy inference image: {e}")
        return None

def _copy_model_performance_images(experiment_id: str, figures_path: str, comparison_dir: Path) -> List[str]:
    """Copy model performance images and return list of static paths"""
    model_performance_images_path = f"{figures_path}/model_performances/*.png"
    performance_images = glob.glob(model_performance_images_path)
    model_performance_images = []
    
    for i, source_path in enumerate(performance_images):
        source_perf = Path(source_path)
        dest_perf = comparison_dir / f"model_performance_{experiment_id}_{i}.png"
        
        try:
            shutil.copy2(source_perf, dest_perf)
            model_performance_images.append(f"comparison/model_performance_{experiment_id}_{i}.png")
            logger.info(f"Copied model performance image: {source_perf} -> {dest_perf}")
        except Exception as e:
            logger.error(f"Failed to copy model performance image {source_path}: {e}")
    
    return model_performance_images

def _create_comparison_data_structure(experiment_id: str, main_inference_image: Optional[str], model_performance_images: List[str]) -> Dict:
    """Create the comparison data structure with actual experiment data"""
    experiment_data_path = f"./experiments/{experiment_id}/data.json"
    
    try:
        with open(experiment_data_path, 'r') as f:
            experiment_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Experiment data file not found: {experiment_data_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in experiment data file: {e}")
        return {}
    
    # Extract groups data from the JSON
    groups_data = []
    for group in experiment_data.get('groups', []):
        group_info = {
            "name": group.get('name'),
            "rats": group.get('rats', []),
            "regions": group.get('regions', []),
            "model_name": group.get('best_model_name'),  # Template expects 'model_name'
            "expected_rmse": group.get('expected_rmse'),
            "average_density": group.get('average_density'),
            "roi_associations": group.get('roi_associations', {})
        }
        groups_data.append(group_info)

    print(main_inference_image)
    print(model_performance_images)
    return {
        "experiment_id": experiment_data.get('experiment_id'),
        "experiment_name": experiment_data.get('experiment_name'),
        "experimenter_name": experiment_data.get('experimenter_name'),
        "experiment_date": experiment_data.get('experiment_date'),
        "groups": groups_data,
        "main_inference_image": main_inference_image,
        "model_performance_images": model_performance_images
    }

def get_comparison_data(experiment_id):
    """Get comparison data for an experiment by copying images and creating data structure"""
    figures_path = f"./figures/experiment_figures/{experiment_id}/"

    # Setup directory and copy images
    comparison_dir = _setup_comparison_directory()
    main_inference_image = _copy_inference_image(experiment_id, figures_path, comparison_dir)
    model_performance_images = _copy_model_performance_images(experiment_id, figures_path, comparison_dir)

    # Create and return the data structure with actual experiment data
    return _create_comparison_data_structure(experiment_id, main_inference_image, model_performance_images)

