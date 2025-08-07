"""
🐭 Rat Brain GUI - Flask Application

Web app frontend to explore structured rat brain image datasets and interact 
with existing segmentation + feature map models. Built for non-technical local users.

🚨 Critical TODOs to Complete:
Processing logic: Implement actual segmentation and feature extraction
Comparison plots: Implement matplotlib-based statistical comparisons

"""


import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import logging
from PIL import Image, ImageEnhance
import io
from config import Config

import base64

from werkzeug.utils import secure_filename
from io import BytesIO

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.imageio import numpy_to_tif, tif_to_numpy
from src.tracers.DLTracer import DLTracer
from src.NNs.Unet import UNetModel
from src.utils.imageio import generate_image_outer_mask

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


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


def scan_available_rats() -> List[str]:
    """
    Scan the data directory for available rat folders
    
    Returns:
        List of rat IDs found in the data directory
    """
    rats = []
    if Config.DATA_DIR.exists():
        for item in Config.DATA_DIR.iterdir():
            if item.is_dir():
                rats.append(item.name)
    logger.info(f"Found {len(rats)} rats: {rats}")
    return rats

def get_rat_regions(rat_id: str) -> List[str]:
    """
    Get available regions for a specific rat
    
    Args:
        rat_id: The rat identifier
        
    Returns:
        List of region names for the rat
    """
    rat_dir = validate_safe_path(Config.DATA_DIR, rat_id)
    regions = []
    if rat_dir and rat_dir.exists():
        for item in rat_dir.iterdir():
            if item.is_dir():
                regions.append(item.name)
    logger.info(f"Found {len(regions)} regions for rat {rat_id}: {regions}")
    return regions


def get_rat_subregions(rat_id: str, slice_name : str) -> List[str]:
    """
    Get available subregions for a specific rat_id + slice_name (bregma)
    
    Args:
        rat_id: The rat identifier
        
    Returns:
        List of region names for the rat
    """
    rat_dir = validate_safe_path(Config.DATA_DIR, rat_id, slice_name)
    regions = []
    if rat_dir and rat_dir.exists():
        for item in rat_dir.iterdir():
            if item.is_dir():
                regions.append(item.name)
    logger.info(f"Found {len(regions)} regions for rat {rat_id}: {regions}")
    return regions

def get_rat_metadata(rat_id: str) -> Optional[Dict]:
    """
    Load metadata for a specific rat
    
    Args:
        rat_id: The rat identifier
        
    Returns:
        Dictionary containing metadata or None if not found
    """
    metadata_file = validate_safe_path(Config.DATA_DIR, rat_id, Config.METADATA_FILE)
    if metadata_file and metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in metadata file: {metadata_file}")
    return None

def find_image_file(rat_id: str, slice : str, region: str) -> Optional[Path]:
    """
    Find the image file for a specific rat and region
    
    Args:
        rat_id: The rat identifier
        region: The region name
        
    Returns:
        Path to the image file or None if not found
    """
    
    # Validate path is safe
    region_dir = validate_safe_path(Config.DATA_DIR, rat_id, slice, region)
    if not region_dir or not region_dir.exists():
        return None
        
    for pattern in Config.IMAGE_PATTERNS:
        for file_path in region_dir.glob(pattern):
            if file_path.is_file():
                return file_path
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


# Flask routes
@app.route('/')
def index():
    """Index page - shows list of available rats"""
    rats = scan_available_rats()
    return render_template('index.html', rats=rats)




@app.route('/quicktrace', methods=['GET', 'POST'])
def quicktrace_page():
    """Quicktrace page - allow user to submit image path to trace"""
    if request.method == 'POST':
        image_path = request.form.get('image_path')
        if not image_path:
            return render_template('quicktrace.html', error="No file path provided.")

        try:
            # Validate the path is safe
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                return render_template('quicktrace.html', error="File does not exist.")
            image_path = str(image_path_obj)

            original_image_jpg_path = os.path.join('./static', 'quicktrace', 'original_image.jpg')
            convert_tif_to_jpg_and_save(Path(image_path), original_image_jpg_path, use_raw_path=True)
            
            # Generate tracing
            image = tif_to_numpy(image_path)[:, :, :1]
            mask = generate_image_outer_mask(image)
            tracer = DLTracer(Config.USED_SEGMENTATION_MODEL_PATH, UNetModel, 128, tracer_name="gui_tracer")
            trace = tracer.trace(image, mask).squeeze()

            output_tif_path = os.path.join('./static', 'quicktrace', 'traced_result.tif')
            numpy_to_tif(trace, output_tif_path)

            output_trace_jpg_path = os.path.join('./static', 'quicktrace', 'traced_result.jpg')
            convert_tif_to_jpg_and_save(Path(output_tif_path), output_trace_jpg_path, use_raw_path=True)

            return render_template('quicktrace.html', 
                                original_image=os.path.abspath(original_image_jpg_path),
                                result_image=os.path.abspath(output_trace_jpg_path),
                                original_image_file_path=os.path.abspath(image_path), 
                                high_quality_file_path=os.path.abspath(output_tif_path))


        except Exception as e:
            return render_template('quicktrace.html', error=f"Processing error: {str(e)}")

    return render_template('quicktrace.html')

@app.route('/view/<rat_id>')
def rat_page(rat_id: str):
    """Rat page - shows regions for a specific rat"""
    regions = get_rat_regions(rat_id)
    metadata = get_rat_metadata(rat_id)
    if not regions:
        return render_template('error.html', message=f"Rat {rat_id} not found")
    return render_template('rat.html', rat_id=rat_id, regions=regions, metadata=metadata)

@app.route('/view/<rat_id>/<slice_name>')
def slice_page(rat_id: str, slice_name: str):
    """Slice page - shows subregions for a specific rat and slice"""
    subregions = get_rat_subregions(rat_id, slice_name)
    metadata = get_rat_metadata(rat_id)

    if not subregions:
        return render_template('error.html', message=f"Slice {slice_name} not found for rat {rat_id}")

    return render_template('rat.html', 
                         rat_id=rat_id, 
                         regions=subregions, 
                         metadata=metadata,
                         slice_name=slice_name)


@app.route('/view/<rat_id>/<slice_name>/<region>')
def region_page(rat_id: str, slice_name: str, region: str):
    """Region page - shows image, segmentation, and feature map"""

    # Validate image URL path
    image_url_path = validate_safe_path(Config.DATA_DIR, rat_id, slice_name, region, "th.tif")
    if not image_url_path:
        return render_template('error.html', message=f"ROI image not found for {rat_id}/{region}")

    # Convert TIF to JPG for browser compatibility
    image_filename = f"{rat_id}_{slice_name}_{region}_img.jpg"
    image_filename_result = convert_tif_to_jpg_and_save(image_url_path, image_filename, lumin_scale=2.0)
    if not image_filename_result:
        return render_template('error.html', message=f"Failed to convert image for {rat_id}/{region}")
    
    # Check for cached segmentation and convert if exists
    seg_cache_path = get_cached_segmentation_path(rat_id, slice_name, region)
    has_segmentation = seg_cache_path and seg_cache_path.exists()
    segmentation_filename = None
    if has_segmentation:
        seg_filename = f"{rat_id}_{slice_name}_{region}_seg.jpg"
        segmentation_filename = convert_tif_to_jpg_and_save(seg_cache_path, seg_filename, lumin_scale=5.0)
    
    # Check for cached feature map and convert if exists
    feature_cache_path = get_cached_feature_map_path(rat_id, slice_name, region)
    has_feature_map = feature_cache_path and feature_cache_path.exists()
    feature_map_filename = None
    if has_feature_map:
        feature_filename = f"{rat_id}_{slice_name}_{region}_features.jpg"
        feature_map_filename = convert_tif_to_jpg_and_save(feature_cache_path, feature_filename, lumin_scale=15.0)
    
   
    # Get directory paths - resolve to absolute paths and simplify
    image_dir_path = str(image_url_path.parent.resolve()).replace('\\', '/') if image_url_path else None
    seg_dir_path = str(seg_cache_path.parent.resolve()).replace('\\', '/') if seg_cache_path else None
    feature_dir_path = str(feature_cache_path.parent.resolve()).replace('\\', '/') if feature_cache_path else None
    
    return render_template('region.html', 
                         rat_id=rat_id, 
                         slice_name=slice_name,
                         region=region,
                         image_filename=image_filename_result,
                         segmentation_filename=segmentation_filename,
                         feature_map_filename=feature_map_filename,
                         has_segmentation=has_segmentation,
                         has_feature_map=has_feature_map,
                         image_dir_path=image_dir_path,
                         seg_dir_path=seg_dir_path,
                         feature_dir_path=feature_dir_path)


@app.route('/compare', methods=['GET', 'POST'])
def compare_page():
    """Group-based comparison page - allows creating multiple groups for comparison"""
    if request.method == 'GET':
        rats = scan_available_rats()
        regions = Config.PREDEFINED_REGIONS
        return render_template('compare.html', rats=rats, regions=regions)
    
    elif request.method == 'POST':
        # Handle group-based comparison form submission
        # !!! TODO: Implement the group-based comparison logic here
        # This will receive JSON data with multiple groups containing rats and regions
        
        # For now, return a placeholder response
        return jsonify({
            'status': 'placeholder',
            'message': 'Group-based comparison logic needs to be implemented',
            'groups': request.get_json() if request.is_json else {}
        })


@app.route('/api/rats')
def api_rats():
    """JSON API endpoint for getting available rats"""
    rats = scan_available_rats()
    return jsonify(rats)

@app.route('/api/rats/<rat_id>/regions')
def api_regions(rat_id: str):
    """JSON API endpoint for getting regions for a rat"""
    regions = get_rat_regions(rat_id)
    return jsonify(regions)

@app.route('/api/compare/groups', methods=['POST'])
def api_compare_groups():
    """API endpoint for group-based comparison"""
    try:
        data = request.get_json()
        if not data or 'groups' not in data:
            return jsonify({'error': 'Invalid request format'}), 400
        
        groups = data['groups']
        if len(groups) < 2:
            return jsonify({'error': 'At least 2 groups required for comparison'}), 400
        
        # !!! TODO: Implement actual comparison logic here
        # This should:
        # 1. Load data for each group's rats and regions
        # 2. Calculate summary statistics
        # 3. Generate comparison visualizations
        # 4. Perform statistical tests
        
        # Placeholder response
        result = {
            'status': 'success',
            'message': 'Comparison completed (placeholder)',
            'groups_analyzed': len(groups),
            'summary': {
                'total_rats': sum(len(g.get('rats', [])) for g in groups),
                'total_regions': sum(len(g.get('regions', [])) for g in groups)
            },
            'results': {
                'statistics': 'Placeholder - statistical analysis would be here',
                'visualizations': 'Placeholder - graphs would be generated here',
                'tests': 'Placeholder - statistical tests would be performed here'
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in group comparison: {e}")
        return jsonify({'error': f'Comparison failed: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', message="Internal server error"), 500


# Application startup
def create_app():
    """Application factory function"""
    
    if not Config.DATA_DIR.exists():
        logger.error(f"Data directory does not exist: {Config.DATA_DIR}")
        raise FileNotFoundError(f"Data directory not found: {Config.DATA_DIR}")
    
    if not Config.CACHE_DIR.exists():
        logger.error(f"Cache directory does not exist: {Config.CACHE_DIR}")
        raise FileNotFoundError(f"Cache directory not found: {Config.CACHE_DIR}")
    
    logger.info("Rat Brain GUI application initialized")
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
