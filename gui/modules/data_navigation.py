from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os
import glob

import numpy as np

from config import Config

import logging
logger = logging.getLogger(__name__)


import json

from .data_loading import validate_safe_path

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







def get_directories_with_tif_images(root_directory):
    """
    Returns a list of directory names that are exactly 2 levels below the root
    and contain at least one .tif or .tiff image file.
    
    Args:
        root_directory (str): Path to the root directory to search
    
    Returns:
        list: List of directory names (not full paths) that meet the criteria
    """
    matching_directories = []
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(root_directory):
        # Calculate the depth relative to root_directory
        relative_path = os.path.relpath(root, root_directory)
        depth = len(relative_path.split(os.sep)) if relative_path != '.' else 0
        
        # Only consider directories that are exactly 2 levels deep
        if depth == 3:
            # Check if this directory contains any .tif or .tiff files
            tif_files = glob.glob(os.path.join(root, "*.tif")) + glob.glob(os.path.join(root, "*.tiff"))
            
            if tif_files:
                # Get just the directory name (not the full path)
                dir_name = os.path.basename(root)
                matching_directories.append(dir_name)
    
    return [d for d in np.unique(matching_directories)]

def scan_available_experiments() -> List[Dict]:
    """
    Scan the experiments directory for available experiment folders
    
    Returns:
        List of dictionaries containing experiment information with keys:
        - experiment_id: The experiment ID (folder name)
        - experiment_date: Date from data.json if available
        - experimenter_name: Name from data.json if available
        - experiment_name: Name from data.json if available
    """
    experiments = []
    experiments_dir = Path("./experiments")
    
    if experiments_dir.exists():
        for item in experiments_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                experiment_info = {
                    'experiment_id': item.name,
                    'experiment_date': None,
                    'experimenter_name': None,
                    'experiment_name': None
                }
                
                # Try to load data.json for additional info
                data_file = item / "data.json"
                if data_file.exists():
                    try:
                        with open(data_file, 'r') as f:
                            data = json.load(f)
                            experiment_info['experiment_date'] = data.get('experiment_date')
                            experiment_info['experimenter_name'] = data.get('experimenter_name')
                            experiment_info['experiment_name'] = data.get('experiment_name')
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Could not parse data.json for experiment {item.name}: {e}")
                
                experiments.append(experiment_info)
    
    # Sort by experiment ID (numerically, handling 4-digit string format)
    experiments.sort(key=lambda x: int(x['experiment_id']))
    logger.info(f"Found {len(experiments)} experiments: {[exp['experiment_id'] for exp in experiments]}")
    return experiments