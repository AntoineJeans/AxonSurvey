import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
from typing import List, Tuple, Optional, Dict, Any, Union

# Global styling configuration
plt.rcParams["font.family"] = 'Calibri'

# Constants for consistent styling
STYLE_CONFIG = {
    'dark': {
        'style': 'dark_background',
        'contour_size': 4,
        'figsize': (18, 6),
        'title_fontsize': 28,
        'label_fontsize': 25,
        'tick_fontsize': 15,
        'legend_fontsize': 20,
        'legend_title_fontsize': 25,
        'legend_facecolor': '#004e52'
    },
    'light': {
        'style': 'seaborn-v0_8-white',
        'contour_size': 2,
        'figsize': (12, 8),
        'title_fontsize': 20,
        'label_fontsize': 16,
        'tick_fontsize': 12,
        'legend_fontsize': 14,
        'legend_title_fontsize': 16,
        'legend_facecolor': 'white'
    }
}

# Color schemes
COLOR_SCHEMES = {
    'model_types': {
        "baseline": '#FFFFFF',
        "threshold": '#f0aa6c',
        "DL": '#7fdb92',
        "other": '#a3a3a3'
    },
    'predictors': ['#FFFFFF', '#f0aa6c', '#7fdb92'],
    'default': ['#FFFFFF']
}

def apply_style(style_name: str = 'dark') -> None:
    """Apply consistent styling to matplotlib plots."""
    config = STYLE_CONFIG[style_name]
    plt.style.use(config['style'])

def setup_figure(style_name: str = 'dark', figsize: Optional[Tuple[int, int]] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Create a figure with consistent styling and contour formatting."""
    config = STYLE_CONFIG[style_name]
    figsize = figsize or config['figsize']
    fig, ax = plt.subplots(figsize=figsize)
    
    # Apply contour styling
    for spine in ax.spines.values():
        spine.set_linewidth(config['contour_size'])
    
    return fig, ax

def format_axes(ax: plt.Axes, labels: List[str], title: str, ylabel: str, 
                style_name: str = 'dark', rotation: float = 12.0) -> None:
    """Format axes with consistent styling including ticks, labels, and title."""
    config = STYLE_CONFIG[style_name]
    
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=config['tick_fontsize'], rotation=rotation)
    ax.tick_params(axis="y", labelsize=config['tick_fontsize'])
    ax.set_ylabel(ylabel, fontsize=config['label_fontsize'])
    ax.set_title(title, fontsize=config['title_fontsize'], pad=30)
    plt.ylim(bottom=0)

def create_legend(ax: plt.Axes, handles: List, title: str = "Legend", 
                 style_name: str = 'dark') -> None:
    """Create a legend with consistent styling and positioning."""
    config = STYLE_CONFIG[style_name]
    ax.legend(
        handles=handles,
        title=title,
        title_fontsize=config['legend_title_fontsize'],
        fontsize=config['legend_fontsize'],
        loc='upper right',
        bbox_to_anchor=(1.0, 1.0),
        facecolor=config['legend_facecolor']
    )

def save_plot(title: str, save_path: Optional[str] = None) -> None:
    """Save plot with consistent naming and high-quality settings."""
    plt.tight_layout()
    if save_path is not None:
        save_file = os.path.join(save_path, f"{title}.png")
        plt.savefig(save_file, transparent=True, dpi=300, bbox_inches='tight')
    plt.style.use('classic')

def get_model_colors(model_types: Optional[List[str]] = None) -> List[str]:
    """Get color mapping for different model types."""
    if model_types is None:
        return COLOR_SCHEMES['default'] * 10  # Fallback for unknown number
    
    type_to_color = COLOR_SCHEMES['model_types']
    return [type_to_color.get(t, type_to_color['other']) for t in model_types]

def create_box_data(expected: float, lower: float, upper: float) -> Dict[str, Any]:
    """Create standardized box plot data structure for matplotlib."""
    return {
        'med': expected,
        'q1': lower,
        'q3': upper,
        'whislo': lower,
        'whishi': upper,
        'fliers': []
    }

def display_model_bounds(expected: List[float], lower: List[float], upper: List[float], 
                        labels: List[str], title: str, metric_name: str, 
                        save_path: Optional[str] = None, nice: bool = True, 
                        model_types: Optional[List[str]] = None) -> None:
    """Display model performance bounds as bar charts or box plots."""
    
    if nice:
        apply_style('dark')
        fig, ax = setup_figure('dark')
        
        # Get colors and create bars
        model_colors = get_model_colors(model_types)
        bars = ax.bar(range(1, len(expected) + 1), expected, 
                     color=model_colors[:len(expected)], linewidth=0)
        
        # Add legend if model types are provided
        if model_types is not None:
            type_to_color = COLOR_SCHEMES['model_types']
            existing_types = np.unique(model_types)
            legend_patches = [
                mpatches.Patch(color=type_to_color[label], label=label) 
                for label in type_to_color.keys() if label in existing_types
            ]
            create_legend(ax, legend_patches, "Extractor types")
        
        # Format axes
        format_axes(ax, labels, title, f"Expected model {metric_name}")
        
    else:
        apply_style('light')
        fig, ax = setup_figure('light', (12, 6))
        
        # Create box plots
        boxes = [create_box_data(e, l, u) for e, l, u in zip(expected, lower, upper)]
        for i, box in enumerate(boxes):
            ax.bxp([box], positions=[i+1], showfliers=False)
        
        # Format axes with different rotation for compactness
        format_axes(ax, labels, title, f"Expected model {metric_name}", 'light', 20.0)
        plt.grid(True, alpha=0.3)
    
    save_plot(title, save_path)

def display_inference_bounds(confidence_bounds: List[List[Tuple[float, float, float]]], 
                           labels: List[str], title: str, 
                           predictor_names: Optional[List[str]] = None,
                           property_name: str = "Density", 
                           save_path: Optional[str] = None) -> None:
    """Display inference confidence bounds as grouped box plots."""
    
    apply_style('dark')
    
    # Calculate global bounds for consistent y-axis
    all_bounds = [bound for group in confidence_bounds for bound in group]
    total_minimum = min([l for _, _, l in all_bounds])
    total_maximum = max([u for _, u, _ in all_bounds])
    
    fig, ax = setup_figure('dark')
    
    if predictor_names is None:
        # Single predictor case - more compact
        confidence_bounds = [c[0] for c in confidence_bounds]
        boxes = [create_box_data(e, l, u) for e, l, u in confidence_bounds]
        
        for i, box in enumerate(boxes):
            ax.bxp([box], positions=[i+1], showfliers=False, widths=0.6)
    
    else:
        # Multiple predictors case
        colors = COLOR_SCHEMES['predictors']
        num_groups = len(confidence_bounds)
        num_predictors = len(predictor_names)
        group_width = 0.8
        box_width = group_width / num_predictors
        
        legend_handles = []
        
        for group_idx, group_bounds in enumerate(confidence_bounds):
            group_center = group_idx + 1
            
            for pred_idx, (pred_name, bounds) in enumerate(zip(predictor_names, group_bounds)):
                e, l, u = bounds
                color = colors[pred_idx % len(colors)]
                
                # Calculate position within the group
                offset = (pred_idx - (num_predictors - 1) / 2) * box_width
                position = group_center + offset
                
                # Create box plot
                box_data = create_box_data(e, l, u)
                bp = ax.bxp([box_data], positions=[position], showfliers=False, 
                           patch_artist=True, widths=box_width*0.8)
                
                # Set color for the box
                for patch in bp['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(1.0)
                
                # Add to legend handles (only once per predictor)
                if group_idx == 0:
                    legend_handles.append(mpatches.Patch(color=color, label=pred_name))
        
        # Add legend
        create_legend(ax, legend_handles, "Estimator types")
    
    # Format axes with improved y-axis limits
    format_axes(ax, labels, title, f"Expected {property_name}")
    ax.set_ylim(total_minimum * 0.95, total_maximum * 1.05)  # More compact y-axis
    plt.grid(True, alpha=0.2)
    
    save_plot(title, save_path)

def display_inference_points(group_data: List[List[float]], labels: List[str], 
                           title: str, property_name: str = "Density", 
                           save_path: Optional[str] = None) -> None:
    """Display inference points as scatter plots with background ranges."""
    
    apply_style('dark')
    
    # Calculate total range for y-axis
    all_points = [point for group in group_data for point in group]
    total_maximum = max(all_points)
    
    # Use more compact figure size
    fig, ax = setup_figure('dark', (12, 6))
    
    # Display points with improved spacing
    for i, (label, points) in enumerate(zip(labels, group_data)):
        x_position = i + 1
        
        # Calculate min and max for this group
        group_min = min(points)
        group_max = max(points)
        
        # Add background box with reduced alpha for better visibility
        box_width = 0.3  # More compact
        box_height = group_max - group_min
        box_x = x_position - box_width/2
        box_y = group_min
        
        rect = mpatches.Rectangle((box_x, box_y), box_width, box_height, 
                                facecolor='#f0aa6c', alpha=0.5, edgecolor='none')
        ax.add_patch(rect)
        
        # Plot points with smaller size for better density
        ax.scatter([x_position] * len(points), points, color='#FFFFFF', s=100, alpha=0.8)
        
        # Add mean line
        mean_val = np.mean(points)
        ax.plot([x_position-0.25, x_position+0.25], [mean_val, mean_val], 
                color='#f0aa6c', linestyle='--', alpha=0.7, linewidth=2)
    
    # Format axes with more compact limits
    format_axes(ax, labels, title, f"Expected {property_name}")
    ax.set_ylim(0, total_maximum * 1.05)  # Start from 0 for better interpretation
    ax.set_xlim(0.5, len(labels) + 0.5)
    plt.grid(True, alpha=0.2)
    
    save_plot(title, save_path)

def plot_single_variable_fitness(estimated: List[float], real_property: List[float], 
                                estimated_name: str = "Estimated Value",
                                real_name: str = "Real Value",
                                display_stats: bool = False) -> None:
    """Plot correlation between estimated and real values with trend line."""
    
    assert len(estimated) == len(real_property), 'estimated and predicted value lists not same length'
    
    # Compute Pearson correlation coefficient
    r, p_value = pearsonr(estimated, real_property)
    
    apply_style('light')
    fig, ax = setup_figure('light', (8, 6))
    
    # Create scatter plot with improved styling
    ax.scatter(estimated, real_property, alpha=0.7, edgecolors='k', s=80)
    
    # Add trend line for better interpretation
    z = np.polyfit(estimated, real_property, 1)
    p = np.poly1d(z)
    ax.plot(estimated, p(estimated), "r--", alpha=0.8, linewidth=2)
    
    ax.set_xlabel(estimated_name, fontsize=14)
    ax.set_ylabel(real_name, fontsize=14)
    
    if display_stats:
        title = f"{estimated_name} vs {real_name}: R={r:.3f}, p={p_value:.2e}"
    else:
        title = f"{estimated_name} vs {real_name}"
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.style.use('classic')

def display_model_rmse(expected: List[float], labels: List[str], title: str) -> None:
    """Display RMSE values as bar chart with value labels."""
    
    apply_style('light')
    fig, ax = setup_figure('light', (10, 6))
    
    bars = ax.bar(labels, expected, color="skyblue", alpha=0.8)
    
    # Add value labels on bars for better readability
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylim(0, max(expected) * 1.15)
    ax.set_ylabel("Expected Density", fontsize=14)
    ax.set_title(title, fontsize=16, pad=20)
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    plt.style.use('classic')

def display_test_bias_for_group(biases: List[float], alpha: float, 
                              name_list: List[str], group_label: str) -> None:
    """Display test bias statistics as bar chart with threshold line."""
    
    apply_style('light')
    fig, ax = setup_figure('light', (10, 6))
    
    bias_values = [1 - bias for bias in biases]
    bars = ax.bar(name_list, bias_values, color="skyblue", alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Bias statistic", fontsize=14)
    ax.tick_params(axis='x', rotation=20)
    
    # Add threshold line
    threshold = 1 - alpha
    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=2, 
               label=f"Threshold = {threshold:.3f}")
    
    title = f"Bias for models in {group_label} regions"
    ax.set_title(title, fontsize=16, pad=20)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.show()
    plt.style.use('classic')
        