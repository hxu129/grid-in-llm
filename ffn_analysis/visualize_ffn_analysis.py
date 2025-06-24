"""
Enhanced FFN Analysis Visualization with Maze Overlay

This module provides visualization tools for FFN activation analysis with support for
overlaying activation heatmaps on actual maze structures for more intuitive interpretation.

Key Features:
- Traditional grid-based heatmaps
- NEW: Maze overlay heatmaps showing activations on actual maze structure
- Row/column coordinate labels for precise position reference
- Layer-wise analysis and comparison
- Neuron specialization analysis
- Single neuron detailed views

Usage:
    # Run complete analysis with both traditional and maze overlay visualizations
    python visualize_ffn_analysis.py
    
    # Or use individual functions programmatically
    from visualize_ffn_analysis import load_ffn_analysis_results, visualize_all_layers_with_maze
    results = load_ffn_analysis_results()
    visualize_all_layers_with_maze(results)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os
import json
import sys
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as patches

# Setup paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)  # For neuron_heatmap_visualizer

# Add data/maze to path for maze imports
maze_path = os.path.join(project_root, 'data', 'maze')
sys.path.append(maze_path)

from neuron_heatmap_visualizer import plot_neuron_heatmaps, plot_single_neuron_heatmap

try:
    from maze_visualizer import MazeVisualizer
    MAZE_VISUALIZER_AVAILABLE = True
except ImportError:
    print("Warning: MazeVisualizer not available. Maze overlay functionality will be limited.")
    MAZE_VISUALIZER_AVAILABLE = False
    MazeVisualizer = None

def load_ffn_analysis_results(results_path: str = "ffn_position_analysis_results.npz") -> Dict:
    """Load the FFN analysis results."""
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    data = np.load(results_path, allow_pickle=True)
    
    # Extract metadata
    grid_size = int(data['grid_size'])
    n_positions = int(data['n_positions'])
    n_layers = int(data['n_layers'])
    ffn_size = int(data['ffn_size'])
    metadata = data['metadata'].item()
    
    # Extract layer matrices
    layer_matrices = {}
    for layer_name in metadata['layer_names']:
        layer_matrices[layer_name] = data[layer_name]
    
    print(f"Loaded results: {n_layers} layers, {grid_size}x{grid_size} grid, {ffn_size} neurons per layer")
    print(f"Layer matrices: {list(layer_matrices.keys())}")
    
    return {
        'grid_size': grid_size,
        'n_positions': n_positions,
        'n_layers': n_layers,
        'ffn_size': ffn_size,
        'metadata': metadata,
        'layer_matrices': layer_matrices
    }

def load_maze_data(grid_size: int, maze_data_dir: str = "data/maze/maze_nav_data") -> Dict:
    """Load maze structure data for the given grid size."""
    try:
        # Use the already defined project_root from module initialization
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # Parent directory (project root)
        
        # Build absolute path to maze dataset
        dataset_path = os.path.join(project_root, maze_data_dir, f'maze_nav_dataset_{grid_size}.json')
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Maze dataset not found: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            
        maze_data = dataset['maze_data']
        print(f"Loaded maze data: {grid_size}x{grid_size} maze with {maze_data['num_nodes']} nodes")
        return maze_data
        
    except Exception as e:
        print(f"Error loading maze data: {e}")
        print("Falling back to simple grid visualization without maze structure.")
        return None

def create_maze_overlay_heatmap(activation_matrix: np.ndarray, neuron_idx: int, 
                               maze_data: Dict, grid_size: int,
                               title: str = None, figsize: tuple = (10, 8),
                               cmap: str = 'RdYlBu_r', alpha: float = 0.9) -> plt.Figure:
    """
    Create a heatmap overlaid on the actual maze structure by drawing walls as lines.
    
    Args:
        activation_matrix: Matrix of shape (n_positions, n_neurons)
        neuron_idx: Index of neuron to visualize
        maze_data: Maze structure data from dataset
        grid_size: Size of the grid
        title: Custom title for the plot
        figsize: Figure size
        cmap: Colormap for activation heatmap
        alpha: Transparency of activation overlay
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 1. Prepare activation data for a continuous heatmap
    neuron_activations = activation_matrix[:, neuron_idx]
    activation_grid = neuron_activations.reshape((grid_size, grid_size))
    
    # Mask NaN values to prevent them from being plotted
    masked_activations = np.ma.masked_where(np.isnan(activation_grid), activation_grid)
    
    # 2. Draw the continuous activation heatmap
    # The 'extent' argument maps the grid to data coordinates.
    im = ax.imshow(masked_activations, cmap=cmap, origin='lower', 
                  interpolation='nearest', alpha=alpha,
                  extent=(-0.5, grid_size - 0.5, -0.5, grid_size - 0.5))
    
    # 3. Draw maze walls as lines based on adjacency matrix
    if maze_data is not None:
        adj_matrix = np.array(maze_data['adjacency_matrix'])
        wall_color = '#2C3E50'  # Dark, clear wall color
        wall_lw = 5  # Increased wall thickness
        
        for r in range(grid_size):
            for c in range(grid_size):
                node_id = r * grid_size + c
                
                # Check for wall to the right
                if c < grid_size - 1:
                    neighbor_id = r * grid_size + (c + 1)
                    if adj_matrix[node_id, neighbor_id] == 0:
                        ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], color=wall_color, linewidth=wall_lw)
                
                # Check for wall below (or above in 'lower' origin)
                if r < grid_size - 1:
                    neighbor_id = (r + 1) * grid_size + c
                    if adj_matrix[node_id, neighbor_id] == 0:
                        ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color=wall_color, linewidth=wall_lw)

    # 4. Draw the outer border of the maze
    ax.add_patch(patches.Rectangle((-0.5, -0.5), grid_size, grid_size, 
                                   fill=False, edgecolor=wall_color, lw=wall_lw))
    
    # Add node position labels (optional, smaller font)
    if grid_size <= 12:  # Only for smaller mazes
        for position in range(grid_size * grid_size):
            row, col = position // grid_size, position % grid_size
            
            # Add position label at the center of the cell
            ax.text(col, row, str(position), 
                   ha='center', va='center', fontsize=8, 
                   color='black', weight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7),
                   zorder=4)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Activation Level', rotation=270, labelpad=15)
    
    # Set title
    if title is None:
        title = f"Neuron {neuron_idx} Activation Heatmap on Maze"
    ax.set_title(title, fontsize=12, pad=10)
    
    # Set axis properties for an NxN grid
    ax.set_xlim(-1, grid_size)
    ax.set_ylim(-1, grid_size)
    ax.set_aspect('equal')
    
    # Add row and column labels on the perimeter
    col_labels = [str(col) for col in range(grid_size)]
    ax.set_xticks(np.arange(grid_size))
    ax.set_xticklabels(col_labels)
    ax.tick_params(axis='x', which='major', labelsize=8, pad=2)
    
    row_labels = [str(row) for row in range(grid_size)]
    ax.set_yticks(np.arange(grid_size))
    ax.set_yticklabels(row_labels)
    ax.tick_params(axis='y', which='major', labelsize=8, pad=2)
    
    # Add labels for axes
    ax.set_xlabel('Column', fontsize=10, labelpad=5)
    ax.set_ylabel('Row', fontsize=10, labelpad=5)
    
    return fig

def create_maze_overlay_grid(results: Dict, layer_name: str, 
                           neuron_indices: List[int], maze_data: Dict = None,
                           figsize: tuple = (16, 12), cols: int = 4,
                           cmap: str = 'RdYlBu_r', alpha: float = 0.9) -> plt.Figure:
    """
    Create a grid of maze overlay heatmaps for multiple neurons.
    
    Args:
        results: Results from load_ffn_analysis_results
        layer_name: Name of the layer to analyze
        neuron_indices: List of neuron indices to visualize
        maze_data: Maze structure data (if None, will try to load)
        figsize: Figure size
        cols: Number of columns in the grid
        cmap: Colormap for activation heatmaps
        alpha: Transparency of activation overlay
    
    Returns:
        matplotlib Figure object
    """
    grid_size = results['grid_size']
    
    # Load maze data if not provided
    if maze_data is None:
        maze_data = load_maze_data(grid_size)
    
    # Get the activation matrix
    if layer_name not in results['layer_matrices']:
        raise ValueError(f"Layer {layer_name} not found in results")
    
    matrix = results['layer_matrices'][layer_name]  # (n_positions, n_neurons)
    
    # Calculate grid layout
    n_plots = len(neuron_indices)
    rows = (n_plots + cols - 1) // cols
    
    # Create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Create heatmap for each neuron
    for i, neuron_idx in enumerate(neuron_indices):
        ax = axes[i]
        
        # 1. Prepare and draw continuous activation heatmap
        neuron_activations = matrix[:, neuron_idx]
        activation_grid = neuron_activations.reshape((grid_size, grid_size))
        masked_activations = np.ma.masked_where(np.isnan(activation_grid), activation_grid)
        
        ax.imshow(masked_activations, cmap=cmap, origin='lower', 
                  alpha=alpha, zorder=1, interpolation='nearest',
                  extent=(-0.5, grid_size - 0.5, -0.5, grid_size - 0.5))
        
        # 2. Draw maze walls as lines
        if maze_data is not None:
            adj_matrix = np.array(maze_data['adjacency_matrix'])
            wall_color = '#2C3E50'
            wall_lw = 5 # Increased wall thickness for grid view

            for r in range(grid_size):
                for c in range(grid_size):
                    node_id = r * grid_size + c
                    if c < grid_size - 1:
                        neighbor_id = r * grid_size + (c + 1)
                        if adj_matrix[node_id, neighbor_id] == 0:
                            ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], color=wall_color, linewidth=wall_lw)
                    if r < grid_size - 1:
                        neighbor_id = (r + 1) * grid_size + c
                        if adj_matrix[node_id, neighbor_id] == 0:
                            ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color=wall_color, linewidth=wall_lw)
            
            # Draw outer border
            ax.add_patch(patches.Rectangle((-0.5, -0.5), grid_size, grid_size,
                                           fill=False, edgecolor=wall_color, lw=wall_lw))

        # Set title
        ax.set_title(f"Neuron {neuron_idx}", fontsize=10)
        
        # Set axis properties
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.axis('off') # Keep it clean for the grid view
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f"{layer_name.upper()}: FFN Neuron Activations on Maze Structure", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def analyze_neuron_specialization(results: Dict, layer_name: str, top_k: int = 20) -> Dict:
    """
    Analyze which neurons are most specialized for specific positions.
    
    Args:
        results: Results from load_ffn_analysis_results
        layer_name: Name of the layer to analyze (e.g., 'layer_0')
        top_k: Number of top specialized neurons to return
        
    Returns:
        Dictionary with specialization analysis
    """
    matrix = results['layer_matrices'][layer_name]  # (n_positions, n_neurons)
    
    # Remove NaN values for analysis
    valid_mask = ~np.isnan(matrix)
    
    # Calculate specialization metrics for each neuron
    neuron_specialization = []
    
    for neuron_idx in range(matrix.shape[1]):
        neuron_activations = matrix[:, neuron_idx]
        valid_activations = neuron_activations[~np.isnan(neuron_activations)]
        
        if len(valid_activations) > 1:
            # Calculate variance as a measure of specialization
            # Higher variance = more specialized (some positions activate much more than others)
            variance = np.var(valid_activations)
            max_activation = np.max(valid_activations)
            max_position = np.argmax(neuron_activations)
            
            # Calculate how concentrated the activation is
            # (difference between max and mean)
            concentration = max_activation - np.mean(valid_activations)
            
            neuron_specialization.append({
                'neuron_idx': neuron_idx,
                'variance': variance,
                'max_activation': max_activation,
                'max_position': max_position,
                'concentration': concentration,
                'n_valid_positions': len(valid_activations)
            })
    
    # Sort by concentration (most specialized first)
    neuron_specialization.sort(key=lambda x: x['concentration'], reverse=True)
    
    return {
        'layer_name': layer_name,
        'top_specialized_neurons': neuron_specialization[:top_k],
        'all_neurons': neuron_specialization
    }

def visualize_layer_overview_with_maze(results: Dict, layer_name: str, 
                                     n_neurons_to_show: int = 100,
                                     figsize: tuple = (16, 12),
                                     maze_data: Dict = None):
    """Visualize an overview of the most interesting neurons in a layer overlaid on maze."""
    
    grid_size = results['grid_size']
    
    # Load maze data if not provided
    if maze_data is None:
        maze_data = load_maze_data(grid_size)
    
    # Analyze specialization
    specialization = analyze_neuron_specialization(results, layer_name, top_k=n_neurons_to_show)
    top_neurons = specialization['top_specialized_neurons']
    
    if not top_neurons:
        print(f"No valid neurons found for {layer_name}")
        return None
    
    # Get the neuron indices to visualize
    neuron_indices = [neuron['neuron_idx'] for neuron in top_neurons[:n_neurons_to_show]]
    
    # Create maze overlay grid
    fig = create_maze_overlay_grid(
        results, layer_name, neuron_indices, maze_data,
        figsize=figsize, cols=4, alpha=0.8
    )
    
    return fig

def visualize_all_layers_with_maze(results: Dict, neurons_per_layer: int = 9,
                                 save_dir: str = "ffn_visualizations"):
    """Create maze overlay visualizations for all layers."""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load maze data once
    grid_size = results['grid_size']
    maze_data = load_maze_data(grid_size)
    
    layer_names = sorted(results['layer_matrices'].keys())
    
    for layer_name in layer_names:
        print(f"Creating maze overlay visualization for {layer_name}...")
        
        try:
            fig = visualize_layer_overview_with_maze(
                results, 
                layer_name, 
                n_neurons_to_show=neurons_per_layer,
                figsize=(12, 9),
                maze_data=maze_data
            )
            
            if fig is not None:
                # Save the figure
                save_path = os.path.join(save_dir, f"{layer_name}_maze_overlay.png")
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved {save_path}")
                plt.close(fig)  # Close to save memory
        
        except Exception as e:
            print(f"Error creating maze overlay visualization for {layer_name}: {e}")

def create_single_neuron_maze_overlay(results: Dict, layer_name: str, neuron_idx: int,
                                    save_path: str = None, figsize: tuple = (10, 8)):
    """Create a single neuron maze overlay heatmap and optionally save it."""
    
    grid_size = results['grid_size']
    maze_data = load_maze_data(grid_size)
    
    if layer_name not in results['layer_matrices']:
        raise ValueError(f"Layer {layer_name} not found in results")
    
    matrix = results['layer_matrices'][layer_name]
    
    # Create the visualization
    fig = create_maze_overlay_heatmap(
        matrix, neuron_idx, maze_data, grid_size,
        title=f"{layer_name.upper()} Neuron {neuron_idx} on Maze",
        figsize=figsize
    )
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved single neuron maze overlay: {save_path}")
    
    return fig

def visualize_layer_overview(results: Dict, layer_name: str, 
                           n_neurons_to_show: int = 16,
                           figsize: tuple = (16, 12)):
    """Visualize an overview of the most interesting neurons in a layer."""
    
    # Analyze specialization
    specialization = analyze_neuron_specialization(results, layer_name, top_k=n_neurons_to_show)
    top_neurons = specialization['top_specialized_neurons']
    
    if not top_neurons:
        print(f"No valid neurons found for {layer_name}")
        return None
    
    # Get the neuron indices to visualize
    neuron_indices = [neuron['neuron_idx'] for neuron in top_neurons[:n_neurons_to_show]]
    
    # Get the matrix
    matrix = results['layer_matrices'][layer_name]
    
    # Create custom titles showing specialization info
    titles = []
    for neuron in top_neurons[:n_neurons_to_show]:
        max_pos = neuron['max_position']
        row, col = max_pos // results['grid_size'], max_pos % results['grid_size']
        titles.append(f"N{neuron['neuron_idx']} (max@{max_pos}:[{row},{col}])")
    
    # Create visualization
    # Need to extract specific neurons and present as (n_positions, selected_neurons)
    selected_matrix = matrix[:, neuron_indices]  # Shape: (n_positions, selected_neurons)
    
    fig = plot_neuron_heatmaps(
        selected_matrix,  # Shape: (n_positions, selected_neurons)
        neuron_indices=list(range(len(neuron_indices))),  # Index into selected matrix
        grid_size=results['grid_size'],
        figsize=figsize,
        cols=4,
        title_prefix="",
        cmap='RdYlBu_r',  # Red-Yellow-Blue colormap, reversed
        show_colorbar=True
    )
    
    # Update titles
    for i, title in enumerate(titles):
        if i < len(fig.axes) and i < len(titles):
            fig.axes[i].set_title(title, fontsize=10)
    
    fig.suptitle(f"{layer_name.upper()}: Most Position-Specialized FFN Neurons", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def visualize_all_layers_overview(results: Dict, neurons_per_layer: int = 9,
                                save_dir: str = "ffn_visualizations"):
    """Create overview visualizations for all layers."""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    layer_names = sorted(results['layer_matrices'].keys())
    
    for layer_name in layer_names:
        print(f"Creating visualization for {layer_name}...")
        
        try:
            fig = visualize_layer_overview(
                results, 
                layer_name, 
                n_neurons_to_show=neurons_per_layer,
                figsize=(12, 9)
            )
            
            if fig is not None:
                # Save the figure
                save_path = os.path.join(save_dir, f"{layer_name}_overview.png")
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved {save_path}")
                plt.close(fig)  # Close to save memory
        
        except Exception as e:
            print(f"Error creating visualization for {layer_name}: {e}")

def compare_layers_specialization(results: Dict, position: int) -> None:
    """Compare how different layers respond to a specific position."""
    
    layer_names = sorted(results['layer_matrices'].keys())
    n_layers = len(layer_names)
    
    fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(4 * ((n_layers + 1) // 2), 8))
    if n_layers == 1:
        axes = [axes]
    elif axes.ndim == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    row, col = position // results['grid_size'], position % results['grid_size']
    
    for i, layer_name in enumerate(layer_names):
        matrix = results['layer_matrices'][layer_name]
        position_activations = matrix[position, :]  # All neurons for this position
        
        # Remove NaN values
        valid_activations = position_activations[~np.isnan(position_activations)]
        
        if len(valid_activations) > 0:
            axes[i].hist(valid_activations, bins=50, alpha=0.7, color=f'C{i}')
            axes[i].set_title(f"{layer_name}\nPosition {position} [{row},{col}]")
            axes[i].set_xlabel("Normalized Activation")
            axes[i].set_ylabel("# Neurons")
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(layer_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f"Neuron Activation Distributions for Position {position}", 
                 fontsize=14, fontweight='bold', y=1.02)
    return fig

def create_layer_summary_statistics(results: Dict) -> Dict:
    """Create summary statistics for all layers."""
    
    summary = {}
    layer_names = sorted(results['layer_matrices'].keys())
    
    for layer_name in layer_names:
        matrix = results['layer_matrices'][layer_name]
        
        # Calculate coverage (non-NaN entries)
        valid_mask = ~np.isnan(matrix)
        coverage = np.mean(valid_mask) * 100
        
        # Calculate statistics for valid entries
        valid_data = matrix[valid_mask]
        
        if len(valid_data) > 0:
            summary[layer_name] = {
                'coverage_percent': coverage,
                'mean_activation': np.mean(valid_data),
                'std_activation': np.std(valid_data),
                'min_activation': np.min(valid_data),
                'max_activation': np.max(valid_data),
                'n_positions_with_data': np.sum(np.any(valid_mask, axis=1)),
                'n_neurons_with_data': np.sum(np.any(valid_mask, axis=0))
            }
        else:
            summary[layer_name] = {
                'coverage_percent': 0,
                'mean_activation': np.nan,
                'std_activation': np.nan,
                'min_activation': np.nan,
                'max_activation': np.nan,
                'n_positions_with_data': 0,
                'n_neurons_with_data': 0
            }
    
    return summary

def print_analysis_summary(results: Dict):
    """Print a comprehensive summary of the analysis."""
    print("\n" + "="*80)
    print("FFN POSITION ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"Grid Size: {results['grid_size']}x{results['grid_size']} ({results['n_positions']} positions)")
    print(f"Number of Layers: {results['n_layers']}")
    print(f"FFN Size per Layer: {results['ffn_size']} neurons")
    print(f"Total FFN Neurons: {results['n_layers'] * results['ffn_size']}")
    
    # Layer-wise summary
    summary = create_layer_summary_statistics(results)
    
    print(f"\nLayer-wise Coverage and Statistics:")
    print("-" * 60)
    print(f"{'Layer':<10} {'Coverage':<10} {'Positions':<10} {'Neurons':<10} {'Mean±Std':<15}")
    print("-" * 60)
    
    for layer_name in sorted(summary.keys()):
        stats = summary[layer_name]
        coverage = f"{stats['coverage_percent']:.1f}%"
        positions = f"{stats['n_positions_with_data']}/{results['n_positions']}"
        neurons = f"{stats['n_neurons_with_data']}/{results['ffn_size']}"
        mean_std = f"{stats['mean_activation']:.2f}±{stats['std_activation']:.2f}"
        
        print(f"{layer_name:<10} {coverage:<10} {positions:<10} {neurons:<10} {mean_std:<15}")
    
    # Find most specialized neurons across all layers
    print(f"\nTop Position-Specialized Neurons (across all layers):")
    print("-" * 60)
    
    all_specialized = []
    for layer_name in sorted(results['layer_matrices'].keys()):
        specialization = analyze_neuron_specialization(results, layer_name, top_k=5)
        for neuron in specialization['top_specialized_neurons']:
            neuron['layer'] = layer_name
            all_specialized.append(neuron)
    
    # Sort by concentration across all layers
    all_specialized.sort(key=lambda x: x['concentration'], reverse=True)
    
    print(f"{'Layer':<8} {'Neuron':<8} {'Max Pos':<8} {'Grid Loc':<10} {'Concentration':<12}")
    print("-" * 60)
    
    for neuron in all_specialized[:10]:  # Top 10
        layer = neuron['layer']
        neuron_idx = neuron['neuron_idx']
        max_pos = neuron['max_position']
        row, col = max_pos // results['grid_size'], max_pos % results['grid_size']
        grid_loc = f"[{row},{col}]"
        concentration = neuron['concentration']
        
        print(f"{layer:<8} {neuron_idx:<8} {max_pos:<8} {grid_loc:<10} {concentration:<12.3f}")

def main():
    """Main function to run the complete visualization analysis."""
    
    # Load results
    results_path = "ffn_position_analysis_results.npz"
    
    if not os.path.exists(results_path):
        print(f"Results file {results_path} not found!")
        print("Please run ffn_position_analysis.py first to generate the data.")
        return
    
    print("Loading FFN analysis results...")
    results = load_ffn_analysis_results(results_path)
    
    # Print comprehensive summary
    print_analysis_summary(results)
    
    # Create traditional heatmap visualizations for all layers
    print("\nCreating traditional layer overview visualizations...")
    visualize_all_layers_overview(results, neurons_per_layer=12)
    
    # Create NEW maze overlay visualizations for all layers
    print("\nCreating maze overlay visualizations...")
    visualize_all_layers_with_maze(results, neurons_per_layer=12)
    
    # Example: Compare layers for a specific position
    print("\nCreating position comparison across layers...")
    example_position = 27  # Middle-ish position
    fig = compare_layers_specialization(results, example_position)
    fig.savefig("ffn_visualizations/position_comparison_example.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("\nVisualization complete! Check the 'ffn_visualizations/' directory for results.")
    print("New maze overlay files are saved with '_maze_overlay.png' suffix.")
    
    return results

if __name__ == "__main__":
    results = main() 