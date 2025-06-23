import numpy as np
import matplotlib.pyplot as plt
from neuron_heatmap_visualizer import plot_neuron_heatmaps, plot_single_neuron_heatmap
from typing import Dict, List, Optional
import os

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
    selected_matrix = matrix[:, neuron_indices]  # Shape: (64, len(neuron_indices))
    
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
    
    # Create visualizations for all layers
    print("\nCreating layer overview visualizations...")
    visualize_all_layers_overview(results, neurons_per_layer=12)
    
    # Example: Compare layers for a specific position
    print("\nCreating position comparison across layers...")
    example_position = 27  # Middle-ish position
    fig = compare_layers_specialization(results, example_position)
    fig.savefig("ffn_visualizations/position_comparison_example.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("\nVisualization complete! Check the 'ffn_visualizations/' directory for results.")
    
    return results

if __name__ == "__main__":
    results = main() 