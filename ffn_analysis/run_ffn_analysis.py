#!/usr/bin/env python3
"""
Complete FFN Position Analysis Pipeline

This script runs the full analysis pipeline:
1. Collect FFN activations when generating position tokens
2. Create normalized matrices for each layer
3. Visualize the results with heatmaps
4. Generate summary statistics

Usage:
    python run_ffn_analysis.py
"""

import os
import sys
import argparse
from ffn_position_analysis import FFNActivationCollector
from visualize_ffn_analysis import main as visualize_main, load_maze_data
from neuron_heatmap_visualizer import plot_single_neuron_heatmap, plot_neuron_heatmaps
import numpy as np
import matplotlib.pyplot as plt

def find_representative_neurons_and_save_images(matrices: dict, grid_size: int, 
                                              maze_data: dict = None,
                                              save_dir: str = "representative_neurons"):
    """
    For each position, find the most representative neuron and save its activation image.
    
    Args:
        matrices: Dictionary of layer matrices from analysis
        grid_size: Size of the maze grid
        maze_data: Dictionary containing maze data for overlay
        save_dir: Directory to save representative neuron images
    """
    print(f"\nFinding representative neurons for each position...")
    
    # Create grid-size specific directory
    grid_base_dir = f"grid_{grid_size}x{grid_size}"
    save_dir = os.path.join(grid_base_dir, save_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    n_positions = grid_size * grid_size
    representative_neurons = {}
    
    # Process each layer type
    for matrix_name, matrix in matrices.items():
        if matrix is None or matrix.size == 0:
            continue
            
        layer_dir = os.path.join(save_dir, matrix_name)
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir)
        
        layer_representatives = {}
        
        print(f"Processing {matrix_name} ({matrix.shape})...")
        
        # For each position, find the neuron with highest average activation
        for position in range(n_positions):
            position_row = matrix[position, :]  # All neurons for this position
            
            # Skip positions with no data
            if np.all(np.isnan(position_row)):
                continue
            
            # Find neuron with highest activation for this position
            valid_neurons = ~np.isnan(position_row)
            if np.any(valid_neurons):
                valid_activations = position_row[valid_neurons]
                max_activation_idx = np.argmax(valid_activations)
                # Map back to original neuron index
                neuron_indices = np.where(valid_neurons)[0]
                best_neuron_idx = neuron_indices[max_activation_idx]
                max_activation_value = valid_activations[max_activation_idx]
                
                layer_representatives[position] = {
                    'neuron_idx': best_neuron_idx,
                    'activation_value': max_activation_value,
                    'grid_pos': (position // grid_size, position % grid_size)
                }
                
                # Create and save heatmap for this neuron
                try:
                    neuron_activations = matrix[:, best_neuron_idx:best_neuron_idx+1]  # Shape: (n_positions, 1)
                    
                    fig = plot_single_neuron_heatmap(
                        neuron_activations,
                        neuron_idx=0,  # Index into the single neuron matrix
                        grid_size=grid_size,
                        maze_data=maze_data,
                        title=f"{matrix_name} - Pos {position} [{position//grid_size},{position%grid_size}]\nNeuron {best_neuron_idx} (act: {max_activation_value:.3f})",
                        figsize=(8, 6),
                        cmap='RdYlBu_r'
                    )
                    
                    # Save the image
                    save_path = os.path.join(layer_dir, f"pos_{position:02d}_neuron_{best_neuron_idx}.png")
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"Warning: Could not create image for position {position}, neuron {best_neuron_idx}: {e}")
        
        representative_neurons[matrix_name] = layer_representatives
        print(f"  Found {len(layer_representatives)} representative neurons")
    
    # Save summary information
    summary_path = os.path.join(save_dir, "representative_neurons_summary.npz")
    np.savez_compressed(summary_path, **representative_neurons)
    
    print(f"Representative neuron analysis complete!")
    print(f"Images saved in: {save_dir}/")
    print(f"Summary saved in: {summary_path}")
    
    return representative_neurons

def save_layer_heatmaps(matrices: dict, grid_size: int, maze_data: dict, grid_base_dir: str):
    """
    Generate and save heatmaps for all neurons in each layer.
    
    Args:
        matrices: Dictionary of layer matrices from analysis.
        grid_size: Size of the maze grid.
        maze_data: Dictionary containing maze data for overlay.
        grid_base_dir: Base directory for saving outputs for this grid size.
    """
    heatmap_dir = os.path.join(grid_base_dir, 'neuron_heatmaps')
    if not os.path.exists(heatmap_dir):
        os.makedirs(heatmap_dir)
    
    num_layers = 0
    # Determine the number of layers from the matrices keys
    while f'layer_{num_layers}_fc' in matrices or f'layer_{num_layers}_proj' in matrices:
        num_layers += 1

    print(f"  Found {num_layers} layers to process for heatmaps.")

    for i in range(num_layers):
        for layer_type in ['fc', 'proj']:
            matrix_key = f'layer_{i}_{layer_type}'
            data = matrices.get(matrix_key)
            
            if data is not None and data.size > 0:
                print(f"  Generating heatmap for {matrix_key}...")
                plot_neuron_heatmaps(
                    data, 
                    title_prefix=f'layer {i} {layer_type}', 
                    maze_data=maze_data, 
                    grid_size=grid_size, 
                    save_path=os.path.join(heatmap_dir, f'l{i}_{layer_type}_heatmap.png')
                )

def run_complete_analysis(max_samples: int = 100, 
                         task: str = 'maze',
                         model_path: str = 'out-maze-nav',
                         grid_size: int = 8,
                         normalization: str = 'z_score',
                         skip_collection: bool = False,
                         skip_heatmaps: bool = False,
                         skip_visualization: bool = False,
                         save_representative_neurons: bool = True):
    """
    Run the complete FFN position analysis pipeline.
    
    Args:
        max_samples: Number of validation samples to process
        task: Task to analyze ('maze', 'path')
        model_path: Path to the trained model directory
        grid_size: Size of the maze grid
        normalization: Normalization method ('z_score', 'min_max', 'none')
        skip_collection: Skip data collection if results already exist
        skip_heatmaps: Skip heatmap generation step
        skip_visualization: Skip visualization step
    """
    
    # Create grid-size specific directory for all outputs
    grid_base_dir = f"grid_{grid_size}x{grid_size}_{task}"
    if not os.path.exists(grid_base_dir):
        os.makedirs(grid_base_dir)
    
    results_file = os.path.join(grid_base_dir, "ffn_position_analysis_results.npz")
    
    print("="*80)
    print("FFN POSITION ANALYSIS PIPELINE")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Output directory: {grid_base_dir}/")
    print(f"Max samples: {max_samples}")
    print(f"Normalization: {normalization}")
    print()
    
    # Step 1: Data Collection
    matrices = None
    if not skip_collection or not os.path.exists(results_file):
        print("STEP 1: Collecting FFN activations...")
        print("-" * 40)
        
        try:
            collector = FFNActivationCollector(
                model_path=model_path, grid_size=grid_size, task=task
            )
            collector.collect_activations(max_samples=max_samples)
            matrices = collector.normalize_and_create_matrices(normalization=normalization)
            save_path = collector.save_results(matrices, save_path=results_file)
            print(f"✓ Data collection complete! Results saved to {save_path}")
            
        except Exception as e:
            print(f"✗ Error during data collection: {e}")
            return False
    else:
        print("STEP 1: Skipping data collection (results file exists)")

    # Load matrices from file if not already in memory
    if matrices is None:
        if os.path.exists(results_file):
            print("  Loading activation matrices from file...")
            data = np.load(results_file, allow_pickle=True)
            matrices = {key: data[key] for key in data.keys() if key.startswith('layer_')}
        else:
            print(f"✗ Error: Results file not found at '{results_file}'. Please run collection first.")
            return False
    
    print()

    # Load maze data once for all subsequent steps
    print("  Loading maze data for visualizations...")
    maze_data = load_maze_data(grid_size)
    if maze_data is None:
        print("  Warning: Could not load maze data. Visualizations will not have maze overlays.")
    print()
    
    # Step 2: Generate Layer Heatmaps
    if not skip_heatmaps:
        print("STEP 2: Generating layer heatmaps...")
        print("-" * 40)
        try:
            save_layer_heatmaps(matrices, grid_size, maze_data, grid_base_dir)
            print("✓ Layer heatmaps saved.")
        except Exception as e:
            print(f"✗ Error during heatmap generation: {e}")
            return False
    else:
        print("STEP 2: Skipping layer heatmaps generation")

    print()

    # Step 3: Main Visualization
    if not skip_visualization:
        print("STEP 3: Creating main visualizations...")
        print("-" * 40)
        
        try:
            original_cwd = os.getcwd()
            os.chdir(grid_base_dir)
            visualize_main()
            os.chdir(original_cwd)
            print("✓ Main visualization complete!")
            
        except Exception as e:
            print(f"✗ Error during main visualization: {e}")
            # Restore CWD on error
            if 'original_cwd' in locals() and os.getcwd() != original_cwd:
                os.chdir(original_cwd)
            return False
    else:
        print("STEP 3: Skipping main visualization")
    
    print()
    
    # Step 4: Representative Neurons Analysis
    if save_representative_neurons:
        print("STEP 4: Finding representative neurons...")
        print("-" * 40)
        
        try:
            find_representative_neurons_and_save_images(
                matrices, grid_size, maze_data=maze_data,
                save_dir=os.path.join("..", grid_base_dir, "representative_neurons")
            )
            print("✓ Representative neurons analysis complete!")
            
        except Exception as e:
            print(f"✗ Error during representative neurons analysis: {e}")
            return False
    else:
        print("STEP 4: Skipping representative neurons analysis")
    
    print()
    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved in: {results_file}")
    print(f"Heatmaps saved in: {os.path.join(grid_base_dir, 'neuron_heatmaps')}/")
    print(f"Visualizations saved in: {os.path.join(grid_base_dir, 'ffn_visualizations')}/")
    if save_representative_neurons:
        print(f"Representative neurons saved in: {os.path.join(grid_base_dir, 'representative_neurons')}/")
    print()
    print("To load results later:")
    print(f"  import numpy as np")
    print(f"  data = np.load('{results_file}')")
    print(f"  layer_0_fc_matrix = data['layer_0_fc']  # First linear layer")
    print(f"  layer_0_proj_matrix = data['layer_0_proj']  # Second linear layer")
    print()
    
    return True

def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the complete FFN position analysis pipeline."
    )
    
    parser.add_argument('--max-samples', type=int, default=1000,
                        help="Number of validation samples to process")
    parser.add_argument('--grid-size', type=int, default=8,
                        help="Size of the maze grid (e.g., 8 for 8x8)")
    parser.add_argument('--model-path', type=str, default=os.path.join('..', 'out-maze-nav'),
                        help="Path to the trained model directory")
    parser.add_argument('--task', type=str, default='maze',
                        choices=['maze', 'path'],
                        help="Task to analyze")
    parser.add_argument('--normalization', type=str, default='z_score',
                        choices=['z_score', 'min_max', 'none'],
                        help="Normalization method for activation matrices")
    parser.add_argument('--skip-collection', action='store_true',
                        help="Skip data collection and use existing results file")
    parser.add_argument('--skip-heatmaps', action='store_true',
                        help="Skip the layer heatmaps generation step")
    parser.add_argument('--skip-visualization', action='store_true',
                        help="Skip the main visualization step")
    parser.add_argument('--no-representative-neurons', action='store_false', dest='save_representative_neurons',
                        help="Do not run the representative neurons analysis")

    args = parser.parse_args()
    
    # Start the pipeline
    run_complete_analysis(
        max_samples=args.max_samples,
        grid_size=args.grid_size,
        model_path=args.model_path,
        task=args.task,
        normalization=args.normalization,
        skip_collection=args.skip_collection,
        skip_heatmaps=args.skip_heatmaps,
        skip_visualization=args.skip_visualization,
        save_representative_neurons=args.save_representative_neurons
    )

if __name__ == "__main__":
    main() 