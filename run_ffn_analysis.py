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
from visualize_ffn_analysis import main as visualize_main

def run_complete_analysis(max_samples: int = 100, 
                         completion_length: int = 8,
                         model_path: str = 'out-maze-nav',
                         grid_size: int = 8,
                         normalization: str = 'z_score',
                         skip_collection: bool = False,
                         skip_visualization: bool = False):
    """
    Run the complete FFN position analysis pipeline.
    
    Args:
        max_samples: Number of validation samples to process
        completion_length: Number of tokens to generate per sample
        model_path: Path to the trained model directory
        grid_size: Size of the maze grid
        normalization: Normalization method ('z_score', 'min_max', 'none')
        skip_collection: Skip data collection if results already exist
        skip_visualization: Skip visualization step
    """
    
    results_file = "ffn_position_analysis_results.npz"
    
    print("="*80)
    print("FFN POSITION ANALYSIS PIPELINE")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Max samples: {max_samples}")
    print(f"Completion length: {completion_length}")
    print(f"Normalization: {normalization}")
    print()
    
    # Step 1: Data Collection
    if not skip_collection or not os.path.exists(results_file):
        print("STEP 1: Collecting FFN activations...")
        print("-" * 40)
        
        try:
            # Initialize collector
            collector = FFNActivationCollector(
                model_path=model_path,
                grid_size=grid_size
            )
            
            # Collect activations
            collector.collect_activations(
                max_samples=max_samples,
                completion_length=completion_length
            )
            
            # Create normalized matrices
            matrices = collector.normalize_and_create_matrices(normalization=normalization)
            
            # Save results
            save_path = collector.save_results(matrices, save_path=results_file)
            print(f"✓ Data collection complete! Results saved to {save_path}")
            
        except Exception as e:
            print(f"✗ Error during data collection: {e}")
            return False
    else:
        print("STEP 1: Skipping data collection (results file exists)")
    
    print()
    
    # Step 2: Visualization
    if not skip_visualization:
        print("STEP 2: Creating visualizations...")
        print("-" * 40)
        
        try:
            results = visualize_main()
            print("✓ Visualization complete!")
            
        except Exception as e:
            print(f"✗ Error during visualization: {e}")
            return False
    else:
        print("STEP 2: Skipping visualization")
    
    print()
    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved in: {results_file}")
    print("Visualizations saved in: ffn_visualizations/")
    print()
    print("To load results later:")
    print(f"  import numpy as np")
    print(f"  data = np.load('{results_file}')")
    print(f"  layer_0_matrix = data['layer_0']  # Shape: (64, 768)")
    print()
    
    return True

def main():
    """Main function with command line argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Run FFN position analysis for maze navigation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--max-samples', type=int, default=100,
                       help='Number of validation samples to process')
    parser.add_argument('--completion-length', type=int, default=8,
                       help='Number of tokens to generate per sample')
    parser.add_argument('--model-path', type=str, default='out-maze-nav',
                       help='Path to the trained model directory')
    parser.add_argument('--grid-size', type=int, default=8,
                       help='Size of the maze grid')
    parser.add_argument('--normalization', type=str, default='z_score',
                       choices=['z_score', 'min_max', 'none'],
                       help='Normalization method for activations')
    parser.add_argument('--skip-collection', action='store_true',
                       help='Skip data collection if results file exists')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='Skip visualization step')
    parser.add_argument('--quick', action='store_true',
                       help='Quick run with fewer samples (for testing)')
    
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        args.max_samples = 20
        args.completion_length = 5
        print("Quick mode enabled: reduced samples and completion length")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist!")
        print("Please train the maze navigation model first or specify correct path.")
        return 1
    
    # Check if checkpoint exists
    ckpt_path = os.path.join(args.model_path, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"Error: Model checkpoint '{ckpt_path}' does not exist!")
        return 1
    
    # Run the analysis
    success = run_complete_analysis(
        max_samples=args.max_samples,
        completion_length=args.completion_length,
        model_path=args.model_path,
        grid_size=args.grid_size,
        normalization=args.normalization,
        skip_collection=args.skip_collection,
        skip_visualization=args.skip_visualization
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 