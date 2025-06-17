"""
Enhanced Maze Navigation Visualizer

Specialized visualization tools for maze navigation GPT training data.
Provides detailed visualizations with node labels, paths, and sequence analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from matplotlib.colors import ListedColormap
import sys

# Add maze path for imports
maze_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'maze')
sys.path.append(maze_path)
from maze_visualizer import MazeVisualizer


class MazeNavVisualizer(MazeVisualizer):
    """Enhanced visualizer for maze navigation training data."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 12), dpi: int = 100):
        super().__init__(figsize, dpi)
        
        # Enhanced color scheme for navigation analysis
        self.nav_colors = {
            'sequence_highlight': '#FF6B6B',  # Coral red
            'path_numbers': '#4ECDC4',        # Teal
            'direction_arrows': '#45B7D1',    # Blue
            'step_markers': '#96CEB4',        # Mint green
        }
    
    def visualize_navigation_sequence(self, maze_data: Dict, sequence_data: Dict, 
                                    save_path: str = None, show_plot: bool = True) -> plt.Figure:
        """
        Create a detailed visualization of a navigation sequence.
        
        Args:
            maze_data: Maze data dictionary
            sequence_data: Sequence data with path, moves, etc.
            save_path: Path to save visualization
            show_plot: Whether to display the plot
        
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(20, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.3)
        
        # Left panel: Maze with labeled path
        ax_maze = fig.add_subplot(gs[0, 0])
        self._draw_detailed_navigation(ax_maze, maze_data, sequence_data)
        
        # Right panel: Sequence breakdown
        ax_seq = fig.add_subplot(gs[0, 1])
        self._draw_sequence_breakdown(ax_seq, sequence_data)
        
        # Overall title
        start_node = sequence_data['start_node']
        end_node = sequence_data['end_node']
        path_length = len(sequence_data['path'])
        
        fig.suptitle(
            f"Navigation Sequence Analysis: {start_node} → {end_node}\n"
            f"Path Length: {path_length} nodes, Sequence Length: {sequence_data['length']} tokens",
            fontsize=16, y=0.95
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            print(f"Navigation sequence visualization saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def _draw_detailed_navigation(self, ax, maze_data: Dict, sequence_data: Dict):
        """Draw maze with detailed navigation path and step numbers."""
        size = maze_data['config']['size']
        
        # Draw basic maze structure
        self._draw_maze_structure(ax, maze_data, show_grid=True, show_node_labels=True)
        
        # Draw path with step numbers
        path = sequence_data['path']
        if len(path) > 1:
            # Draw path line
            path_coords = []
            for node_id in path:
                row, col = node_id // size, node_id % size
                path_coords.append((2*col + 1, 2*row + 1))
            
            x_coords = [coord[0] for coord in path_coords]
            y_coords = [coord[1] for coord in path_coords]
            
            # Draw thick path line
            ax.plot(x_coords, y_coords, 
                   color=self.nav_colors['sequence_highlight'], 
                   linewidth=4, alpha=0.8, zorder=5)
            
            # Add step numbers on path
            for i, (x, y) in enumerate(path_coords):
                # Step number with background circle
                circle = patches.Circle((x, y), 0.3, 
                                      facecolor=self.nav_colors['step_markers'], 
                                      edgecolor='white', linewidth=2, zorder=6)
                ax.add_patch(circle)
                
                ax.text(x, y, str(i), ha='center', va='center', 
                       fontsize=8, weight='bold', color='black', zorder=7)
            
            # Add directional arrows between steps
            for i in range(len(path_coords) - 1):
                x1, y1 = path_coords[i]
                x2, y2 = path_coords[i + 1]
                
                # Calculate arrow position (75% along the segment)
                arrow_x = x1 + 0.75 * (x2 - x1)
                arrow_y = y1 + 0.75 * (y2 - y1)
                
                dx = x2 - x1
                dy = y2 - y1
                
                if abs(dx) > 0.1 or abs(dy) > 0.1:  # Avoid zero-length arrows
                    ax.arrow(arrow_x - 0.15*dx, arrow_y - 0.15*dy, 
                            0.3*dx, 0.3*dy,
                            head_width=0.15, head_length=0.1,
                            fc=self.nav_colors['direction_arrows'], 
                            ec=self.nav_colors['direction_arrows'],
                            alpha=0.9, zorder=6)
        
        ax.set_title("Maze with Navigation Path", fontsize=14, pad=20)
    
    def _draw_sequence_breakdown(self, ax, sequence_data: Dict):
        """Draw a breakdown of the navigation sequence."""
        ax.axis('off')
        
        # Extract sequence information
        sequence = sequence_data['sequence']
        path = sequence_data['path']
        
        # Create text breakdown
        y_pos = 0.95
        line_height = 0.05
        
        # Title
        ax.text(0.05, y_pos, "Sequence Breakdown:", fontsize=14, weight='bold', 
               transform=ax.transAxes)
        y_pos -= line_height * 1.5
        
        # Start and end
        ax.text(0.05, y_pos, f"Start Node: {sequence_data['start_node']}", 
               fontsize=12, transform=ax.transAxes)
        y_pos -= line_height
        
        ax.text(0.05, y_pos, f"End Node: {sequence_data['end_node']}", 
               fontsize=12, transform=ax.transAxes)
        y_pos -= line_height * 1.5
        
        # Path breakdown
        ax.text(0.05, y_pos, "Path Steps:", fontsize=12, weight='bold', 
               transform=ax.transAxes)
        y_pos -= line_height
        
        # Show path with step numbers
        for i, node in enumerate(path):
            step_text = f"Step {i}: Node {node}"
            if i < len(path) - 1:
                # Add direction if available
                if 'moves' in sequence_data and i < len(sequence_data['moves']):
                    direction = sequence_data['moves'][i]
                    step_text += f" → {direction}"
            
            ax.text(0.1, y_pos, step_text, fontsize=10, transform=ax.transAxes)
            y_pos -= line_height * 0.8
            
            if y_pos < 0.2:  # Stop if we run out of space
                ax.text(0.1, y_pos, "... (truncated)", fontsize=10, 
                       style='italic', transform=ax.transAxes)
                break
        
        # Add sequence format at bottom
        y_pos = 0.15
        ax.text(0.05, y_pos, "GPT Sequence Format:", fontsize=12, weight='bold', 
               transform=ax.transAxes)
        y_pos -= line_height
        
        # Show first few tokens of sequence
        seq_preview = str(sequence[:8]) + "..." if len(sequence) > 8 else str(sequence)
        ax.text(0.05, y_pos, seq_preview, fontsize=10, family='monospace',
               transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", 
                                               facecolor='lightgray', alpha=0.5))
    
    def create_training_overview(self, dataset: Dict, save_path: str = None, 
                               show_plot: bool = True) -> plt.Figure:
        """Create an overview visualization of the training dataset."""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Maze structure (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        maze_data = dataset['maze_data']
        self._draw_maze_structure(ax1, maze_data, show_grid=True, show_node_labels=True)
        ax1.set_title(f"Maze Structure ({maze_data['config']['size']}x{maze_data['config']['size']})", 
                     fontsize=12)
        
        # 2. Path length distribution (top-middle)
        ax2 = fig.add_subplot(gs[0, 1])
        train_lengths = [seq['length'] for seq in dataset['train']['sequences']]
        test_lengths = [seq['length'] for seq in dataset['test']['sequences']]
        
        ax2.hist(train_lengths, bins=20, alpha=0.7, label='Train', color='blue')
        ax2.hist(test_lengths, bins=20, alpha=0.7, label='Test', color='orange')
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Sequence Length Distribution')
        ax2.legend()
        
        # 3. Dataset statistics (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        stats_text = f"""Dataset Statistics:
        
Maze Size: {maze_data['config']['size']}x{maze_data['config']['size']}
Total Nodes: {maze_data['num_nodes']}
Vocabulary Size: {dataset['config']['vocab_size']}

Training Sequences: {dataset['stats']['train_pairs']:,}
Test Sequences: {dataset['stats']['test_pairs']:,}
Total Pairs: {dataset['stats']['total_pairs']:,}

Average Train Length: {dataset['stats']['avg_train_length']:.1f}
Average Test Length: {dataset['stats']['avg_test_length']:.1f}
Max Sequence Length: {dataset['stats']['max_sequence_length']}

Seed: {dataset['config']['seed']}
"""
        
        ax3.text(0.05, 0.95, stats_text, fontsize=11, family='monospace',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
        
        # 4. Sample paths visualization (bottom row)
        # Get sample sequences of different lengths
        train_seqs = dataset['train']['sequences']
        short_seq = next((s for s in train_seqs if s['length'] <= 10), None)
        medium_seq = next((s for s in train_seqs if 10 < s['length'] <= 30), None)
        long_seq = next((s for s in train_seqs if s['length'] > 30), None)
        
        samples = [seq for seq in [short_seq, medium_seq, long_seq] if seq is not None]
        
        for i, seq in enumerate(samples[:3]):
            ax = fig.add_subplot(gs[1, i])
            
            # Create temp maze data for this sample
            temp_maze_data = maze_data.copy()
            temp_maze_data['config']['start_pos'] = [
                seq['start_node'] // maze_data['config']['size'],
                seq['start_node'] % maze_data['config']['size']
            ]
            temp_maze_data['config']['end_pos'] = [
                seq['end_node'] // maze_data['config']['size'],
                seq['end_node'] % maze_data['config']['size']
            ]
            
            # Draw maze with path
            self._draw_maze_structure(ax, temp_maze_data, show_grid=False, show_node_labels=False)
            self._draw_solution_path(ax, temp_maze_data, seq['path'], 'sample', show_arrows=True)
            
            ax.set_title(f"Sample {i+1}: {seq['start_node']}→{seq['end_node']}\nLength: {seq['length']}", 
                        fontsize=10)
        
        fig.suptitle("Maze Navigation Training Dataset Overview", fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            print(f"Training overview saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig


def create_comprehensive_visualization(dataset_path: str, output_dir: str = "maze_nav_data"):
    """Create comprehensive visualizations for maze navigation dataset."""
    print("Creating comprehensive visualizations...")
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    visualizer = MazeNavVisualizer(figsize=(16, 16))
    
    # 1. Training overview
    overview_path = os.path.join(output_dir, "training_overview.png")
    visualizer.create_training_overview(dataset, save_path=overview_path, show_plot=False)
    
    # 2. Detailed sequence examples
    train_sequences = dataset['train']['sequences']
    
    # Pick diverse examples
    examples = []
    if train_sequences:
        # Short path
        short_paths = [s for s in train_sequences if len(s['path']) <= 5]
        if short_paths:
            examples.append(('short', short_paths[0]))
        
        # Medium path
        medium_paths = [s for s in train_sequences if 5 < len(s['path']) <= 15]
        if medium_paths:
            examples.append(('medium', medium_paths[0]))
        
        # Long path
        long_paths = [s for s in train_sequences if len(s['path']) > 15]
        if long_paths:
            examples.append(('long', long_paths[0]))
    
    # Create detailed visualizations for each example
    for category, seq_data in examples:
        detail_path = os.path.join(output_dir, f"navigation_detail_{category}.png")
        visualizer.visualize_navigation_sequence(
            dataset['maze_data'], seq_data, 
            save_path=detail_path, show_plot=False
        )
    
    print(f"Comprehensive visualizations saved to {output_dir}/")
    print("Generated files:")
    print(f"  - training_overview.png")
    for category, _ in examples:
        print(f"  - navigation_detail_{category}.png")


if __name__ == "__main__":
    # Example usage
    dataset_path = "maze_nav_data/maze_nav_dataset.json"
    if os.path.exists(dataset_path):
        create_comprehensive_visualization(dataset_path)
    else:
        print("Dataset not found. Please run maze_nav.py first.") 