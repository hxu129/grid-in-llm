"""
Path Visualizer for Maze Navigation

Visualizes generated paths from the maze navigation GPT model on the actual maze.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import pickle
import os
from typing import List, Dict, Tuple, Optional, Union


class PathVisualizer:
    """Visualizes generated paths on maze grids."""
    
    def __init__(self, maze_size, maze_data_dir: str = "data/maze/maze_nav_data"):
        """
        Initialize the path visualizer.
        
        Args:
            maze_size: Size of the maze (e.g., 8, 12, 15, 20)
            maze_data_dir: Directory containing maze data files
        """
        self.maze_data_dir = maze_data_dir
        self.maze_size = maze_size
        self.maze_data = None
        self.meta = None
        
        # Load maze data and metadata
        self._load_maze_data()
        
        # Colors for visualization
        self.colors = {
            'wall': '#2C3E50',        # Dark blue-gray
            'path': '#FFFFFF',        # White
            'start': '#E74C3C',       # Red
            'end': '#27AE60',         # Green
            'generated_path': '#3498DB',  # Blue
            'step_numbers': '#F39C12',    # Orange
            'current_pos': '#9B59B6'      # Purple
        }
    
    def _load_maze_data(self):
        """Load maze data and metadata from files based on maze size."""
        try:
            # Load metadata
            meta_path = os.path.join(self.maze_data_dir, f'meta_{self.maze_size}.pkl')
            with open(meta_path, 'rb') as f:
                self.meta = pickle.load(f)
            
            # Load maze dataset
            dataset_path = os.path.join(self.maze_data_dir, f'maze_nav_dataset_{self.maze_size}.json')
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
                self.maze_data = dataset['maze_data']
            
            print(f"Loaded maze data: {self.maze_size}x{self.maze_size} maze with {self.maze_data['num_nodes']} nodes")
            
        except Exception as e:
            print(f"Error loading maze data for size {self.maze_size}: {e}")
            print("Attempting to generate maze data on-the-fly...")
            try:
                self._generate_maze_data()
            except Exception as gen_error:
                print(f"Failed to generate maze data: {gen_error}")
                raise RuntimeError(f"Could not load or generate maze data for size {self.maze_size}")
    
    def _generate_maze_data(self):
        """Generate maze data on-the-fly for the specified size."""
        try:
            # Import the maze generation modules
            import sys
            maze_module_path = os.path.join(os.path.dirname(self.maze_data_dir), '.')
            if maze_module_path not in sys.path:
                sys.path.insert(0, maze_module_path)
            
            from maze_nav import MazeNavDataGenerator, MazeNavConfig
            
            # Generate maze data
            config = MazeNavConfig(maze_size=self.maze_size, seed=42, max_pairs=100)  # Limited pairs for visualization
            generator = MazeNavDataGenerator(config)
            dataset = generator.generate_data()
            
            # Extract maze data
            self.maze_data = dataset['maze_data']
            
            print(f"Generated maze data: {self.maze_size}x{self.maze_size} maze with {self.maze_data['num_nodes']} nodes")
            
        except ImportError as e:
            raise ImportError(f"Could not import maze generation modules: {e}. "
                            f"Please ensure the maze modules are available.")
        except Exception as e:
            raise RuntimeError(f"Failed to generate maze data: {e}")
    
    def parse_generated_path(self, path_string: str, skip_first_n: int = 2) -> Tuple[List[int], List[str]]:
        """
        Parse a generated path string into nodes and directions.
        
        Args:
            path_string: Generated path string like "19 28 19 left 18 left 17 down 27..."
            skip_first_n: Number of tokens to skip from the beginning (default 2)
        
        Returns:
            Tuple of (node_sequence, direction_sequence)
        """
        tokens = path_string.strip().split()
        
        # Skip the first n tokens as instructed
        tokens = tokens[skip_first_n:]
        
        nodes = []
        directions = []
        
        # Parse alternating pattern: node, direction, node, direction, ...
        for i, token in enumerate(tokens):
            if i % 2 == 0:  # Even indices are nodes
                try:
                    node_id = int(token)
                    nodes.append(node_id)
                except ValueError:
                    # If we can't parse as int, we might have hit the end
                    break
            else:  # Odd indices are directions
                if token in ['up', 'down', 'left', 'right']:
                    directions.append(token)
                else:
                    # If we hit an unknown direction, stop parsing
                    break
        
        return nodes, directions
    
    def node_to_position(self, node_id: int) -> Tuple[int, int]:
        """Convert node ID to (row, col) position in the maze."""
        if self.maze_size is None:
            raise ValueError("Maze size not loaded")
        return node_id // self.maze_size, node_id % self.maze_size
    
    def position_to_node(self, row: int, col: int) -> int:
        """Convert (row, col) position to node ID."""
        if self.maze_size is None:
            raise ValueError("Maze size not loaded")
        return row * self.maze_size + col
    
    def create_maze_grid(self) -> np.ndarray:
        """Create a 2D grid representation of the maze for visualization."""
        if self.maze_data is None:
            raise ValueError("Maze data not loaded")
        
        # Create grid: 2*size+1 x 2*size+1 to include walls
        grid_size = 2 * self.maze_size + 1
        grid = np.ones((grid_size, grid_size))  # 1 = wall, 0 = path
        
        # Mark all valid nodes as paths
        for node_id in range(self.maze_data['num_nodes']):
            row, col = self.node_to_position(node_id)
            grid_row, grid_col = 2 * row + 1, 2 * col + 1
            grid[grid_row, grid_col] = 0  # 0 = path
        
        # Add connections between adjacent nodes
        adj_matrix = np.array(self.maze_data['adjacency_matrix'])
        for i in range(self.maze_data['num_nodes']):
            for j in range(i + 1, self.maze_data['num_nodes']):
                if adj_matrix[i][j] == 1:  # Connected
                    row1, col1 = self.node_to_position(i)
                    row2, col2 = self.node_to_position(j)
                    
                    # Find the wall cell between these two nodes
                    wall_row = row1 + row2 + 1  # +1 because grid is offset
                    wall_col = col1 + col2 + 1
                    
                    if 0 <= wall_row < grid_size and 0 <= wall_col < grid_size:
                        grid[wall_row, wall_col] = 0  # Remove wall
        
        return grid
    
    def visualize_path(self, path_string: str, title: str = None, 
                      save_path: str = None, show_plot: bool = True,
                      figsize: Tuple[int, int] = (12, 12),
                      show_node_labels: bool = True) -> plt.Figure:
        """
        Visualize a generated path on the maze.
        
        Args:
            path_string: Generated path string
            title: Title for the plot
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            figsize: Figure size
            show_node_labels: Whether to show node ID labels on each grid cell
        
        Returns:
            matplotlib Figure object
        """
        if self.maze_data is None:
            raise ValueError("Maze data not loaded")
        
        # Parse the path
        nodes, directions = self.parse_generated_path(path_string)
        
        if not nodes:
            raise ValueError("No valid nodes found in path string")
        
        print(f"Parsed path: {len(nodes)} nodes, {len(directions)} directions")
        print(f"Start node: {nodes[0]}, End node: {nodes[-1]}")
        
        # Create the maze grid
        grid = self.create_maze_grid()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Display the maze
        ax.imshow(grid, cmap='gray_r', origin='upper')
        
        # Add node ID labels on each valid node
        if show_node_labels:
            for node_id in range(self.maze_data['num_nodes']):
                row, col = self.node_to_position(node_id)
                # Convert to grid coordinates
                grid_row, grid_col = 2 * row + 1, 2 * col + 1
                
                # Add text label with node ID
                ax.text(grid_col, grid_row, str(node_id), 
                       ha='center', va='center', 
                       fontsize=9 if self.maze_size <= 24 else 4, color='black', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", 
                               facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=0.5),
                       zorder=3)
        
        # Plot the generated path
        path_coords = []
        for node_id in nodes:
            row, col = self.node_to_position(node_id)
            # Convert to grid coordinates (add 1 for offset)
            grid_row, grid_col = 2 * row + 1, 2 * col + 1
            path_coords.append((grid_col, grid_row))  # matplotlib uses (x, y)
        
        # Draw path line
        if len(path_coords) > 1:
            x_coords = [coord[0] for coord in path_coords]
            y_coords = [coord[1] for coord in path_coords]
            
            ax.plot(x_coords, y_coords, 
                   color=self.colors['generated_path'], 
                   linewidth=3, alpha=0.8, zorder=5)
        
        # Mark start and end points
        if path_coords:
            start_x, start_y = path_coords[0]
            end_x, end_y = path_coords[-1]
            
            # Start point (larger circle)
            ax.scatter(start_x, start_y, 
                      color=self.colors['start'], 
                      s=200, zorder=6, 
                      edgecolor='white', linewidth=2,
                      label=f'Start (Node {nodes[0]})')
            
            # End point (larger circle)
            ax.scatter(end_x, end_y, 
                      color=self.colors['end'], 
                      s=200, zorder=6,
                      edgecolor='white', linewidth=2,
                      label=f'End (Node {nodes[-1]})')
        
        # Add step numbers on path (for shorter paths)
        if len(nodes) <= 20:  # Only show numbers for short paths
            for i, (x, y) in enumerate(path_coords):
                circle = patches.Circle((x, y), 0.3, 
                                      facecolor=self.colors['step_numbers'], 
                                      edgecolor='white', linewidth=1, zorder=7)
                ax.add_patch(circle)
                
                ax.text(x, y, str(i), ha='center', va='center', 
                       fontsize=8, weight='bold', color='white', zorder=8)
        
        # Add directional arrows
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
                        head_width=0.4, head_length=0.3,
                        fc=self.colors['current_pos'], 
                        ec=self.colors['current_pos'],
                        alpha=0.7, zorder=6)
        
        # Formatting
        ax.set_xlim(-0.5, grid.shape[1] - 0.5)
        ax.set_ylim(grid.shape[0] - 0.5, -0.5)  # Flip y-axis
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Title and legend
        if title is None:
            title = f"Generated Path Visualization\nPath Length: {len(nodes)} nodes"
        ax.set_title(title, fontsize=14, pad=20)
        
        if len(nodes) <= 20:  # Only show legend for shorter paths
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        # Add path info as text
        info_text = f"Nodes: {nodes[:5]}{'...' if len(nodes) > 5 else ''}\n"
        info_text += f"Directions: {directions[:5]}{'...' if len(directions) > 5 else ''}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to: {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        
        return fig


# Example usage and utility functions
def visualize_generated_path(path_string: str, 
                           maze_size: int,
                           maze_data_dir: str = "data/maze/maze_nav_data",
                           title: str = None,
                           save_path: str = None,
                           show_node_labels: bool = True):
    """
    Convenience function to visualize a generated path.
    
    Args:
        path_string: Generated path string from the model
        maze_size: Size of the maze (e.g., 8, 12, 15, 20)
        maze_data_dir: Directory containing maze data
        title: Optional title for the plot
        save_path: Optional path to save the figure
        show_node_labels: Whether to show node ID labels on each grid cell
    """
    visualizer = PathVisualizer(maze_size=maze_size, maze_data_dir=maze_data_dir)
    return visualizer.visualize_path(path_string, title=title, save_path=save_path, 
                                   show_node_labels=show_node_labels)


if __name__ == "__main__":
    # Example usage
    example_path = "19 28 19 left 18 left 17 down 27 down 37 left 36 up 26 up 16 up 6 left 5"
    
    # Example with default 15x15 maze
    visualizer = PathVisualizer(maze_size=15)
    visualizer.visualize_path(
        example_path,
        title="Example Generated Path (15x15 Maze)",
        save_path="example_path_visualization_15x15.png",
        show_node_labels=True
    )
    
    # Example with 8x8 maze
    try:
        visualizer_8x8 = PathVisualizer(maze_size=8)
        # You would use a path appropriate for 8x8 maze here
        example_path_8x8 = "0 right 1 down 9 right 10"  # Example for 8x8
        visualizer_8x8.visualize_path(
            example_path_8x8,
            title="Example Generated Path (8x8 Maze)",
            save_path="example_path_visualization_8x8.png",
            show_node_labels=True
        )
    except Exception as e:
        print(f"Could not visualize 8x8 maze: {e}")
    
    # Using the convenience function
    visualize_generated_path(
        example_path,
        maze_size=15,
        title="Using Convenience Function",
        show_node_labels=True
    ) 