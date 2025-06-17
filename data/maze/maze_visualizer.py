"""
Maze Visualization System

Provides comprehensive visualization tools for mazes and their solutions.
Supports multiple algorithms, path overlays, and batch visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec


class MazeVisualizer:
    """
    Comprehensive maze visualization system.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 10), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        
        # Color schemes
        self.colors = {
            'wall': '#2C3E50',           # Dark blue-gray
            'passage': '#ECF0F1',        # Light gray
            'start': '#E74C3C',          # Red
            'end': '#27AE60',            # Green
            'bfs_path': '#3498DB',       # Blue
            'astar_path': '#E67E22',     # Orange  
            'dfs_path': '#9B59B6',       # Purple
            'grid_lines': '#BDC3C7',     # Light gray
            'node_labels': '#34495E'     # Dark gray
        }
        
        # Algorithm styles
        self.path_styles = {
            'bfs': {'color': self.colors['bfs_path'], 'linewidth': 3, 'alpha': 0.8, 'label': 'BFS (Optimal)'},
            'astar': {'color': self.colors['astar_path'], 'linewidth': 3, 'alpha': 0.8, 'label': 'A* (Optimal)'},
            'dfs': {'color': self.colors['dfs_path'], 'linewidth': 3, 'alpha': 0.8, 'label': 'DFS (Complete)'}
        }
    
    def _load_maze_data(self, maze_source: Union[str, Dict]) -> Dict:
        """Load maze data from file or dictionary."""
        if isinstance(maze_source, str):
            with open(maze_source, 'r') as f:
                return json.load(f)
        return maze_source
    
    def _get_maze_walls(self, maze_data: Dict) -> np.ndarray:
        """
        Convert adjacency matrix to wall representation for visualization.
        Returns a grid where True indicates walls.
        """
        size = maze_data['config']['size']
        adj_matrix = np.array(maze_data['adjacency_matrix'])
        
        # Create a larger grid for walls (2*size + 1)
        grid_size = 2 * size + 1
        walls = np.ones((grid_size, grid_size), dtype=bool)
        
        # Mark cell spaces as passages
        for row in range(size):
            for col in range(size):
                walls[2*row + 1, 2*col + 1] = False
        
        # Mark connections between cells
        for node1 in range(len(adj_matrix)):
            for node2 in range(node1 + 1, len(adj_matrix)):
                if adj_matrix[node1][node2] == 1:
                    # Get positions
                    row1, col1 = node1 // size, node1 % size
                    row2, col2 = node2 // size, node2 % size
                    
                    # Mark the wall between cells as passage
                    wall_row = row1 + row2 + 1
                    wall_col = col1 + col2 + 1
                    walls[wall_row, wall_col] = False
        
        return walls
    
    def _draw_maze_structure(self, ax, maze_data: Dict, show_grid: bool = True, show_node_labels: bool = False):
        """Draw the basic maze structure."""
        size = maze_data['config']['size']
        walls = self._get_maze_walls(maze_data)
        
        # Create color map
        cmap = ListedColormap([self.colors['passage'], self.colors['wall']])
        
        # Draw maze
        ax.imshow(walls, cmap=cmap, origin='lower')
        
        # Draw grid lines if requested
        if show_grid:
            for i in range(size + 1):
                ax.axhline(y=2*i - 0.5, color=self.colors['grid_lines'], linewidth=0.5, alpha=0.5)
                ax.axvline(x=2*i - 0.5, color=self.colors['grid_lines'], linewidth=0.5, alpha=0.5)
        
        # Add node labels if requested
        if show_node_labels:
            # Adaptive font size based on maze size
            if size <= 10:
                fontsize = 8
                bbox_props = dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8)
            elif size <= 20:
                fontsize = 6
                bbox_props = dict(boxstyle="round,pad=0.05", facecolor='white', alpha=0.7)
            else:
                fontsize = 4
                bbox_props = dict(boxstyle="round,pad=0.02", facecolor='white', alpha=0.6)
            
            for node_id in range(size * size):
                row, col = node_id // size, node_id % size
                ax.text(2*col + 1, 2*row + 1, str(node_id), 
                       ha='center', va='center', fontsize=fontsize, 
                       color=self.colors['node_labels'], weight='bold',
                       bbox=bbox_props, zorder=4)
        
        # Mark start and end positions (only if not showing node labels)
        if not show_node_labels:
            start_pos = maze_data['config']['start_pos']
            end_pos = maze_data['config']['end_pos']
            
            # Start position (S)
            ax.add_patch(patches.Circle((2*start_pos[1] + 1, 2*start_pos[0] + 1), 
                                       0.4, color=self.colors['start'], zorder=5))
            ax.text(2*start_pos[1] + 1, 2*start_pos[0] + 1, 'S', 
                   ha='center', va='center', fontsize=12, color='white', weight='bold', zorder=6)
            
            # End position (E)
            ax.add_patch(patches.Circle((2*end_pos[1] + 1, 2*end_pos[0] + 1), 
                                       0.4, color=self.colors['end'], zorder=5))
            ax.text(2*end_pos[1] + 1, 2*end_pos[0] + 1, 'E', 
                   ha='center', va='center', fontsize=12, color='white', weight='bold', zorder=6)
        
        # Set axis properties
        ax.set_xlim(-0.5, 2*size + 0.5)
        ax.set_ylim(-0.5, 2*size + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _draw_solution_path(self, ax, maze_data: Dict, solution_path: List[int], 
                           algorithm: str = 'bfs', show_arrows: bool = True):
        """Draw a solution path on the maze."""
        if not solution_path:
            return
        
        size = maze_data['config']['size']
        style = self.path_styles.get(algorithm, self.path_styles['bfs'])
        
        # Convert node IDs to grid coordinates
        path_coords = []
        for node_id in solution_path:
            row, col = node_id // size, node_id % size
            path_coords.append((2*col + 1, 2*row + 1))
        
        # Draw path lines
        if len(path_coords) > 1:
            x_coords = [coord[0] for coord in path_coords]
            y_coords = [coord[1] for coord in path_coords]
            
            ax.plot(x_coords, y_coords, 
                   color=style['color'], linewidth=style['linewidth'], 
                   alpha=style['alpha'], zorder=3, label=style['label'])
            
            # Add arrows to show direction
            if show_arrows and len(path_coords) > 1:
                for i in range(0, len(path_coords) - 1, max(1, len(path_coords) // 8)):
                    if i + 1 < len(path_coords):
                        dx = x_coords[i + 1] - x_coords[i]
                        dy = y_coords[i + 1] - y_coords[i]
                        if dx != 0 or dy != 0:  # Avoid zero-length arrows
                            ax.arrow(x_coords[i], y_coords[i], dx*0.3, dy*0.3,
                                   head_width=0.2, head_length=0.15, 
                                   fc=style['color'], ec=style['color'], 
                                   alpha=0.8, zorder=4)
    
    def visualize_maze(self, maze_source: Union[str, Dict], 
                      title: str = "Maze", 
                      save_path: Optional[str] = None,
                      show_grid: bool = True,
                      show_node_labels: bool = False,
                      show_plot: bool = True) -> plt.Figure:
        """
        Visualize a single maze without solution.
        
        Args:
            maze_source: Maze file path or maze data dictionary
            title: Plot title
            save_path: Path to save the visualization
            show_grid: Whether to show grid lines
            show_node_labels: Whether to show node IDs
            show_plot: Whether to display the plot
        
        Returns:
            matplotlib Figure object
        """
        maze_data = self._load_maze_data(maze_source)
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Draw maze structure
        self._draw_maze_structure(ax, maze_data, show_grid, show_node_labels)
        
        # Set title
        size = maze_data['config']['size']
        ax.set_title(f"{title}\nSize: {size}x{size}", fontsize=14, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            print(f"Maze visualization saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def visualize_solution(self, maze_source: Union[str, Dict], 
                          solution_path: List[int],
                          algorithm: str = 'bfs',
                          title: str = None,
                          save_path: Optional[str] = None,
                          show_grid: bool = True,
                          show_arrows: bool = True,
                          show_plot: bool = True) -> plt.Figure:
        """
        Visualize a maze with its solution path.
        
        Args:
            maze_source: Maze file path or maze data dictionary
            solution_path: List of node IDs representing the solution
            algorithm: Algorithm used ('bfs', 'astar', 'dfs')
            title: Plot title (auto-generated if None)
            save_path: Path to save the visualization
            show_grid: Whether to show grid lines
            show_arrows: Whether to show directional arrows
            show_plot: Whether to display the plot
        
        Returns:
            matplotlib Figure object
        """
        maze_data = self._load_maze_data(maze_source)
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Draw maze structure
        self._draw_maze_structure(ax, maze_data, show_grid)
        
        # Draw solution path
        self._draw_solution_path(ax, maze_data, solution_path, algorithm, show_arrows)
        
        # Set title
        if title is None:
            size = maze_data['config']['size']
            path_length = len(solution_path)
            title = f"Maze Solution ({algorithm.upper()})\nSize: {size}x{size}, Path Length: {path_length}"
        
        ax.set_title(title, fontsize=14, pad=20)
        
        # Add legend
        if solution_path:
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            print(f"Solution visualization saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def compare_algorithms(self, maze_source: Union[str, Dict],
                          solutions: Dict[str, List[int]],
                          title: str = None,
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> plt.Figure:
        """
        Compare multiple algorithm solutions on the same maze.
        
        Args:
            maze_source: Maze file path or maze data dictionary
            solutions: Dictionary mapping algorithm names to solution paths
            title: Plot title (auto-generated if None)
            save_path: Path to save the visualization
            show_plot: Whether to display the plot
        
        Returns:
            matplotlib Figure object
        """
        maze_data = self._load_maze_data(maze_source)
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Draw maze structure
        self._draw_maze_structure(ax, maze_data, show_grid=True)
        
        # Draw all solution paths
        for algorithm, path in solutions.items():
            if path:
                self._draw_solution_path(ax, maze_data, path, algorithm, show_arrows=False)
        
        # Set title
        if title is None:
            size = maze_data['config']['size']
            algorithms = list(solutions.keys())
            title = f"Algorithm Comparison\nSize: {size}x{size}, Algorithms: {', '.join(algorithms)}"
        
        ax.set_title(title, fontsize=14, pad=20)
        
        # Add legend
        if any(solutions.values()):
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            print(f"Algorithm comparison saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_maze_gallery(self, maze_files: List[str], 
                           cols: int = 3,
                           title: str = "Maze Gallery",
                           save_path: Optional[str] = None,
                           show_plot: bool = True) -> plt.Figure:
        """
        Create a gallery of multiple mazes.
        
        Args:
            maze_files: List of maze file paths
            cols: Number of columns in the gallery
            title: Gallery title
            save_path: Path to save the gallery
            show_plot: Whether to display the plot
        
        Returns:
            matplotlib Figure object
        """
        if not maze_files:
            raise ValueError("No maze files provided")
        
        rows = (len(maze_files) + cols - 1) // cols
        
        fig = plt.figure(figsize=(cols * 4, rows * 4), dpi=self.dpi)
        gs = gridspec.GridSpec(rows, cols, hspace=0.3, wspace=0.1)
        
        for idx, maze_file in enumerate(maze_files):
            row = idx // cols
            col = idx % cols
            
            ax = fig.add_subplot(gs[row, col])
            
            try:
                maze_data = self._load_maze_data(maze_file)
                self._draw_maze_structure(ax, maze_data, show_grid=False)
                
                # Set subplot title
                size = maze_data['config']['size']
                filename = os.path.basename(maze_file).replace('.json', '')
                ax.set_title(f"{filename}\n{size}x{size}", fontsize=10)
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\n{os.path.basename(maze_file)}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Error", fontsize=10)
        
        # Hide empty subplots
        for idx in range(len(maze_files), rows * cols):
            row = idx // cols
            col = idx % cols
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
        
        fig.suptitle(title, fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            print(f"Maze gallery saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig


def visualize_from_solution_data(solution_data: Dict, 
                                visualizer: MazeVisualizer = None,
                                save_dir: str = None) -> plt.Figure:
    """
    Create visualization from solution data (from maze_solver output).
    
    Args:
        solution_data: Solution data from solve_maze_from_file
        visualizer: MazeVisualizer instance (creates new if None)
        save_dir: Directory to save visualization
    
    Returns:
        matplotlib Figure object
    """
    if visualizer is None:
        visualizer = MazeVisualizer()
    
    if not solution_data['solution_found']:
        print("No solution found, cannot visualize path")
        return None
    
    maze_file = solution_data['maze_file']
    algorithm = solution_data['algorithm']
    path = solution_data['path']
    
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{os.path.basename(maze_file).replace('.json', '')}_{algorithm}_solution.png"
        save_path = os.path.join(save_dir, filename)
    
    return visualizer.visualize_solution(
        maze_file, path, algorithm, 
        save_path=save_path, show_plot=True
    )


if __name__ == "__main__":
    # Example usage
    print("Testing maze visualization...")
    
    # Create a test maze
    from maze_generator import MazeGenerator, MazeConfig, save_maze
    
    config = MazeConfig(size=8, seed=123)
    generator = MazeGenerator(config)
    maze_data = generator.generate_maze()
    
    # Save test maze
    test_file = save_maze(maze_data, "viz_test_maze.json", "maze_output")
    
    # Create visualizer
    visualizer = MazeVisualizer(figsize=(8, 8))
    
    # Test basic maze visualization
    print("1. Basic maze visualization...")
    visualizer.visualize_maze(test_file, "Test Maze", 
                             save_path="maze_output/test_maze_viz.png", 
                             show_plot=False)
    
    # Test solution visualization
    print("2. Solution visualization...")
    from maze_solver import solve_maze_from_file
    
    solution = solve_maze_from_file(test_file, 'bfs')
    if solution['solution_found']:
        visualizer.visualize_solution(test_file, solution['path'], 'bfs',
                                     save_path="maze_output/test_solution_viz.png",
                                     show_plot=False)
    
    print("Visualization tests completed!") 