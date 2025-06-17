"""
Maze Generator

Generates square mazes without cycles and provides adjacency matrix representation.
Supports flexible start and end positions.
"""

import numpy as np
import random
from typing import Tuple, List, Dict, Set, Optional
import json
import os
from dataclasses import dataclass


@dataclass
class MazeConfig:
    """Configuration for maze generation."""
    size: int  # Square maze size (size x size)
    start_pos: Optional[Tuple[int, int]] = None  # (row, col), None for random
    end_pos: Optional[Tuple[int, int]] = None    # (row, col), None for random
    seed: Optional[int] = None
    visualize: bool = False  # Whether to create visualization
    save_visualization: bool = False  # Whether to save visualization to file


class MazeGenerator:
    """
    Generates square mazes using a modified DFS algorithm to ensure no cycles.
    """
    
    def __init__(self, config: MazeConfig):
        self.config = config
        self.size = config.size
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
        
        # Directions: up, right, down, left
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.direction_names = ['up', 'right', 'down', 'left']
    
    def _is_valid_cell(self, row: int, col: int) -> bool:
        """Check if cell coordinates are within maze bounds."""
        return 0 <= row < self.size and 0 <= col < self.size
    
    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cells."""
        neighbors = []
        for dr, dc in self.directions:
            nr, nc = row + dr, col + dc
            if self._is_valid_cell(nr, nc):
                neighbors.append((nr, nc))
        return neighbors
    
    def _cell_to_node_id(self, row: int, col: int) -> int:
        """Convert (row, col) cell coordinates to unique node ID."""
        return row * self.size + col
    
    def _node_id_to_cell(self, node_id: int) -> Tuple[int, int]:
        """Convert node ID back to (row, col) cell coordinates."""
        return node_id // self.size, node_id % self.size
    
    def generate_maze(self) -> Dict:
        """
        Generate a maze without cycles using DFS.
        Returns maze data including adjacency matrix and metadata.
        """
        # Initialize visited set and edges
        visited = set()
        edges = set()  # Set of (node1, node2) tuples representing connections
        
        # Choose start position
        if self.config.start_pos is None:
            start_row = random.randint(0, self.size - 1)
            start_col = random.randint(0, self.size - 1)
        else:
            start_row, start_col = self.config.start_pos
        
        if not self._is_valid_cell(start_row, start_col):
            raise ValueError(f"Invalid start position: ({start_row}, {start_col})")
        
        # DFS to create maze without cycles
        stack = [(start_row, start_col)]
        visited.add((start_row, start_col))
        
        while stack:
            current_row, current_col = stack.pop()
            
            # Get unvisited neighbors
            neighbors = self._get_neighbors(current_row, current_col)
            unvisited_neighbors = [(r, c) for r, c in neighbors if (r, c) not in visited]
            
            if unvisited_neighbors:
                # Put current cell back on stack
                stack.append((current_row, current_col))
                
                # Choose random unvisited neighbor
                next_row, next_col = random.choice(unvisited_neighbors)
                visited.add((next_row, next_col))
                
                # Add edge (both directions for undirected graph)
                node1 = self._cell_to_node_id(current_row, current_col)
                node2 = self._cell_to_node_id(next_row, next_col)
                edges.add((min(node1, node2), max(node1, node2)))
                
                # Continue from new cell
                stack.append((next_row, next_col))
        
        # Choose end position (must be different from start and reachable)
        reachable_cells = list(visited)
        if self.config.end_pos is None:
            # Remove start position from candidates
            end_candidates = [cell for cell in reachable_cells if cell != (start_row, start_col)]
            if not end_candidates:
                raise ValueError("Cannot find valid end position")
            end_row, end_col = random.choice(end_candidates)
        else:
            end_row, end_col = self.config.end_pos
            if not self._is_valid_cell(end_row, end_col):
                raise ValueError(f"Invalid end position: ({end_row}, {end_col})")
            if (end_row, end_col) not in visited:
                raise ValueError(f"End position ({end_row}, {end_col}) is not reachable from start")
        
        # Build adjacency matrix
        num_nodes = self.size * self.size
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        
        for node1, node2 in edges:
            adjacency_matrix[node1][node2] = 1
            adjacency_matrix[node2][node1] = 1  # Undirected graph
        
        # Create node position mapping
        node_positions = {}
        for node_id in range(num_nodes):
            row, col = self._node_id_to_cell(node_id)
            node_positions[node_id] = {'row': row, 'col': col}
        
        start_node = self._cell_to_node_id(start_row, start_col)
        end_node = self._cell_to_node_id(end_row, end_col)
        
        maze_data = {
            'config': {
                'size': self.size,
                'start_pos': [start_row, start_col],
                'end_pos': [end_row, end_col],
                'seed': self.config.seed
            },
            'adjacency_matrix': adjacency_matrix.tolist(),
            'node_positions': node_positions,
            'start_node': start_node,
            'end_node': end_node,
            'num_nodes': num_nodes,
            'num_edges': len(edges),
            'reachable_cells': len(visited)
        }
        
        return maze_data
    
    def visualize_maze(self, maze_data: Dict) -> str:
        """Create a text visualization of the maze."""
        adj_matrix = np.array(maze_data['adjacency_matrix'])
        start_node = maze_data['start_node']
        end_node = maze_data['end_node']
        
        # Create grid representation
        grid = [['#' for _ in range(self.size * 2 + 1)] for _ in range(self.size * 2 + 1)]
        
        # Mark cells
        for node_id in range(maze_data['num_nodes']):
            row, col = self._node_id_to_cell(node_id)
            grid_row, grid_col = row * 2 + 1, col * 2 + 1
            
            if node_id == start_node:
                grid[grid_row][grid_col] = 'S'
            elif node_id == end_node:
                grid[grid_row][grid_col] = 'E'
            else:
                grid[grid_row][grid_col] = ' '
        
        # Mark connections
        for node1 in range(maze_data['num_nodes']):
            row1, col1 = self._node_id_to_cell(node1)
            for node2 in range(node1 + 1, maze_data['num_nodes']):
                if adj_matrix[node1][node2] == 1:
                    row2, col2 = self._node_id_to_cell(node2)
                    # Mark the wall between cells as open
                    mid_row = (row1 + row2) * 2 // 2 + 1
                    mid_col = (col1 + col2) * 2 // 2 + 1
                    grid[mid_row][mid_col] = ' '
        
        return '\n'.join(''.join(row) for row in grid)
    
    def create_visualization(self, maze_data: Dict, output_dir: str = "maze_output") -> Optional[str]:
        """Create graphical visualization of the maze using MazeVisualizer."""
        try:
            from minGPT.projects.maze_nav.maze_visualizer import MazeVisualizer
            
            visualizer = MazeVisualizer(figsize=(8, 8))
            
            # Create visualization filename
            seed_str = f"_seed{self.config.seed}" if self.config.seed is not None else ""
            filename = f"maze_{self.size}x{self.size}{seed_str}_viz.png"
            save_path = os.path.join(output_dir, filename) if self.config.save_visualization else None
            
            # Create visualization
            fig = visualizer.visualize_maze(
                maze_data, 
                title=f"Generated Maze (Size: {self.size}x{self.size})",
                save_path=save_path,
                show_plot=self.config.visualize and not self.config.save_visualization,
                show_grid=True
            )
            
            return save_path
            
        except ImportError:
            print("Warning: Visualization requires matplotlib. Skipping visualization.")
            return None
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
            return None


def save_maze(maze_data: Dict, filename: str, output_dir: str = "maze_output") -> str:
    """Save maze data to file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Convert numpy arrays to lists for JSON serialization
    maze_data_copy = maze_data.copy()
    
    with open(filepath, 'w') as f:
        json.dump(maze_data_copy, f, indent=2)
    
    return filepath


if __name__ == "__main__":
    # Example usage with visualization
    config = MazeConfig(
        size=10, 
        seed=42, 
        visualize=True,  # Show visualization
        save_visualization=True  # Also save to file
    )
    generator = MazeGenerator(config)
    maze = generator.generate_maze()
    
    print("Generated maze:")
    print(generator.visualize_maze(maze))
    print(f"\nMaze info:")
    print(f"Size: {maze['config']['size']}x{maze['config']['size']}")
    print(f"Start: {maze['config']['start_pos']} (node {maze['start_node']})")
    print(f"End: {maze['config']['end_pos']} (node {maze['end_node']})")
    print(f"Reachable cells: {maze['reachable_cells']}")
    print(f"Total edges: {maze['num_edges']}")
    
    # Save the maze
    filename = save_maze(maze, "example_maze_with_viz.json")
    print(f"\nMaze saved to: {filename}")
    
    # Create visualization
    viz_path = generator.create_visualization(maze)
    if viz_path:
        print(f"Visualization saved to: {viz_path}") 