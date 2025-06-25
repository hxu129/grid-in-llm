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
    algorithm: str = 'dfs'  # Algorithm to use: 'dfs', 'wilson', or 'kruskal'
    start_pos: Optional[Tuple[int, int]] = None  # (row, col), None for random
    end_pos: Optional[Tuple[int, int]] = None    # (row, col), None for random
    seed: Optional[int] = None
    visualize: bool = False  # Whether to create visualization
    save_visualization: bool = False  # Whether to save visualization to file


class _DisjointSetUnion:
    """Helper class for the Disjoint Set Union (DSU) data structure."""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.num_sets = n

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            self.num_sets -= 1
            return True
        return False


class MazeGenerator:
    """
    Generates square mazes using a modified DFS algorithm to ensure no cycles.
    """
    
    def __init__(self, config: MazeConfig):
        self.config = config
        self.size = config.size
        if config.algorithm not in ['dfs', 'wilson', 'kruskal']:
            raise ValueError(f"Invalid algorithm: {config.algorithm}. Choose 'dfs', 'wilson', or 'kruskal'.")

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
        Generate a maze using the configured algorithm.
        Returns maze data including adjacency matrix and metadata.
        """
        if self.config.algorithm == 'dfs':
            edges = self._generate_dfs_edges()
        elif self.config.algorithm == 'wilson':
            edges = self._generate_wilson_edges()
        elif self.config.algorithm == 'kruskal':
            edges = self._generate_kruskal_edges()
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

        return self._finalize_maze_data(edges)

    def _generate_dfs_edges(self) -> Set[Tuple[int, int]]:
        """Generate maze edges using Depth-First Search (DFS)."""
        visited = set()
        edges = set()
        
        if self.config.start_pos is None:
            start_row, start_col = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
        else:
            start_row, start_col = self.config.start_pos
        
        if not self._is_valid_cell(start_row, start_col):
            raise ValueError(f"Invalid start position for DFS: ({start_row}, {start_col})")

        stack = [(start_row, start_col)]
        visited.add((start_row, start_col))
        
        while stack:
            current_row, current_col = stack[-1]
            
            neighbors = self._get_neighbors(current_row, current_col)
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            if unvisited_neighbors:
                next_row, next_col = random.choice(unvisited_neighbors)
                visited.add((next_row, next_col))
                
                node1 = self._cell_to_node_id(current_row, current_col)
                node2 = self._cell_to_node_id(next_row, next_col)
                edges.add((min(node1, node2), max(node1, node2)))
                
                stack.append((next_row, next_col))
            else:
                stack.pop()
        
        return edges

    def _generate_wilson_edges(self) -> Set[Tuple[int, int]]:
        """
        Generate maze edges using Wilson's algorithm for a uniform spanning tree.
        """
        all_cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        maze_cells = set()
        edges = set()

        # Start with one random cell in the maze
        first_cell = random.choice(all_cells)
        maze_cells.add(first_cell)
        
        while len(maze_cells) < len(all_cells):
            # Pick a random starting cell not yet in the maze
            start_cell = random.choice(all_cells)
            while start_cell in maze_cells:
                start_cell = random.choice(all_cells)
            
            # Perform a random walk
            path = [start_cell]
            current_cell = start_cell
            
            while current_cell not in maze_cells:
                neighbors = self._get_neighbors(current_cell[0], current_cell[1])
                next_cell = random.choice(neighbors)
                
                # If the walk intersects itself, erase the loop
                if next_cell in path:
                    path = path[:path.index(next_cell) + 1]
                else:
                    path.append(next_cell)
                current_cell = next_cell
            
            # Add the loop-erased path to the maze
            for i in range(len(path) - 1):
                cell1, cell2 = path[i], path[i+1]
                maze_cells.add(cell1)
                
                node1 = self._cell_to_node_id(cell1[0], cell1[1])
                node2 = self._cell_to_node_id(cell2[0], cell2[1])
                edges.add((min(node1, node2), max(node1, node2)))
        
        return edges

    def _generate_kruskal_edges(self) -> Set[Tuple[int, int]]:
        """Generate maze edges using Randomized Kruskal's algorithm."""
        num_nodes = self.size * self.size
        dsu = _DisjointSetUnion(num_nodes)
        edges = set()
        
        # Create a list of all possible edges
        all_possible_edges = []
        for r in range(self.size):
            for c in range(self.size):
                node1 = self._cell_to_node_id(r, c)
                # Add edge to the right
                if c + 1 < self.size:
                    node2 = self._cell_to_node_id(r, c + 1)
                    all_possible_edges.append((node1, node2))
                # Add edge down
                if r + 1 < self.size:
                    node2 = self._cell_to_node_id(r + 1, c)
                    all_possible_edges.append((node1, node2))
        
        # Randomize the order of edges
        random.shuffle(all_possible_edges)
        
        for node1, node2 in all_possible_edges:
            if dsu.union(node1, node2):
                edges.add((min(node1, node2), max(node1, node2)))
                if len(edges) == num_nodes - 1:
                    break
        
        return edges

    def _finalize_maze_data(self, edges: Set[Tuple[int, int]]) -> Dict:
        """Build the final maze data structure from a set of edges."""
        # Determine start position
        if self.config.start_pos is None:
            start_row, start_col = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
        else:
            start_row, start_col = self.config.start_pos
        if not self._is_valid_cell(start_row, start_col):
            raise ValueError(f"Invalid start position: ({start_row}, {start_col})")

        # Determine end position
        if self.config.end_pos is None:
            all_cells = [(r, c) for r in range(self.size) for c in range(self.size)]
            end_candidates = [cell for cell in all_cells if cell != (start_row, start_col)]
            if not end_candidates:
                raise ValueError("Cannot find a valid end position different from the start.")
            end_row, end_col = random.choice(end_candidates)
        else:
            end_row, end_col = self.config.end_pos
        if not self._is_valid_cell(end_row, end_col):
            raise ValueError(f"Invalid end position: ({end_row}, {end_col})")

        # Build adjacency matrix
        num_nodes = self.size * self.size
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        for node1, node2 in edges:
            adjacency_matrix[node1][node2] = 1
            adjacency_matrix[node2][node1] = 1

        # Create node position mapping
        node_positions = {i: {'row': r, 'col': c} for i, (r, c) in enumerate( (self._node_id_to_cell(i) for i in range(num_nodes)) )}
        
        start_node = self._cell_to_node_id(start_row, start_col)
        end_node = self._cell_to_node_id(end_row, end_col)
        
        maze_data = {
            'config': {
                'size': self.size,
                'algorithm': self.config.algorithm,
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
            'reachable_cells': num_nodes # Both algorithms connect the whole maze
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