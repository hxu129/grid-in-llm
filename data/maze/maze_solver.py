"""
Maze Solver

Implements shortest path algorithms to solve mazes represented as adjacency matrices.
Outputs solutions as sequences of states (node IDs) and provides conversion to offsets.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import deque
import heapq
import json
import os


class MazeSolver:
    """
    Solves mazes using various shortest path algorithms.
    """
    
    def __init__(self, maze_data: Dict):
        self.maze_data = maze_data
        self.adjacency_matrix = np.array(maze_data['adjacency_matrix'])
        self.size = maze_data['config']['size']
        self.start_node = maze_data['start_node']
        self.end_node = maze_data['end_node']
        self.node_positions = maze_data['node_positions']
        
        # Direction mappings for offset conversion
        self.direction_vectors = {
            (-1, 0): 'up',
            (0, 1): 'right', 
            (1, 0): 'down',
            (0, -1): 'left'
        }
    
    def _get_neighbors(self, node_id: int) -> List[int]:
        """Get neighboring nodes from adjacency matrix."""
        neighbors = []
        for neighbor_id in range(len(self.adjacency_matrix)):
            if self.adjacency_matrix[node_id][neighbor_id] == 1:
                neighbors.append(neighbor_id)
        return neighbors
    
    def _node_to_position(self, node_id: int) -> Tuple[int, int]:
        """Convert node ID to (row, col) position."""
        # Try both integer and string keys for compatibility
        if node_id in self.node_positions:
            pos = self.node_positions[node_id]
        elif str(node_id) in self.node_positions:
            pos = self.node_positions[str(node_id)]
        else:
            raise KeyError(f"Node {node_id} not found in node_positions")
        return pos['row'], pos['col']
    
    def _manhattan_distance(self, node1: int, node2: int) -> int:
        """Calculate Manhattan distance between two nodes."""
        row1, col1 = self._node_to_position(node1)
        row2, col2 = self._node_to_position(node2)
        return abs(row1 - row2) + abs(col1 - col2)
    
    def solve_bfs(self) -> Optional[List[int]]:
        """
        Solve maze using Breadth-First Search.
        Returns path as list of node IDs from start to end.
        """
        if self.start_node == self.end_node:
            return [self.start_node]
        
        queue = deque([self.start_node])
        visited = {self.start_node}
        parent = {self.start_node: None}
        
        while queue:
            current = queue.popleft()
            
            for neighbor in self._get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
                    
                    if neighbor == self.end_node:
                        # Reconstruct path
                        path = []
                        node = self.end_node
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        return path[::-1]  # Reverse to get start-to-end path
        
        return None  # No path found
    
    def solve_dfs(self) -> Optional[List[int]]:
        """
        Solve maze using Depth-First Search.
        Returns path as list of node IDs from start to end.
        Note: DFS doesn't guarantee shortest path, but included for completeness.
        """
        if self.start_node == self.end_node:
            return [self.start_node]
        
        stack = [self.start_node]
        visited = {self.start_node}
        parent = {self.start_node: None}
        
        while stack:
            current = stack.pop()
            
            if current == self.end_node:
                # Reconstruct path
                path = []
                node = self.end_node
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return path[::-1]  # Reverse to get start-to-end path
            
            for neighbor in self._get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    stack.append(neighbor)
        
        return None  # No path found
    
    def solve_astar(self) -> Optional[List[int]]:
        """
        Solve maze using A* algorithm.
        Returns shortest path as list of node IDs from start to end.
        """
        if self.start_node == self.end_node:
            return [self.start_node]
        
        # Priority queue: (f_score, node_id)
        open_set = [(self._manhattan_distance(self.start_node, self.end_node), self.start_node)]
        closed_set = set()
        
        # g_score: cost from start to node
        g_score = {self.start_node: 0}
        # f_score: g_score + heuristic
        f_score = {self.start_node: self._manhattan_distance(self.start_node, self.end_node)}
        # parent: for path reconstruction
        parent = {}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == self.end_node:
                # Reconstruct path
                path = []
                node = self.end_node
                while node in parent:
                    path.append(node)
                    node = parent[node]
                path.append(self.start_node)
                return path[::-1]  # Reverse to get start-to-end path
            
            closed_set.add(current)
            
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score[current] + 1  # All edges have weight 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    parent[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._manhattan_distance(neighbor, self.end_node)
                    
                    # Add to open set if not already there
                    if not any(item[1] == neighbor for item in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def solve(self, algorithm: str = 'bfs') -> Optional[List[int]]:
        """
        Solve maze using specified algorithm.
        
        Args:
            algorithm: 'bfs', 'dfs', or 'astar'
        
        Returns:
            Path as list of node IDs from start to end, or None if no path exists.
        """
        if algorithm == 'bfs':
            return self.solve_bfs()
        elif algorithm == 'dfs':
            return self.solve_dfs()
        elif algorithm == 'astar':
            return self.solve_astar()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use 'bfs', 'dfs', or 'astar'.")
    
    def path_to_positions(self, path: List[int]) -> List[Tuple[int, int]]:
        """Convert path of node IDs to path of (row, col) positions."""
        return [self._node_to_position(node_id) for node_id in path]
    
    def path_to_offsets(self, path: List[int]) -> List[str]:
        """
        Convert path of node IDs to sequence of movement directions.
        
        Args:
            path: List of node IDs representing the solution path
        
        Returns:
            List of direction strings ('up', 'down', 'left', 'right')
        """
        if len(path) < 2:
            return []
        
        offsets = []
        positions = self.path_to_positions(path)
        
        for i in range(1, len(positions)):
            prev_row, prev_col = positions[i-1]
            curr_row, curr_col = positions[i]
            
            offset = (curr_row - prev_row, curr_col - prev_col)
            
            if offset in self.direction_vectors:
                offsets.append(self.direction_vectors[offset])
            else:
                raise ValueError(f"Invalid move from {positions[i-1]} to {positions[i]}")
        
        return offsets
    
    def validate_path(self, path: List[int]) -> bool:
        """
        Validate that a path is valid (all consecutive nodes are connected).
        
        Args:
            path: List of node IDs
        
        Returns:
            True if path is valid, False otherwise
        """
        if not path:
            return False
        
        if path[0] != self.start_node or path[-1] != self.end_node:
            return False
        
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            if self.adjacency_matrix[current][next_node] != 1:
                return False
        
        return True
    
    def visualize_solution(self, path: List[int], algorithm: str = 'bfs', 
                          save_dir: str = None, show_plot: bool = True) -> Optional[str]:
        """
        Create visualization of the solution path.
        
        Args:
            path: Solution path as list of node IDs
            algorithm: Algorithm used for the solution
            save_dir: Directory to save visualization (optional)
            show_plot: Whether to display the plot
        
        Returns:
            Path to saved visualization file, or None if not saved
        """
        try:
            from minGPT.projects.maze_nav.maze_visualizer import MazeVisualizer
            
            visualizer = MazeVisualizer(figsize=(10, 10))
            
            save_path = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                size = self.maze_data['config']['size']
                seed = self.maze_data['config'].get('seed', 'unknown')
                filename = f"solution_{size}x{size}_seed{seed}_{algorithm}.png"
                save_path = os.path.join(save_dir, filename)
            
            # Create visualization
            fig = visualizer.visualize_solution(
                self.maze_data, path, algorithm,
                save_path=save_path, show_plot=show_plot,
                show_arrows=True
            )
            
            return save_path
            
        except ImportError:
            print("Warning: Visualization requires matplotlib. Skipping visualization.")
            return None
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
            return None


def solve_maze_from_file(maze_file: str, algorithm: str = 'bfs', 
                        visualize: bool = False, save_visualization: bool = False,
                        viz_output_dir: str = "maze_output") -> Dict:
    """
    Load maze from file and solve it.
    
    Args:
        maze_file: Path to maze JSON file
        algorithm: Solving algorithm ('bfs', 'dfs', 'astar')
        visualize: Whether to create and show visualization
        save_visualization: Whether to save visualization to file
        viz_output_dir: Directory to save visualizations
    
    Returns:
        Dictionary containing solution data
    """
    with open(maze_file, 'r') as f:
        maze_data = json.load(f)
    
    solver = MazeSolver(maze_data)
    path = solver.solve(algorithm)
    
    if path is None:
        solution_data = {
            'maze_file': maze_file,
            'algorithm': algorithm,
            'solution_found': False,
            'path': None,
            'path_positions': None,
            'offsets': None,
            'path_length': 0,
            'visualization_path': None
        }
    else:
        positions = solver.path_to_positions(path)
        offsets = solver.path_to_offsets(path)
        
        # Create visualization if requested
        viz_path = None
        if visualize or save_visualization:
            viz_path = solver.visualize_solution(
                path, algorithm, 
                save_dir=viz_output_dir if save_visualization else None,
                show_plot=visualize and not save_visualization
            )
        
        solution_data = {
            'maze_file': maze_file,
            'algorithm': algorithm,
            'solution_found': True,
            'path': path,
            'path_positions': positions,
            'offsets': offsets,
            'path_length': len(path),
            'num_moves': len(offsets),
            'start_node': maze_data['start_node'],
            'end_node': maze_data['end_node'],
            'is_valid': solver.validate_path(path),
            'visualization_path': viz_path
        }
    
    return solution_data


def compare_algorithms_with_viz(maze_file: str, algorithms: List[str] = ['bfs', 'astar', 'dfs'],
                               save_dir: str = "maze_output", show_plot: bool = True) -> Dict:
    """
    Solve maze with multiple algorithms and create comparison visualization.
    
    Args:
        maze_file: Path to maze JSON file
        algorithms: List of algorithms to compare
        save_dir: Directory to save visualization
        show_plot: Whether to display the plot
    
    Returns:
        Dictionary containing all solutions and comparison data
    """
    with open(maze_file, 'r') as f:
        maze_data = json.load(f)
    
    solver = MazeSolver(maze_data)
    solutions = {}
    paths_for_viz = {}
    
    # Solve with each algorithm
    for algorithm in algorithms:
        path = solver.solve(algorithm)
        if path:
            positions = solver.path_to_positions(path)
            offsets = solver.path_to_offsets(path)
            
            solutions[algorithm] = {
                'path': path,
                'path_positions': positions,
                'offsets': offsets,
                'path_length': len(path),
                'num_moves': len(offsets),
                'is_valid': solver.validate_path(path)
            }
            paths_for_viz[algorithm] = path
        else:
            solutions[algorithm] = {
                'path': None,
                'path_positions': None,
                'offsets': None,
                'path_length': 0,
                'num_moves': 0,
                'is_valid': False
            }
    
    # Create comparison visualization
    viz_path = None
    if paths_for_viz:
        try:
            from minGPT.projects.maze_nav.maze_visualizer import MazeVisualizer
            
            visualizer = MazeVisualizer(figsize=(12, 10))
            
            save_path = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                size = maze_data['config']['size']
                seed = maze_data['config'].get('seed', 'unknown')
                alg_str = '_'.join(algorithms)
                filename = f"comparison_{size}x{size}_seed{seed}_{alg_str}.png"
                save_path = os.path.join(save_dir, filename)
            
            fig = visualizer.compare_algorithms(
                maze_data, paths_for_viz,
                save_path=save_path, show_plot=show_plot
            )
            
            viz_path = save_path
            
        except ImportError:
            print("Warning: Visualization requires matplotlib. Skipping comparison visualization.")
        except Exception as e:
            print(f"Warning: Comparison visualization failed: {e}")
    
    return {
        'maze_file': maze_file,
        'algorithms': algorithms,
        'solutions': solutions,
        'comparison_visualization': viz_path
    }


if __name__ == "__main__":
    # Example usage - requires a maze file to exist
    import sys
    
    if len(sys.argv) > 1:
        maze_file = sys.argv[1]
        algorithm = sys.argv[2] if len(sys.argv) > 2 else 'bfs'
        visualize = len(sys.argv) > 3 and sys.argv[3].lower() == 'true'
        
        try:
            # Solve with visualization
            solution = solve_maze_from_file(maze_file, algorithm, 
                                          visualize=visualize, 
                                          save_visualization=True)
            
            print(f"Maze: {solution['maze_file']}")
            print(f"Algorithm: {solution['algorithm']}")
            print(f"Solution found: {solution['solution_found']}")
            
            if solution['solution_found']:
                print(f"Path length: {solution['path_length']} nodes")
                print(f"Number of moves: {solution['num_moves']}")
                print(f"Path validity: {solution['is_valid']}")
                print(f"Path (nodes): {solution['path']}")
                print(f"Path (positions): {solution['path_positions']}")
                print(f"Offsets: {solution['offsets']}")
                if solution['visualization_path']:
                    print(f"Visualization saved to: {solution['visualization_path']}")
            
            # Also create algorithm comparison
            if solution['solution_found']:
                print("\nCreating algorithm comparison...")
                comparison = compare_algorithms_with_viz(maze_file, show_plot=False)
                if comparison['comparison_visualization']:
                    print(f"Comparison visualization saved to: {comparison['comparison_visualization']}")
        
        except FileNotFoundError:
            print(f"Maze file not found: {maze_file}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python maze_solver.py <maze_file> [algorithm] [visualize]")
        print("Algorithms: bfs, dfs, astar")
        print("Visualize: true/false") 