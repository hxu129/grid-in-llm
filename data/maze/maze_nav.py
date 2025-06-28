"""
Maze Navigation GPT Training Data Generator

Creates training data for a GPT model to learn maze navigation.
Generates all possible paths between reachable node pairs in a large maze.
"""

import numpy as np
import json
import random
import os
import sys
import pickle
import itertools
import logging
import time
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple, Optional
from collections import deque
from dataclasses import dataclass

# Add the maze directory to path to import existing modules
maze_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'maze')
sys.path.append(maze_path)
from maze_generator import MazeGenerator, MazeConfig
from maze_solver import MazeSolver


@dataclass
class MazeNavConfig:
    """Configuration for maze navigation training data generation."""
    maze_size: int = 50
    seed: Optional[int] = None
    output_dir: str = "maze_nav_data"
    max_pairs: Optional[int] = None
    train_ratio: float = 0.8
    algorithm: str = "wilson" # "wilson", "dfs", or "kruskal"
    # Free world mode parameters
    free_world_mode: bool = False
    num_random_paths: int = 50000
    free_world_output_dir: str = "path_int_data"


class MazeNavDataGenerator:
    """Generates training data for maze navigation GPT model."""
    
    def __init__(self, config: MazeNavConfig):
        self.config = config
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
        
        # Simple vocabulary: nodes (0 to N-1) + directions + special tokens
        self.direction_tokens = ["up", "right", "down", "left"]
        self.direction_to_offset = {
            "up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)
        }
        
        self.maze_data = None
        self.vocab_size = None
        self.token_to_id = {}
        self.id_to_token = {}
    
    def generate_data(self) -> Dict:
        """Generate complete training dataset."""
        if self.config.free_world_mode:
            logging.info(f"Generating {self.config.maze_size}x{self.config.maze_size} free world (no walls)...")
            self.maze_data = self._create_free_world()
        else:
            logging.info(f"Generating {self.config.maze_size}x{self.config.maze_size} maze...")
            # Generate maze
            maze_config = MazeConfig(size=self.config.maze_size, seed=self.config.seed, algorithm=self.config.algorithm)
            generator = MazeGenerator(maze_config)
            self.maze_data = generator.generate_maze()
        
        # Setup vocabulary
        num_nodes = self.maze_data['num_nodes']
        vocab = list(range(num_nodes)) + self.direction_tokens + ["\n", "<PAD>"]
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        self.vocab_size = len(vocab)
        self.padding_token_id = self.token_to_id['<PAD>']
        
        logging.info(f"Generated {'free world' if self.config.free_world_mode else 'maze'} with {num_nodes} nodes, vocab size: {self.vocab_size}")
        
        if self.config.free_world_mode:
            # Generate random paths with node coverage
            all_pairs = self._generate_random_pairs_with_coverage()
        else:
            # Find all reachable pairs
            all_pairs = self._find_reachable_pairs()
            if self.config.max_pairs and len(all_pairs) > self.config.max_pairs:
                all_pairs = random.sample(all_pairs, self.config.max_pairs)
        
        # Split train/test
        if self.config.free_world_mode:
            train_pairs, test_pairs = self._split_pairs_free_world(all_pairs)
        else:
            train_pairs, test_pairs = self._split_pairs(all_pairs)
        
        # Generate sequences
        if self.config.free_world_mode:
            train_sequences = self._generate_sequences_free_world(train_pairs)
            test_sequences = self._generate_sequences_free_world(test_pairs)
        else:
            train_sequences = self._generate_sequences(train_pairs)
            test_sequences = self._generate_sequences(test_pairs)
        
        # Create dataset
        dataset = {
            'config': {
                'maze_size': self.config.maze_size,
                'seed': self.config.seed,
                'vocab_size': self.vocab_size,
                'vocab_encoding': 'mixed',  # For compatibility
                'free_world_mode': self.config.free_world_mode
            },
            'maze_data': self.maze_data,
            'vocabulary': {
                'token_to_id': self.token_to_id,
                'id_to_token': self.id_to_token,
                'direction_tokens': self.direction_tokens,
                'direction_name_to_id': {d: self.token_to_id[d] for d in self.direction_tokens}
            },
            'train': {'sequences': train_sequences, 'count': len(train_sequences)},
            'test': {'sequences': test_sequences, 'count': len(test_sequences)},
            'stats': {
                'total_pairs': len(all_pairs),
                'train_pairs': len(train_pairs),
                'test_pairs': len(test_pairs),
                'avg_train_length': float(np.mean([s['length'] for s in train_sequences])) if train_sequences else 0,
                'avg_test_length': float(np.mean([s['length'] for s in test_sequences])) if test_sequences else 0,
                'max_sequence_length': max([s['length'] for s in train_sequences + test_sequences]) if (train_sequences or test_sequences) else 0
            }
        }
        
        logging.info(f"Generated {len(train_sequences)} train, {len(test_sequences)} test sequences")
        return dataset
    
    def _find_reachable_pairs(self) -> List[Tuple[int, int]]:
        """Find all pairs of reachable nodes."""
        num_nodes = self.maze_data['num_nodes']
        # The maze is a spanning tree, so all nodes are reachable from one another.
        # This is much faster than running BFS from every node.
        # We use itertools.permutations to get all ordered pairs.
        all_pairs = list(itertools.permutations(range(num_nodes), 2))
        return all_pairs
    
    def _split_pairs(self, all_pairs: List[Tuple[int, int]]) -> Tuple[List, List]:
        """Split pairs: direct connections to train, others with random sampling."""
        adj_matrix = np.array(self.maze_data['adjacency_matrix'])
        direct = [(s, e) for s, e in all_pairs if adj_matrix[s][e] == 1]
        others = [(s, e) for s, e in all_pairs if adj_matrix[s][e] == 0]
        
        # Random sampling without replacement for training
        train_size = int(len(others) * self.config.train_ratio)
        # train_others = random.choices(others, k=train_size)  # sampling with replacement
        train_others = random.sample(others, k=train_size)  # sampling without replacement
        train_pairs = direct + train_others
        
        # For test, use remaining others (without replacement to avoid overlap)
        train_others_set = set(train_others)
        remaining_others = [pair for pair in others if pair not in train_others_set]
        test_pairs = remaining_others
        
        return train_pairs, test_pairs
    
    def _generate_sequences(self, pairs: List[Tuple[int, int]]) -> List[Dict]:
        """Generate training sequences from node pairs in parallel."""
        
        num_processes = cpu_count()
        logging.info(f"Generating sequences in parallel with {num_processes} processes...")

        # Convert adjacency matrix to a NumPy array for efficient processing
        adj_matrix = np.array(self.maze_data['adjacency_matrix'])

        # Prepare arguments for the worker function
        args = [(
            start, end, 
            adj_matrix, 
            self.maze_data['num_nodes'], 
            self.config.maze_size,
            self.direction_to_offset
        ) for start, end in pairs]
        
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(_process_pair_worker, args)
            
        # Filter out None results (for paths that couldn't be found, though this shouldn't happen in a connected maze)
        sequences = [res for res in results if res is not None]
        return sequences
    
    def _find_shortest_path(self, start: int, end: int) -> Optional[List[int]]:
        """Find shortest path using BFS."""
        if start == end:
            return [start]
        
        adj_matrix = np.array(self.maze_data['adjacency_matrix'])
        num_nodes = self.maze_data['num_nodes']
        
        queue = deque([start])
        visited = {start}
        parent = {start: None}
        
        while queue:
            current = queue.popleft()
            for neighbor in range(num_nodes):
                if adj_matrix[current][neighbor] == 1 and neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
                    
                    if neighbor == end:
                        # Reconstruct path
                        path = []
                        node = end
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        return path[::-1]
        
        return None
    
    def _path_to_sequence(self, path: List[int]) -> List:
        """Convert path to training sequence: [start, end, start, move1, node1, move2, node2, ..., newline]."""
        if len(path) < 2:
            return []
        
        sequence = [path[0], path[-1], path[0]]  # start, end, start
        
        for i in range(len(path) - 1):
            # Get movement direction
            from_row, from_col = path[i] // self.config.maze_size, path[i] % self.config.maze_size
            to_row, to_col = path[i + 1] // self.config.maze_size, path[i + 1] % self.config.maze_size
            offset = (to_row - from_row, to_col - from_col)
            
            for direction, dir_offset in self.direction_to_offset.items():
                if offset == dir_offset:
                    sequence.extend([direction, path[i + 1]])
                    break
        
        sequence.append("\n")
        return sequence
    
    def _create_free_world(self) -> Dict:
        """Create a free world (no walls) with full connectivity between adjacent cells."""
        size = self.config.maze_size
        num_nodes = size * size
        
        # Create adjacency matrix for a grid world with no walls
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        
        for row in range(size):
            for col in range(size):
                current_node = row * size + col
                
                # Connect to adjacent cells (up, right, down, left)
                for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < size and 0 <= new_col < size:
                        neighbor_node = new_row * size + new_col
                        adjacency_matrix[current_node][neighbor_node] = 1
        
        # Create maze_data structure compatible with existing code
        maze_data = {
            'size': size,
            'num_nodes': num_nodes,
            'adjacency_matrix': adjacency_matrix.tolist(),
            'grid': [[1 for _ in range(size)] for _ in range(size)],  # All cells are open
            'free_world': True,
            'config': {
                'size': size,
                'seed': self.config.seed,
                'algorithm': 'free_world'
            }
        }
        
        return maze_data
    
    def _generate_random_pairs_with_coverage(self) -> List[Tuple[int, int]]:
        """Generate random start-end pairs ensuring all nodes are covered."""
        num_nodes = self.maze_data['num_nodes']
        all_nodes = set(range(num_nodes))
        covered_nodes = set()
        pairs = []
        
        # Generate random pairs until we reach the desired count
        target_pairs = min(self.config.num_random_paths, num_nodes * num_nodes)
        
        while len(pairs) < target_pairs:
            # If we haven't covered all nodes yet, prioritize uncovered nodes
            if len(covered_nodes) < num_nodes:
                uncovered = list(all_nodes - covered_nodes)
                if uncovered:
                    # Pick an uncovered node as start or end
                    if random.random() < 0.5:
                        start = random.choice(uncovered)
                        end = random.randint(0, num_nodes - 1)
                    else:
                        start = random.randint(0, num_nodes - 1)
                        end = random.choice(uncovered)
                else:
                    # All nodes covered, generate completely random pairs
                    start = random.randint(0, num_nodes - 1)
                    end = random.randint(0, num_nodes - 1)
            else:
                # All nodes covered, generate completely random pairs
                start = random.randint(0, num_nodes - 1)
                end = random.randint(0, num_nodes - 1)
            
            # Avoid same start and end
            if start != end:
                pairs.append((start, end))
                covered_nodes.add(start)
                covered_nodes.add(end)
        
        logging.info(f"Generated {len(pairs)} random pairs covering {len(covered_nodes)}/{num_nodes} nodes")
        return pairs
    
    def _split_pairs_free_world(self, all_pairs: List[Tuple[int, int]]) -> Tuple[List, List]:
        """Split pairs for free world mode."""
        # Simple random split for free world
        random.shuffle(all_pairs)
        train_size = int(len(all_pairs) * self.config.train_ratio)
        train_pairs = all_pairs[:train_size]
        test_pairs = all_pairs[train_size:]
        return train_pairs, test_pairs
    
    def _generate_sequences_free_world(self, pairs: List[Tuple[int, int]]) -> List[Dict]:
        """Generate training sequences for free world mode."""
        num_processes = cpu_count()
        logging.info(f"Generating free world sequences in parallel with {num_processes} processes...")

        # For free world, we don't need maze constraints, just generate random valid paths
        args = [(
            start, end, 
            self.config.maze_size,
            self.direction_to_offset
        ) for start, end in pairs]
        
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(_process_free_world_pair_worker, args)
            
        # Filter out None results
        sequences = [res for res in results if res is not None]
        return sequences
    
    def _generate_random_path_free_world(self, start: int, end: int) -> List[int]:
        """Generate a random path in free world from start to end."""
        if start == end:
            return [start]
        
        size = self.config.maze_size
        start_row, start_col = start // size, start % size
        end_row, end_col = end // size, end % size
        
        path = [start]
        current_row, current_col = start_row, start_col
        
        # Generate a random path that eventually reaches the target
        max_steps = size ** 4
        step_count = 0
        
        while (current_row, current_col) != (end_row, end_col) and step_count < max_steps:
            # Choose direction that gets us closer to target with some randomness
            possible_moves = []
            
            # Add all valid moves
            for direction, (dr, dc) in self.direction_to_offset.items():
                new_row, new_col = current_row + dr, current_col + dc
                if 0 <= new_row < size and 0 <= new_col < size:
                    possible_moves.append((direction, new_row, new_col))
            
            if not possible_moves:
                break
            
            # Weight moves towards the target
            weighted_moves = []
            for direction, new_row, new_col in possible_moves:
                # Calculate distance to target
                dist_to_target = abs(new_row - end_row) + abs(new_col - end_col)
                current_dist = abs(current_row - end_row) + abs(current_col - end_col)
                
                # Prefer moves that get closer, but allow some randomness
                if dist_to_target < current_dist:
                    weight = 3  # Higher weight for good moves
                else:
                    weight = 1  # Lower weight for random moves
                
                weighted_moves.extend([(direction, new_row, new_col)] * weight)
            
            # Choose a move
            direction, new_row, new_col = random.choice(weighted_moves)
            current_row, current_col = new_row, new_col
            new_node = current_row * size + current_col
            path.append(new_node)
            step_count += 1
        
        # If we didn't reach the target, add a direct path to it
        if (current_row, current_col) != (end_row, end_col):
            path.append(end)
        
        return path
    
    def save_all_files(self, dataset: Dict):
        """Save all required output files."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save main dataset
        with open(os.path.join(self.config.output_dir, f"maze_nav_dataset_{self.config.maze_size}.json"), 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Save meta.pkl
        stoi = {str(token): token_id for token, token_id in self.token_to_id.items()}
        itos = {token_id: str(token) for token, token_id in self.token_to_id.items()}
        
        meta = {
            'vocab_size': self.vocab_size,
            'stoi': stoi,
            'itos': itos,
            'config': dataset['config'],
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'direction_tokens': self.direction_tokens,
            'padding_token_id': self.padding_token_id
        }
        
        with open(os.path.join(self.config.output_dir, f'meta_{self.config.maze_size}.pkl'), 'wb') as f:
            pickle.dump(meta, f)
        
        # Save binary data
        def save_binary(sequences, filename):
            # shuffle before saving
            random.shuffle(sequences)
            tokens = []
            for seq_data in sequences:
                token_ids = [self.token_to_id[token] for token in seq_data['sequence']]
                tokens.extend(token_ids)
            
            if tokens:
                max_id = max(tokens)
                dtype = np.uint16 if max_id < 65536 else np.uint32
                np.array(tokens, dtype=dtype).tofile(os.path.join(self.config.output_dir, filename))
            return len(tokens)
        
        train_tokens = save_binary(dataset['train']['sequences'], f'train_{self.config.maze_size}.bin')
        val_tokens = save_binary(dataset['test']['sequences'], f'val_{self.config.maze_size}.bin')
        
        # Save PyTorch format
        pytorch_data = dataset.copy()
        pytorch_data['train_data'] = [[self.token_to_id[t] for t in s['sequence']] for s in dataset['train']['sequences']]
        pytorch_data['test_data'] = [[self.token_to_id[t] for t in s['sequence']] for s in dataset['test']['sequences']]
        
        with open(os.path.join(self.config.output_dir, f"maze_nav_pytorch_{self.config.maze_size}.json"), 'w') as f:
            json.dump(pytorch_data, f, indent=2)
        
        return train_tokens, val_tokens
    
    def visualize_maze(self, save_path: str = None):
        """Create a visualization of the maze with node labels."""
        if not self.maze_data:
            logging.warning("No maze data available for visualization")
            return None
        
        try:
            from maze_nav_visualizer import MazeNavVisualizer
            
            visualizer = MazeNavVisualizer(figsize=(16, 16))
            fig = visualizer.visualize_maze(
                self.maze_data, 
                title=f"Maze with Node Labels (Size: {self.config.maze_size}x{self.config.maze_size})",
                save_path=save_path,
                show_grid=True,
                show_node_labels=True,
                show_plot=False
            )
            return save_path
            
        except ImportError:
            logging.warning("Warning: Visualization requires matplotlib. Skipping visualization.")
            return None
        except Exception as e:
            logging.warning(f"Warning: Visualization failed: {e}")
            return None
    
    def visualize_sample_paths(self, dataset: Dict, num_samples: int = 3, save_path: str = None):
        """Visualize sample navigation paths on the maze."""
        if not self.maze_data:
            logging.warning("No maze data available for visualization")
            return None
        
        try:
            from maze_nav_visualizer import MazeNavVisualizer
            
            # Get sample sequences
            train_sequences = dataset['train']['sequences']
            if not train_sequences:
                logging.warning("No training sequences available")
                return None
            
            # Sample from different lengths
            samples = []
            short_seqs = [s for s in train_sequences if s['length'] <= 10]
            medium_seqs = [s for s in train_sequences if 10 < s['length'] <= 20]
            long_seqs = [s for s in train_sequences if s['length'] > 20]
            
            if short_seqs: samples.append(short_seqs[0])
            if medium_seqs: samples.append(medium_seqs[0])
            if long_seqs: samples.append(long_seqs[0])
            
            visualizer = MazeNavVisualizer(figsize=(20, 8))
            
            # Create detailed visualizations for each sample
            for i, seq_data in enumerate(samples[:num_samples]):
                detail_path = save_path.replace('.png', f'_detail_{i+1}.png') if save_path else None
                visualizer.visualize_navigation_sequence(
                    self.maze_data, seq_data, 
                    save_path=detail_path, show_plot=False
                )
            
            return save_path
            
        except Exception as e:
            logging.warning(f"Warning: Sample paths visualization failed: {e}")
            return None


def _find_shortest_path_static(start: int, end: int, adj_matrix: np.ndarray, num_nodes: int) -> Optional[List[int]]:
    """Static version of find_shortest_path for multiprocessing."""
    if start == end:
        return [start]
    
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    
    while queue:
        current = queue.popleft()
        # Find neighbors using the adjacency matrix
        neighbors = np.where(adj_matrix[current] == 1)[0]
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
                
                if neighbor == end:
                    path = []
                    node = end
                    while node is not None:
                        path.append(node)
                        node = parent.get(node)
                    return path[::-1]
    return None

def _path_to_sequence_static(path: List[int], maze_size: int, direction_to_offset: dict) -> List:
    """Static version of _path_to_sequence for multiprocessing."""
    if len(path) < 2:
        return []
    
    sequence = [path[0], path[-1], path[0]]  # start, end, start
    
    for i in range(len(path) - 1):
        from_row, from_col = path[i] // maze_size, path[i] % maze_size
        to_row, to_col = path[i+1] // maze_size, path[i+1] % maze_size
        offset = (to_row - from_row, to_col - from_col)
        
        for direction, dir_offset in direction_to_offset.items():
            if offset == dir_offset:
                sequence.extend([direction, path[i + 1]])
                break
    
    sequence.append("\n")
    return sequence

def _process_pair_worker(start, end, adj_matrix, num_nodes, maze_size, direction_to_offset):
    """Worker function for parallel sequence generation."""
    path = _find_shortest_path_static(start, end, adj_matrix, num_nodes)
    if path and len(path) > 1:
        # Convert numpy integers in path to standard Python integers for JSON serialization
        path = [int(node) for node in path]
        sequence = _path_to_sequence_static(path, maze_size, direction_to_offset)
        return {
            'start_node': start,
            'end_node': end,
            'path': path,
            'sequence': sequence,
            'length': len(sequence)
        }
    return None


def _generate_random_path_free_world_static(start: int, end: int, maze_size: int) -> List[int]:
    """
    Static version of _generate_random_path_free_world for multiprocessing.
    Generates a self-avoiding random walk. If the walk gets stuck, it completes
    the path with a deterministic shortest walk to the target.
    """
    if start == end:
        return [start]

    path = [start]
    visited_in_path = {start}

    current_row, current_col = start // maze_size, start % maze_size
    end_row, end_col = end // maze_size, end % maze_size

    max_steps = maze_size ** 4
    direction_to_offset = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}

    for step_count in range(max_steps):
        if (current_row, current_col) == (end_row, end_col):
            break  # Target reached

        # Find valid, unvisited neighbors
        possible_moves = []
        for direction, (dr, dc) in direction_to_offset.items():
            new_row, new_col = current_row + dr, current_col + dc
            new_node = new_row * maze_size + new_col
            if 0 <= new_row < maze_size and 0 <= new_col < maze_size and new_node not in visited_in_path:
                possible_moves.append((direction, new_row, new_col, new_node))

        if not possible_moves:
            # The self-avoiding walk is trapped
            break

        # Weight moves towards the target
        weighted_moves = []
        for direction, new_row, new_col, new_node in possible_moves:
            dist_to_target = abs(new_row - end_row) + abs(new_col - end_col)
            current_dist = abs(current_row - end_row) + abs(current_col - end_col)
            weight = 3 if dist_to_target < current_dist else 1
            weighted_moves.extend([(direction, new_row, new_col, new_node)] * weight)

        if not weighted_moves:
            # This can happen if all moves increase distance, pick one at random
            direction, new_row, new_col, new_node = random.choice(possible_moves)
        else:
            direction, new_row, new_col, new_node = random.choice(weighted_moves)

        current_row, current_col = new_row, new_col
        path.append(new_node)
        visited_in_path.add(new_node)

    # The path ends where it ends, either by reaching the target or by getting trapped.
    # The deterministic rescue walk is no longer needed with the new logic.
    return path


def _process_free_world_pair_worker(start, end, maze_size, direction_to_offset):
    """Worker function for parallel free world sequence generation."""
    path = _generate_random_path_free_world_static(start, end, maze_size)
    if path and len(path) > 1:
        # Convert to sequence format
        sequence = _path_to_sequence_static(path, maze_size, direction_to_offset)
        return {
            'start_node': start,
            'end_node': path[-1],  # Use the actual end node from the path
            'path': path,
            'sequence': sequence,
            'length': len(sequence)
        }
    return None


def main():
    """Generate maze navigation training data."""
<<<<<<< HEAD
    config = MazeNavConfig(
        maze_size=10,
        seed=41,
        output_dir="maze_nav_data",
        max_pairs=10000000, 
        train_ratio=0.8
=======
    # load config from maze_gen_config.py
    from maze_gen_config import (
        maze_size, max_pairs, train_ratio, seed, output_dir, algorithm,
        free_world_mode, num_random_paths, free_world_output_dir
    )
    config = MazeNavConfig()
    config.maze_size = maze_size
    config.max_pairs = max_pairs
    config.train_ratio = train_ratio
    config.seed = seed
    config.algorithm = algorithm
    config.free_world_mode = free_world_mode
    config.num_random_paths = num_random_paths
    
    # Use different output directory for free world mode
    if free_world_mode:
        config.output_dir = free_world_output_dir
    else:
        config.output_dir = output_dir
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Setup Logging
    log_file = os.path.join(config.output_dir, f"generation_{config.maze_size}x{config.maze_size}_{time.strftime('%Y%m%d-%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
>>>>>>> Debug
    )
    
    logging.info("="*50)
    logging.info("STARTING MAZE NAVIGATION DATASET GENERATION")
    logging.info("="*50)
    logging.info("Configuration:")
    for key, value in vars(config).items():
        logging.info(f"  {key}: {value}")
    logging.info("-" * 50)

    generator = MazeNavDataGenerator(config)
    dataset = generator.generate_data()
    train_tokens, val_tokens = generator.save_all_files(dataset)
    
    # Generate visualizations
    logging.info("\nGenerating visualizations...")
    
    # Maze visualization with node labels
    maze_viz_path = os.path.join(config.output_dir, f"maze_with_labels_{config.maze_size}x{config.maze_size}.png")
    generator.visualize_maze(save_path=maze_viz_path)
    
    # Sample paths visualization
    paths_viz_path = os.path.join(config.output_dir, f"sample_paths_{config.maze_size}x{config.maze_size}.png")
    generator.visualize_sample_paths(dataset, num_samples=3, save_path=paths_viz_path)
    
    # Print summary
    summary = "\n" + "="*50
    summary += f"\n{'FREE WORLD' if config.free_world_mode else 'MAZE'} NAVIGATION DATASET SUMMARY"
    summary += "\n" + "="*50
    summary += f"\nMode: {'Free World (No Walls)' if config.free_world_mode else 'Maze'}"
    summary += f"\nGrid size: {config.maze_size}x{config.maze_size}"
    summary += f"\nTotal nodes: {dataset['maze_data']['num_nodes']}"
    summary += f"\nVocabulary size: {dataset['config']['vocab_size']}"
    if config.free_world_mode:
        summary += f"\nRandom paths generated: {config.num_random_paths}"
    summary += f"\nTraining sequences: {dataset['stats']['train_pairs']}"
    summary += f"\nTest sequences: {dataset['stats']['test_pairs']}"
    summary += f"\nAverage sequence length: {dataset['stats']['avg_train_length']:.1f}"
    summary += f"\nMax sequence length: {dataset['stats']['max_sequence_length']}"
    summary += f"\nTraining tokens: {train_tokens:,}"
    summary += f"\nValidation tokens: {val_tokens:,}"
    summary += f"\nFiles saved to: {config.output_dir}/"
    
    # Show visualization info
    summary += f"\n\nVisualizations generated:"
    summary += f"\n  Maze with node labels: {maze_viz_path}"
    summary += f"\n  Sample navigation paths:"
    for i in range(3):
        detail_path = paths_viz_path.replace('.png', f'_detail_{i+1}.png')
        summary += f"\n    - Navigation detail {i+1}: {detail_path}"
    logging.info(summary)

    logging.info("\n" + "="*50)
    logging.info("DATASET GENERATION COMPLETE")
    logging.info("="*50)


if __name__ == "__main__":
    main() 