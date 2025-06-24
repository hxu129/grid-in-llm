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
    algorithm: str = "wilson" # "wilson" or "dfs"


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
        print(f"Generating {self.config.maze_size}x{self.config.maze_size} maze...")
        
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
        
        print(f"Generated maze with {num_nodes} nodes, vocab size: {self.vocab_size}")
        
        # Find all reachable pairs
        all_pairs = self._find_reachable_pairs()
        if self.config.max_pairs and len(all_pairs) > self.config.max_pairs:
            all_pairs = random.sample(all_pairs, self.config.max_pairs)
        
        # Split train/test: direct connections go to train, rest split 50/50
        train_pairs, test_pairs = self._split_pairs(all_pairs)
        
        # Generate sequences
        train_sequences = self._generate_sequences(train_pairs)
        test_sequences = self._generate_sequences(test_pairs)
        
        # Create dataset
        dataset = {
            'config': {
                'maze_size': self.config.maze_size,
                'seed': self.config.seed,
                'vocab_size': self.vocab_size,
                'vocab_encoding': 'mixed'  # For compatibility
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
        
        print(f"Generated {len(train_sequences)} train, {len(test_sequences)} test sequences")
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
        print(f"Generating sequences in parallel with {num_processes} processes...")

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
            print("No maze data available for visualization")
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
            print("Warning: Visualization requires matplotlib. Skipping visualization.")
            return None
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
            return None
    
    def visualize_sample_paths(self, dataset: Dict, num_samples: int = 3, save_path: str = None):
        """Visualize sample navigation paths on the maze."""
        if not self.maze_data:
            print("No maze data available for visualization")
            return None
        
        try:
            from maze_nav_visualizer import MazeNavVisualizer
            
            # Get sample sequences
            train_sequences = dataset['train']['sequences']
            if not train_sequences:
                print("No training sequences available")
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
            print(f"Warning: Sample paths visualization failed: {e}")
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


def main():
    """Generate maze navigation training data."""
    # load config from maze_gen_config.py
    from maze_gen_config import maze_size, max_pairs, train_ratio, seed, output_dir, algorithm
    config = MazeNavConfig()
    config.maze_size = maze_size
    config.max_pairs = max_pairs
    config.train_ratio = train_ratio
    config.seed = seed
    config.output_dir = output_dir
    config.algorithm = algorithm
    
    generator = MazeNavDataGenerator(config)
    dataset = generator.generate_data()
    train_tokens, val_tokens = generator.save_all_files(dataset)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Maze visualization with node labels
    maze_viz_path = os.path.join(config.output_dir, f"maze_with_labels_{config.maze_size}x{config.maze_size}.png")
    generator.visualize_maze(save_path=maze_viz_path)
    
    # Sample paths visualization
    paths_viz_path = os.path.join(config.output_dir, f"sample_paths_{config.maze_size}x{config.maze_size}.png")
    generator.visualize_sample_paths(dataset, num_samples=3, save_path=paths_viz_path)
    
    # Print summary
    print("\n" + "="*50)
    print("MAZE NAVIGATION DATASET SUMMARY")
    print("="*50)
    print(f"Maze size: {config.maze_size}x{config.maze_size}")
    print(f"Total nodes: {dataset['maze_data']['num_nodes']}")
    print(f"Vocabulary size: {dataset['config']['vocab_size']}")
    print(f"Training sequences: {dataset['stats']['train_pairs']}")
    print(f"Test sequences: {dataset['stats']['test_pairs']}")
    print(f"Average sequence length: {dataset['stats']['avg_train_length']:.1f}")
    print(f"Max sequence length: {dataset['stats']['max_sequence_length']}")
    print(f"Training tokens: {train_tokens:,}")
    print(f"Validation tokens: {val_tokens:,}")
    print(f"Files saved to: {config.output_dir}/")
    
    # Show visualization info
    print(f"\nVisualizations generated:")
    print(f"  Maze with node labels: {maze_viz_path}")
    print(f"  Sample navigation paths:")
    for i in range(3):
        detail_path = paths_viz_path.replace('.png', f'_detail_{i+1}.png')
        print(f"    - Navigation detail {i+1}: {detail_path}")


if __name__ == "__main__":
    main() 