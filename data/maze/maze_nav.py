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
from typing import List, Dict, Tuple, Set, Optional
from collections import deque
from dataclasses import dataclass
import itertools

# Add the maze directory to path to import existing modules
maze_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'maze')
sys.path.append(maze_path)
from maze_generator import MazeGenerator, MazeConfig
from maze_solver import MazeSolver


@dataclass
class MazeNavConfig:
    """Configuration for maze navigation training data generation."""
    maze_size: int = 50  # Size of the maze (50x50 = 2500 nodes)
    seed: Optional[int] = None
    train_test_split: float = 0.5  # 50% train, 50% test
    output_dir: str = "maze_nav_data"
    max_pairs: Optional[int] = None  # Limit number of pairs if needed
    vocab_encoding: str = "mixed"  # "numeric" or "mixed" (numbers + words)


class MazeNavDataGenerator:
    """Generates training data for maze navigation GPT model."""
    
    def __init__(self, config: MazeNavConfig):
        self.config = config
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
        
        # Vocabulary setup
        self.direction_tokens = ["up", "right", "down", "left"]
        self.direction_to_offset = {
            "up": (-1, 0),
            "right": (0, 1), 
            "down": (1, 0),
            "left": (0, -1)
        }
        
        # Will be set when maze is generated
        self.maze_data = None
        self.vocab_size = None
        self.token_to_id = {}
        self.id_to_token = {}
        
    def generate_large_maze(self) -> Dict:
        """Generate a large complex maze."""
        print(f"Generating {self.config.maze_size}x{self.config.maze_size} maze...")
        
        maze_config = MazeConfig(
            size=self.config.maze_size,
            seed=self.config.seed
        )
        generator = MazeGenerator(maze_config)
        self.maze_data = generator.generate_maze()
        
        print(f"Generated maze with {self.maze_data['num_nodes']} nodes and {self.maze_data['num_edges']} edges")
        return self.maze_data
    
    def setup_vocabulary(self):
        """Setup vocabulary mapping for the GPT model."""
        # Node IDs: 0 to num_nodes-1
        num_nodes = self.maze_data['num_nodes']
        
        if self.config.vocab_encoding == "numeric":
            # Pure numeric encoding: nodes are 0-(N-1), directions are N, N+1, N+2, N+3
            vocab = list(range(num_nodes)) + list(range(num_nodes, num_nodes + 4))
            self.token_to_id = {i: i for i in vocab}
            self.id_to_token = {i: i for i in vocab}
            # Map direction names to numeric IDs
            self.direction_name_to_id = {
                "up": num_nodes,
                "right": num_nodes + 1,
                "down": num_nodes + 2,
                "left": num_nodes + 3
            }
        else:
            # Mixed encoding: keep nodes as numbers, directions as strings
            vocab = list(range(num_nodes)) + self.direction_tokens
            self.token_to_id = {token: i for i, token in enumerate(vocab)}
            self.id_to_token = {i: token for token, i in self.token_to_id.items()}
            self.direction_name_to_id = {
                name: self.token_to_id[name] for name in self.direction_tokens
            }
        
        self.vocab_size = len(vocab)
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Node tokens: 0-{num_nodes-1}")
        print(f"Direction tokens: {self.direction_tokens}")
        
        # Create encode and decode functions for compatibility with sample.py
        if self.config.vocab_encoding == "numeric":
            self.encode = lambda tokens: [token if isinstance(token, int) else self.direction_name_to_id[token] for token in tokens]
            self.decode = lambda token_ids: [self.id_to_token[token_id] for token_id in token_ids]
        else:
            self.encode = lambda tokens: [self.token_to_id[token] for token in tokens]
            self.decode = lambda token_ids: [self.id_to_token[token_id] for token_id in token_ids]
    
    def find_all_reachable_pairs(self) -> List[Tuple[int, int]]:
        """Find all pairs of nodes that are reachable from each other."""
        print("Finding all reachable node pairs...")
        
        # Use BFS from each node to find all reachable nodes
        adj_matrix = np.array(self.maze_data['adjacency_matrix'])
        num_nodes = self.maze_data['num_nodes']
        
        all_pairs = []
        
        for start_node in range(num_nodes):
            # BFS to find all reachable nodes from start_node
            visited = set()
            queue = deque([start_node])
            visited.add(start_node)
            reachable = []
            
            while queue:
                current = queue.popleft()
                reachable.append(current)
                
                # Find neighbors
                for neighbor in range(num_nodes):
                    if adj_matrix[current][neighbor] == 1 and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Add pairs (start_node, end_node) for all reachable nodes except self
            for end_node in reachable:
                if start_node != end_node:
                    all_pairs.append((start_node, end_node))
        
        print(f"Found {len(all_pairs)} total reachable pairs")
        
        # Limit pairs if specified
        if self.config.max_pairs and len(all_pairs) > self.config.max_pairs:
            print(f"Limiting to {self.config.max_pairs} pairs")
            all_pairs = random.sample(all_pairs, self.config.max_pairs)
        
        return all_pairs
    
    def compute_shortest_path(self, start_node: int, end_node: int) -> Optional[List[int]]:
        """Compute shortest path between two nodes using BFS."""
        if start_node == end_node:
            return [start_node]
        
        adj_matrix = np.array(self.maze_data['adjacency_matrix'])
        num_nodes = self.maze_data['num_nodes']
        
        # BFS for shortest path
        queue = deque([start_node])
        visited = {start_node}
        parent = {start_node: None}
        
        while queue:
            current = queue.popleft()
            
            # Find neighbors
            for neighbor in range(num_nodes):
                if adj_matrix[current][neighbor] == 1 and neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
                    
                    if neighbor == end_node:
                        # Reconstruct path
                        path = []
                        node = end_node
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        return path[::-1]  # Reverse to get start-to-end
        
        return None  # No path found
    
    def node_to_position(self, node_id: int) -> Tuple[int, int]:
        """Convert node ID to (row, col) position."""
        return node_id // self.config.maze_size, node_id % self.config.maze_size
    
    def get_movement_direction(self, from_node: int, to_node: int) -> str:
        """Get movement direction from one node to another."""
        from_row, from_col = self.node_to_position(from_node)
        to_row, to_col = self.node_to_position(to_node)
        
        offset = (to_row - from_row, to_col - from_col)
        
        for direction, dir_offset in self.direction_to_offset.items():
            if offset == dir_offset:
                return direction
        
        raise ValueError(f"Invalid movement from node {from_node} to {to_node}")
    
    def path_to_sequence(self, path: List[int]) -> List:
        """Convert a path to GPT training sequence format."""
        if len(path) < 2:
            return []
        
        start_node = path[0]
        end_node = path[-1]
        
        # Start with [start_node] [end_node] [start_node]
        sequence = [start_node, end_node, start_node]
        
        # Add [move] [intermediate_node] pairs for each step
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            
            # Get movement direction
            movement = self.get_movement_direction(current_node, next_node)
            
            # Add movement token (convert to ID if using numeric encoding)
            if self.config.vocab_encoding == "numeric":
                movement_id = self.direction_name_to_id[movement]
                sequence.append(movement_id)
            else:
                sequence.append(movement)
            
            # Add the resulting node
            sequence.append(next_node)
        
        return sequence
    
    def split_train_test(self, all_pairs: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Split pairs into train and test sets with special rules."""
        print("Splitting pairs into train and test sets...")
        
        adj_matrix = np.array(self.maze_data['adjacency_matrix'])
        
        # Find directly connected pairs (distance = 1)
        directly_connected = []
        other_pairs = []
        
        for start, end in all_pairs:
            if adj_matrix[start][end] == 1:  # Directly connected
                directly_connected.append((start, end))
            else:
                other_pairs.append((start, end))
        
        print(f"Directly connected pairs: {len(directly_connected)}")
        print(f"Other pairs: {len(other_pairs)}")
        
        # All directly connected pairs go to training
        train_pairs = directly_connected.copy()
        
        # Split remaining pairs 50/50
        random.shuffle(other_pairs)
        split_point = len(other_pairs) // 2
        train_pairs.extend(other_pairs[:split_point])
        test_pairs = other_pairs[split_point:]
        
        print(f"Train pairs: {len(train_pairs)}")
        print(f"Test pairs: {len(test_pairs)}")
        
        return train_pairs, test_pairs
    
    def generate_training_data(self) -> Dict:
        """Generate complete training dataset."""
        print("Generating maze navigation training data...")
        
        # Generate maze
        self.generate_large_maze()
        self.setup_vocabulary()
        
        # Find all reachable pairs
        all_pairs = self.find_all_reachable_pairs()
        
        # Split into train/test
        train_pairs, test_pairs = self.split_train_test(all_pairs)
        
        # Generate sequences for training set
        print("Generating training sequences...")
        train_sequences = []
        for i, (start, end) in enumerate(train_pairs):
            if i % 1000 == 0:
                print(f"Processing training pair {i+1}/{len(train_pairs)}")
            
            path = self.compute_shortest_path(start, end)
            if path:
                sequence = self.path_to_sequence(path)
                if sequence:
                    train_sequences.append({
                        'start_node': start,
                        'end_node': end,
                        'path': path,
                        'sequence': sequence,
                        'length': len(sequence)
                    })
        
        # Generate sequences for test set
        print("Generating test sequences...")
        test_sequences = []
        for i, (start, end) in enumerate(test_pairs):
            if i % 1000 == 0:
                print(f"Processing test pair {i+1}/{len(test_pairs)}")
            
            path = self.compute_shortest_path(start, end)
            if path:
                sequence = self.path_to_sequence(path)
                if sequence:
                    test_sequences.append({
                        'start_node': start,
                        'end_node': end,
                        'path': path,
                        'sequence': sequence,
                        'length': len(sequence)
                    })
        
        # Store sequences for visualization
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        
        # Create dataset
        dataset = {
            'config': {
                'maze_size': self.config.maze_size,
                'seed': self.config.seed,
                'vocab_size': self.vocab_size,
                'vocab_encoding': self.config.vocab_encoding
            },
            'maze_data': self.maze_data,
            'vocabulary': {
                'token_to_id': self.token_to_id,
                'id_to_token': self.id_to_token,
                'direction_tokens': self.direction_tokens,
                'direction_name_to_id': self.direction_name_to_id
            },
            'train': {
                'sequences': train_sequences,
                'count': len(train_sequences)
            },
            'test': {
                'sequences': test_sequences,
                'count': len(test_sequences)
            },
            'stats': {
                'total_pairs': len(all_pairs),
                'train_pairs': len(train_pairs),
                'test_pairs': len(test_pairs),
                'avg_train_length': np.mean([s['length'] for s in train_sequences]) if train_sequences else 0,
                'avg_test_length': np.mean([s['length'] for s in test_sequences]) if test_sequences else 0,
                'max_sequence_length': max([s['length'] for s in train_sequences + test_sequences]) if (train_sequences or test_sequences) else 0
            }
        }
        
        print(f"Generated {len(train_sequences)} training sequences")
        print(f"Generated {len(test_sequences)} test sequences")
        print(f"Average training sequence length: {dataset['stats']['avg_train_length']:.1f}")
        print(f"Average test sequence length: {dataset['stats']['avg_test_length']:.1f}")
        print(f"Maximum sequence length: {dataset['stats']['max_sequence_length']}")
        
        return dataset
    
    def save_dataset(self, dataset: Dict, filename: str = "maze_nav_dataset.json"):
        """Save dataset to file."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        filepath = os.path.join(self.config.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Dataset saved to: {filepath}")
        return filepath
    
    def save_meta_pkl(self, dataset: Dict):
        """Save meta.pkl file for compatibility with train.py and sample.py."""
        print("Saving meta.pkl file...")
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        meta_path = os.path.join(self.config.output_dir, 'meta.pkl')
        
        # Create stoi and itos mappings compatible with sample.py
        if self.config.vocab_encoding == "numeric":
            # For numeric encoding, map numbers to themselves and directions to their IDs
            stoi = {}
            itos = {}
            
            # Add nodes
            for i in range(self.maze_data['num_nodes']):
                stoi[str(i)] = i
                itos[i] = str(i)
            
            # Add directions
            for direction, idx in self.direction_name_to_id.items():
                stoi[direction] = idx
                itos[idx] = direction
        else:
            # For mixed encoding, use the existing token mappings
            stoi = {}
            itos = {}
            
            for token, token_id in self.token_to_id.items():
                stoi[str(token)] = token_id
                itos[token_id] = str(token)
        
        meta = {
            'vocab_size': self.vocab_size,
            'stoi': stoi,
            'itos': itos,
            'config': dataset['config'],
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'direction_tokens': self.direction_tokens
        }
        
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        print(f"Meta file saved to: {meta_path}")
        return meta_path
    
    def save_binary_data(self, dataset: Dict):
        """Save training data in binary format compatible with train.py."""
        print("Saving binary training data...")
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Convert sequences to token IDs and flatten them
        def sequences_to_binary(sequences, filename):
            all_tokens = []
            
            for seq_data in sequences:
                sequence = seq_data['sequence']
                # Convert to token IDs
                token_ids = self.encode(sequence)
                all_tokens.extend(token_ids)
            
            if all_tokens:
                # Convert to numpy array with appropriate dtype
                max_token_id = max(all_tokens)
                if max_token_id < 65536:  # uint16 range
                    dtype = np.uint16
                else:
                    dtype = np.uint32
                
                tokens_array = np.array(all_tokens, dtype=dtype)
                filepath = os.path.join(self.config.output_dir, filename)
                tokens_array.tofile(filepath)
                print(f"Saved {len(all_tokens)} tokens to {filepath}")
                return len(all_tokens)
            else:
                print(f"No tokens to save for {filename}")
                return 0
        
        # Save train and validation data
        train_tokens = sequences_to_binary(dataset['train']['sequences'], 'train.bin')
        val_tokens = sequences_to_binary(dataset['test']['sequences'], 'val.bin')
        
        print(f"Training tokens: {train_tokens:,}")
        print(f"Validation tokens: {val_tokens:,}")
        
        return train_tokens, val_tokens
    
    def create_pytorch_dataset(self, dataset: Dict) -> Dict:
        """Create PyTorch-ready dataset format."""
        print("Creating PyTorch-ready dataset...")
        
        def sequences_to_tensors(sequences):
            # Convert sequences to list of token ID lists
            tensor_data = []
            for seq_data in sequences:
                sequence = seq_data['sequence']
                # Convert tokens to IDs if needed
                if self.config.vocab_encoding == "mixed":
                    token_ids = [self.token_to_id[token] for token in sequence]
                else:
                    token_ids = sequence  # Already numeric
                tensor_data.append(token_ids)
            return tensor_data
        
        pytorch_dataset = {
            'vocab_size': dataset['vocabulary']['vocab_size'] if 'vocab_size' in dataset['vocabulary'] else self.vocab_size,
            'token_to_id': dataset['vocabulary']['token_to_id'],
            'id_to_token': dataset['vocabulary']['id_to_token'],
            'train_data': sequences_to_tensors(dataset['train']['sequences']),
            'test_data': sequences_to_tensors(dataset['test']['sequences']),
            'config': dataset['config'],
            'stats': dataset['stats']
        }
        
        return pytorch_dataset
    
    def visualize_maze_with_labels(self, save_path: str = None, show_paths: bool = False) -> Optional[str]:
        """Create a detailed visualization of the maze with node labels."""
        if not self.maze_data:
            print("No maze data available for visualization")
            return None
        
        try:
            from maze_nav_visualizer import MazeNavVisualizer
            
            visualizer = MazeNavVisualizer(figsize=(16, 16))
            
            # Create visualization with node labels
            fig = visualizer.visualize_maze(
                self.maze_data, 
                title=f"Maze with Node Labels (Size: {self.config.maze_size}x{self.config.maze_size})",
                save_path=save_path,
                show_grid=True,
                show_node_labels=True,  # Enable node labels
                show_plot=False  # Don't show plot immediately
            )
            
            return save_path
            
        except ImportError:
            print("Warning: Visualization requires matplotlib. Skipping visualization.")
            return None
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
            return None
    
    def visualize_sample_paths(self, num_samples: int = 3, save_path: str = None) -> Optional[str]:
        """Visualize sample navigation paths on the maze with detailed analysis."""
        if not self.maze_data:
            print("No maze data available for visualization")
            return None
        
        try:
            from maze_nav_visualizer import MazeNavVisualizer
            
            # Get some sample training sequences
            sample_sequences = []
            if hasattr(self, 'train_sequences') and self.train_sequences:
                # Take samples from different length ranges
                short_seqs = [s for s in self.train_sequences if s['length'] <= 10]
                medium_seqs = [s for s in self.train_sequences if 10 < s['length'] <= 20]
                long_seqs = [s for s in self.train_sequences if s['length'] > 20]
                
                # Sample from each category
                samples = []
                if short_seqs: samples.append(short_seqs[0])
                if medium_seqs: samples.append(medium_seqs[0])
                if long_seqs: samples.append(long_seqs[0])
                
                sample_sequences = samples[:num_samples]
            
            if not sample_sequences:
                print("No sample sequences available")
                return None
            
            visualizer = MazeNavVisualizer(figsize=(20, 8))
            
            # Create individual detailed visualizations for each sample
            for i, seq_data in enumerate(sample_sequences):
                individual_path = save_path.replace('.png', f'_detail_{i+1}.png') if save_path else None
                visualizer.visualize_navigation_sequence(
                    self.maze_data, seq_data, 
                    save_path=individual_path, show_plot=False
                )
            
            return save_path
            
        except Exception as e:
            print(f"Warning: Sample paths visualization failed: {e}")
            return None
    
    def create_comprehensive_visualization(self) -> Optional[str]:
        """Create comprehensive training dataset visualization."""
        if not hasattr(self, 'train_sequences'):
            print("No training sequences available for comprehensive visualization")
            return None
        
        try:
            from maze_nav_visualizer import create_comprehensive_visualization
            
            # Build dataset for visualization
            dataset = {
                'maze_data': self.maze_data,
                'config': {
                    'vocab_size': len(set().union(*[seq['sequence'] for seq in self.train_sequences])),
                    'seed': self.config.seed
                },
                'train': {'sequences': self.train_sequences},
                'test': {'sequences': getattr(self, 'test_sequences', [])},
                'stats': {
                    'train_pairs': len(self.train_sequences),
                    'test_pairs': len(getattr(self, 'test_sequences', [])),
                    'total_pairs': len(self.train_sequences) + len(getattr(self, 'test_sequences', [])),
                    'avg_train_length': sum(s['length'] for s in self.train_sequences) / len(self.train_sequences),
                    'avg_test_length': (sum(s['length'] for s in getattr(self, 'test_sequences', [])) / 
                                       len(getattr(self, 'test_sequences', []))) if getattr(self, 'test_sequences', []) else 0,
                    'max_sequence_length': max(s['length'] for s in self.train_sequences)
                }
            }
            
            create_comprehensive_visualization('temp_dataset.json', self.config.output_dir)
            
            # Save temporary dataset
            import json
            temp_path = os.path.join(self.config.output_dir, 'temp_dataset.json')
            with open(temp_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            create_comprehensive_visualization(temp_path, self.config.output_dir)
            
            # Clean up temp file
            os.remove(temp_path)
            
            return self.config.output_dir
            
        except Exception as e:
            print(f"Warning: Comprehensive visualization failed: {e}")
            return None


def main():
    """Main function to generate maze navigation training data."""
    # Configuration
    config = MazeNavConfig(
        maze_size=10,  # Start with smaller maze for testing
        seed=42,
        train_test_split=0.5,
        output_dir="maze_nav_data",
        max_pairs=1000,  # Smaller for testing
        vocab_encoding="mixed"  # "numeric" or "mixed"
    )
    
    # Generate data
    generator = MazeNavDataGenerator(config)
    dataset = generator.generate_training_data()
    
    # Save dataset files
    dataset_path = generator.save_dataset(dataset)
    
    # Save meta.pkl for train.py compatibility
    meta_path = generator.save_meta_pkl(dataset)
    
    # Save binary data for train.py compatibility
    train_tokens, val_tokens = generator.save_binary_data(dataset)
    
    # Create PyTorch-ready version
    pytorch_dataset = generator.create_pytorch_dataset(dataset)
    pytorch_path = generator.save_dataset(pytorch_dataset, "maze_nav_pytorch.json")
    
    # Generate visualizations
    print("\nGenerating comprehensive visualizations...")
    
    # Create maze visualization with node labels
    maze_viz_path = os.path.join(config.output_dir, f"maze_with_labels_{config.maze_size}x{config.maze_size}.png")
    generator.visualize_maze_with_labels(save_path=maze_viz_path)
    
    # Create comprehensive training overview
    generator.create_comprehensive_visualization()
    
    # Create detailed sample paths visualization
    paths_viz_path = os.path.join(config.output_dir, f"sample_paths_{config.maze_size}x{config.maze_size}.png")
    generator.visualize_sample_paths(num_samples=3, save_path=paths_viz_path)
    
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
    print(f"Dataset saved to: {dataset_path}")
    print(f"Meta file saved to: {meta_path}")
    print(f"Binary training data: train.bin ({train_tokens:,} tokens)")
    print(f"Binary validation data: val.bin ({val_tokens:,} tokens)")
    print(f"PyTorch dataset saved to: {pytorch_path}")
    
    # Show visualization info
    print(f"\nVisualizations generated:")
    print(f"  Maze with node labels: {maze_viz_path}")
    print(f"  Training overview: {os.path.join(config.output_dir, 'training_overview.png')}")
    print(f"  Detailed navigation examples:")
    for i in range(3):
        detail_path = paths_viz_path.replace('.png', f'_detail_{i+1}.png')
        if os.path.exists(detail_path):
            print(f"    - Navigation detail {i+1}: {detail_path}")
    
    # Show example sequences
    print("\nExample training sequences:")
    for i, seq in enumerate(dataset['train']['sequences'][:3]):
        print(f"  {i+1}: {seq['sequence']}")
        print(f"     Start: {seq['start_node']}, End: {seq['end_node']}, Length: {seq['length']}")


if __name__ == "__main__":
    main() 