import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from contextlib import nullcontext
import random

# Add parent directory to Python path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPTConfig, GPT
import ecco
from typing import Dict, List, Tuple, Optional
import re

class FFNActivationCollector:
    """
    Collects FFN activations from maze navigation model when generating position tokens.
    Focuses on intermediate activations (after GELU, before second linear layer).
    """
    
    def __init__(self, model_path: str = '../out-maze-nav', grid_size: int = 8):
        """
        Initialize the activation collector.
        
        Args:
            model_path: Path to the trained model directory
            grid_size: Size of the maze grid
        """
        self.grid_size = grid_size
        self.n_positions = grid_size * grid_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Get the directory where this script is located
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)  # Parent directory
        
        # Load model and setup
        self.model, self.tokenizer = self._load_model(model_path)
        self.n_layers = self.model.config.n_layer
        self.n_embd = self.model.config.n_embd
        self.ffn_size = 4 * self.n_embd  # Size of FFN intermediate layer
        
        # Storage for activations
        self.position_activations_fc = defaultdict(lambda: defaultdict(list))  # First linear layer (after GELU)
        self.position_activations_proj = defaultdict(lambda: defaultdict(list))  # Second linear layer (after c_proj)
        self.hooks = {}
        
        print(f"Model loaded: {self.n_layers} layers, {self.n_embd} embedding dim, {self.ffn_size} FFN size")
        print(f"Grid size: {grid_size}x{grid_size} = {self.n_positions} positions")
    
    def _load_model(self, model_path: str) -> Tuple[GPT, object]:
        """Load the trained maze navigation model and tokenizer."""
        # Load model (following logit_lens.ipynb pattern)
        ckpt_path = os.path.join(model_path, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        
        # Clean up state dict
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        
        # Load tokenizer metadata
        meta_path = os.path.join(self.project_root, 'data', checkpoint['config']['dataset'], f'meta_{self.grid_size}.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        stoi, itos = meta['stoi'], meta['itos']
        
        def encode(s):
            if isinstance(s, str):
                tokens = s.split()
                return [stoi[token] if token in stoi else int(token) if token.isdigit() else stoi.get(token, 0) for token in tokens]
            return s
        
        def decode(l):
            result = []
            for i in l:
                token = itos.get(i, str(i))
                result.append(token)
            return ' '.join(result)
        
        # Create tokenizer-like object
        class MazeTokenizer:
            def __init__(self, encode_fn, decode_fn):
                self.encode = encode_fn
                self.decode = decode_fn
                
            def convert_ids_to_tokens(self, ids):
                if not isinstance(ids, list):
                    ids = [ids]
                return [itos.get(i, str(i)) for i in ids]
        
        tokenizer = MazeTokenizer(encode, decode)
        
        return model, tokenizer
    
    def _load_validation_data(self, val_path: str = None, 
                            max_samples: int = 100) -> List[List[int]]:
        """Load validation data following logit_lens.ipynb pattern."""
        if val_path is None:
            val_path = os.path.join(self.project_root, "data", "maze", "maze_nav_data", f"train_{self.grid_size}.bin")
        
        val_data = np.fromfile(val_path, dtype=np.uint16)
        
        # Split data by end-of-sequence marker
        eos_marker = self.grid_size ** 2 + 4
        path_indices = np.where(val_data == eos_marker)[0]
        paths = np.split(val_data, path_indices + 1)
        
        # Convert to list of lists, filtering out any empty paths from split
        all_paths = [p.tolist()[:-1] for p in paths if p.size > 1]
        
        # Shuffle and select a subset of paths
        random.shuffle(all_paths)
        return all_paths[:max_samples]
    
    def _register_ffn_hooks(self):
        """Register hooks to capture FFN activations from both linear layers for a whole sequence."""
        self._remove_hooks()  # Clear any existing hooks

        def create_hook(layer_idx, activation_type):
            def hook_fn(module, input, output):
                if self._current_path is None:
                    return

                # Choose the right activation tensor based on the hook type
                if activation_type == 'fc': # Before c_proj (after GELU)
                    activations_tensor = input[0]
                else: # 'proj', after c_proj
                    activations_tensor = output

                activations = activations_tensor.detach().float().cpu().numpy()[0]  # (seq_len, n_neurons)
                path = self._current_path
                
                # Determine which dictionary to store activations in
                target_activations_dict = self.position_activations_fc if activation_type == 'fc' else self.position_activations_proj

                for i in range(activations.shape[0]): # Loop over sequence length
                    target_token_id = path[i + 1]  # The token being predicted
                    if self._is_position_token(target_token_id):
                        position = target_token_id
                        activation_at_pos_i = activations[i, :]
                        layer_key = f"layer_{layer_idx}"
                        target_activations_dict[layer_key][position].append(activation_at_pos_i.copy())
            
            return hook_fn
        
        # Register hooks for both linear layers in each transformer layer
        for layer_idx in range(self.n_layers):
            # Hook for first linear layer (input to c_proj = after GELU)
            fc_layer_name = f"transformer.h.{layer_idx}.mlp.c_proj"
            fc_module = dict(self.model.named_modules())[fc_layer_name]
            fc_hook = fc_module.register_forward_hook(create_hook(layer_idx, 'fc'))
            self.hooks[fc_layer_name + "_fc"] = fc_hook
            
            # Hook for second linear layer (output of c_proj)
            proj_layer_name = f"transformer.h.{layer_idx}.mlp.c_proj"
            proj_module = dict(self.model.named_modules())[proj_layer_name]
            proj_hook = proj_module.register_forward_hook(create_hook(layer_idx, 'proj'))
            self.hooks[proj_layer_name + "_proj"] = proj_hook
            
        print(f"Registered hooks for {len(self.hooks)} FFN operations ({self.n_layers} layers × 2 linear operations each)")
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks = {}
    
    def _is_position_token(self, token_id: int) -> bool:
        """Check if a token represents a grid position (0 to grid_size^2-1)."""
        token_str = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        try:
            pos = int(token_str)
            return 0 <= pos < self.n_positions
        except (ValueError, TypeError):
            return False
    
    def collect_activations(self, max_samples: int = 100):
        """
        Main method to collect FFN activations. Processes full paths at once.
        
        Args:
            max_samples: Maximum number of validation samples to process.
        """
        print("Loading validation data...")
        validation_paths = self._load_validation_data(max_samples=max_samples)
        
        print(f"Processing {len(validation_paths)} validation paths...")
        
        # Register hooks
        self._register_ffn_hooks()
        
        processed_samples = 0
        self._current_path = None
        
        try:
            with torch.no_grad():
                for sample_idx, path in enumerate(validation_paths):
                    if sample_idx % 10 == 0:
                        print(f"Processing sample {sample_idx}/{len(validation_paths)}")
                    
                    if len(path) < 2:
                        continue

                    try:
                        # Input is the path sequence, excluding the last token
                        input_ids = torch.tensor([path[:-1]], dtype=torch.long, device=self.device)
                        self._current_path = path  # Provide full path for hooks

                        # Forward pass to trigger hooks
                        if self.device == 'cuda' and torch.cuda.is_bf16_supported():
                            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                                self.model(input_ids=input_ids)
                        else:
                            self.model(input_ids=input_ids)
                        
                        processed_samples += 1
                        
                    except Exception as e:
                        print(f"Warning: Error processing sample {sample_idx}: {e}")
                        continue
        
        finally:
            # Clean up
            self._current_path = None
            self._remove_hooks()
        
        print(f"Processed {processed_samples} samples")
        self._print_collection_summary()
    
    def _print_collection_summary(self):
        """Print summary of collected activations."""
        print("\nActivation Collection Summary:")
        print("=" * 60)
        
        print("First Linear Layer (after GELU):")
        print("-" * 30)
        for layer_key in sorted(self.position_activations_fc.keys()):
            layer_data = self.position_activations_fc[layer_key]
            total_activations = sum(len(acts) for acts in layer_data.values())
            unique_positions = len(layer_data)
            
            print(f"{layer_key}: {total_activations} activations across {unique_positions} unique positions")
            
            # Show position coverage
            if layer_data:
                positions_with_data = sorted(layer_data.keys())
                print(f"  Positions: {positions_with_data[:10]}{'...' if len(positions_with_data) > 10 else ''}")
                
                # Show activation counts per position
                pos_counts = {pos: len(acts) for pos, acts in layer_data.items()}
                avg_count = np.mean(list(pos_counts.values()))
                print(f"  Avg activations per position: {avg_count:.1f}")
        
        print("\nSecond Linear Layer (after c_proj):")
        print("-" * 30)
        for layer_key in sorted(self.position_activations_proj.keys()):
            layer_data = self.position_activations_proj[layer_key]
            total_activations = sum(len(acts) for acts in layer_data.values())
            unique_positions = len(layer_data)
            
            print(f"{layer_key}: {total_activations} activations across {unique_positions} unique positions")
            
            # Show position coverage
            if layer_data:
                positions_with_data = sorted(layer_data.keys())
                print(f"  Positions: {positions_with_data[:10]}{'...' if len(positions_with_data) > 10 else ''}")
                
                # Show activation counts per position
                pos_counts = {pos: len(acts) for pos, acts in layer_data.items()}
                avg_count = np.mean(list(pos_counts.values()))
                print(f"  Avg activations per position: {avg_count:.1f}")
    
    def normalize_and_create_matrices(self, normalization: str = 'z_score') -> Dict[str, np.ndarray]:
        """
        Normalize activations and create position-neuron matrices for both linear layers.
        
        Args:
            normalization: 'z_score', 'min_max', or 'none'
            
        Returns:
            Dictionary mapping layer names to (n_positions, n_neurons) matrices
        """
        print(f"\nCreating position-neuron matrices with {normalization} normalization...")
        
        matrices = {}
        
        # Process first linear layer (after GELU)
        print("Processing first linear layer activations...")
        for layer_key in sorted(self.position_activations_fc.keys()):
            layer_data = self.position_activations_fc[layer_key]
            
            if not layer_data:
                print(f"Warning: No data for {layer_key}_fc")
                continue
            
            # Initialize matrix: (n_positions, n_neurons)
            matrix = np.full((self.n_positions, self.ffn_size), np.nan)
            
            # Fill matrix with averaged activations
            for position, activations_list in layer_data.items():
                if activations_list:
                    # Average multiple activations for the same position
                    avg_activation = np.mean(activations_list, axis=0)
                    matrix[position, :] = avg_activation
            
            # Apply normalization
            matrix = self._apply_normalization(matrix, normalization, self.ffn_size)
            
            # Store matrix
            matrices[f"{layer_key}_fc"] = matrix
            
            # Print statistics
            valid_entries = ~np.isnan(matrix)
            coverage = np.mean(valid_entries) * 100
            print(f"{layer_key}_fc: {matrix.shape} matrix, {coverage:.1f}% coverage")
        
        # Process second linear layer (after c_proj)
        print("Processing second linear layer activations...")
        for layer_key in sorted(self.position_activations_proj.keys()):
            layer_data = self.position_activations_proj[layer_key]
            
            if not layer_data:
                print(f"Warning: No data for {layer_key}_proj")
                continue
            
            # Initialize matrix: (n_positions, n_embd)
            matrix = np.full((self.n_positions, self.n_embd), np.nan)
            
            # Fill matrix with averaged activations
            for position, activations_list in layer_data.items():
                if activations_list:
                    # Average multiple activations for the same position
                    avg_activation = np.mean(activations_list, axis=0)
                    matrix[position, :] = avg_activation
            
            # Apply normalization
            matrix = self._apply_normalization(matrix, normalization, self.n_embd)
            
            # Store matrix
            matrices[f"{layer_key}_proj"] = matrix
            
            # Print statistics
            valid_entries = ~np.isnan(matrix)
            coverage = np.mean(valid_entries) * 100
            print(f"{layer_key}_proj: {matrix.shape} matrix, {coverage:.1f}% coverage")
        
        return matrices
    
    def _apply_normalization(self, matrix: np.ndarray, normalization: str, n_neurons: int) -> np.ndarray:
        """Apply normalization to a matrix."""
        if normalization == 'z_score':
            # Z-score normalization per neuron (across positions)
            for neuron_idx in range(n_neurons):
                neuron_data = matrix[:, neuron_idx]
                valid_data = neuron_data[~np.isnan(neuron_data)]
                if len(valid_data) > 1:
                    mean_val = np.mean(valid_data)
                    std_val = np.std(valid_data)
                    if std_val > 0:
                        matrix[:, neuron_idx] = (neuron_data - mean_val) / std_val
        
        elif normalization == 'min_max':
            # Min-max normalization per neuron
            for neuron_idx in range(n_neurons):
                neuron_data = matrix[:, neuron_idx]
                valid_data = neuron_data[~np.isnan(neuron_data)]
                if len(valid_data) > 1:
                    min_val = np.min(valid_data)
                    max_val = np.max(valid_data)
                    if max_val > min_val:
                        matrix[:, neuron_idx] = (neuron_data - min_val) / (max_val - min_val)
        
        return matrix
    
    def save_results(self, matrices: Dict[str, np.ndarray], 
                    save_path: str = "ffn_position_analysis_results.npz"):
        """Save the analysis results."""
        print(f"\nSaving results to {save_path}...")
        
        # Prepare data for saving
        save_data = {
            'grid_size': self.grid_size,
            'n_positions': self.n_positions,
            'n_layers': self.n_layers,
            'ffn_size': self.ffn_size,
            **matrices  # Unpack all layer matrices
        }
        
        # Save metadata
        metadata = {
            'layer_names': list(matrices.keys()),
            'matrix_shape': f"({self.n_positions}, {self.ffn_size})",
            'description': "FFN neuron activations when generating specific grid positions"
        }
        save_data['metadata'] = metadata
        
        np.savez_compressed(save_path, **save_data)
        print(f"Results saved! Matrices for {len(matrices)} layers")
        
        return save_path

def main():
    """Main execution function."""
    print("Starting FFN Position Analysis...")
    print("=" * 60)
    
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, 'out-maze-nav')
    
    # Initialize collector
    collector = FFNActivationCollector(
        model_path=model_path,
        grid_size=8  # Change this for different maze sizes
    )
    
    # Collect activations
    collector.collect_activations(
        max_samples=500,  # Can use more samples now
    )
    
    # Create normalized matrices
    matrices = collector.normalize_and_create_matrices(normalization='z_score')
    
    # Save results
    save_path = collector.save_results(matrices)
    
    print(f"\nAnalysis complete! Results saved to {save_path}")
    print("\nTo load results later:")
    print(f"data = np.load('{save_path}')")
    print("layer_0_matrix = data['layer_0']")
    
    return matrices, save_path

if __name__ == "__main__":
    matrices, save_path = main() 